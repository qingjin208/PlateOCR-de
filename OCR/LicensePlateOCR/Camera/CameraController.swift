@preconcurrency import AVFoundation
import SwiftUI
import CoreImage
import ImageIO
import os

/// Retains the buffer until asynchronous recognition finishes.
private struct CameraFrame: @unchecked Sendable { let buffer: CVPixelBuffer }

/// Drops frames while recognition is busy instead of creating a backlog, and gates the
/// whole capture callback.
///
/// `enabled` exists so a camera that is idle (tab just opened), paused, or suspended can
/// reject a frame **on the capture queue**, before any `Task { @MainActor }` hop. The
/// previous code hopped to the main actor first and failed a `guard` inside — ~5 pointless
/// main-thread wake-ups per second for as long as the camera tab was on screen, even with
/// detection switched off.
///
/// `OSAllocatedUnfairLock` is Apple's recommended replacement for a bare `NSLock` in Swift
/// concurrency code: it is `Sendable`, and `withLock` cannot leave the lock held across an
/// `await` (which is exactly the bug a manual lock/unlock pair invites here). Also note
/// `os_unfair_lock` must never be used from Swift directly — `OSAllocatedUnfairLock` is the
/// supported wrapper.
private final class FrameGate: @unchecked Sendable {
    private struct State {
        var busy = false
        var lastStart = Date.distantPast
        var enabled = false
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// Cheap read used on the capture queue before touching the buffer.
    var isEnabled: Bool { state.withLock { $0.enabled } }

    /// Mirrors `active && liveEnabled && !paused` onto the capture queue.
    func setEnabled(_ enabled: Bool) {
        state.withLock { $0.enabled = enabled }
    }

    /// Returns true when this frame should be analysed (at most one in flight, ≥0.2 s apart).
    func begin() -> Bool {
        state.withLock { s in
            guard s.enabled, !s.busy,
                  Date().timeIntervalSince(s.lastStart) >= 0.2 else { return false }
            s.busy = true
            s.lastStart = Date()
            return true
        }
    }

    func finish() {
        state.withLock { $0.busy = false }
    }
}

struct CameraRecognitionLogEntry: Identifiable, Equatable {
    /// Stable across vote updates (the aggregator's entry id), so a row is refreshed in
    /// place instead of being torn down and re-created on every analysis cycle.
    let id: UUID
    let date: Date
    let result: PlateResult
    /// Frames that agreed on `result.text`, and all frames that saw this plate.
    let votes: Int
    let totalVotes: Int

    /// Equality is deliberately limited to what the row actually renders — identity,
    /// timestamp, text, counts and the corrected-image instance. `PlateResult` carries a
    /// fresh `UUID` per instance and an array of candidates, so a structural compare would
    /// report a change on *every* analysis cycle; this lets the controller skip the
    /// published assignment (and the whole list's re-render) when nothing visible moved.
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
            && lhs.date == rhs.date
            && lhs.votes == rhs.votes
            && lhs.totalVotes == rhs.totalVotes
            && lhs.result.text == rhs.result.text
            && lhs.result.confidence == rhs.result.confidence
            && lhs.result.correctedPlateImage === rhs.result.correctedPlateImage
    }
}

@MainActor @Observable
final class CameraController: NSObject {
    nonisolated let session = AVCaptureSession()
    private nonisolated let sessionQueue = DispatchQueue(label: "licenseplate.camera.session")
    private nonisolated let frameGate = FrameGate()
    private var recognizer: LicensePlateRecognizing
    private var didSetup = false
    @ObservationIgnored private var configurationTask: Task<Bool, Never>?
    private var active = false
    private var generation = 0
    private var analysisIdentifier = 0
    private var lastPrinted: [String: Date] = [:]
    var isAuthorized = false
    var setupFailed = false
    var liveResults: [PlateResult] = []
    var liveDetections: [PlateDetection] = []
    var recognitionHistory: [CameraRecognitionLogEntry] = []
    var isAnalyzing = false
    var elapsedMilliseconds = 0
    var errorMessage: String?
    /// Live triage values (see the status pill): which detector implementation actually
    /// runs, and the model's best raw anchor score before the confidence threshold. On a
    /// device where the PC script detects the same scene, a Core ML name means the ONNX
    /// path never compiled in / failed to init; a near-zero top score means the model
    /// never saw a plate-like input (preprocessing/orientation), not a threshold problem.
    var detectorName: String = ""
    var lastTopScore: Float?
    /// Analysis cycles started since the app launched. Diagnostic only: together with the
    /// latency pill it separates the two ways the feed can look "dead" — a counter that
    /// stays at 0 means the capture pipeline never delivered a frame the gate accepted
    /// (session/permission/source problem), while a rising counter with 0 ms latency means
    /// every analysis threw before finishing (detector init/run failure).
    var frameCount = 0
    var mode: RecognitionMode = .accurate
    var liveEnabled = true {
        didSet {
            generation += 1
            if !liveEnabled {
                // Close the voting session first: this lists a confident one-frame sighting
                // and drops every live track, so the next session cannot revive a plate the
                // user saw half a minute ago.
                publishHistory(voteAggregator.finish())
                liveResults = []
                liveDetections = []
                isAnalyzing = false
                trackedPlates.removeAll()
                resultTracks.removeAll()
                stableResults = []
                resultCycle = 0
                analysisCycle = 0
                foldedFrameIdentifier = -1
                previewView?.clearOverlay()
            }
            refreshAnalysisGate()
        }
    }
    /// When true, analysis is halted (frozen) but the camera keeps showing the last frame
    /// and the rolling recognition history stays on screen. The detection boxes are cleared
    /// on pause: no further `draw` happens while frozen, so leaving them would show a stale
    /// snapshot on a stopped picture. Toggled by the pause button — distinct from
    /// `liveEnabled`, which fully stops the live feature and clears everything.
    ///
    /// `isAnalyzing` is cleared here rather than in the view: it drives the status pill and
    /// the scan beam, and a paused pipeline must not keep claiming "Live Detection" just because
    /// the button that flipped the flag happened to also reset it (any other caller — a
    /// scene-phase handler, a future auto-pause — would have left it stuck true).
    var paused = false {
        didSet {
            guard paused != oldValue else { return }
            if paused {
                isAnalyzing = false
                // Pause freezes the feed, so no later frame will redraw the overlay.
                // Drop the boxes now instead of leaving a stale snapshot; the rolling
                // recognition-history list is intentionally left untouched.
                previewView?.clearOverlay()
            }
            refreshAnalysisGate()
        }
    }
    weak var previewView: PreviewView?
    /// Diagnostic: echoes the exact frame the SDK analyzed (as-is, rotation included)
    /// into a corner thumbnail, with the same normalized boxes drawn on it. If the box
    /// fits the plate on the echo but not on the preview → preview mapping bug; if the
    /// echo itself is rotated → the output connection's rotation did not apply.
    var debugEchoEnabled = false {
        didSet {
            if !debugEchoEnabled { previewView?.showDebugFrame(image: nil, detections: []) }
        }
    }

    /// Unified-logging handles. `print` writes to stdout, which is slow, has no levels and
    /// is invisible to the Console app; `Logger` + `OSSignposter` is what Apple's logging
    /// guidance recommends, and the signpost interval shows up directly in the Instruments
    /// Points of Interest track — the only reliable way to measure a per-frame change.
    private nonisolated static let logger = Logger(subsystem: "com.itlogica.PlateOCR",
                                                   category: "recognition")
    private nonisolated static let signposter = OSSignposter(subsystem: "com.itlogica.PlateOCR",
                                                             category: "recognition")

    /// Tells the capture queue whether frames are worth analysing at all.
    /// Called from every state change that can start or stop the live pipeline.
    private func refreshAnalysisGate() {
        frameGate.setEnabled(active && liveEnabled && !paused)
    }

    // MARK: - Temporal detection stability
    //
    // The pose model occasionally emits a one-frame phantom box (e.g. a quad regressed
    // onto the bumper *below* the real plate, or onto a plate-like reflection). Showing
    // every raw detection makes those phantoms flash on the live feed. Instead, boxes are
    // tracked across analysis cycles: a box is only drawn once it has been seen in
    // `confirmationHits` consecutive cycles at the same place (IoU association), its
    // geometry is exponentially smoothed, and a track survives `maxMissedCycles`
    // unmatched cycles so stable boxes don't flicker.
    //
    // Expiry is counted in ANALYSIS CYCLES, not wall-clock seconds. A single ONNX +
    // PaddleOCR pass on two plates can take well over 0.45 s on an A14 device; a
    // time-based TTL shorter than that frame latency retires the track before its
    // second sighting, so `hits` can never reach `confirmationHits` and no box is ever
    // drawn. Cycle counting makes confirmation independent of device speed.

    /// One plate box tracked across consecutive analysis cycles.
    private struct TrackedPlate {
        var rect: CGRect
        var quad: [CGPoint]
        var confidence: Float
        var hits: Int
        /// Analysis-cycle index of the last match (never a wall-clock date — see above).
        var lastSeenCycle: Int
    }

    @ObservationIgnored private var trackedPlates: [TrackedPlate] = []
    /// Monotonic counter incremented once per tracker fold (see `stabilizeDetections`).
    @ObservationIgnored private var analysisCycle = 0

    /// OCR is run independently on every frame, so a single blurry/garbled pass used to
    /// replace the label text with a different reading — or with nothing at all. These
    /// tracks stabilise *what the chip says*, mirroring what `trackedPlates` does for the
    /// box geometry. Kept separate because detection and OCR can disagree in a frame.
    private struct TrackedResult {
        var rect: CGRect
        var corners: [CGPoint]?
        /// The original reading: text, candidates and rectified image. Its confidence is
        /// the raw one, used for text-switch decisions; display uses `displayConfidence`.
        var reading: PlateResult
        /// Consecutive cycles a *different* text was read while this one stayed listed.
        var dissent: Int
        /// Consecutive cycles this reading was actually matched. A brand-new reading
        /// starts at 1 and is withheld from the overlay until it repeats (or is strong
        /// enough on its own) — the same bar the history vote applies.
        var hits: Int
        var lastSeenCycle: Int
        /// Exponentially smoothed confidence shown on the chip.
        var displayConfidence: Float
    }

    @ObservationIgnored private var resultTracks: [TrackedResult] = []
    /// Published, stabilized readings consumed by BOTH overlay draws of a cycle.
    @ObservationIgnored private var stableResults: [PlateResult] = []
    /// Result tracking advances on its own final-publish cadence. It must NOT piggyback
    /// on `analysisCycle` (driven by the intermediate detections callback): if that
    /// callback is skipped, expiry would stall and stale labels would never retire.
    @ObservationIgnored private var resultCycle = 0
    /// Identifier of the analysis frame whose detections were already folded into the
    /// tracker by the intermediate detection callback (so the final publish of the same
    /// frame reuses the tracks instead of double-counting `hits`).
    @ObservationIgnored private var foldedFrameIdentifier = -1

    /// Cross-frame voting for the history list (see `PlateVoteAggregator`). Kept separate
    /// from `trackedPlates`, which only stabilises the *overlay geometry*: this one decides
    /// what the list shows, and runs on the results rather than on raw detections.
    /// `@ObservationIgnored` because it is written every analysis cycle and must not make
    /// the whole view tree re-evaluate — `recognitionHistory` is the published output.
    @ObservationIgnored private var voteAggregator = PlateVoteAggregator()

    /// Cycles a box must be seen consecutively before it is drawn.
    private static let confirmationHits = 2
    /// A track is retired after this many unmatched analysis cycles. Counting cycles
    /// instead of seconds keeps confirmation working on slow devices (a 1 s/frame phone
    /// still confirms on its second frame).
    ///
    /// Kept short on purpose: while the phone is moving, a retired plate's frozen box
    /// would otherwise sit at an old screen position for several seconds (the box has no
    /// motion of its own — only the next detection can move it). Two cycles still bridges
    /// a single missed detection so a transient occlusion does not blink the box off.
    private static let maxMissedCycles = 2
    /// Minimum IoU to associate an incoming detection with an existing track.
    private static let associationIoU: CGFloat = 0.3
    /// Weight of the *new* observation when smoothing track geometry (0...1) while the
    /// camera is steady. Heavy smoothing kills single-frame keypoint jitter.
    private static let smoothing: CGFloat = 0.5
    /// Weight used while the phone is moving (large box displacement): follow quickly so
    /// the outline does not trail the plate across the screen.
    private static let movingSmoothing: CGFloat = 0.85
    /// Normalized centre displacement above which a track is considered to be moving.
    /// A plate jumping more than ~4% of the frame between cycles cannot be sensor jitter.
    private static let movingThreshold: CGFloat = 0.04
    /// Cycles a reading may be missing before its label is removed. At the camera's
    /// 0.2 s gate that is roughly 0.4 s: long enough to bridge a blurry frame, short
    /// enough that a departing car does not leave a stale chip at an old screen spot.
    private static let resultMissedCycles = 2
    /// A rival text must be read this many consecutive cycles — or beat the listed
    /// reading by this much raw confidence — before it replaces the label.
    private static let resultSwitchHits = 2
    private static let resultSwitchConfidenceGap: Float = 0.12
    /// Weight of the new confidence in the displayed EMA.
    private static let resultConfidenceSmoothing: Float = 0.35
    /// A reading must be confirmed this many cycles — OR be this confident in a single
    /// pass — before the overlay draws it. Mirrors `PlateVoteSettings.minimumVotes` /
    /// `provisionalConfidence`, so the live overlay and the history list apply the same
    /// admission bar. Without this, a one-frame misread from a shaky hand (e.g. `CS83`
    /// read off a bumper at 63%) is drawn immediately while the history correctly drops it.
    private static let resultConfirmationHits = 2
    private static let resultProvisionalConfidence: Float = 0.85

    private nonisolated static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isEmpty else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    /// Association score for the detection and result trackers. Plain IoU breaks when the
    /// phone is shaken: a plate can move by more than its own width between two
    /// accurate-mode frames (~1.9 s apart), so `IoU < associationIoU` and the same plate
    /// spawns several tracks — the "boxes 3, plates 1" / three labels for two cars case.
    /// When direct IoU is too low, retry on boxes dilated by half their own size, but only
    /// if the two boxes have comparable dimensions: that bridges shake-induced motion while
    /// keeping two genuinely different plates (different size/distance) apart. Returns 0
    /// when the pair does not associate, so callers can keep a single `bestScore` threshold.
    private nonisolated static func associationScore(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let direct = iou(a, b)
        if direct >= associationIoU { return direct }
        let wRatio = max(a.width, b.width) / max(min(a.width, b.width), 1e-6)
        let hRatio = max(a.height, b.height) / max(min(a.height, b.height), 1e-6)
        guard wRatio <= 1.8, hRatio <= 1.8 else { return 0 }
        let da = a.insetBy(dx: -a.width * 0.5, dy: -a.height * 0.5)
        let db = b.insetBy(dx: -b.width * 0.5, dy: -b.height * 0.5)
        let expanded = iou(da, db)
        // Slightly discount the fallback so a directly-overlapping candidate always wins
        // the best-match contest; still clear `associationIoU` so it can be selected.
        return expanded >= associationIoU ? expanded * 0.9 : 0
    }

    private nonisolated static func lerp(_ a: CGPoint, _ b: CGPoint, _ t: CGFloat) -> CGPoint {
        CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
    }

    private nonisolated static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(a.x - b.x, a.y - b.y)
    }

    private nonisolated static func lerp(_ a: CGRect, _ b: CGRect, _ t: CGFloat) -> CGRect {
        CGRect(x: a.minX + (b.minX - a.minX) * t,
               y: a.minY + (b.minY - a.minY) * t,
               width: a.width + (b.width - a.width) * t,
               height: a.height + (b.height - a.height) * t)
    }

    /// Folds the current frame's raw detections into the tracker and returns the
    /// stabilized set to draw: confirmed tracks (smoothed, `hits >= confirmationHits`),
    /// including confirmed tracks that missed this cycle but are still fresh.
    private func stabilizeDetections(_ incoming: [PlateDetection]) -> [PlateDetection] {
        analysisCycle += 1
        let cycle = analysisCycle
        trackedPlates.removeAll { cycle - $0.lastSeenCycle > Self.maxMissedCycles }

        var matchedIndices = Set<Int>()
        var nextTracks: [TrackedPlate] = []
        for detection in incoming {
            var bestIndex = -1
            var bestScore: CGFloat = 0
            for (index, track) in trackedPlates.enumerated() where !matchedIndices.contains(index) {
                let score = Self.associationScore(detection.rect, track.rect)
                if score > bestScore { bestScore = score; bestIndex = index }
            }
            let incomingQuad = detection.plateQuad
            if bestIndex >= 0 {
                matchedIndices.insert(bestIndex)
                var track = trackedPlates[bestIndex]
                let centreShift = Self.distance(CGPoint(x: detection.rect.midX, y: detection.rect.midY),
                                                CGPoint(x: track.rect.midX, y: track.rect.midY))
                // Phone moving → follow the new position almost immediately; steady →
                // heavy smoothing to kill keypoint jitter. This removes the "box lags
                // behind, then catches up" feel while walking around a car.
                let t = centreShift > Self.movingThreshold ? Self.movingSmoothing : Self.smoothing
                track.rect = Self.lerp(track.rect, detection.rect, t)
                // Pose corners have a fixed order, so point-wise smoothing is safe.
                if track.quad.count == incomingQuad.count {
                    track.quad = zip(track.quad, incomingQuad).map { Self.lerp($0, $1, t) }
                } else {
                    track.quad = incomingQuad
                }
                track.confidence = max(track.confidence, detection.confidence)
                track.hits += 1
                track.lastSeenCycle = cycle
                nextTracks.append(track)
            } else {
                nextTracks.append(TrackedPlate(rect: detection.rect, quad: incomingQuad,
                                               confidence: detection.confidence, hits: 1,
                                               lastSeenCycle: cycle))
            }
        }
        // Tracks that missed this cycle keep their last geometry (frozen) and age out
        // after `maxMissedCycles` unmatched cycles — that's the anti-flicker persistence.
        for (index, track) in trackedPlates.enumerated() where !matchedIndices.contains(index) {
            nextTracks.append(track)
        }
        trackedPlates = nextTracks
        return confirmedTracks()
    }

    /// The currently confirmed tracks, reboxed as `PlateDetection`s for the overlay.
    private func confirmedTracks() -> [PlateDetection] {
        trackedPlates
            .filter { $0.hits >= Self.confirmationHits }
            .map { PlateDetection(rect: $0.rect, confidence: $0.confidence, label: "license_plate",
                                  keypoints: $0.quad, keypointConfidences: nil) }
    }

    /// Folds one frame's OCR readings into the result tracks. Geometry and confidence
    /// are smoothed; the text changes only on repeated or clearly stronger evidence, and
    /// a track survives a few unread cycles so one failed OCR pass cannot blink the chip
    /// off. Advances its own cycle counter so a skipped detections callback cannot stall
    /// expiry.
    private func stabilizeResults(_ incoming: [PlateResult]) {
        resultCycle += 1
        let cycle = resultCycle
        resultTracks.removeAll { cycle - $0.lastSeenCycle > Self.resultMissedCycles }
        var matchedIndices = Set<Int>()
        var nextTracks: [TrackedResult] = []
        for reading in incoming {
            var bestIndex = -1
            var bestScore: CGFloat = 0
            for (index, track) in resultTracks.enumerated() where !matchedIndices.contains(index) {
                let score = Self.associationScore(reading.boundingBox, track.rect)
                if score > bestScore { bestScore = score; bestIndex = index }
            }
            if bestIndex >= 0 {
                matchedIndices.insert(bestIndex)
                var track = resultTracks[bestIndex]
                let centreShift = Self.distance(CGPoint(x: reading.boundingBox.midX, y: reading.boundingBox.midY),
                                                CGPoint(x: track.rect.midX, y: track.rect.midY))
                let t: CGFloat = centreShift > Self.movingThreshold ? Self.movingSmoothing : Self.smoothing
                track.rect = Self.lerp(track.rect, reading.boundingBox, t)
                if let oldCorners = track.corners, let newCorners = reading.corners,
                   oldCorners.count == newCorners.count {
                    track.corners = zip(oldCorners, newCorners).map { Self.lerp($0, $1, t) }
                } else {
                    track.corners = reading.corners
                }
                track.lastSeenCycle = cycle
                track.hits += 1
                let weight = Self.resultConfidenceSmoothing
                track.displayConfidence = track.displayConfidence * (1 - weight)
                                          + reading.confidence * weight
                if reading.text == track.reading.text {
                    track.dissent = 0
                    if reading.confidence > track.reading.confidence { track.reading = reading }
                } else if reading.confidence >= track.reading.confidence + Self.resultSwitchConfidenceGap {
                    // A clearly stronger rival (e.g. the correct reading after a one-frame
                    // misread) wins immediately.
                    track.reading = reading
                    track.dissent = 0
                } else {
                    // A one-glyph OCR wobble must not flip the chip; only a run of dissent.
                    track.dissent += 1
                    if track.dissent >= Self.resultSwitchHits {
                        track.reading = reading
                        track.dissent = 0
                    }
                }
                nextTracks.append(track)
            } else {
                nextTracks.append(TrackedResult(rect: reading.boundingBox,
                                                corners: reading.corners,
                                                reading: reading,
                                                dissent: 0,
                                                hits: 1,
                                                lastSeenCycle: cycle,
                                                displayConfidence: reading.confidence))
            }
        }
        // Unread tracks keep their last stable reading (frozen) and age out after
        // `resultMissedCycles` — that is what bridges a single bad OCR frame.
        for (index, track) in resultTracks.enumerated() where !matchedIndices.contains(index) {
            nextTracks.append(track)
        }
        resultTracks = nextTracks
        // Only confirmed tracks reach the overlay. A reading seen once at modest
        // confidence is kept in `resultTracks` (so a repeat on the next cycle can confirm
        // it) but is withheld here — otherwise a shake-induced one-frame misread is drawn
        // immediately even though the history vote correctly refuses to list it. This is
        // the same two-bar admission rule: repeats `resultConfirmationHits` times, OR a
        // single pass at `resultProvisionalConfidence`+.
        stableResults = resultTracks
            .filter { $0.hits >= Self.resultConfirmationHits
                      || $0.reading.confidence >= Self.resultProvisionalConfidence }
            .map { displayResult(for: $0) }
    }

    /// Rebuilds a result with smoothed geometry/confidence while keeping the winning
    /// reading's text, candidates and rectified image intact.
    private func displayResult(for track: TrackedResult) -> PlateResult {
        PlateResult(text: track.reading.text,
                    confidence: track.displayConfidence,
                    boundingBox: track.rect,
                    imageSize: track.reading.imageSize,
                    candidates: track.reading.candidates,
                    correctedPlateImage: track.reading.correctedPlateImage,
                    corners: track.corners,
                    votes: track.reading.votes)
    }

    /// Single publish path for the live overlay. `updateTracks` is true only for the
    /// intermediate detection callback (the one place per analysis cycle that folds raw
    /// detections into the box tracker); the final publish of the same frame reuses those
    /// tracks. `results` is the already-stabilized set from `stabilizeResults`.
    /// `draw` is false for the intermediate callback: it folds the tracker but defers the
    /// single paint to the final callback, so a cycle never paints two different states.
    private func publishLiveOverlay(detections: [PlateDetection], frameIdentifier: Int,
                                    results: [PlateResult], draw: Bool = true) {
        let updateTracks = foldedFrameIdentifier != frameIdentifier
        let stabilized: [PlateDetection]
        if updateTracks {
            stabilized = stabilizeDetections(detections)
            foldedFrameIdentifier = frameIdentifier
        } else {
            stabilized = confirmedTracks()
        }
        liveDetections = stabilized
        guard draw else { return }
        previewView?.draw(detections: stabilized, results: results)
    }

    /// Shared process-wide context for the debug echo (Apple: reuse CIContext).
    /// A computed property also avoids a non-Sendable global `static let`.
    private nonisolated static var debugImageContext: CIContext { SDKGraphics.ciContext }

    /// Downscaled CGImage of the analysed buffer, in the same upright orientation the
    /// detector saw it (so the echo can be compared against the live preview directly).
    private nonisolated func debugThumbnail(from buffer: CVPixelBuffer,
                                            orientation: CGImagePropertyOrientation) -> CGImage? {
        let width = CGFloat(CVPixelBufferGetWidth(buffer))
        let height = CGFloat(CVPixelBufferGetHeight(buffer))
        guard width > 0, height > 0 else { return nil }
        // `oriented` changes the extent (and its origin); normalize to (0,0) before scaling.
        var image = CIImage(cvPixelBuffer: buffer).oriented(orientation)
        image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX,
                                                       y: -image.extent.minY))
        let maxDimension: CGFloat = 360
        let scale = min(maxDimension / image.extent.width, maxDimension / image.extent.height, 1)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return Self.debugImageContext.createCGImage(scaled, from: scaled.extent)
    }

    /// Mirrors the aggregator's listed set into the published history array.
    ///
    /// Equality-gated on purpose: the array is rebuilt on every analysis cycle (~5 fps) and
    /// `@Observable` notifies on *every* assignment, which would re-evaluate the whole
    /// history list — up to 300 rows — five times a second for no visible change.
    private func publishHistory(_ aggregates: [PlateVoteAggregate]) {
        let entries = aggregates.map {
            CameraRecognitionLogEntry(id: $0.id, date: $0.firstSeen,
                                      result: $0.result, votes: $0.votes,
                                      totalVotes: $0.totalVotes)
        }
        guard entries != recognitionHistory else { return }
        recognitionHistory = entries
    }

    func clearRecognitionHistory() {
        recognitionHistory.removeAll()
        lastPrinted.removeAll()
        voteAggregator.reset()
        // Invalidate the frame that may be in flight: its final publish only checks the
        // generation token, so without this bump a pass that started before the tap would
        // re-list its plates (and re-draw the labels) a few hundred ms later.
        generation += 1
        // Reset everything the list was derived from, so "Clear History" is a real reset and not a
        // one-second-later reappearance:
        //  • the overlay tracker held boxes whose `hits` were accumulated for the old
        //    session — dropping them means the next frame re-confirms from scratch;
        //  • the on-screen result cards/labels are part of the same accumulated state.
        // The capture session itself keeps running (this is not 停止).
        trackedPlates.removeAll()
        resultTracks.removeAll()
        stableResults = []
        resultCycle = 0
        analysisCycle = 0
        foldedFrameIdentifier = -1
        liveResults = []
        liveDetections = []
        isAnalyzing = false
        previewView?.clearOverlay()
    }

    init(recognizer: LicensePlateRecognizing, mode: RecognitionMode = .accurate) {
        self.recognizer = recognizer
        self.mode = mode
        // Triage: record which detector is actually in the pipeline from the start. The
        // old code only learned the name after a *successful* pass (see `performAnalysis`),
        // so until then `detectorName` stayed "" and the "Detector not using ONNX" badge fired with
        // an empty reason text — a false alarm while the ONNX detector was still fine.
        self.detectorName = (recognizer as? LicensePlateRecognizer)?.detectorInfo ?? ""
        super.init()
    }

    /// Swaps the recognition backend when the user changes the plate region in the
    /// header menu. Camera configuration is region-agnostic (still NV12), so only the
    /// analyzer changes; stale per-region detections and votes are discarded.
    func setRecognizer(_ newRecognizer: LicensePlateRecognizing) {
        recognizer = newRecognizer
        detectorName = (newRecognizer as? LicensePlateRecognizer)?.detectorInfo ?? ""
        lastTopScore = nil
        clearRecognitionHistory()
    }

    @discardableResult
    func setup() async -> Bool {
        active = true
        refreshAnalysisGate()
        let activationGeneration = generation
        let granted: Bool
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: granted = true
        case .notDetermined: granted = await AVCaptureDevice.requestAccess(for: .video)
        default: granted = false
        }
        isAuthorized = granted
        guard granted, active, generation == activationGeneration, !Task.isCancelled else { return false }
        if !didSetup {
            let ownsConfiguration = configurationTask == nil
            if ownsConfiguration {
                configurationTask = Task {
                    await withCheckedContinuation { continuation in
                        sessionQueue.async { continuation.resume(returning: self.configureSession()) }
                    }
                }
            }
            guard let task = configurationTask else { return false }
            let ok = await task.value
            if ownsConfiguration {
                setupFailed = !ok
                didSetup = ok
                configurationTask = nil
            }
            if !ok { errorMessage = "Camera initialization failed"; return false }
        }
        guard active, generation == activationGeneration, !Task.isCancelled else { return false }
        previewView?.ensurePortraitOrientation()
        sessionQueue.async { if !self.session.isRunning { self.session.startRunning() } }
        return true
    }

    private nonisolated func configureSession() -> Bool {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera), session.canAddInput(input) else { return false }
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "licenseplate.camera.frames"))
        guard session.canAddOutput(output) else { return false }
        session.addInput(input)
        session.addOutput(output)
        // NOTE: we deliberately do NOT set the data-output connection's `videoRotationAngle`.
        // That hardware rotation is documented as unreliable — the front camera ignores it,
        // some rear configurations do too, and a connection can still deliver landscape
        // buffers even after 90° was requested. Instead the *analysis* path keeps the raw
        // landscape buffer and hands it to the SDK with `orientation: .right`, so the one
        // render that produces the detector's CGImage also rotates it (no second full-frame
        // pass, no extra 8 MB buffer). The *preview* layer keeps its own portrait rotation
        // (set in `CameraPreview.ensurePortraitOrientation`), which is a separate, reliable
        // display path.
        return true
    }

    /// Halts the capture session and cancels in-flight analysis, but keeps `liveResults`
    /// so the result cards can be shown again when the camera tab returns. Used when
    /// switching to another media source or when the app is backgrounded — this is the
    /// "pause the hardware" path, distinct from a full teardown that would also wipe the
    /// recognition history. Keeping the controller (and its `AVCaptureVideoPreviewLayer`)
    /// mounted across tab switches avoids the black flash caused by tearing the preview
    /// layer down and rebuilding it.
    func suspendSession() {
        active = false
        generation += 1
        // A frame caught mid-analysis will never publish (its completion guards on
        // `generation == token`), so reset the "analyzing" indicator here — the same reason
        // `liveEnabled` and `paused` reset it in their didSet blocks.
        isAnalyzing = false
        refreshAnalysisGate()
        // Freeze the history here too: the preview keeps its last frame, so no further
        // analysis (and therefore no track expiry) will run until the camera returns.
        publishHistory(voteAggregator.finish())
        trackedPlates.removeAll()
        resultTracks.removeAll()
        stableResults = []
        resultCycle = 0
        analysisCycle = 0
        foldedFrameIdentifier = -1
        previewView?.clearOverlay()
        sessionQueue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    // MARK: - Frame analysis (off the main actor)

    /// Heavy recognition runs on a detached background task so the UI thread is never
    /// blocked by ONNX inference + OCR. All state/overlay writes hop back to the main
    /// actor explicitly (see `captureOutput`). The detection overlay mapping
    /// (`PreviewView.toLayer`, aspect-fill) is untouched — only the scheduling changed.
    private nonisolated func performAnalysis(frame: CameraFrame, token: Int, frameIdentifier: Int,
                                             mode: RecognitionMode,
                                             recognizer: LicensePlateRecognizing) async {
        let started = Date()
        let signpost = Self.signposter.beginInterval("frame")
        // Both cleanups are deferred: a cancellation between the render and the publish
        // would otherwise leave the gate busy forever and silently stop live detection.
        defer { Self.signposter.endInterval("frame", signpost) }
        defer { frameGate.finish() }
        do {
            // The raw buffer is the sensor's native landscape frame. Hand it to the SDK
            // as-is with `orientation: .right` and let `CIImage.oriented` rotate it inside
            // the render that already has to happen — the old path first rotated into a
            // brand-new 8 MB BGRA buffer, then had the SDK render the same frame a second
            // time. Detections still come back in upright portrait space, so the overlay
            // mapping, the corrector and the vote tracker are unaffected.
            let result = try await recognizer.recognizeFrame(frame.buffer, orientation: .right,
                                                             mode: mode, tiled: false) {
                [weak self] detections, _ in
                Task { @MainActor in
                    guard let self, self.active, self.liveEnabled, !self.paused,
                          self.generation == token, self.analysisIdentifier == frameIdentifier else { return }
                    // Fold this frame's detections into the tracker, but do NOT paint yet.
                    // Painting here used to draw provisional geometry with the *previous*
                    // cycle's label, then the final callback repainted with this cycle's
                    // data. That double draw is the "wrong box, then flashes to the right
                    // place" symptom. One paint per cycle (the final callback below) keeps
                    // box and label sourced from the same frame.
                    self.publishLiveOverlay(detections: detections, frameIdentifier: frameIdentifier,
                                            results: self.stableResults, draw: false)
                }
            }
            let latency = Int(Date().timeIntervalSince(started) * 1000)
            await MainActor.run { [weak self] in
                guard let self, self.active, self.liveEnabled, !self.paused, self.generation == token else { return }
                self.isAnalyzing = false
                self.elapsedMilliseconds = latency
                // Triage values for the status pill (see `lastTopScore`).
                if let plateRecognizer = recognizer as? LicensePlateRecognizer {
                    self.detectorName = plateRecognizer.detectorInfo
                }
                #if canImport(onnxruntime) || canImport(onnxruntime_mobile_objc) || canImport(onnxruntime_objc)
                self.lastTopScore = (recognizer as? LicensePlateRecognizer)?.detectorTopScore
                #endif
                // Keep the raw output for diagnostics, but draw the cross-frame stabilized
                // set: neither the text nor its displayed confidence jumps between frames.
                self.liveResults = result.plates
                self.stabilizeResults(result.plates)
                self.publishLiveOverlay(detections: result.detections, frameIdentifier: frameIdentifier,
                                        results: self.stableResults)
                self.errorMessage = nil
                // Diagnostic echo: the analysed (upright) buffer + the same boxes, so the
                // preview-vs-buffer alignment question can be answered on device.
                if self.debugEchoEnabled {
                    self.previewView?.showDebugFrame(
                        image: self.debugThumbnail(from: frame.buffer, orientation: .right),
                        detections: result.detections)
                }
                // Cross-frame voting decides what the history lists: a reading must repeat
                // and hold a majority before it appears, so a one-frame misread never gets
                // its own row, and a plate seen at 1s/3s/5s stays a single row whose text
                // and image follow the winning vote (not whichever frame finished last).
                self.publishHistory(self.voteAggregator.ingest(result.plates, at: Date()))
                // Throttled per text — diagnostics only, not the list.
                for plate in result.plates {
                    let now = Date()
                    if now.timeIntervalSince(self.lastPrinted[plate.text] ?? .distantPast) >= 2 {
                        let box = String(describing: plate.boundingBox)
                        Self.logger.debug(
                            "plate=\(plate.text, privacy: .public) confidence=\(Double(plate.confidence)) box=\(box, privacy: .public) latency=\(latency)ms")
                        self.lastPrinted[plate.text] = now
                    }
                }
                self.lastPrinted = self.lastPrinted.filter { Date().timeIntervalSince($0.value) < 10 }
            }
        } catch {
            await MainActor.run { [weak self] in
                guard let self, self.active, self.generation == token else { return }
                self.isAnalyzing = false
                // Refresh the detector triage even on a failed pass: with the
                // unavailable-stub every pass throws before finishing, so the
                // success-only refresh left `detectorName` empty forever and the
                // "Detector not using ONNX" badge could never show *why* it fired.
                if let plateRecognizer = recognizer as? LicensePlateRecognizer {
                    self.detectorName = plateRecognizer.detectorInfo
                }
                self.errorMessage = error.localizedDescription
                self.liveResults = []
                self.liveDetections = []
                self.trackedPlates.removeAll()
                self.resultTracks.removeAll()
                self.stableResults = []
                self.resultCycle = 0
                self.analysisCycle = 0
                self.foldedFrameIdentifier = -1
                self.previewView?.clearOverlay()
            }
        }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        // First line on the capture queue: an idle, paused or suspended camera drops the
        // frame here. `begin()` also keeps at most one analysis in flight and enforces the
        // 0.2 s spacing, so a dropped frame costs no rotation, no inference and no OCR.
        guard frameGate.isEnabled,
              let buffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              frameGate.begin() else { return }
        let frame = CameraFrame(buffer: buffer)
        // Mint the generation token and check gating on the main actor (cheap), then run the
        // heavy recognition off the main actor so the scan beam, controls and latency never
        // stall. Final state/overlay writes happen back on the main actor.
        Task { @MainActor in
            guard self.active, self.liveEnabled, !self.paused else { self.frameGate.finish(); return }
            self.analysisIdentifier += 1
            self.frameCount += 1
            let token = self.generation
            let frameIdentifier = self.analysisIdentifier
            self.isAnalyzing = true
            let mode = self.mode
            let recognizer = self.recognizer
            Task.detached(priority: .userInitiated) { [weak self] in
                await self?.performAnalysis(frame: frame, token: token, frameIdentifier: frameIdentifier,
                                           mode: mode, recognizer: recognizer)
            }
        }
    }
}
