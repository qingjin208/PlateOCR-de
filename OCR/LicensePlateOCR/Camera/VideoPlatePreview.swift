/*
 License Plate OCR SDK — Demo App: video preview with plate overlay

 Abstract:
 Plays the selected clip and paints the plates found in each *sampled* frame on top of
 the video. The SDK only recognizes a handful of frames (default 50), so the overlay
 interpolates geometry between the two samples bracketing the playback head. A plate
 therefore glides with the car instead of jumping from one stale sample to the next —
 the persistent "flicker" reported on moving footage.

 The overlay geometry is anchored to `AVPlayerLayer.videoRect`, which already accounts
 for the layer bounds, `videoGravity` and the track's preferred transform, so normalized
 (top-left, 0...1) points from the SDK map straight onto the displayed picture.
 */

import AVFoundation
import SwiftUI
import Foundation

struct VideoPlatePreview: UIViewRepresentable {
    let player: AVPlayer
    let samples: [PlateVideoSample]
    /// Cross-frame majority-voted readings. Only the *text/confidence* is borrowed from
    /// these; geometry still comes from the interpolated samples.
    let results: [PlateResult]

    func makeUIView(context: Context) -> VideoPlateView {
        let view = VideoPlateView()
        view.attach(player)
        view.samples = samples
        view.results = results
        return view
    }

    func updateUIView(_ uiView: VideoPlateView, context: Context) {
        uiView.attach(player)
        uiView.samples = samples
        uiView.results = results
    }

    static func dismantleUIView(_ uiView: VideoPlateView, coordinator: ()) {
        uiView.detach()
    }
}

/// One plate to draw at the current playback time, with geometry already interpolated
/// between the two bracketing samples.
private struct VideoDrawItem {
    var quad: [CGPoint]
    var box: CGRect
    var text: String
    var confidence: Float
    var recognized: Bool
    /// Fade factor at a track's birth/death (a plate present in only one bracket sample).
    var alpha: CGFloat
}

final class VideoPlateView: UIView {
    /// Sampled frames. A fresh recognition pass resets the cached sample order so the
    /// next render recomputes interpolation; an unrelated SwiftUI re-render (play/pause,
    /// progress, settings) leaves it alone. Previously every `updateUIView` re-assignment
    /// cleared the selection and rebuilt every layer — a visible flash.
    var samples: [PlateVideoSample] = [] {
        didSet {
            guard !Self.sameSamples(oldValue, samples) else { return }
            sortedSamples = samples.sorted { $0.time < $1.time }
            rebuildFrozenReadings()
            if samples.isEmpty { clearLayers() }
            render(at: playerLayer.player?.currentTime().seconds ?? 0)
            refreshSoon()
        }
    }

    /// Majority-voted readings (`PlateVoteAggregator.consolidate`). Only text and
    /// confidence are borrowed; geometry still comes from the interpolated samples.
    /// Equality-gated for the same reason as `samples`.
    var results: [PlateResult] = [] {
        didSet {
            guard !Self.sameReadings(oldValue, results) else { return }
            rebuildFrozenReadings()
            render(at: playerLayer.player?.currentTime().seconds ?? 0)
        }
    }

    /// One resolved reading for a sampled plate. The text is decided once (from a
    /// cross-sample track's majority vote) and frozen; playback only looks it up. Without
    /// this, `drawItems` re-derives text from each frame's box on every tick, and as the
    /// head crosses a sample boundary the same car flips between the voted spelling and
    /// that frame's raw OCR reading (`MAB1234` / `MAB123A`).
    private struct FrozenReading {
        let text: String
        let confidence: Float
    }
    /// Indexed by `sortedSamples` position, then plate position within that sample.
    private var frozenReadings: [[Int: FrozenReading]] = []

    private static func sameReadings(_ a: [PlateResult], _ b: [PlateResult]) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { $0.text == $1.text && $0.confidence == $1.confidence }
    }

    private static func sameSamples(_ a: [PlateVideoSample], _ b: [PlateVideoSample]) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { $0.id == $1.id && $0.time == $1.time }
    }

    private var sortedSamples: [PlateVideoSample] = []
    private var timeObserver: Any?
    private weak var observedPlayer: AVPlayer?

    private let plateContainerLayer = CALayer()
    /// Labels in their own container always paint above the plate fills, regardless of
    /// the order in which the reused shape/label pools grow or shrink.
    private let labelContainerLayer = CALayer()
    private let detectionLayer = CAShapeLayer()
    /// Layers are reused across renders: only their path/frame/string is updated. Tearing
    /// them down and rebuilding each tick is itself a flash even when geometry is smooth.
    private var plateLayers: [CAShapeLayer] = []
    private var labels: [CATextLayer] = []

    override class var layerClass: AnyClass { AVPlayerLayer.self }

    var playerLayer: AVPlayerLayer {
        (layer as? AVPlayerLayer) ?? AVPlayerLayer()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        playerLayer.videoGravity = .resizeAspect
        detectionLayer.fillColor = nil
        detectionLayer.strokeColor = PlateOverlayPalette.unifiedColor.cgColor
        detectionLayer.lineWidth = PlateOverlayPalette.detectionStrokeWidth
        layer.addSublayer(detectionLayer)
        layer.addSublayer(plateContainerLayer)
        layer.addSublayer(labelContainerLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        if let timeObserver, let observedPlayer {
            observedPlayer.removeTimeObserver(timeObserver)
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        detectionLayer.frame = bounds
        plateContainerLayer.frame = bounds
        labelContainerLayer.frame = bounds
        render(at: playerLayer.player?.currentTime().seconds ?? 0)
    }

    // MARK: - Player

    func attach(_ player: AVPlayer) {
        guard playerLayer.player !== player else { return }
        detach()
        playerLayer.player = player
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(value: 1, timescale: 30), queue: .main
        ) { [weak self] time in
            let seconds = time.seconds
            Task { @MainActor in self?.render(at: seconds) }
        }
        observedPlayer = player
        if samples.isEmpty { clearLayers() }
        render(at: player.currentTime().seconds)
        refreshSoon()
    }

    /// `AVPlayerLayer.videoRect` is empty until the item is ready and laid out. Delayed
    /// repaints make sure the overlay appears even when the player is paused at t=0.
    private func refreshSoon() {
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 200_000_000)
            self?.render(at: self?.playerLayer.player?.currentTime().seconds ?? 0)
            try? await Task.sleep(nanoseconds: 400_000_000)
            self?.render(at: self?.playerLayer.player?.currentTime().seconds ?? 0)
        }
    }

    func detach() {
        if let timeObserver, let observedPlayer {
            observedPlayer.removeTimeObserver(timeObserver)
        }
        timeObserver = nil
        observedPlayer = nil
    }

    // MARK: - Interpolation

    /// Builds the plates to draw at `seconds` by interpolating between the two samples
    /// that bracket the playback head. Same plate is paired by box IoU; a plate seen in
    /// only one bracket sample fades in/out so it never blinks on/off at a sample boundary.
    private func drawItems(at seconds: Double) -> [VideoDrawItem] {
        guard !sortedSamples.isEmpty, seconds.isFinite else { return [] }
        let indexA: Int
        let a: PlateVideoSample
        let indexB: Int?
        let b: PlateVideoSample?
        var alpha: CGFloat = 0

        if seconds <= sortedSamples.first!.time {
            indexA = 0
            a = sortedSamples[0]
            indexB = sortedSamples.count > 1 ? 1 : nil
            b = sortedSamples.count > 1 ? sortedSamples[1] : nil
        } else if seconds >= sortedSamples.last!.time {
            indexA = sortedSamples.count - 1
            a = sortedSamples[indexA]
            indexB = nil
            b = nil
        } else {
            var i = 0
            while i + 1 < sortedSamples.count, sortedSamples[i + 1].time < seconds { i += 1 }
            indexA = i
            a = sortedSamples[i]
            indexB = i + 1
            let next = sortedSamples[i + 1]
            b = next
            let span = next.time - a.time
            alpha = span > 0 ? CGFloat(max(0, min(1, (seconds - a.time) / span))) : 0
        }

        var items: [VideoDrawItem] = []
        var usedB = Set<Int>()

        for (ia, plateA) in a.plates.enumerated() {
            var bestB: Int?
            var bestIoU: CGFloat = Self.pairIoU
            if let b {
                for (jb, plateB) in b.plates.enumerated() where !usedB.contains(jb) {
                    let score = Self.iou(plateA.boundingBox, plateB.boundingBox)
                    if score > bestIoU { bestIoU = score; bestB = jb }
                }
            }
            if let jb = bestB, let b {
                usedB.insert(jb)
                let plateB = b.plates[jb]
                let box = Self.lerp(plateA.boundingBox, plateB.boundingBox, alpha)
                let quadA = Self.quad(for: plateA)
                let quadB = Self.quad(for: plateB)
                let quad = Self.lerpQuad(quadA, quadB, alpha)
                let reading = frozenReading(sampleIndex: indexA, plateIndex: ia, fallback: plateA)
                items.append(VideoDrawItem(quad: quad, box: box,
                                           text: reading.text, confidence: reading.confidence,
                                           recognized: true, alpha: 1))
            } else {
                // Present only in A: fade out as the head advances toward B.
                let fade = 1 - alpha
                if fade > Self.minAlpha {
                    let reading = frozenReading(sampleIndex: indexA, plateIndex: ia, fallback: plateA)
                    items.append(VideoDrawItem(quad: Self.quad(for: plateA),
                                               box: plateA.boundingBox,
                                               text: reading.text, confidence: reading.confidence,
                                               recognized: true, alpha: fade))
                }
            }
        }
        if let b {
            for (jb, plateB) in b.plates.enumerated() where !usedB.contains(jb) {
                // Present only in B: fade in.
                if alpha > Self.minAlpha {
                    let reading = frozenReading(sampleIndex: indexB ?? 0,
                                                plateIndex: jb, fallback: plateB)
                    items.append(VideoDrawItem(quad: Self.quad(for: plateB),
                                               box: plateB.boundingBox,
                                               text: reading.text, confidence: reading.confidence,
                                               recognized: true, alpha: alpha))
                }
            }
        }
        return items
    }

    private static let pairIoU: CGFloat = 0.15
    private static let minAlpha: CGFloat = 0.08

    private static func quad(for plate: PlateResult) -> [CGPoint] {
        if let corners = plate.corners, corners.count == 4 { return corners }
        let r = plate.boundingBox
        return [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.maxX, y: r.minY),
                CGPoint(x: r.maxX, y: r.maxY), CGPoint(x: r.minX, y: r.maxY)]
    }

    private static func lerp(_ a: CGRect, _ b: CGRect, _ t: CGFloat) -> CGRect {
        CGRect(x: a.minX + (b.minX - a.minX) * t,
               y: a.minY + (b.minY - a.minY) * t,
               width: a.width + (b.width - a.width) * t,
               height: a.height + (b.height - a.height) * t)
    }

    private static func lerpQuad(_ a: [CGPoint], _ b: [CGPoint], _ t: CGFloat) -> [CGPoint] {
        guard a.count == b.count else { return t < 0.5 ? a : b }
        return zip(a, b).map { CGPoint(x: $0.x + ($1.x - $0.x) * t,
                                       y: $0.y + ($1.y - $0.y) * t) }
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isEmpty else { return 0 }
        let area = inter.width * inter.height
        let union = a.width * a.height + b.width * b.height - area
        return union > 0 ? area / union : 0
    }

    // MARK: - Overlay

    /// Reuses existing layers and only mutates their geometry/style. `videoRect` must be
    /// valid before touching layers: clearing on a not-yet-ready rect used to blank the
    /// preview between the clear and the delayed repaint.
    private func render(at seconds: Double) {
        let videoRect = playerLayer.videoRect
        guard !videoRect.isEmpty, videoRect.width > 0, videoRect.height > 0 else { return }
        let items = drawItems(at: seconds)

        func toLayer(_ point: CGPoint) -> CGPoint {
            CGPoint(x: videoRect.minX + point.x * videoRect.width,
                    y: videoRect.minY + point.y * videoRect.height)
        }
        func toLayer(_ rect: CGRect) -> CGRect {
            CGRect(x: videoRect.minX + rect.minX * videoRect.width,
                   y: videoRect.minY + rect.minY * videoRect.height,
                   width: rect.width * videoRect.width,
                   height: rect.height * videoRect.height)
        }

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        // Resize the reused layer pools to match the current item count.
        while plateLayers.count < items.count {
            let shape = CAShapeLayer()
            shape.lineJoin = .round
            plateContainerLayer.addSublayer(shape)
            plateLayers.append(shape)
            let text = CATextLayer()
            text.fontSize = 13
            text.contentsScale = traitCollection.displayScale
            text.foregroundColor = UIColor.white.cgColor
            text.alignmentMode = .left
            labelContainerLayer.addSublayer(text)
            labels.append(text)
        }
        while plateLayers.count > items.count {
            plateLayers.removeLast().removeFromSuperlayer()
            labels.removeLast().removeFromSuperlayer()
        }

        let detectionPath = CGMutablePath()
        for (index, item) in items.enumerated() {
            guard item.quad.count == 4 else { continue }
            let points = item.quad.map(toLayer)
            let path = CGMutablePath()
            path.addLines(between: points)
            path.closeSubpath()

            let color = PlateOverlayPalette.color(for: item.text)
            let shape = plateLayers[index]
            shape.path = path
            shape.fillColor = color.cgColor(alpha: PlateOverlayPalette.fillAlpha * item.alpha)
            shape.strokeColor = color.cgColor(alpha: item.alpha)
            shape.lineWidth = PlateOverlayPalette.strokeWidth
            shape.opacity = 1

            detectionPath.addRect(toLayer(item.box))

            let label = labels[index]
            let minX = points.map(\.x).min() ?? 0
            let minY = points.map(\.y).min() ?? 0
            let maxX = points.map(\.x).max() ?? 0
            let width = min(bounds.width, max(140, maxX - minX))
            label.string = "\(item.text)  \(Int(item.confidence * 100))%"
            label.backgroundColor = color.cgColor(alpha: 0.95 * item.alpha)
            label.frame = CGRect(x: max(0, min(bounds.width - width, minX)),
                                 y: max(0, minY - 21), width: width, height: 21)
            label.opacity = Float(item.alpha)
        }
        detectionLayer.path = detectionPath
        CATransaction.commit()
    }

    private func clearLayers() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        plateLayers.forEach { $0.removeFromSuperlayer() }
        plateLayers = []
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        detectionLayer.path = nil
        CATransaction.commit()
    }

    /// IoU used to link the same physical plate across sample frames into one track.
    /// Deliberately looser than the interpolation pairing: samples are far apart in time,
    /// so a moving car's boxes overlap far less than two adjacent video frames would.
    private static let trackIoU: CGFloat = 0.12

    /// Resolves one stable text per sampled plate and caches it. Called only when the
    /// recognition output changes — never during playback.
    ///
    /// The same car is linked across sample frames by box IoU into a track (one-to-one,
    /// best match first, and never linking two plates from the *same* frame). Each track's
    /// readings are then majority-voted; ties go to the highest-confidence observation.
    /// Playback looks up the frozen text by (sample, plate) instead of re-deriving it from
    /// the current box, so crossing a sample boundary can no longer flip a label between
    /// a voted spelling and one frame's raw OCR reading.
    private func rebuildFrozenReadings() {
        struct Observation {
            let text: String
            let confidence: Float
            let box: CGRect
        }

        // Flatten to time order so a track can only ever consume *earlier* samples.
        var flat: [(sample: Int, plate: Int, obs: Observation)] = []
        for (si, sample) in sortedSamples.enumerated() {
            for (pi, plate) in sample.plates.enumerated() {
                flat.append((si, pi, Observation(text: plate.text,
                                                 confidence: plate.confidence,
                                                 box: plate.boundingBox)))
            }
        }

        var trackOfFlat = [Int](repeating: -1, count: flat.count)
        var trackObservations: [[Observation]] = []
        var trackLatestBox: [CGRect] = []
        var trackLastSample: [Int] = []

        for (fi, entry) in flat.enumerated() {
            var bestTrack: Int?
            var bestIoU = Self.trackIoU
            // Only tracks last seen in an *earlier* sample are eligible. Without this,
            // two cars present in the same frame would merge into a single track.
            for ti in trackObservations.indices where trackLastSample[ti] < entry.sample {
                let score = Self.iou(trackLatestBox[ti], entry.obs.box)
                if score > bestIoU { bestIoU = score; bestTrack = ti }
            }
            if let ti = bestTrack {
                trackOfFlat[fi] = ti
                trackObservations[ti].append(entry.obs)
                trackLatestBox[ti] = entry.obs.box
                trackLastSample[ti] = entry.sample
            } else {
                trackOfFlat[fi] = trackObservations.count
                trackObservations.append([entry.obs])
                trackLatestBox.append(entry.obs.box)
                trackLastSample.append(entry.sample)
            }
        }

        var trackText: [String] = []
        var trackConfidence: [Float] = []
        for observations in trackObservations {
            var votes: [String: Int] = [:]
            var confidence: [String: Float] = [:]
            // The highest-confidence sighting is this track's representative box: it is
            // the frame the model read most clearly, so it is the best anchor for matching
            // the globally consolidated reading.
            var representative: Observation = observations[0]
            for obs in observations {
                votes[obs.text, default: 0] += 1
                confidence[obs.text] = max(confidence[obs.text] ?? 0, obs.confidence)
                if obs.confidence > representative.confidence { representative = obs }
            }
            let winner = votes.max {
                if $0.value != $1.value { return $0.value < $1.value }
                return (confidence[$0.key] ?? 0) < (confidence[$1.key] ?? 0)
            }?.key ?? ""

            // Align with the result list's authoritative spelling. This lookup is done
            // ONCE here (not per playback frame): match the track's representative box
            // against the consolidated results and borrow that text/confidence. Without
            // the freeze this same lookup ran every tick with each frame's box, which is
            // what made a crossing plate flip between two spellings.
            var finalText = winner
            var finalConfidence = confidence[winner] ?? 0
            if !results.isEmpty {
                var bestIoU: CGFloat = 0.15
                var best: PlateResult?
                for candidate in results {
                    let score = Self.iou(representative.box, candidate.boundingBox)
                    if score > bestIoU { bestIoU = score; best = candidate }
                }
                if let best {
                    finalText = best.text
                    finalConfidence = max(finalConfidence, best.confidence)
                }
            }
            trackText.append(finalText)
            trackConfidence.append(finalConfidence)
        }

        var table = [[Int: FrozenReading]](repeating: [:], count: sortedSamples.count)
        for (fi, entry) in flat.enumerated() {
            let ti = trackOfFlat[fi]
            guard trackObservations.indices.contains(ti) else { continue }
            table[entry.sample][entry.plate] = FrozenReading(text: trackText[ti],
                                                             confidence: trackConfidence[ti])
        }
        frozenReadings = table
    }

    /// Looks up the frozen reading for one sampled plate. Falls back to that plate's own
    /// OCR output only when no table entry exists (e.g. samples changed mid-render).
    private func frozenReading(sampleIndex: Int, plateIndex: Int,
                               fallback plate: PlateResult) -> (text: String, confidence: Float) {
        guard frozenReadings.indices.contains(sampleIndex),
              let reading = frozenReadings[sampleIndex][plateIndex] else {
            return (plate.text, plate.confidence)
        }
        return (reading.text, reading.confidence)
    }
}
