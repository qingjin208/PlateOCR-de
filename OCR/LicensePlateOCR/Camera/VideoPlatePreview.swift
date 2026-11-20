/*
 License Plate OCR SDK — Demo App: video preview with plate overlay

 Abstract:
 Plays the selected clip and paints the plates found in each *sampled* frame on top of
 the video. Every sampled frame carries its own normalized coordinates, so the overlay
 switches as playback reaches that sample — drawing the aggregate result list instead
 would place boxes in the wrong coordinate space.

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
    /// these; geometry still comes from the sample currently on screen.
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

final class VideoPlateView: UIView {
    /// Sampled frames plus the id currently drawn. A fresh recognition pass resets the
    /// selection so it repaints immediately; an unrelated SwiftUI re-render (play/pause,
    /// progress, settings) leaves the selection alone. Previously every `updateUIView`
    /// re-assignment cleared `currentSampleID` and rebuilt every layer — a visible flash.
    var samples: [PlateVideoSample] = [] {
        didSet {
            guard !Self.sameSamples(oldValue, samples) else { return }
            currentSampleID = nil
            // `renderOverlay` now bails out before clearing when videoRect is not ready,
            // so an empty sample set must explicitly drop the previous pass's layers.
            if samples.isEmpty {
                clearLayers()
                return
            }
            updateOverlay(for: playerLayer.player?.currentTime().seconds ?? 0)
            refreshSoon()
        }
    }

    /// Majority-voted readings (`PlateVoteAggregator.consolidate`). Only text and
    /// confidence are borrowed from these; geometry stays with the visible sample.
    /// Equality-gated for the same reason as `samples`: SwiftUI re-assigns on every body
    /// evaluation, and an unconditional repaint would flash the overlay for no reason.
    var results: [PlateResult] = [] {
        didSet {
            guard !Self.sameReadings(oldValue, results) else { return }
            renderOverlay()
        }
    }

    private static func sameReadings(_ a: [PlateResult], _ b: [PlateResult]) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { $0.text == $1.text && $0.confidence == $1.confidence }
    }

    /// Identity of a recognition pass: ids and timestamps are enough to tell a new pass
    /// from a re-assignment of the same array. Payload changes always come with new ids.
    private static func sameSamples(_ a: [PlateVideoSample], _ b: [PlateVideoSample]) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { $0.id == $1.id && $0.time == $1.time }
    }

    private var currentSampleID: Int?
    private var timeObserver: Any?
    private weak var observedPlayer: AVPlayer?
    /// Four independent callers can request a repaint (time observer, delayed refresh,
    /// results update, layout). Without this guard each one tears down and rebuilds every
    /// CALayer, which reads as a flash even when the visible content did not change.
    private var lastRenderSignature: String?

    private let plateContainerLayer = CALayer()
    private let detectionLayer = CAShapeLayer()
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
        renderOverlay()
    }

    // MARK: - Player

    func attach(_ player: AVPlayer) {
        guard playerLayer.player !== player else { return }
        detach()
        playerLayer.player = player
        // Follow playback so the overlay matches the frame currently on screen.
        // Hop to the main actor explicitly: `updateOverlay` touches CALayers, and a
        // directly-captured call is a concurrency violation under strict checking.
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(value: 1, timescale: 20), queue: .main
        ) { [weak self] time in
            let seconds = time.seconds
            Task { @MainActor in self?.updateOverlay(for: seconds) }
        }
        observedPlayer = player
        currentSampleID = nil
        if samples.isEmpty { clearLayers() }
        updateOverlay(for: player.currentTime().seconds)
        refreshSoon()
    }

    /// `AVPlayerLayer.videoRect` is empty until the item is ready and the layer has been
    /// laid out. A couple of delayed repaints make sure the overlay appears even when the
    /// player is paused at t=0 (the periodic observer only ticks during playback).
    private func refreshSoon() {
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 200_000_000)
            self?.renderOverlay()
            try? await Task.sleep(nanoseconds: 400_000_000)
            self?.renderOverlay()
        }
    }

    func detach() {
        if let timeObserver, let observedPlayer {
            observedPlayer.removeTimeObserver(timeObserver)
        }
        timeObserver = nil
        observedPlayer = nil
    }

    // MARK: - Overlay

    /// Picks the sampled frame nearest to the playback position. Switching has a
    /// hysteresis band: the current sample is kept until a rival is clearly nearer, so a
    /// playback head hovering around a midpoint cannot flip between two samples on
    /// consecutive 20 Hz ticks (every flip tore down and rebuilt all overlay layers).
    private func updateOverlay(for seconds: Double) {
        guard !samples.isEmpty, seconds.isFinite else {
            if currentSampleID != nil {
                currentSampleID = nil
                clearLayers()
            }
            return
        }
        guard let nearest = samples.min(by: { abs($0.time - seconds) < abs($1.time - seconds) }) else {
            return
        }
        if let currentID = currentSampleID,
           let current = samples.first(where: { $0.id == currentID }),
           current.id != nearest.id {
            // 25% of the local sample spacing, capped at 0.25 s: wide enough to absorb
            // observer jitter, short enough that a real seek still switches promptly.
            let hold = min(0.25, spacing(around: current.time) * 0.25)
            if abs(nearest.time - seconds) + hold >= abs(current.time - seconds) {
                return
            }
        }
        guard nearest.id != currentSampleID else { return }
        currentSampleID = nearest.id
        renderOverlay()
    }

    /// Gap between the two samples surrounding `time` (1 s when only one sample exists).
    private func spacing(around time: Double) -> Double {
        guard samples.count > 1 else { return 1 }
        var bestDistance = Double.infinity
        var gap = 1.0
        for i in 0..<(samples.count - 1) {
            let a = samples[i].time, b = samples[i + 1].time
            let distance = abs((a + b) / 2 - time)
            if distance < bestDistance {
                bestDistance = distance
                gap = b - a
            }
        }
        return gap > 0 ? gap : 1
    }

    /// Tears down every overlay layer. Called only when a repaint can immediately follow,
    /// or when the sample set became empty — never on a not-yet-ready `videoRect`, which
    /// used to leave a blank preview between the clear and the delayed repaint.
    private func clearLayers() {
        lastRenderSignature = nil
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        plateLayers.forEach { $0.removeFromSuperlayer() }
        plateLayers = []
        detectionLayer.path = nil
        CATransaction.commit()
    }

    /// Quantised to what is visible, so repeated repaint requests from the observer,
    /// delayed refresh and layout are no-ops instead of layer teardowns.
    private func renderSignature() -> String {
        guard let id = currentSampleID,
              let sample = samples.first(where: { $0.id == id }) else { return "empty" }
        let videoRect = playerLayer.videoRect
        guard !videoRect.isEmpty, videoRect.width > 0, videoRect.height > 0 else { return "novideo" }
        func rect(_ r: CGRect) -> String {
            String(format: "%.2f,%.2f,%.2f,%.2f", r.minX, r.minY, r.width, r.height)
        }
        var signature = "\(id)|" + String(format: "%.1f,%.1f", bounds.width, bounds.height)
                        + "|" + rect(videoRect)
        for detection in sample.detections {
            signature += rect(detection.rect)
            for corner in detection.plateQuad {
                signature += String(format: "%.2f,%.2f,", corner.x, corner.y)
            }
            signature += ";"
        }
        signature += "|"
        for plate in sample.plates {
            let reading = stableReading(for: plate)
            signature += reading.text + String(format: ":%d;", Int(reading.confidence * 100))
        }
        return signature
    }

    private func renderOverlay() {
        let signature = renderSignature()
        guard signature != lastRenderSignature else { return }
        guard let id = currentSampleID,
              let sample = samples.first(where: { $0.id == id }) else { return }
        let videoRect = playerLayer.videoRect
        guard !videoRect.isEmpty, videoRect.width > 0, videoRect.height > 0 else { return }
        clearLayers()
        // Set after clearLayers(), which invalidates the signature.
        lastRenderSignature = signature

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
        func quadPath(_ quad: [CGPoint]) -> CGPath? {
            guard quad.count == 4 else { return nil }
            let path = CGMutablePath()
            path.addLines(between: quad.map(toLayer))
            path.closeSubpath()
            return path
        }

        // Keep the plain detection rectangle, then tint the plate area on top of it.
        let detectionPath = CGMutablePath()
        for detection in sample.detections {
            detectionPath.addRect(toLayer(detection.rect))
        }

        var usedResultIndices = Set<Int>()
        for detection in sample.detections {
            let matched = sample.plates.indices.first { i in
                !usedResultIndices.contains(i) && (detection.rect.contains(sample.plates[i].boundingBox)
                    || detection.rect.intersects(sample.plates[i].boundingBox))
            }
            if let matched { usedResultIndices.insert(matched) }
            let result = matched.map { sample.plates[$0] }
            let key = PlateOverlayPalette.key(text: result?.text, box: detection.rect)
            // Resolve the quad without nested-optional chaining (`result?.corners` is an
            // optional-of-optional property access, which makes `??` ambiguous).
            var quad = detection.plateQuad
            if let result, let corners = result.corners, corners.count == 4 {
                quad = corners
            }
            guard let path = quadPath(quad) else { continue }
            let color = PlateOverlayPalette.color(for: key)
            addPlateLayer(path: path, color: color, recognized: result != nil)
            if let result {
                let reading = stableReading(for: result)
                addLabel(for: result, color: color, quad: quad,
                         text: reading.text, confidence: reading.confidence)
            }
        }
        // Plates carved out of a merged crop have no dedicated detection rectangle.
        for (index, plate) in sample.plates.enumerated() where !usedResultIndices.contains(index) {
            guard let path = quadPath(plate.plateQuad) else { continue }
            let color = PlateOverlayPalette.color(
                for: PlateOverlayPalette.key(text: plate.text, box: plate.boundingBox))
            addPlateLayer(path: path, color: color, recognized: true)
            let reading = stableReading(for: plate)
            addLabel(for: plate, color: color, quad: plate.plateQuad,
                     text: reading.text, confidence: reading.confidence)
        }

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        detectionLayer.path = detectionPath
        CATransaction.commit()
    }

    private func addPlateLayer(path: CGPath, color: PlateOverlayColor, recognized: Bool) {
        let plate = CAShapeLayer()
        plate.path = path
        plate.fillColor = color.cgColor(alpha: recognized ? PlateOverlayPalette.fillAlpha
                                                         : PlateOverlayPalette.fillAlpha * 0.4)
        plate.strokeColor = color.cgColor
        plate.lineWidth = recognized ? PlateOverlayPalette.strokeWidth
                                     : PlateOverlayPalette.strokeWidth * 0.6
        plate.lineJoin = .round
        plateContainerLayer.addSublayer(plate)
        plateLayers.append(plate)
    }

    /// Majority-voted text/confidence for one per-frame reading. OCR on neighbouring
    /// sampled frames can flip a single glyph (`MAB1234` / `MAB123A`); borrowing the voted
    /// reading keeps the label still while the box still follows the visible frame.
    private func stableReading(for plate: PlateResult) -> (text: String, confidence: Float) {
        guard !results.isEmpty else { return (plate.text, plate.confidence) }
        let box = plate.boundingBox
        var bestIoU: CGFloat = 0.2
        var best: PlateResult?
        for candidate in results {
            let other = candidate.boundingBox
            let inter = box.intersection(other)
            guard !inter.isEmpty, inter.width > 0, inter.height > 0 else { continue }
            let area = inter.width * inter.height
            let union = box.width * box.height + other.width * other.height - area
            guard union > 0 else { continue }
            let score = area / union
            if score > bestIoU {
                bestIoU = score
                best = candidate
            }
        }
        guard let best else { return (plate.text, plate.confidence) }
        return (best.text, best.confidence)
    }

    private func addLabel(for plate: PlateResult, color: PlateOverlayColor, quad: [CGPoint],
                          text: String, confidence: Float) {
        let videoRect = playerLayer.videoRect
        let points = quad.map { CGPoint(x: videoRect.minX + $0.x * videoRect.width,
                                        y: videoRect.minY + $0.y * videoRect.height) }
        let minX = points.map(\.x).min() ?? 0
        let minY = points.map(\.y).min() ?? 0
        let maxX = points.map(\.x).max() ?? 0
        let width = min(bounds.width, max(140, maxX - minX))
        let label = CATextLayer()
        label.string = "\(text)  \(Int(confidence * 100))%"
        label.fontSize = 13
        label.contentsScale = traitCollection.displayScale
        label.foregroundColor = UIColor.white.cgColor
        label.backgroundColor = color.cgColor(alpha: 0.95)
        label.frame = CGRect(x: max(0, min(bounds.width - width, minX)),
                             y: max(0, minY - 21), width: width, height: 21)
        plateContainerLayer.addSublayer(label)
        labels.append(label)
    }
}
