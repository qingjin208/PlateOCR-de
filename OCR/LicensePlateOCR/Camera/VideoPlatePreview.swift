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

struct VideoPlatePreview: UIViewRepresentable {
    let player: AVPlayer
    let samples: [PlateVideoSample]

    func makeUIView(context: Context) -> VideoPlateView {
        let view = VideoPlateView()
        view.attach(player)
        view.samples = samples
        return view
    }

    func updateUIView(_ uiView: VideoPlateView, context: Context) {
        uiView.attach(player)
        uiView.samples = samples
    }

    static func dismantleUIView(_ uiView: VideoPlateView, coordinator: ()) {
        uiView.detach()
    }
}

final class VideoPlateView: UIView {
    /// Sampled frames plus the id currently drawn. Setting `samples` resets the selection
    /// so a fresh recognition pass repaints immediately.
    var samples: [PlateVideoSample] = [] {
        didSet {
            currentSampleID = nil
            updateOverlay(for: playerLayer.player?.currentTime().seconds ?? 0)
            refreshSoon()
        }
    }

    private var currentSampleID: Int?
    private var timeObserver: Any?
    private weak var observedPlayer: AVPlayer?

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
        detectionLayer.strokeColor = UIColor.systemYellow.cgColor
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

    /// Picks the sampled frame nearest to the playback position and redraws when it changes.
    private func updateOverlay(for seconds: Double) {
        guard !samples.isEmpty, seconds.isFinite else {
            if currentSampleID != nil {
                currentSampleID = nil
                renderOverlay()
            }
            return
        }
        guard let nearest = samples.min(by: { abs($0.time - seconds) < abs($1.time - seconds) }) else {
            return
        }
        guard nearest.id != currentSampleID else { return }
        currentSampleID = nearest.id
        renderOverlay()
    }

    private func renderOverlay() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        labels.forEach { $0.removeFromSuperlayer() }
        labels = []
        plateLayers.forEach { $0.removeFromSuperlayer() }
        plateLayers = []
        detectionLayer.path = nil
        CATransaction.commit()

        guard let id = currentSampleID,
              let sample = samples.first(where: { $0.id == id }) else { return }
        let videoRect = playerLayer.videoRect
        guard !videoRect.isEmpty, videoRect.width > 0, videoRect.height > 0 else { return }

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
            if let result { addLabel(for: result, color: color, quad: quad) }
        }
        // Plates carved out of a merged crop have no dedicated detection rectangle.
        for (index, plate) in sample.plates.enumerated() where !usedResultIndices.contains(index) {
            guard let path = quadPath(plate.plateQuad) else { continue }
            let color = PlateOverlayPalette.color(
                for: PlateOverlayPalette.key(text: plate.text, box: plate.boundingBox))
            addPlateLayer(path: path, color: color, recognized: true)
            addLabel(for: plate, color: color, quad: plate.plateQuad)
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

    private func addLabel(for plate: PlateResult, color: PlateOverlayColor, quad: [CGPoint]) {
        let videoRect = playerLayer.videoRect
        let points = quad.map { CGPoint(x: videoRect.minX + $0.x * videoRect.width,
                                        y: videoRect.minY + $0.y * videoRect.height) }
        let minX = points.map(\.x).min() ?? 0
        let minY = points.map(\.y).min() ?? 0
        let maxX = points.map(\.x).max() ?? 0
        let width = min(bounds.width, max(140, maxX - minX))
        let label = CATextLayer()
        label.string = "\(plate.text)  \(Int(plate.confidence * 100))%"
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
