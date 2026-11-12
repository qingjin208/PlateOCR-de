/*
 License Plate OCR SDK

 Abstract:
 Opt-in strengthening of the recognition pipeline ("enhanced mode").

 The SDK's default behaviour is deliberately conservative: it has to keep working for
 every existing caller and it was tuned against real captures. The switches below bundle
 the improvements that need field validation, so an App can turn them on (and off again)
 with a single toggle instead of forking the pipeline.

 Each sub-switch is only honoured while `isEnabled` is `true` — that keeps one master
 switch in the App while still allowing a single improvement to be isolated during a
 real-device test.
 */

import Foundation

public struct RecognitionEnhancements: Sendable, Equatable {

    /// Master switch. `false` = legacy behaviour for every sub-switch below.
    public var isEnabled: Bool

    /// Fold-aware scoring in the plate validator.
    ///
    /// The structural parser generates several city/series splits and picks the best one.
    /// Scoring used to reward only "city code is a real registration code", so a split
    /// that reached its answer by *folding* confusable characters (`B`→`8`, `S`→`5`, …)
    /// tied with the correct split and won by being generated first. With this on, every
    /// fold costs points, so a reading that reads the characters as they are wins.
    /// Example fixed by this switch: `B AB 123E` used to come back as `B-A 8123E`.
    public var foldAwareScoring: Bool

    /// Strict numeric part: the identification number never starts with `0`
    /// (FZV §10 numbers run 1…9999), so a leading zero means the OCR read the wrong
    /// glyph — usually an `O`/`Q`/`D` that was folded into the number, or a stray oval
    /// stamp. Rejecting those readings prevents `B-AB 0123` from being accepted.
    public var strictNumericPart: Bool

    /// Pose-keypoint quads for the overlays.
    ///
    /// A plate that owns (almost) the whole detection crop inherits the detection's four
    /// pose corners, so the App tints the actual plate outline instead of an axis-aligned
    /// box. It also stops the OCR observation box from being mapped linearly back into
    /// the frame when OCR ran on a perspective-corrected image — those two coordinate
    /// spaces are not linearly related.
    public var keypointQuads: Bool

    /// Adaptive cleaning of the rectified plate image before OCR.
    ///
    /// Masks the two geometrically-determined interference sources on a German plate:
    /// the EU blue strip (its "D" otherwise lands inside the read text) and circular
    /// registration/inspection seals (otherwise read as `O`/`B`/`8`/`D`/`E`). Both are
    /// detected adaptively — masking happens ONLY when the feature is actually found,
    /// so plates without seals (new cars, front plates only, blurry shots) are passed
    /// through untouched. Parameters calibrated on real crops (Scripts/plate_cleaner_
    /// calibration.py): 4/13 seals masked, 0 characters damaged, EU strip correct in
    /// every crop that has one.
    public var plateImageCleaning: Bool

    /// - Parameters:
    ///   - isEnabled: Master switch (default `false`, i.e. legacy behaviour).
    ///   - foldAwareScoring: See `foldAwareScoring`.
    ///   - strictNumericPart: See `strictNumericPart`.
    ///   - keypointQuads: See `keypointQuads`.
    public init(isEnabled: Bool = false,
                foldAwareScoring: Bool = true,
                strictNumericPart: Bool = true,
                keypointQuads: Bool = true,
                plateImageCleaning: Bool = true) {
        self.isEnabled = isEnabled
        self.foldAwareScoring = foldAwareScoring
        self.strictNumericPart = strictNumericPart
        self.keypointQuads = keypointQuads
        self.plateImageCleaning = plateImageCleaning
    }

    /// Legacy behaviour: every strengthening switch bypassed.
    public static let off = RecognitionEnhancements(isEnabled: false)

    /// Everything on. The Demo app starts here; a production caller should only opt in
    /// after validating against its own capture conditions.
    public static let on = RecognitionEnhancements(isEnabled: true)

    // MARK: - Effective flags
    //
    // A sub-switch only counts while the master switch is on, so call sites read these
    // instead of the raw properties.

    public var effectiveFoldAwareScoring: Bool { isEnabled && foldAwareScoring }
    public var effectiveStrictNumericPart: Bool { isEnabled && strictNumericPart }
    public var effectiveKeypointQuads: Bool { isEnabled && keypointQuads }
    public var effectivePlateImageCleaning: Bool { isEnabled && plateImageCleaning }

    /// Short human-readable state, for on-device diagnostics.
    public var summary: String {
        guard isEnabled else { return "关闭" }
        var parts: [String] = []
        if foldAwareScoring { parts.append("折叠排序") }
        if strictNumericPart { parts.append("数字段严格") }
        if keypointQuads { parts.append("角点四边形") }
        return parts.isEmpty ? "已开启（无生效子项）" : "已开启（\(parts.joined(separator: "、"))）"
    }
}
