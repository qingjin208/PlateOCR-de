/*
 License Plate OCR SDK

 Abstract:
 Cross-frame voting. Turns per-frame OCR results into a de-duplicated plate list where
 each physical plate appears once and its text follows the majority reading instead of
 whichever frame happened to be OCR'd last.

 Why this exists:
 Every frame is OCR'd independently, so a plate that stays in view for five seconds
 yields five readings — and some of them are wrong by a single glyph (`M-AB 1234` read
 as `M-AB 123A`). Both readings pass the format validator (both are structurally legal
 German plates), so without voting the UI lists every variant as its own result and a
 one-frame misread looks as authoritative as the correct reading.

 Two entry points:
 - `PlateVoteAggregator` (live camera): associates readings spatially across analysis
   cycles (IoU), so each tracked plate accumulates votes and is listed once. A reading
   is only listed after it repeats and holds a majority; a later, clearly stronger
   reading can still replace it.
 - `consolidate(_:)` (video): sampled frames are too far apart in time for spatial
   association, so votes are counted per text and near-identical texts (one confusable
   glyph apart, same city code) are merged into the majority reading.

 Association is spatial on purpose: two physically distinct plates can read the same
 text (a misread of one equals a legal reading of the other), and two same-city plates
 one digit apart are common on a car park. Merging by text alone would silently drop a
 car; merging by track keeps them separate.
 */

import Foundation
import CoreGraphics

// MARK: - Settings

/// Thresholds for cross-frame voting. Defaults are tuned for the live camera's ~5 fps
/// analysis cadence; a caller that analyses less often should lower `minimumVotes` (or
/// raise `provisionalConfidence`) so a plate that is only seen once is not dropped.
public struct PlateVoteSettings: Sendable, Equatable {
    /// Minimum IoU between a reading's box and a track's box to count as the same plate.
    public var associationIoU: CGFloat
    /// Votes a reading needs before it is listed.
    public var minimumVotes: Int
    /// Share of a track's votes the reading must hold (0...1).
    public var minimumAgreement: Double
    /// A track that disappears after a single sighting is still listed when that reading's
    /// OCR confidence is at least this high (protects plates that only flashed by).
    public var provisionalConfidence: Float
    /// How long a track survives without a matching detection.
    ///
    /// Must comfortably exceed one analysis cycle or the whole voting mechanism silently
    /// stops working: `ingest` expires stale tracks *before* absorbing the frame, so with
    /// a timeout shorter than the cycle time every track would be retired on every frame
    /// and no reading would ever reach `minimumVotes` (the "history stays empty on a slow
    /// phone" symptom). A cycle is `max(0.2 s gate, inference + OCR)`, and a thermally
    /// throttled device can easily take 1 s — hence 2 s rather than 1 s.
    public var trackTimeout: TimeInterval
    /// Extra votes a rival reading needs before it replaces an already-listed one.
    public var overrideMargin: Int
    /// How soon after a plate left the frame a differently-read sighting still counts as
    /// the same plate. Guards the "confusable variant" merge: a plate that comes back
    /// within this window may be read one glyph differently (and the stronger reading
    /// corrects the row), while a different car arriving later must get its own row.
    public var variantAdoptionWindow: TimeInterval
    /// Upper bound on listed plates (oldest dropped first).
    public var maximumEntries: Int

    public init(associationIoU: CGFloat = 0.3,
                minimumVotes: Int = 2,
                minimumAgreement: Double = 0.5,
                provisionalConfidence: Float = 0.85,
                trackTimeout: TimeInterval = 2.0,
                overrideMargin: Int = 2,
                variantAdoptionWindow: TimeInterval = 5.0,
                maximumEntries: Int = 300) {
        self.associationIoU = associationIoU
        self.minimumVotes = minimumVotes
        self.minimumAgreement = minimumAgreement
        self.provisionalConfidence = provisionalConfidence
        self.trackTimeout = trackTimeout
        self.overrideMargin = overrideMargin
        self.variantAdoptionWindow = variantAdoptionWindow
        self.maximumEntries = maximumEntries
    }

    public static let `default` = PlateVoteSettings()
}

// MARK: - Aggregate

/// One listed plate: the winning reading plus how it was reached.
public struct PlateVoteAggregate: Identifiable, Sendable {
    /// Stable across updates, so a SwiftUI list row is refreshed instead of recreated.
    public let id: UUID
    /// The majority reading (carries its own corrected image).
    public let result: PlateResult
    /// Votes for that reading.
    public let votes: Int
    /// All readings seen for this plate, including the rejected variants.
    public let totalVotes: Int
    public let firstSeen: Date
    public let lastSeen: Date

    /// 0...1 — how one-sided the vote was. 1.0 = every frame agreed.
    public var agreement: Double { totalVotes > 0 ? Double(votes) / Double(totalVotes) : 0 }

    public init(id: UUID, result: PlateResult, votes: Int, totalVotes: Int,
                firstSeen: Date, lastSeen: Date) {
        self.id = id
        self.result = result
        self.votes = votes
        self.totalVotes = totalVotes
        self.firstSeen = firstSeen
        self.lastSeen = lastSeen
    }
}

// MARK: - Aggregator

/// Accumulates per-frame readings into one entry per physical plate.
public struct PlateVoteAggregator: Sendable {

    private struct Track {
        let id: UUID
        var rect: CGRect
        var firstSeen: Date
        var lastSeen: Date
        /// text -> votes.
        var votes: [String: Int]
        /// text -> highest-confidence frame result carrying that text (keeps its image).
        var best: [String: PlateResult]
        /// Listed entry this track is currently confirming, if any.
        var entryID: UUID?
        var totalVotes: Int
    }

    private struct Entry {
        let id: UUID
        var text: String
        var votes: Int
        var totalVotes: Int
        var best: PlateResult
        var firstSeen: Date
        var lastSeen: Date
        /// Live track currently attached to this entry — used to keep two plates that are
        /// in frame at the same time from being merged into one entry.
        var boundTrack: UUID?
    }

    public private(set) var settings: PlateVoteSettings
    private var tracks: [Track] = []
    private var entries: [Entry] = []

    public init(settings: PlateVoteSettings = .default) {
        self.settings = settings
    }

    /// Clears everything (history reset).
    public mutating func reset() {
        tracks.removeAll()
        entries.removeAll()
    }

    /// The listed plates, oldest first (the UI reverses for "newest on top").
    public func aggregates() -> [PlateVoteAggregate] {
        entries
            .sorted { $0.firstSeen < $1.firstSeen }
            .map {
                PlateVoteAggregate(id: $0.id,
                                   result: $0.best.withVotes($0.votes),
                                   votes: $0.votes,
                                   totalVotes: $0.totalVotes,
                                   firstSeen: $0.firstSeen,
                                   lastSeen: $0.lastSeen)
            }
    }

    /// Folds one frame's results into the tracks and returns the full listed set.
    ///
    /// Returning the whole set (rather than just the new entries) keeps the caller simple:
    /// the history list *is* this set, so a later vote that corrects a reading updates the
    /// existing row instead of appending a second one.
    @discardableResult
    public mutating func ingest(_ results: [PlateResult], at date: Date = Date()) -> [PlateVoteAggregate] {
        expireTracks(before: date)
        for result in results { absorb(result, at: date) }
        return aggregates()
    }

    /// Closes the current session: every live track is treated as having left the frame,
    /// so a confident single sighting still gets listed, and nothing is left behind for
    /// the next session to revive.
    ///
    /// Call this when the camera stops or the app goes to the background. Without it two
    /// things go wrong: the last plate that was only seen once never reaches the list
    /// (provisional confirmation happens on expiry, and a frozen camera never expires
    /// anything), and on restart the first `ingest` expires the stale tracks and lists a
    /// plate the user saw half a minute ago.
    @discardableResult
    public mutating func finish() -> [PlateVoteAggregate] {
        let live = tracks
        tracks.removeAll()
        for track in live { retire(track) }
        return aggregates()
    }

    // MARK: Track handling

    private mutating func absorb(_ result: PlateResult, at date: Date) {
        var matchIndex: Int?
        var bestIoU = settings.associationIoU
        for (index, track) in tracks.enumerated() {
            let score = Self.iou(track.rect, result.boundingBox)
            if score >= bestIoU { bestIoU = score; matchIndex = index }
        }

        if let index = matchIndex {
            tracks[index].rect = Self.smooth(tracks[index].rect, result.boundingBox)
            tracks[index].lastSeen = date
            tracks[index].votes[result.text, default: 0] += 1
            tracks[index].totalVotes += 1
            let existing = tracks[index].best[result.text]
            if existing == nil || result.confidence > (existing?.confidence ?? 0) {
                tracks[index].best[result.text] = result
            }
        } else {
            tracks.append(Track(id: UUID(),
                                rect: result.boundingBox,
                                firstSeen: date,
                                lastSeen: date,
                                votes: [result.text: 1],
                                best: [result.text: result],
                                entryID: nil,
                                totalVotes: 1))
            matchIndex = tracks.count - 1
        }

        if let index = matchIndex { evaluate(trackIndex: index) }
    }

    /// Drops tracks that went away, freeing their entry so the same plate can re-attach
    /// to it when it comes back into view (instead of being listed twice).
    private mutating func expireTracks(before date: Date) {
        var survivors: [Track] = []
        survivors.reserveCapacity(tracks.count)
        for track in tracks {
            if date.timeIntervalSince(track.lastSeen) > settings.trackTimeout {
                retire(track)
            } else {
                survivors.append(track)
            }
        }
        tracks = survivors
    }

    /// Releases a track that is no longer being observed: frees the entry it was bound to
    /// and, when the plate only ever showed up once, lists it if that reading was
    /// confident. Shared by `expireTracks` (normal timeout) and `finish` (session end).
    private mutating func retire(_ track: Track) {
        for index in entries.indices where entries[index].boundTrack == track.id {
            entries[index].boundTrack = nil
        }
        // A plate that only showed up in one analysis cycle: list it when that single
        // reading was confident, so fast-moving cars are not silently dropped.
        if track.entryID == nil,
           let top = Self.strongest(track),
           top.votes == 1,
           let best = track.best[top.text],
           best.confidence >= settings.provisionalConfidence {
            confirm(track: track, text: top.text, bindTrack: false)
        }
    }

    /// Decides whether the track's current vote leader should be listed, replaced, or
    /// left alone.
    private mutating func evaluate(trackIndex: Int) {
        guard tracks.indices.contains(trackIndex) else { return }
        let track = tracks[trackIndex]
        guard let top = Self.strongest(track), top.votes > 0 else { return }
        let agreement = track.totalVotes > 0 ? Double(top.votes) / Double(track.totalVotes) : 0

        if let entryID = track.entryID,
           let index = entries.firstIndex(where: { $0.id == entryID }) {
            // Already listed and still voting the same text: refresh the counts in place so
            // the row's "5/6 帧" keeps growing instead of freezing at the confirming frame.
            if entries[index].text == top.text {
                entries[index].votes = max(entries[index].votes, top.votes)
                entries[index].totalVotes = max(entries[index].totalVotes, track.totalVotes)
                entries[index].lastSeen = track.lastSeen
                if let best = track.best[top.text], best.confidence > entries[index].best.confidence {
                    entries[index].best = best
                }
                return
            }
            // A different reading won the vote. Only a clearly stronger rival replaces the
            // listed text, so the row follows the majority instead of the latest frame.
            guard top.votes >= entries[index].votes + settings.overrideMargin,
                  agreement >= settings.minimumAgreement,
                  let best = track.best[top.text] else { return }
            entries[index].text = top.text
            entries[index].best = best
            entries[index].votes = top.votes
            entries[index].totalVotes = max(entries[index].totalVotes, track.totalVotes)
            entries[index].lastSeen = track.lastSeen
            return
        }

        guard top.votes >= settings.minimumVotes,
              agreement >= settings.minimumAgreement else { return }

        if let id = confirm(track: track, text: top.text, bindTrack: true) {
            tracks[trackIndex].entryID = id
        }
    }

    /// Adds the track's winning reading to the listed set — merging into an existing
    /// entry for the same plate when that entry is not already taken by a live track.
    @discardableResult
    private mutating func confirm(track: Track, text: String, bindTrack: Bool) -> UUID? {
        guard let best = track.best[text] else { return nil }
        let votes = track.votes[text] ?? 0

        let mergeIndex = entries.firstIndex {
            Self.isSamePlate($0.text, text) && ($0.boundTrack == nil || $0.boundTrack == track.id)
        }
        if let index = mergeIndex {
            var entry = entries[index]
            if entry.text == text {
                // Same reading: the returning / re-confirmed sighting simply adds votes.
                entry.votes += votes
                entry.totalVotes += track.totalVotes
                if best.confidence > entry.best.confidence { entry.best = best }
            } else if votes >= entry.votes,
                      track.firstSeen.timeIntervalSince(entry.lastSeen) <= settings.variantAdoptionWindow {
                // A confusable variant (the plate left and came back reading one glyph
                // differently) that is at least as strong as what the row holds, and soon
                // enough after the plate left to be the same vehicle. Adopt text, image and
                // counts *together* — updating `best` alone would leave a row showing one
                // text with another text's vote count, and the next `evaluate` would
                // compare against a text the row no longer displays.
                entry.text = text
                entry.best = best
                entry.votes = votes
                entry.totalVotes = track.totalVotes
            } else {
                // A weaker variant of the listed reading: it did see this plate, so it
                // counts in the denominator, but it must not touch the listed text or the
                // agreeing count (otherwise a one-off misread would inflate agreement).
                entry.totalVotes += track.totalVotes
            }
            entry.lastSeen = max(entry.lastSeen, track.lastSeen)
            if bindTrack { entry.boundTrack = track.id }
            entries[index] = entry
            return entry.id
        }

        let entry = Entry(id: UUID(),
                          text: text,
                          votes: votes,
                          totalVotes: track.totalVotes,
                          best: best,
                          firstSeen: track.firstSeen,
                          lastSeen: track.lastSeen,
                          boundTrack: bindTrack ? track.id : nil)
        entries.append(entry)
        trimEntries()
        return entry.id
    }

    private mutating func trimEntries() {
        guard entries.count > settings.maximumEntries else { return }
        let overflow = entries.count - settings.maximumEntries
        // Drop by `firstSeen`, not by array position: an entry that was merged into or
        // adopted from a returning track keeps its original slot in the array but may
        // carry a much newer timestamp, so `prefix(overflow)` could evict a young row and
        // keep a stale one — and `aggregates()` sorts by `firstSeen`, so the eviction
        // order has to match what the list actually shows.
        let removed = Set(entries.sorted { $0.firstSeen < $1.firstSeen }.prefix(overflow).map(\.id))
        entries.removeAll { removed.contains($0.id) }
        for index in tracks.indices {
            if let entryID = tracks[index].entryID, removed.contains(entryID) {
                tracks[index].entryID = nil
            }
        }
    }

    // MARK: Helpers

    private static func strongest(_ track: Track) -> (text: String, votes: Int)? {
        var best: (text: String, votes: Int)?
        for (text, votes) in track.votes {
            guard let current = best else { best = (text, votes); continue }
            let confidence = track.best[text]?.confidence ?? 0
            let currentConfidence = track.best[current.text]?.confidence ?? 0
            if votes > current.votes
                || (votes == current.votes && confidence > currentConfidence)
                || (votes == current.votes && confidence == currentConfidence && text < current.text) {
                best = (text, votes)
            }
        }
        return best
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return 0 }
        let area = intersection.width * intersection.height
        let union = a.width * a.height + b.width * b.height - area
        return union > 0 ? area / union : 0
    }

    private static func smooth(_ a: CGRect, _ b: CGRect) -> CGRect {
        let t: CGFloat = 0.5
        return CGRect(x: a.minX + (b.minX - a.minX) * t,
                      y: a.minY + (b.minY - a.minY) * t,
                      width: a.width + (b.width - a.width) * t,
                      height: a.height + (b.height - a.height) * t)
    }

    /// Whether two canonical readings can be the same physical plate: identical, or one
    /// confusable glyph apart inside the same registration code.
    ///
    /// Deliberately narrow — `M-AB 1234` and `M-AB 1235` are **not** the same plate (two
    /// cars can be parked side by side), while `M-AB 1234` and `M-AB 123A` are (4/A is a
    /// known confusion pair, see `GermanPlateValidator`'s position-aware folding).
    static func isSamePlate(_ a: String, _ b: String) -> Bool {
        if a == b { return true }
        let charsA = Array(a), charsB = Array(b)
        guard charsA.count == charsB.count else { return false }
        let cityA = String(a.prefix { $0 != "-" })
        let cityB = String(b.prefix { $0 != "-" })
        guard cityA == cityB else { return false }

        var difference: (Character, Character)?
        for index in charsA.indices where charsA[index] != charsB[index] {
            guard difference == nil else { return false }
            difference = (charsA[index], charsB[index])
        }
        guard let difference else { return false }
        return confusable(difference.0, difference.1)
    }

    /// Unordered glyph pairs the validator is allowed to fold into one another.
    private static let confusablePairs: Set<String> = {
        let pairs: [(String, String)] = [("0", "O"), ("0", "Q"), ("0", "D"), ("1", "I"),
                                         ("1", "L"), ("2", "Z"), ("4", "A"), ("5", "S"),
                                         ("6", "G"), ("7", "T"), ("8", "B")]
        var set = Set<String>()
        for (left, right) in pairs {
            set.insert(left + right)
            set.insert(right + left)
        }
        return set
    }()

    private static func confusable(_ a: Character, _ b: Character) -> Bool {
        confusablePairs.contains(String(a) + String(b))
    }

    // MARK: - Video consolidation

    /// De-duplicates and majority-votes readings collected across sampled video frames.
    ///
    /// Sampled frames are too far apart for spatial association, so votes are counted per
    /// text: the most-seen reading wins, and near-identical readings (`…1234` vs `…123A`,
    /// same city code) are absorbed into it when they are clearly in the minority — a
    /// one-off misread must not get its own row next to the correct plate. Two readings
    /// that both repeat often are kept apart, because they may be two real plates.
    ///
    /// A reading seen in a single sampled frame is listed only when that sighting was
    /// confident (`provisionalConfidence`), which is the same bar the live aggregator
    /// applies to a plate that flashed by once. Sampled video is sparse (60 frames can
    /// cover a minute), so a passing car legitimately shows up once — but so does a
    /// one-frame misread, and listing it next to the real plates is exactly the "wrong
    /// results shown" case. Pass `minimumVotes: 1` when the caller has already filtered
    /// (e.g. a hand-picked single-frame extraction).
    ///
    /// - Parameters:
    ///   - minimumVotes: Frames that must agree before a reading is listed (default 2).
    ///   - provisionalConfidence: Escape hatch for a plate seen exactly once: listed when
    ///     its OCR confidence reaches this value (default 0.85, mirrors
    ///     `PlateVoteSettings.provisionalConfidence`).
    ///   - mergeRatio: How much stronger the leader must be before a confusable variant
    ///     is absorbed into it (default 4:1).
    ///   - singleFrameFallback: When every reading failed the bars above but the sampled
    ///     frames did contain OCR results, list the best readings with their (weak) vote
    ///     counts instead of returning an empty page. A blank result list after the model
    ///     demonstrably read something is a debugging dead end — the UI shows "1/1 帧"
    ///     so the weakness stays visible. Default `true`.
    public static func consolidate(_ frames: [[PlateResult]],
                                   minimumVotes: Int = 2,
                                   provisionalConfidence: Float = 0.85,
                                   mergeRatio: Double = 4,
                                   singleFrameFallback: Bool = true) -> [PlateResult] {
        var counts: [String: Int] = [:]
        var best: [String: PlateResult] = [:]
        for frame in frames {
            // One vote per text per frame: a merged detection reading the same plate twice
            // must not outvote another frame.
            var seen = Set<String>()
            for result in frame {
                guard seen.insert(result.text).inserted else { continue }
                counts[result.text, default: 0] += 1
                if let existing = best[result.text] {
                    if result.confidence > existing.confidence { best[result.text] = result }
                } else {
                    best[result.text] = result
                }
            }
        }

        let ordered = counts.keys.sorted {
            let leftVotes = counts[$0] ?? 0, rightVotes = counts[$1] ?? 0
            if leftVotes != rightVotes { return leftVotes > rightVotes }
            let leftConfidence = best[$0]?.confidence ?? 0, rightConfidence = best[$1]?.confidence ?? 0
            if leftConfidence != rightConfidence { return leftConfidence > rightConfidence }
            return $0 < $1
        }

        var absorbed = Set<String>()
        var output: [PlateResult] = []
        for text in ordered where !absorbed.contains(text) {
            absorbed.insert(text)
            let leaderVotes = counts[text] ?? 0
            var clusterVotes = leaderVotes
            for other in ordered where !absorbed.contains(other) && isSamePlate(text, other) {
                let votes = counts[other] ?? 0
                // Absorb only a clearly weaker reading: a single sighting against at least
                // three, or a 4:1 minority. Otherwise they may be two distinct plates.
                if Double(votes) * mergeRatio <= Double(leaderVotes) || (votes <= 1 && leaderVotes >= 3) {
                    absorbed.insert(other)
                    clusterVotes += votes
                }
            }
            guard let result = best[text] else { continue }
            // Two agreeing frames, or one sighting that was confident enough to stand on
            // its own (see the doc comment: this is the same escape hatch the live
            // aggregator uses for a plate that only flashed by).
            let listed = clusterVotes >= minimumVotes
                || (clusterVotes == 1 && result.confidence >= provisionalConfidence)
            guard listed else { continue }
            output.append(result.withVotes(clusterVotes))
        }

        // Never return a blank page when the model did read something: list the best
        // readings as-is (their `votes` reflect the weak evidence). This is the escape
        // hatch for videos where the plate only becomes readable in a couple of frames.
        if singleFrameFallback, output.isEmpty, !counts.isEmpty {
            for text in ordered.prefix(3) {
                if let result = best[text] {
                    output.append(result.withVotes(counts[text] ?? 1))
                }
            }
        }

        return output.sorted {
            if $0.votes != $1.votes { return $0.votes > $1.votes }
            return $0.confidence > $1.confidence
        }
    }
}
