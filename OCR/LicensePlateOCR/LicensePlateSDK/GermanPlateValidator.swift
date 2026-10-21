/*
 License Plate OCR SDK

 Abstract:
 German license-plate validation and canonicalization. A cropped plate region often
 contains interference (the blue EU strip, the "D" oval, the city seal, screws, frame
 borders, reflections, or a neighbour plate). This module turns noisy OCR output into a
 clean, format-valid plate string — or rejects it when nothing plausible was read.

 Two things make the difference between "almost right" and "right":
   1. Position-aware confusion folding. The city/series part can only contain letters
      and the numeric part only digits, so an `O` there is a `0` and a `0` here is an
      `O`. That constraint resolves readings that are otherwise ambiguous.
   2. A registration-code preference (see `GermanPlateLexicon`) so the city/series
      boundary is chosen sensibly instead of taking the first split that type-checks.

 Both parsers therefore generate several candidate splits and score them. The scoring
 has two modes, selected by `RecognitionEnhancements.foldAwareScoring` — see
 `score(city:folds:enhancements:)`.
 */

import Foundation

public enum GermanPlateValidator {

    /// Civilian format: `<1–3 city letters> - <1–2 series letters> <1–4 digits> <optional E/H>`.
    /// Example: `B-AB 1234`, `M-XY 99`, `HH-AB 1234`, `K-AB 123Z` (invalid: `Z`).
    ///
    /// The trailing suffix is restricted to `E` (Elektrofahrzeug) and `H` (Oldtimer).
    /// An arbitrary `[A-Z]` used to be accepted, which let OCR noise pass as a plate.
    private static let pattern = "^[A-ZÄÖÜ]{1,3}-[A-ZÄÖÜ]{1,2}[ ]?[0-9]{1,4}[EH]?$"

    /// Whether `text` already matches the canonical German plate format.
    public static func isValid(_ text: String) -> Bool {
        guard let match = text.range(of: pattern, options: .regularExpression),
              match == (text.startIndex..<text.endIndex) else { return false }
        // FZV §10: the distinguishing mark plus the identification number never exceed
        // eight characters; the E/H marker is carried in addition to them.
        var core = text.filter { $0 != "-" && $0 != " " }
        if let last = core.last, suffixLetters.contains(last) { core.removeLast() }
        return core.count <= 8
    }

    // MARK: - OCR confusion tables
    //
    // Applied *by position*, never globally: a digit in the letter region is only ever
    // reinterpreted as a letter, and vice versa. A blind global substitution would corrupt
    // readings that were already correct.

    /// Digits that OCR renders as letters inside the city/series region.
    private static let letterForDigit: [Character: Character] = [
        "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"
    ]

    /// Letters that OCR renders as digits inside the numeric region.
    private static let digitForLetter: [Character: Character] = [
        "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
        "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"
    ]

    /// Legal trailing letters (E = electric, H = historic). Not "any letter".
    private static let suffixLetters: Set<Character> = ["E", "H"]

    // MARK: - Canonicalization

    /// Turns a raw OCR string into a canonical plate string, or `nil` if it cannot be
    /// plausibly interpreted as a German plate.
    ///
    /// Steps (interference handling):
    /// 1. Uppercase and drop stray symbols / frame noise.
    /// 2. Resolve a leading EU-strip artefact (`D` / `EU`) against the reading that keeps
    ///    it as a city code (`D AB 1234` is Dresden, not "A-B 1234").
    /// 3. Use the explicit separators Vision preserved, when there are any.
    /// 4. Otherwise parse the structure positionally: letters, then digits, then an
    ///    optional `E`/`H`, folding confusable characters along the way.
    ///
    /// - Parameter enhancements: Opt-in strengthening (default `.off` = legacy output).
    public static func canonicalize(_ raw: String,
                                    enhancements: RecognitionEnhancements = .off) -> String? {
        let upper = raw.uppercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !upper.isEmpty else { return nil }

        if let resolved = resolveEUStrip(upper, enhancements: enhancements) { return resolved }
        if let separated = parseSeparated(upper, enhancements: enhancements) { return separated }

        let alnum = upper.filter { $0.isLetter || $0.isNumber }
        guard !alnum.isEmpty else { return nil }
        let full = best(from: candidates(in: alnum, enhancements: enhancements))

        // Last resort: a marker that survived the structural parse (`DBAB1234`). Only
        // prefer the stripped reading when it yields a real registration code.
        var alternatives: [Parsed] = []
        let fullHasKnownCity = full.map { GermanPlateLexicon.contains($0.city) } ?? false
        if !fullHasKnownCity {
            for marker in ["EU", "D"] where alnum.hasPrefix(marker) {
                if let stripped = best(from: candidates(in: String(alnum.dropFirst(marker.count)),
                                                       enhancements: enhancements)) {
                    alternatives.append(stripped)
                }
            }
        }
        if let known = alternatives.first(where: { GermanPlateLexicon.contains($0.city) }) {
            return known.text
        }
        return full?.text ?? alternatives.first?.text
    }

    // MARK: - Parsing

    /// A structurally valid reading plus the score used to prefer one split over another.
    private struct Parsed {
        let text: String
        let city: String
        let score: Int
    }

    /// Resolves a leading `D` / `EU` (blue strip) against the same string read as a plate.
    /// `D AB 1234` must stay `D-AB 1234`; `D HH AB 1234` and `EU HH-AB 1234` must lose the
    /// marker. The unstripped reading wins whenever its city code is a real one.
    private static func resolveEUStrip(_ s: String,
                                       enhancements: RecognitionEnhancements) -> String? {
        let tokens = s.split(whereSeparator: { $0 == " " || $0 == "-" }).map(String.init)
        guard tokens.count >= 3, let marker = tokens.first, marker == "D" || marker == "EU" else {
            return nil
        }
        let remainder = tokens.dropFirst().joined()
        guard !remainder.isEmpty else { return nil }
        if let unstripped = best(from: candidates(in: tokens.joined(), enhancements: enhancements)),
           GermanPlateLexicon.contains(unstripped.city) {
            return unstripped.text
        }
        return best(from: candidates(in: remainder, enhancements: enhancements))?.text
    }

    /// Uses the separators Vision preserved (`-`, spaces) to fix the city/series boundary.
    private static func parseSeparated(_ s: String,
                                       enhancements: RecognitionEnhancements) -> String? {
        let tokens = s.split(whereSeparator: { $0 == "-" || $0 == " " || $0 == "." || $0 == "/" })
        guard tokens.count >= 2 else { return nil }
        let head = Array(tokens[0])
        // A city code always contains a letter: this keeps "1 2 3" from parsing as "I-Z 3".
        guard (1...3).contains(head.count), head.contains(where: { $0.isLetter }),
              let headFold = foldToLetters(head) else { return nil }
        let city = headFold.letters
        let rest = Array(tokens.dropFirst().joined().filter { $0.isLetter || $0.isNumber })
        guard !rest.isEmpty else { return nil }

        var readings: [Parsed] = []
        for seriesLength in 1...max(1, min(2, rest.count - 1)) {
            guard rest.count > seriesLength else { continue }
            let seriesChars = Array(rest.prefix(seriesLength))
            // The series is always letters. Folding a digit into it would let a 5-digit
            // number masquerade as a plate (`B AB 12345` → `BA-BI 2345`), so folding is
            // only ever applied to the numeric tail.
            guard seriesChars.allSatisfy({ $0.isLetter }),
                  let seriesFold = foldToLetters(seriesChars),
                  let tail = parseNumberTail(Array(rest.dropFirst(seriesLength)),
                                             enhancements: enhancements) else { continue }
            let series = seriesFold.letters
            let folds = headFold.folds + seriesFold.folds + tail.folds
            let text = "\(city)-\(series) \(tail.digits)\(tail.suffix.map { String($0) } ?? "")"
            guard isValid(text) else { continue }
            readings.append(Parsed(text: text, city: city,
                                   score: score(city: city, folds: folds, enhancements: enhancements)))
        }
        return best(from: readings)?.text
    }

    /// Structural parse with no reliable separator: every plausible city/series boundary
    /// is generated and scored, because OCR often drops the hyphen entirely.
    private static func candidates(in alnum: String,
                                   enhancements: RecognitionEnhancements) -> [Parsed] {
        let chars = Array(alnum)
        guard chars.count >= 4 else { return [] }
        var readings: [Parsed] = []
        for cityLength in 1...min(3, chars.count - 2) {
            let cityChars = Array(chars.prefix(cityLength))
            guard cityChars.contains(where: { $0.isLetter }),
                  let cityFold = foldToLetters(cityChars) else { continue }
            let city = cityFold.letters
            let remainder = Array(chars.dropFirst(cityLength))
            for seriesLength in 1...min(2, remainder.count - 1) {
                let seriesChars = Array(remainder.prefix(seriesLength))
                // See `parseSeparated`: the series must be real letters, otherwise folding
                // would swallow an over-long digit run into the letter region.
                guard seriesChars.allSatisfy({ $0.isLetter }),
                      let seriesFold = foldToLetters(seriesChars),
                      let tail = parseNumberTail(Array(remainder.dropFirst(seriesLength)),
                                                enhancements: enhancements) else { continue }
                let series = seriesFold.letters
                let folds = cityFold.folds + seriesFold.folds + tail.folds
                let text = "\(city)-\(series) \(tail.digits)\(tail.suffix.map { String($0) } ?? "")"
                guard isValid(text) else { continue }
                readings.append(Parsed(text: text, city: city,
                                       score: score(city: city, folds: folds, enhancements: enhancements)))
            }
        }
        return readings
    }

    /// 1–4 digits, folding confusable letters (`O`→`0`, `B`→`8`, …), then an optional
    /// `E`/`H`. Returns `nil` when the remainder cannot be a plate's numeric part.
    ///
    /// `folds` counts how many characters had to be reinterpreted: the caller uses it to
    /// prefer readings that read the glyphs literally.
    private static func parseNumberTail(
        _ chars: [Character],
        enhancements: RecognitionEnhancements
    ) -> (digits: String, suffix: Character?, folds: Int)? {
        guard !chars.isEmpty else { return nil }
        var digits = ""
        var index = 0
        var folds = 0
        while index < chars.count, digits.count < 4 {
            let character = chars[index]
            if character.isNumber {
                digits.append(character)
            } else if let digit = digitForLetter[character] {
                digits.append(digit)
                folds += 1
            } else {
                break
            }
            index += 1
        }
        guard !digits.isEmpty else { return nil }
        // FZV §10 numbers run 1…9999: a leading zero can only come from a misread glyph
        // (typically a folded `O`/`Q`/`D`) or a stray stamp. Only rejected when the App
        // opted into the strict numeric part.
        if enhancements.effectiveStrictNumericPart, digits.hasPrefix("0") { return nil }
        if index == chars.count { return (digits: digits, suffix: nil, folds: folds) }
        guard chars.count - index == 1, suffixLetters.contains(chars[index]) else { return nil }
        return (digits: digits, suffix: chars[index], folds: folds)
    }

    /// Ranks one reading above another.
    ///
    /// Legacy mode only rewards a real registration code (`2`), so two readings that are
    /// both plausible tie and the first generated one wins — which is the split with the
    /// shortest series, i.e. the one that reaches its digits by folding (`AB 123E` loses
    /// to `A` + `8123E`). Enhanced mode adds a penalty per fold, so the literal reading
    /// wins: `B AB 123E` → `B-AB 123E` instead of `B-A 8123E`.
    private static func score(city: String,
                              folds: Int,
                              enhancements: RecognitionEnhancements) -> Int {
        let known = GermanPlateLexicon.contains(city)
        guard enhancements.effectiveFoldAwareScoring else { return known ? 2 : 0 }
        return (known ? 4 : 0) - 2 * folds
    }

    private static func best(from readings: [Parsed]) -> Parsed? {
        var winner: Parsed?
        for reading in readings {
            // No sentinel score: with fold penalties a score can be negative, and a `-1`
            // seed would then discard every reading.
            if let current = winner {
                if reading.score > current.score { winner = reading }
            } else {
                winner = reading
            }
        }
        return winner
    }

    /// Folds a character run into letters, mapping digits that OCR confuses with letters.
    /// `folds` reports how many characters were reinterpreted.
    private static func foldToLetters(_ chars: [Character]) -> (letters: String, folds: Int)? {
        var out = ""
        var folds = 0
        for character in chars {
            if character.isLetter {
                out.append(character)
            } else if let letter = letterForDigit[character] {
                out.append(letter)
                folds += 1
            } else {
                return nil
            }
        }
        guard !out.isEmpty else { return nil }
        return (letters: out, folds: folds)
    }
}
