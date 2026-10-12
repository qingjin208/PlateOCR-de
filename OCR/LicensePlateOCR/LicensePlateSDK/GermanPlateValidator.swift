/*
 License Plate OCR SDK
 
 Abstract:
 German license-plate validation and canonicalization. A cropped plate region often
 contains interference (the blue EU strip, the "D" oval, the city seal, screws, frame
 borders, reflections, or a neighbour plate). This module turns noisy OCR output into a
 clean, format-valid plate string — or rejects it when nothing plausible was read.
 */

import Foundation

public enum GermanPlateValidator {

    /// Loose but realistic civilian format:
    /// `<1–3 city letters> - <1–2 series letters> <1–4 digits> <optional 1 letter>`.
    /// Example: `B-AB 1234`, `M-XY 99`, `HH-AB 1234`, `K-AB 123Z`.
    private static let pattern = "^[A-ZÄÖÜ]{1,3}-[A-ZÄÖÜ]{1,2}[ ]?[0-9]{1,4}[A-Z]?$"

    /// Whether `text` already matches the canonical German plate format.
    public static func isValid(_ text: String) -> Bool {
        text.range(of: pattern, options: .regularExpression) != nil
    }

    /// Turns a raw OCR string into a canonical plate string, or `nil` if it cannot be
    /// plausibly interpreted as a German plate.
    ///
    /// Steps (interference handling):
    /// 1. Uppercase and drop everything that is not a letter/digit (stray symbols, frame noise).
    /// 2. Drop a leading EU-strip artefact (`D` / `EU`) if that makes the remainder valid.
    /// 3. Re-insert the separating hyphen at every valid city/series split.
    public static func canonicalize(_ raw: String) -> String? {
        let upper = raw.uppercased().trimmingCharacters(in: .whitespacesAndNewlines)
        // Preserve explicit city/series separators before flattening noisy OCR.
        if let range = upper.range(of: #"^[A-ZÄÖÜ]{1,3}[ -]+[A-ZÄÖÜ]{1,2}[ ]*[0-9]{1,4}[A-Z]?$"#,
                                   options: .regularExpression) {
            let value = String(upper[range])
            if let separator = value.firstIndex(where: { $0 == " " || $0 == "-" }) {
                let city = value[..<separator]
                let suffix = value[separator...].filter { $0 != " " && $0 != "-" }
                if let digit = suffix.firstIndex(where: \.isNumber) {
                    let candidate = "\(city)-\(suffix[..<digit]) \(suffix[digit...])"
                    if isValid(candidate) { return candidate }
                }
            }
        }
        let alnum = raw.uppercased().unicodeScalars
            .filter { CharacterSet.alphanumerics.contains($0) }
            .map(String.init)
            .joined()
        guard !alnum.isEmpty else { return nil }

        if let direct = tryAsPlate(alnum) { return direct }

        // The blue EU strip shows a "D"; the city seal or reflection sometimes bleeds in.
        for drop in [2, 1] {
            let trimmed = String(alnum.dropFirst(drop))
            if !trimmed.isEmpty, let candidate = tryAsPlate(trimmed) {
                return candidate
            }
        }
        return nil
    }

    /// Attempts to read `s` as `<city><series><digits>[extra]`, inserting the hyphen
    /// (and the conventional space before the digits) at every structurally valid split.
    private static func tryAsPlate(_ s: String) -> String? {
        // Already hyphenated (e.g. Vision kept the dash): re-derive city/series/digits.
        if s.contains("-") {
            let parts = s.components(separatedBy: "-")
            guard parts.count == 2 else { return nil }
            let prefix = parts[0]
            let suffix = parts[1]
            guard let di = suffix.firstIndex(where: { $0.isNumber }) else {
                let flat = "\(prefix)-\(suffix)"
                return isValid(flat) ? flat : nil
            }
            let seriesLetters = String(suffix[..<di])
            let digits = String(suffix[di...])
            let candidate = "\(prefix)-\(seriesLetters) \(digits)"
            return isValid(candidate) ? candidate : nil
        }

        guard let firstDigitIdx = s.firstIndex(where: { $0.isNumber }) else {
            return isValid(s) ? s : nil
        }
        let letters = String(s[..<firstDigitIdx])
        let tail = String(s[firstDigitIdx...])   // digits, optionally followed by a letter

        for cityLen in 1...3 {
            for seriesLen in 1...2 where cityLen + seriesLen == letters.count {
                let city = letters.prefix(cityLen)
                let series = letters.suffix(seriesLen)
                let candidate = "\(city)-\(series) \(tail)"
                if isValid(candidate) { return candidate }
            }
        }
        return nil
    }
}
