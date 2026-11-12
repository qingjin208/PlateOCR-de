# -*- coding: utf-8 -*-
"""重新生成 GermanPlateLexicon.swift：
   = (远程 kennzeichen_de.yaml 的民用码 ∪ 本地已有码) - 特殊/政府牌码。
   数据源: github.com/offene-daten/kennzeichen (CC0 1.0)。"""
import re

YAML_PATH = r'E:\Git\ITLogica\PlateOCR-de\Scripts\kennzeichen_de.yaml'
SWIFT_PATH = r'E:\Git\ITLogica\PlateOCR-de\OCR\LicensePlateOCR\LicensePlateSDK\GermanPlateLexicon.swift'

yaml_text = open(YAML_PATH, encoding='utf-8').read()
entries = re.findall(
    r'^"?([A-ZÄÖÜ]{1,3})"?:\s*\n\s+title:\s*"([^"]*)"\n\s+state:\s*"([^"]*)"',
    yaml_text, re.M)
remote = {c: (t, s) for c, t, s in entries}

# 政府/军警/外交等非民用码（民用牌上不会出现，作为城市提示只会引入偏差）
AUTHORITY_MARKERS = (
    'Landesreg', 'Bundeswehr', 'Bundespolizei', 'NATO', 'Technisches Hilfswerk',
    'Bundesrepublik', 'Wasser- und Schifffahrtsverwaltung', 'Diplomat',
    'Bundestags', 'Bundesrat', 'Bundespräsidialamt', 'Bundeskanzleramt',
    'Bundesministerium', 'Bundesgerichtshof', 'Bundesfinanzhof',
    'Bundesrechnungshof', 'Bundescartellamt', 'Bundesnetzagentur',
    'Bundesamt', 'Bundeskriminalamt', 'Polizeipräsidium', 'Polizei',
    'Regierung ', 'Präsident', 'Landtag', 'Landeskriminalamt',
)
authority = {c for c, (t, s) in remote.items()
             if any(m in t for m in AUTHORITY_MARKERS)}
print('excluded authority codes (%d):' % len(authority), sorted(authority))

remote_civil = set(remote) - authority
print('remote civilian:', len(remote_civil))

swift = open(SWIFT_PATH, encoding='utf-8').read()
local = set(re.findall(r'"([A-ZÄÖÜ]{1,3})"', swift))
merged = local | remote_civil
print('merged:', len(merged))

# 按 Swift 可用标识符分组（ÄÖÜ 按 A/O/U 归组）
def group_key(code):
    first = code[0]
    return {'Ä': 'A', 'Ö': 'O', 'Ü': 'U'}.get(first, first)

groups = {}
for code in sorted(merged, key=lambda c: (group_key(c), c.replace('Ä', 'A').replace('Ö', 'O').replace('Ü', 'U'), len(c))):
    groups.setdefault(group_key(code), []).append(code)

letters_order = sorted(groups.keys())
lines = []
lines.append('''/*
 License Plate OCR SDK — German registration codes

 Abstract:
 The letters before the hyphen on a German plate are the *registration district*
 code (Unterscheidungszeichen), not the federal state. OCR frequently loses or shifts
 the city/series boundary, e.g. `HHAB1234` can be split as `H-HA B 1234` or
 `HH-AB 1234`. Structure alone cannot tell them apart, so the splitter prefers a
 reading whose city code is a real one. The validator also uses the lexicon as an
 anchor for fuzzy prefix recovery (`GermanPlateValidator.anchoredParse`).

 Contents: the complete civilian `Unterscheidungszeichen` list — currently issued
 codes PLUS codes that are no longer issued but remain valid on existing plates
 (Germany never forces re-registration), merged from github.com/offene-daten/
 kennzeichen (CC0 1.0) and the original seed list. Non-civilian plates (Bundeswehr
 `Y`, Bundespolizei `BP`, state-government, diplomatic corps, …) are deliberately
 NOT modelled: they never appear on civilian plates, and hinting at them would
 bias split disambiguation.

 Usage stays *hint-only*: `GermanPlateValidator` still accepts any structurally
 valid split when no code here matches, so a missing entry can never reject a
 real plate.
 */

import Foundation

public enum GermanPlateLexicon {

    /// Known civilian `Unterscheidungszeichen` (issued + historic, still on the road).
    /// Kept uppercase, umlauts as single characters (`TÜ`, `BÖ`, `AÖ`, `LÖ`, `WÜ`).
    public static let cityCodes: Set<String> = [''')

for letter in letters_order:
    lines.append('        // %s' % letter)
    codes = groups[letter]
    for i in range(0, len(codes), 8):
        chunk = ', '.join('"%s"' % c for c in codes[i:i + 8])
        lines.append('        ' + chunk + ',')

lines.append(''']

    /// Umlaut → base vowel (one-substitution distance).
    private static let umlautFold: [Character: Character] = ["Ä": "A", "Ö": "O", "Ü": "U"]

    /// Whether `code` is a known registration code (case-insensitive).
    public static func contains(_ code: String) -> Bool {
        cityCodes.contains(code.uppercased())
    }

    /// Lexicon entries within edit distance ≤ 1 of `code` (single substitution,
    /// deletion, or insertion — umlaut↔base-vowel counts as one substitution).
    /// Used by the validator to recover a *misread* city code: the query is what
    /// OCR produced, the hits are plausible intended codes. Results exclude the
    /// query itself; callers must confirm the remaining structure before trusting
    /// a hit (see `GermanPlateValidator.anchoredParse`).
    public static func fuzzyMatches(for code: String) -> [String] {
        let query = code.uppercased()
        guard (1...3).contains(query.count) else { return [] }
        let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ")
        var candidates = Set<String>()

        // Base variants: fold umlauts to base vowels first — `TU` should reach `TÜ`
        // through the same single-substitution rule as `TUB`→`TÜB`.
        var bases = [query]
        let folded = String(query.map { Self.umlautFold[$0] ?? $0 })
        if folded != query { bases.append(folded) }

        for base in bases {
            let chars = Array(base)
            // Substitutions
            for i in chars.indices {
                for letter in alphabet where letter != chars[i] {
                    var copy = chars
                    copy[i] = letter
                    candidates.insert(String(copy))
                }
            }
            // Deletions
            if chars.count > 1 {
                for i in chars.indices {
                    var copy = chars
                    copy.remove(at: i)
                    candidates.insert(String(copy))
                }
            }
            // Insertions
            for i in 0...chars.count {
                for letter in alphabet {
                    var copy = chars
                    copy.insert(letter, at: i)
                    candidates.insert(String(copy))
                }
            }
        }

        candidates.remove(query)
        return cityCodes.intersection(candidates).sorted()
    }
}
''')

open(SWIFT_PATH, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print('written', SWIFT_PATH)

# 验证：重新解析新文件统计条目
new_swift = open(SWIFT_PATH, encoding='utf-8').read()
new_codes = set(re.findall(r'"([A-ZÄÖÜ]{1,3})"', new_swift))
print('new lexicon count:', len(new_codes))
print('ST in new:', 'ST' in new_codes, '| FFB:', 'FFB' in new_codes, '| TÜ:', 'TÜ' in new_codes)
