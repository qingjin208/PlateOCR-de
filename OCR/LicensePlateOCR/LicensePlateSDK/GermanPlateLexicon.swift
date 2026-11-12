/*
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
    public static let cityCodes: Set<String> = [
        // A
        "A", "AA", "AB", "ABG", "ABI", "AC", "AE", "AH",
        "AIB", "AIC", "AK", "AL", "ALF", "ALZ", "AM", "AN",
        "ANA", "ANG", "ANK", "AÖ", "AP", "APD", "ARN", "ART",
        "AS", "ASL", "ASZ", "AT", "AU", "AUG", "AUR", "AV",
        "AW", "AZ", "AZE",
        // B
        "B", "BA", "BAD", "BAR", "BB", "BBG", "BC", "BCH",
        "BE", "BED", "BER", "BF", "BG", "BGD", "BGL", "BH",
        "BI", "BID", "BIN", "BIR", "BIT", "BIW", "BK", "BKS",
        "BL", "BLB", "BLK", "BM", "BN", "BNA", "BO", "BÖ",
        "BOG", "BOH", "BOR", "BOT", "BR", "BRA", "BRB", "BRG",
        "BRK", "BRL", "BRV", "BS", "BSB", "BSK", "BT", "BTF",
        "BÜD", "BUL", "BÜR", "BÜS", "BÜZ", "BZ",
        // C
        "C", "CA", "CAS", "CB", "CE", "CGL", "CHA", "CLP",
        "CLZ", "CO", "COC", "COE", "CR", "CUX", "CW",
        // D
        "D", "DA", "DAH", "DAN", "DAU", "DBR", "DD", "DE",
        "DEG", "DEL", "DEN", "DGF", "DH", "DI", "DIL", "DIN",
        "DIZ", "DKB", "DL", "DLG", "DM", "DN", "DO", "DON",
        "DU", "DUD", "DÜW", "DW", "DZ",
        // E
        "E", "EA", "EB", "EBE", "EBN", "EBS", "ECK", "ED",
        "EE", "EF", "EG", "EH", "EI", "EIC", "EIL", "EIN",
        "EIS", "EL", "EM", "EMD", "EMS", "EN", "ER", "ERB",
        "ERD", "ERH", "ERK", "ERZ", "ES", "ESB", "ESW", "EU",
        "EW",
        // F
        "F", "FB", "FD", "FDB", "FDS", "FEU", "FF", "FFB",
        "FG", "FI", "FKB", "FL", "FLÖ", "FN", "FO", "FOR",
        "FR", "FRG", "FRI", "FRW", "FS", "FT", "FTL", "FÜ",
        "FÜS", "FW", "FZ",
        // G
        "G", "GA", "GAN", "GAP", "GC", "GD", "GDB", "GE",
        "GEL", "GEO", "GER", "GF", "GG", "GHA", "GHC", "GI",
        "GK", "GL", "GLA", "GM", "GMN", "GN", "GNT", "GÖ",
        "GOA", "GOH", "GP", "GR", "GRA", "GRH", "GRI", "GRM",
        "GRZ", "GS", "GT", "GTH", "GÜ", "GUB", "GUN", "GV",
        "GVM", "GW", "GZ",
        // H
        "H", "HA", "HAB", "HAL", "HAM", "HAS", "HB", "HBN",
        "HBS", "HC", "HCH", "HD", "HDH", "HDL", "HE", "HEB",
        "HEF", "HEI", "HER", "HET", "HF", "HG", "HGN", "HGW",
        "HH", "HHM", "HI", "HIG", "HIP", "HK", "HL", "HM",
        "HMÜ", "HN", "HO", "HOG", "HOH", "HOL", "HOM", "HOR",
        "HÖS", "HOT", "HP", "HR", "HRO", "HS", "HSK", "HST",
        "HU", "HV", "HVL", "HWI", "HX", "HY", "HZ",
        // I
        "IB", "ID", "IG", "IGB", "IK", "IL", "ILL", "IN",
        "IZ",
        // J
        "J", "JE", "JL", "JÜL",
        // K
        "K", "KA", "KB", "KC", "KE", "KEH", "KEL", "KEM",
        "KF", "KG", "KH", "KI", "KIB", "KK", "KL", "KLE",
        "KLZ", "KM", "KN", "KO", "KÖN", "KÖT", "KÖZ", "KR",
        "KRH", "KRU", "KS", "KT", "KU", "KÜN", "KUS", "KW",
        "KY", "KYF",
        // L
        "L", "LA", "LAN", "LAU", "LB", "LBS", "LBZ", "LC",
        "LD", "LDK", "LDS", "LEO", "LER", "LEV", "LF", "LG",
        "LH", "LI", "LIB", "LIF", "LIP", "LL", "LM", "LN",
        "LÖ", "LÖB", "LOS", "LP", "LR", "LRO", "LSZ", "LU",
        "LUK", "LÜN", "LUP", "LWL",
        // M
        "M", "MA", "MAB", "MAI", "MAK", "MAL", "MB", "MC",
        "MD", "ME", "MED", "MEG", "MEI", "MEK", "MER", "MET",
        "MG", "MGH", "MGN", "MH", "MHL", "MI", "MIL", "MK",
        "MKK", "ML", "MM", "MN", "MO", "MOD", "MOL", "MON",
        "MOS", "MQ", "MR", "MS", "MSE", "MSH", "MSP", "MST",
        "MTK", "MTL", "MÜ", "MÜB", "MÜR", "MW", "MY", "MYK",
        "MZ", "MZG",
        // N
        "N", "NAB", "NAI", "NAU", "NB", "ND", "NDH", "NE",
        "NEA", "NEB", "NEC", "NEN", "NES", "NEW", "NF", "NH",
        "NI", "NK", "NL", "NM", "NMB", "NMS", "NÖ", "NOH",
        "NOL", "NOM", "NOR", "NP", "NR", "NT", "NU", "NVP",
        "NW", "NWM", "NY", "NZ",
        // O
        "OA", "OAL", "OB", "OBG", "OC", "OCH", "OD", "OE",
        "OF", "OG", "OH", "OHA", "ÖHR", "OHV", "OHZ", "OK",
        "OL", "OP", "OPR", "OS", "OSL", "OSN", "OTW", "OVI",
        "OVL", "OVP", "OZ",
        // P
        "P", "PA", "PAF", "PAN", "PAR", "PB", "PCH", "PE",
        "PEG", "PF", "PI", "PIR", "PL", "PLÖ", "PM", "PN",
        "PR", "PRÜ", "PS", "PW", "PZ",
        // Q
        "QFT", "QLB",
        // R
        "R", "RA", "RC", "RD", "RDG", "RE", "REG", "REH",
        "REI", "RG", "RH", "RI", "RID", "RIE", "RL", "RM",
        "RN", "RO", "ROD", "ROF", "ROG", "ROH", "ROK", "ROL",
        "ROS", "ROT", "ROW", "RP", "RS", "RSL", "RT", "RU",
        "RÜD", "RÜG", "RV", "RW", "RY", "RZ",
        // S
        "S", "SA", "SAB", "SAD", "SAL", "SAN", "SAW", "SB",
        "SBG", "SBK", "SC", "SCZ", "SDH", "SDL", "SDT", "SE",
        "SEB", "SEE", "SEF", "SEL", "SFA", "SFB", "SFT", "SG",
        "SGH", "SH", "SHA", "SHG", "SHK", "SHL", "SI", "SIG",
        "SIM", "SK", "SL", "SLE", "SLF", "SLK", "SLN", "SLÖ",
        "SLS", "SLÜ", "SLZ", "SM", "SN", "SO", "SOB", "SOG",
        "SOK", "SÖM", "SON", "SP", "SPB", "SPN", "SR", "SRB",
        "SRO", "ST", "STA", "STB", "STD", "STE", "STL", "SU",
        "SUL", "SÜW", "SW", "SWA", "SY", "SZ", "SZB",
        // T
        "T", "TA", "TBB", "TDO", "TE", "TET", "TF", "TG",
        "TI", "TIR", "TO", "TÖL", "TP", "TR", "TS", "TÜ",
        "TUT",
        // U
        "UE", "UEM", "UER", "UFF", "UH", "UL", "UM", "UN",
        "USI",
        // V
        "V", "VAI", "VB", "VEC", "VER", "VG", "VIB", "VIE",
        "VK", "VOH", "VR", "VS",
        // W
        "W", "WA", "WAF", "WAK", "WAN", "WAT", "WB", "WBS",
        "WDA", "WE", "WEB", "WEG", "WEI", "WEL", "WEN", "WER",
        "WES", "WF", "WG", "WHV", "WI", "WIL", "WIS", "WIT",
        "WIZ", "WK", "WL", "WLG", "WM", "WMS", "WN", "WND",
        "WO", "WOB", "WOH", "WOL", "WOR", "WOS", "WR", "WRN",
        "WS", "WSF", "WST", "WSW", "WT", "WTM", "WÜ", "WUG",
        "WÜM", "WUN", "WUR", "WÜS", "WW", "WZ", "WZL",
        // Z
        "Z", "ZE", "ZEL", "ZI", "ZIG", "ZP", "ZR", "ZS",
        "ZW", "ZZ",
]

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
