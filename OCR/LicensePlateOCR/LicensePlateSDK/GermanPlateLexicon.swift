/*
 License Plate OCR SDK — German registration codes

 Abstract:
 The letters before the hyphen on a German plate are the *registration district*
 code (Unterscheidungszeichen), not the federal state. OCR frequently loses or shifts
 the city/series boundary, e.g. `HHAB1234` can be split as `H-HA B 1234` or
 `HH-AB 1234`. Structure alone cannot tell them apart, so the splitter prefers a
 reading whose city code is a real one.

 This is a seed list of well-known codes, deliberately used only as a *tie-breaker*:
 `GermanPlateValidator` still accepts any structurally valid split when no code here
 matches, so a missing entry can never reject a real plate. Extend it freely — the
 full official list has ~400 codes and is published by the Bundesministerium für
 Digitales und Verkehr (FZV Anlage 1).

 Not modelled on purpose: seasonal (Saison), dealer/export, red 06, diplomatic and
 armed-forces plates, whose formats differ from the civilian pattern.
 */

import Foundation

public enum GermanPlateLexicon {

    /// Known civilian `Unterscheidungszeichen`. Kept uppercase.
    public static let cityCodes: Set<String> = [
        // A
        "A", "AA", "AB", "ABG", "ABI", "AC", "AH", "AIB", "AIC", "AK", "AL", "ALF", "ALZ",
        "AM", "AN", "ANA", "ANG", "AÖ", "AP", "APD", "ARN", "AS", "ASL", "ASZ", "AT", "AU",
        "AUG", "AV", "AW", "AZ",
        // B
        "B", "BA", "BAD", "BAR", "BB", "BBG", "BC", "BCH", "BE", "BED", "BER", "BG", "BGD",
        "BGL", "BH", "BI", "BID", "BIR", "BIT", "BK", "BKS", "BL", "BLB", "BLK", "BM", "BN",
        "BNA", "BO", "BÖ", "BOG", "BOH", "BOR", "BOT", "BR", "BRB", "BRG", "BRK", "BRL",
        "BRV", "BS", "BSB", "BSK", "BT", "BÜD", "BUL", "BÜR", "BÜS", "BÜZ", "BZ",
        // C
        "C", "CA", "CAS", "CB", "CE", "CGL", "CHA", "CLP", "CLZ", "CO", "COC", "COE", "CR",
        "CUX", "CW",
        // D
        "D", "DA", "DAH", "DAN", "DAU", "DBR", "DD", "DE", "DEG", "DEL", "DEN", "DGF", "DH",
        "DI", "DIL", "DIN", "DIZ", "DKB", "DL", "DLG", "DM", "DN", "DO", "DON", "DU", "DUD",
        "DÜW", "DW", "DZ",
        // E
        "E", "EB", "EBE", "EBN", "EBS", "ED", "EE", "EF", "EG", "EH", "EIC", "EIL", "EIN",
        "EIS", "EM", "EMD", "EMS", "EN", "ER", "ERB", "ERD", "ERH", "ERK", "ERZ", "ES",
        "ESB", "ESW", "EU", "EW",
        // F
        "F", "FB", "FD", "FDS", "FEU", "FF", "FG", "FL", "FLÖ", "FN", "FO", "FOR", "FR",
        "FRG", "FRI", "FRW", "FS", "FT", "FTL", "FÜ", "FÜS", "FW", "FZ",
        // G
        "G", "GA", "GAP", "GC", "GD", "GDB", "GE", "GHA", "GHC", "GI", "GL", "GLA", "GM",
        "GÖ", "GOA", "GP", "GR", "GRA", "GRH", "GRI", "GRM", "GRZ", "GS", "GT", "GTH", "GÜ",
        "GUB", "GUN", "GW", "GZ",
        // H
        "H", "HA", "HAB", "HAL", "HAM", "HAS", "HB", "HD", "HDH", "HE", "HEF", "HEI", "HER",
        "HF", "HG", "HGW", "HH", "HI", "HL", "HM", "HN", "HO", "HOL", "HOM", "HÖS", "HP",
        "HR", "HRO", "HS", "HSK", "HST", "HU", "HVL", "HX", "HZ",
        // I / J
        "IB", "ID", "IG", "IL", "IN", "IZ", "J", "JE",
        // K
        "K", "KA", "KE", "KEH", "KEL", "KEM", "KF", "KG", "KH", "KI", "KIB", "KL", "KLE",
        "KM", "KN", "KO", "KR", "KRH", "KS", "KT", "KU", "KÜN", "KW", "KY", "KYF",
        // L
        "L", "LA", "LAU", "LB", "LD", "LDK", "LDS", "LEO", "LEV", "LG", "LI", "LIP", "LL",
        "LM", "LN", "LOS", "LP", "LR", "LU", "LUK",
        // M
        "M", "MA", "MB", "MD", "ME", "MEG", "MEI", "MG", "MI", "MIL", "MK", "MKK", "MM",
        "MN", "MOL", "MON", "MOS", "MR", "MS", "MSH", "MTK", "MÜ", "MYK", "MZ",
        // N
        "N", "NB", "ND", "NDH", "NE", "NEA", "NES", "NEW", "NF", "NL", "NM", "NMB", "NMS",
        "NÖ", "NOM", "NP", "NR", "NT", "NU", "NW", "NWM", "NZ",
        // O
        "OA", "OAL", "OB", "OBG", "OC", "OCH", "OD", "OE", "OF", "OG", "OH", "OHA", "OHV",
        "OL", "OP", "OPR", "OS", "OSL", "OSN", "OZ",
        // P
        "P", "PA", "PAN", "PB", "PCH", "PE", "PEG", "PF", "PI", "PIR", "PL", "PLÖ", "PM",
        "PN", "PR", "PS", "PZ",
        // Q / R
        "QLB", "R", "RA", "RC", "RD", "RE", "REG", "RG", "RH", "RI", "RL", "RO", "ROD",
        "ROF", "ROG", "ROH", "ROK", "ROL", "ROT", "ROW", "RS", "RT", "RU", "RÜD", "RÜG",
        "RV", "RW", "RZ",
        // S
        "S", "SA", "SAL", "SAN", "SAW", "SB", "SBG", "SBK", "SC", "SDL", "SE", "SEB", "SEE",
        "SEF", "SEL", "SFB", "SG", "SH", "SHA", "SHG", "SHK", "SHL", "SI", "SIG", "SIM",
        "SK", "SL", "SLF", "SLK", "SLN", "SLÖ", "SM", "SN", "SO", "SOG", "SOK", "SON",
        "SÖM", "SP", "SPN", "SR", "SRO", "SÜW", "SW", "SWA", "SY", "SZ",
        // T
        "T", "TA", "TBB", "TE", "TF", "TI", "TIR", "TO", "TÖL", "TR", "TS", "TÜ", "TUT",
        // U / V
        "UE", "UEM", "UER", "UH", "UL", "UM", "UN", "V", "VB", "VEC", "VER", "VG", "VIB",
        "VIE", "VK", "VR", "VS",
        // W
        "W", "WA", "WAF", "WAK", "WAN", "WB", "WDA", "WE", "WEB", "WEG", "WEI", "WEL",
        "WEN", "WER", "WES", "WF", "WG", "WHV", "WI", "WIL", "WIT", "WK", "WL", "WM", "WN",
        "WND", "WO", "WOB", "WOL", "WOR", "WOS", "WR", "WST", "WÜ", "WUG", "WUN", "WÜS",
        "WZL",
        // Z
        "Z", "ZE", "ZEL", "ZI", "ZP", "ZS", "ZW", "ZZ"
    ]

    /// Whether `code` is a known registration code (case-insensitive).
    public static func contains(_ code: String) -> Bool {
        cityCodes.contains(code.uppercased())
    }
}
