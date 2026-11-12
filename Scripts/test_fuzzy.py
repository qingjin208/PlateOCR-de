# -*- coding: utf-8 -*-
"""模拟 Swift fuzzyMatches 的行为，验证测试用例的唯一性。"""
import re

SWIFT_PATH = r'E:\Git\ITLogica\PlateOCR-de\OCR\LicensePlateOCR\LicensePlateSDK\GermanPlateLexicon.swift'
codes = set(re.findall(r'"([A-ZÄÖÜ]{1,3})"', open(SWIFT_PATH, encoding='utf-8').read()))
print('lexicon:', len(codes))

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ'
FOLD = {'Ä': 'A', 'Ö': 'O', 'Ü': 'U'}

def fuzzy(q):
    bases = [q]
    folded = ''.join(FOLD.get(c, c) for c in q)
    if folded != q:
        bases.append(folded)
    cands = set()
    for base in bases:
        ch = list(base)
        for i in range(len(ch)):
            for L in ALPHA:
                if L != ch[i]:
                    c = ch.copy(); c[i] = L; cands.add(''.join(c))
        if len(ch) > 1:
            for i in range(len(ch)):
                cands.add(''.join(ch[:i] + ch[i+1:]))
        for i in range(len(ch) + 1):
            for L in ALPHA:
                cands.add(''.join(ch[:i] + [L] + ch[i:]))
    cands.discard(q)
    return sorted(cands & codes)

for q in ['TU', 'MUL', 'GKE', 'ST', 'TBB', 'WES', 'WAK', 'X', 'BP', 'ABCD', 'KEBA', 'HHM']:
    print(q, '->', fuzzy(q))
