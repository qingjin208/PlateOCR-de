# -*- coding: utf-8 -*-
"""验证 anchoredParse 的决胜规则：klass 0=变音符替换, 1=删除章形字符, 2=其他编辑。"""
import re

SWIFT_PATH = r'E:\Git\ITLogica\PlateOCR-de\OCR\LicensePlateOCR\LicensePlateSDK\GermanPlateLexicon.swift'
codes = set(re.findall(r'"([A-ZÄÖÜ]{1,3})"', open(SWIFT_PATH, encoding='utf-8').read()))

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ'
FOLD = {'Ä': 'A', 'Ö': 'O', 'Ü': 'U'}
SEAL_GLYPHS = set('DOQB8EÄÖÜ')

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
        for i in range(len(ch)):
            cands.add(''.join(ch[:i] + ch[i+1:]))
        for i in range(len(ch) + 1):
            for L in ALPHA:
                cands.add(''.join(ch[:i] + [L] + ch[i:]))
    cands.discard(q)
    return sorted(cands & codes)

def klass_of(q, hit):
    if len(q) == len(hit):
        diffs = [(a, b) for a, b in zip(q, hit) if a != b]
        if len(diffs) == 1 and (diffs[0] in {('A','Ä'),('Ä','A'),('O','Ö'),('Ö','O'),('U','Ü'),('Ü','U')}):
            return 0
    if len(hit) == len(q) - 1:
        # hit 是 q 删除一个字符的结果，且被删字符是章形字形
        for i in range(len(q)):
            if q[:i] + q[i+1:] == hit and q[i] in SEAL_GLYPHS:
                return 1
    return 2

for q in ['TU', 'STO', 'GKE', 'HHM', 'WES']:
    hits = fuzzy(q)
    by_class = {}
    for h in hits:
        by_class.setdefault(klass_of(q, h), []).append(h)
    print(q, {k: sorted(v) for k, v in sorted(by_class.items())})
print('TU in lexicon:', 'TU' in codes, '| TÜ:', 'TÜ' in codes, '| ST:', 'ST' in codes, '| SO:', 'SO' in codes)
