# -*- coding: utf-8 -*-
"""对比远程 kennzeichen_de.yaml 与现有 GermanPlateLexicon.swift。"""
import re

YAML_PATH = r'E:\Git\ITLogica\PlateOCR-de\Scripts\kennzeichen_de.yaml'
SWIFT_PATH = r'E:\Git\ITLogica\PlateOCR-de\OCR\LicensePlateOCR\LicensePlateSDK\GermanPlateLexicon.swift'

yaml_text = open(YAML_PATH, encoding='utf-8').read()
entry_re = re.compile(r'^"?([A-ZÄÖÜ]{1,3})"?:\s*\n\s+title:\s*"([^"]*)"', re.M)
entries = entry_re.findall(yaml_text)
remote = dict(entries)
print('remote codes:', len(remote))

swift = open(SWIFT_PATH, encoding='utf-8').read()
local = set(re.findall(r'"([A-ZÄÖÜ]{1,3})"', swift))
print('local codes:', len(local))

missing = sorted(set(remote) - local)
extra = sorted(local - set(remote))
print('missing locally (%d):' % len(missing), missing)
print('extra locally (%d):' % len(extra), extra)

for c in ['ST', 'SB', 'TÜ', 'BÖ', 'AÖ', 'WÜ', 'HH', 'B', 'M', 'K', 'PI', 'GI']:
    print(c, 'remote:', c in remote, '| title:', remote.get(c))
