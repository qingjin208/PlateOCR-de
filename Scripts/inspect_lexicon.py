# -*- coding: utf-8 -*-
"""检查远程词表中的特殊/可疑条目。"""
import re

YAML_PATH = r'E:\Git\ITLogica\PlateOCR-de\Scripts\kennzeichen_de.yaml'
yaml_text = open(YAML_PATH, encoding='utf-8').read()
entries = re.findall(
    r'^"?([A-ZÄÖÜ]{1,3})"?:\s*\n\s+title:\s*"([^"]*)"\n\s+state:\s*"([^"]*)"',
    yaml_text, re.M)
remote = {c: (t, s) for c, t, s in entries}

suspects = ['X', 'Y', 'BP', 'BW', 'BY', 'THW', 'LSA', 'NRW', 'MVL', 'NVP',
            'OVP', 'BWL', 'BD', 'BF', 'RP', 'RPL', 'RN', 'RM', 'NY', 'MY',
            'MQ', 'MC', 'MH', 'ML', 'MGH', 'WZ', 'WT', 'WW', 'WS', 'WIS',
            'TF', 'SN', 'SE', 'SU', 'STA']
for c in suspects:
    if c in remote:
        print(c, '->', remote[c])
