#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flow_pv.py — 冷却水流量計 PV 名パーサ。

PV 形式: VA_FLS:{section}_{idx}_{tag}:RATE
  例) VA_FLS:D01_11_XXX:RATE
      VA_FLS:D04_15_084:RATE
  - section : D01〜D12（CCG/温度計と同じセクション表記）
  - idx     : セクション内の連番（2桁）
  - tag     : 個体識別（3桁数字、または未割当を示す "XXX"）
  - sensor_id: "{section}_{idx}"（例 D04_15）

流量計はビーム電流と無関係の機器のため、CCG/温度計/IPと違い**リング(LER/HER)の概念を持たない**
（1本のPVリスト pv_info/FLOW_PV.csv に全セクションがまとまっている）。判定も指示値そのものだけを
見るため、本モジュールはリング判定やビーム相関の仕組みを持たない（他の temp_pv.py 等と違い、
ここが最大の簡略化点）。

このモジュールは PV 名だけから構造を取り出す（kblogrd 不要）。numpy も不要。
"""

import re

_PV_RE = re.compile(r"^VA_FLS:(D\d{2})_(\d+)_([0-9A-Z]+):RATE$")


def parse_pv(pv):
    """流量計 PV 名を辞書に分解。解析できなければ None。

    返り値: {pv, section, idx(int), tag, sensor_id, group_key}
      group_key = (section, tag)  近傍グルーピングのキー（同一セクション内の個体識別用）。
    """
    if not pv:
        return None
    pv = pv.strip()
    m = _PV_RE.match(pv)
    if not m:
        return None
    section, idx_s, tag = m.group(1), m.group(2), m.group(3)
    sensor_id = "%s_%s" % (section, idx_s)
    return {"pv": pv, "section": section, "idx": int(idx_s), "tag": tag,
            "sensor_id": sensor_id, "group_key": (section, tag)}


def parse_all(pv_list):
    """PV 名リスト → 解析できたものの dict リスト（順序保持）。"""
    out = []
    for pv in pv_list:
        d = parse_pv(pv)
        if d:
            out.append(d)
    return out


# ───────────────────────── selftest ─────────────────────────

def _selftest():
    print("=== flow_pv selftest ===")
    samples = [
        "VA_FLS:D01_11_XXX:RATE", "VA_FLS:D04_15_084:RATE", "VA_FLS:D10_02_010:RATE",
        "VA_FLS:D10_08_043:RATE", "bogus:string", "",
    ]
    parsed = parse_all(samples)
    ok = True
    for d in parsed:
        print("  %-26s sec=%s idx=%s tag=%s sensor_id=%s"
              % (d["pv"], d["section"], d["idx"], d["tag"], d["sensor_id"]))
    ok &= (len(parsed) == 4)   # 不正2件は除外

    d0 = parse_pv("VA_FLS:D01_11_XXX:RATE")
    ok &= (d0["section"] == "D01" and d0["idx"] == 11 and d0["tag"] == "XXX"
           and d0["sensor_id"] == "D01_11")
    d1 = parse_pv("VA_FLS:D04_15_084:RATE")
    ok &= (d1["section"] == "D04" and d1["idx"] == 15 and d1["tag"] == "084")
    ok &= (parse_pv("") is None and parse_pv(None) is None and parse_pv("bogus") is None)

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
