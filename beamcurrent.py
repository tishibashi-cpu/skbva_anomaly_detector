#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beamcurrent.py — 蓄積電流（リアルタイム）を EPICS から読む

ダッシュボードのヘッダに「現在のビーム電流」を出すための軽量モジュール。
検知（detector_headless）が走ったタイミングでしか更新されない「最終チェック時点」の
ビーム状態とは別に、こちらは数秒ごとに caget して常に最新値を出す。

EPICS が無い環境（手元など）では None を返し、ダッシュボードを壊さない。
"""

PVS = {"LER": "BMLDCCT:CURRENT", "HER": "BMHDCCT:CURRENT"}

try:
    import epics
except Exception:
    epics = None


def read():
    """{'LER': mA or None, 'HER': mA or None} を返す。epics 不在/取得失敗は None。"""
    out = {"LER": None, "HER": None}
    if epics is None:
        return out
    for ring, pv in PVS.items():
        try:
            v = epics.caget(pv, timeout=1.0)
            out[ring] = float(v) if v is not None else None
        except Exception:
            out[ring] = None
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(read(), ensure_ascii=False))
