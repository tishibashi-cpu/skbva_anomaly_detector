#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_pv.py — イオンポンプ放電電流 PV リストを読む（拡張：開発中）

CCG（圧力）に続く監視対象として、イオンポンプの放電電流を扱うための土台。
PV リスト（pv_info/LER_IP_PV.csv / pv_info/HER_IP_PV.csv、1列目が PV 名）を読み、
リング・側室（D01〜D12）・電源種別を付けて返す。

電源種別が2系統あることに注意:
  - KEK 電源（標準）        : PV 名に `_IP_` を含む。例 VALIP:D01_IP_L01:CUR
  - Agilent 4UHV Controller : PV 名に `_4U_` を含む。D05・D07 セクションのみ。
                              例 VALIP:D05_4U_L01_A01C1:CUR（末尾 A01C1 = 制御器番号・チャンネル）
放電電流の振る舞いが両者で異なるため、将来の異常判定では電源種別ごとに
基準・しきい値・モデルを分ける必要がある（このモジュールは種別を付与するだけ）。
"""

import csv
import os
import re

# PV リストの CSV は pv_info/ にまとめて置く（今後 PV 種別が増えても1か所で管理）。
PV_INFO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pv_info")


def csv_path(ring):
    """そのリングのイオンポンプ PV CSV のパス（pv_info/{ring}_IP_PV.csv）。"""
    return os.path.join(PV_INFO_DIR, "%s_IP_PV.csv" % ring)


SUPPLY_KEK = "KEK"
SUPPLY_4U = "Agilent_4U"

_SEC_RE = re.compile(r"^VA[LH]IP:(D\d+)_")


def ring_of(pv):
    return "LER" if pv.startswith("VAL") else "HER"


def section_of(pv):
    """VALIP:D05_4U_L01_A01C1:CUR → 'D05'。判別不可なら '?'。"""
    m = _SEC_RE.match(pv)
    return m.group(1) if m else "?"


def supply_of(pv):
    """電源種別。PV 名に `_4U_` があれば Agilent、なければ KEK。"""
    return SUPPLY_4U if "_4U_" in pv else SUPPLY_KEK


def load(path):
    """IP PV の CSV を読み、[{pv, ring, section, supply}, ...] を返す。"""
    recs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    for pv in rows[1:]:   # 1行目はヘッダ
        recs.append({
            "pv": pv,
            "ring": ring_of(pv),
            "section": section_of(pv),
            "supply": supply_of(pv),
        })
    return recs


def summary(records):
    """台数の内訳。{total, kek, agilent_4u, by_section:{D01:{kek,4u,total},...}}"""
    by_sec = {}
    kek = a4u = 0
    for r in records:
        s = r["section"]
        d = by_sec.setdefault(s, {"kek": 0, "agilent_4u": 0, "total": 0})
        if r["supply"] == SUPPLY_4U:
            d["agilent_4u"] += 1
            a4u += 1
        else:
            d["kek"] += 1
            kek += 1
        d["total"] += 1
    return {"total": len(records), "kek": kek, "agilent_4u": a4u, "by_section": by_sec}


def count_total(ler_csv, her_csv):
    """(LER本数, HER本数, 合計) を返す（ダッシュボードの台数表示用）。"""
    ler = len(load(ler_csv)) if os.path.isfile(ler_csv) else 0
    her = len(load(her_csv)) if os.path.isfile(her_csv) else 0
    return ler, her, ler + her


if __name__ == "__main__":
    for ring in ("LER", "HER"):
        path = csv_path(ring)
        if not os.path.isfile(path):
            print("%s: %s が見つかりません" % (ring, path))
            continue
        recs = load(path)
        s = summary(recs)
        print("=== %s イオンポンプ電流（計 %d 本 / KEK %d・Agilent_4U %d）===" %
              (ring, s["total"], s["kek"], s["agilent_4u"]))
        for i in range(1, 13):
            d = s["by_section"].get("D%02d" % i)
            if not d:
                continue
            tag = "  ← Agilent 4U" if d["agilent_4u"] else ""
            print("  D%02d: KEK %2d / 4U %2d / 計 %2d%s" %
                  (i, d["kek"], d["agilent_4u"], d["total"], tag))
