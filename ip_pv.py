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


# イオンポンプ PV → 同一地点の CCG PV を導くための正規表現。
#   KEK     : VALIP:D01_IP_L08:CUR        → VALCCG:D01_L08:PRES
#   Agilent : VALIP:D05_4U_L01_A01C1:CUR  → VALCCG:D05_L01:PRES
#   末尾英字（L09A, L16X 等）は保持する。
_PAIR_RE = re.compile(r"^VA([LH])IP:(D\d+)_(?:IP|4U)_([LH]\d+[A-Z]?)(?:_A\d+C\d+)?:CUR$")


def paired_ccg(ip_pv):
    """イオンポンプ電流 PV から、同一地点の CCG 圧力 PV 名を導く。
    変換できなければ None。例: 'VALIP:D01_IP_L08:CUR' → 'VALCCG:D01_L08:PRES'。
    （対応する CCG が実在するかは別途 CCG リストと突き合わせること。）"""
    m = _PAIR_RE.match(ip_pv)
    if not m:
        return None
    return "VA%sCCG:%s_%s:PRES" % (m.group(1), m.group(2), m.group(3))


def build_pairs(ip_pvs, ccg_pvs):
    """イオンポンプ PV 群と CCG PV 群（実在するもの）を突き合わせ、
    同一地点ペアを作る。

    返り値: {"paired": [(ip_pv, ccg_pv), ...],   # 両方そろう地点
             "ip_only": [ip_pv, ...],            # イオンポンプのみの地点
             "ccg_only": [ccg_pv, ...]}          # CCG のみの地点
    """
    ccg_set = set(ccg_pvs)
    paired, ip_only = [], []
    matched_ccg = set()
    for ip in ip_pvs:
        c = paired_ccg(ip)
        if c and c in ccg_set:
            paired.append((ip, c))
            matched_ccg.add(c)
        else:
            ip_only.append(ip)
    ccg_only = [c for c in ccg_pvs if c not in matched_ccg]
    return {"paired": paired, "ip_only": ip_only, "ccg_only": ccg_only}


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
