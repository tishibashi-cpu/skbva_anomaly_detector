#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_pv.py — SuperKEKB ビームパイプ温度計の PV 名パーサ／グルーピング／上下ペア。

温度計 PV は2つの命名規則に対応する（ログ群が異なるため取得側でも区別が要る）:

  (A) ビームパイプ本体（VA/VATemp）:
      VA{L,H}TMP:{センサID}:{位置タグ}:{付帯}
      例) VALTMP:D10M001:QDWNP_4:BL
          VAHTMP:D10_139:QD3E_11:BL
          VAHTMP:D12_233:QEAE_7:NEG
      - ring   : VAL=LER / VAH=HER
      - センサID: D##M###（例 D10M001）または D##_###（例 D10_139）の2形式。両対応。
      - section : センサ ID 先頭の D## （D01〜D12）
      - 位置タグ: QDWNP_4 / QD3E_11 / QEAE_7 など。近傍グルーピングの手がかり。
      - 上下ペア: 連続2本（若番=上 top / 対=下 bottom）。ウィグラー等一部のみ存在。
                  ※全台にあるわけではないので「相対比較の補助」扱い。

  (B) IR（衝突点周辺）フォーカス磁石移動機構（BM/BMOthers）:
      FB_MOVE:{section}:{tag}[:{付帯}]:TEMP
      例) FB_MOVE:D01:QC1L:TEMP
          FB_MOVE:D01:QC1L:BWS:TEMP        （BWS=ビームワイヤスキャナ近傍、付帯あり）
          FB_MOVE:D01:QC1L2:TEMP           （tag に連番が付く個体）
      - ring   : どちらのリングにも一意に属さない共有設備のため "IR" という擬似リングとして扱う。
                 （B/O 層など beam 相関を使う層は、どのビーム電流と比較すべきか未定なので
                 現状は beam=None のまま安全に無効化される。運用時に要相談。）
      - section : D01/D02（IR 両側）。
      - tag     : QC1L / QC1L2 / QC1H / QC1H2 など。
      - suffix  : 付帯（BWS/BWS2）があればそれ、無ければ "TEMP"（区別のため）。

このモジュールは PV 名だけから構造を取り出す（kblogrd 不要）。numpy も不要。
"""

import re

# センサID: D + 2桁 + ( 'M' or '_' ) + 数字
_SENSOR_RE = re.compile(r"^(D\d{2})(M|_)(\d+)$")
# (A) VA{L,H}TMP PV（付帯が無い短い形も許容）
_PV_RE = re.compile(r"^VA(L|H)TMP:([^:]+)(?::([^:]+))?(?::([^:]+))?$")
# (B) FB_MOVE（IR）PV: FB_MOVE:D0x:tag[:sub]:TEMP
_PV_RE_IR = re.compile(r"^FB_MOVE:(D\d{2}):([^:]+)(?::([^:]+))?:TEMP$")

RING_OF = {"L": "LER", "H": "HER"}
RING_OF_HL = {"H": "HER", "L": "LER"}   # IR センサの tag(QC1H/QC1L) からのビームリング推定用


def parse_pv(pv, ring_override=None):
    """温度計 PV 名を辞書に分解。解析できなければ None。

    ring_override: {pv: 'LER'|'HER'} を渡すと、PV 単位で ring を上書きできる
      （一時的にリング間で物理的に付け替えた温度計、IR センサの配線間違いなど）。

    返り値: {pv, ring, natural_ring, sensor_id, section, kind('M'|'_'|None),
             idx(int|None), tag, suffix, group_key, family}
      ring         = 実効リング（上書き適用後。判定・ビーム相関はこちらを使う）
      natural_ring = PV名から機械的に決まるリング（上書き前。アーカイブ上どの
                     <RING>_TEMP_PV.csv に属するか＝取得先の判定に使う。物理的な
                     設置場所が変わっても PV 名自体は変わらないため、取得は常に
                     natural_ring 側のファイルから行う）
      group_key    = (ring, section, tag)  近傍グルーピングのキー（実効リングで束ねる）
      family       = "IR" のとき FB_MOVE（IR衝突点周辺）センサ。(A)形式は None。
    """
    if not pv:
        return None
    pv = pv.strip()

    m = _PV_RE.match(pv)
    if m:
        natural_ring = RING_OF.get(m.group(1))
        ring = (ring_override.get(pv, natural_ring) if ring_override else natural_ring)
        sensor_id = m.group(2)
        tag = m.group(3) or ""
        suffix = m.group(4) or ""
        sm = _SENSOR_RE.match(sensor_id)
        if not sm:
            # ID 形式が想定外でも、section だけは先頭 D## で拾えれば拾う
            section = sensor_id[:3] if re.match(r"^D\d{2}", sensor_id) else "?"
            return {"pv": pv, "ring": ring, "natural_ring": natural_ring, "family": None,
                    "sensor_id": sensor_id, "section": section,
                    "kind": "?", "idx": None, "tag": tag, "suffix": suffix,
                    "group_key": (ring, section, tag)}
        section, kind, idx = sm.group(1), sm.group(2), int(sm.group(3))
        return {"pv": pv, "ring": ring, "natural_ring": natural_ring, "family": None,
                "sensor_id": sensor_id, "section": section,
                "kind": kind, "idx": idx, "tag": tag, "suffix": suffix,
                "group_key": (ring, section, tag)}

    m = _PV_RE_IR.match(pv)
    if m:
        section, tag, sub = m.group(1), m.group(2), m.group(3)
        suffix = sub or "TEMP"
        # ビームリング推定: tag が QC1{H,L}[数字] の形で、H=HER / L=LER（実機の配線規則）。
        # ring_override（{pv: 'LER'|'HER'}）を渡すと、配線間違い等の既知の例外を上書きできる
        # （実データは pv_info/TEMP_RING_OVERRIDE.csv から temp_fetch 側が読み込んで渡す）。
        hl = re.match(r"^QC1([HL])", tag)
        natural_ring = RING_OF_HL.get(hl.group(1)) if hl else None
        ring = natural_ring
        if ring_override and pv in ring_override:
            ring = ring_override[pv]
        return {"pv": pv, "ring": ring, "natural_ring": natural_ring, "family": "IR",
                "sensor_id": pv, "section": section,
                "kind": None, "idx": None, "tag": tag, "suffix": suffix,
                "group_key": (ring, section, tag)}

    return None


def parse_all(pv_list, ring_override=None):
    """PV 名リスト → 解析できたものの dict リスト（順序保持）。"""
    out = []
    for pv in pv_list:
        d = parse_pv(pv, ring_override=ring_override)
        if d:
            out.append(d)
    return out


def group_sensors(parsed):
    """近傍グループ（同 ring/section/tag）ごとに dict をまとめる。
    返り値: {group_key: [parsed, ...]}（各グループ内は idx 昇順）。"""
    groups = {}
    for d in parsed:
        groups.setdefault(d["group_key"], []).append(d)
    for k in groups:
        groups[k].sort(key=lambda d: (d["idx"] is None, d["idx"] if d["idx"] is not None else 0,
                                      d["sensor_id"]))
    return groups


def build_pairs(parsed, require_adjacent=True):
    """上下ペア（連続2本, 若番=上 top / 対=下 bottom）を作る。

    同一 (ring, section, tag) グループ内で idx 昇順に並べ、(0,1),(2,3),… と2本ずつ。
    require_adjacent=True のとき、idx が隣接（差=1）の組だけをペアにする
    （番号が飛んでいる＝物理的に連続でない可能性が高い組は除外）。
    返り値: [{top, bottom, label, section, ring, tag}]
      label = 上側のセンサ ID（例 D10M001）。
    ペアはウィグラー等一部のみ。全台には存在しない点に注意。
    """
    pairs = []
    for key, members in group_sensors(parsed).items():
        ring, section, tag = key
        ms = [d for d in members if d["idx"] is not None]
        for i in range(0, len(ms) - 1, 2):
            a, b = ms[i], ms[i + 1]
            if require_adjacent and (b["idx"] - a["idx"] != 1):
                continue
            pairs.append({"top": a["sensor_id"], "bottom": b["sensor_id"],
                          "top_pv": a["pv"], "bottom_pv": b["pv"],
                          "label": a["sensor_id"], "section": section,
                          "ring": ring, "tag": tag})
    return pairs


# ───────────────────────── selftest / demo ─────────────────────────

def _selftest():
    print("=== temp_pv selftest ===")
    samples = [
        "VALTMP:D10M001:QDWNP_4:BL", "VALTMP:D10M002:QDWNP_4:BL",
        "VALTMP:D10M003:QDWNP_5:BL", "VALTMP:D10M004:QDWNP_5:BL",
        "VAHTMP:D10_139:QD3E_11:BL", "VAHTMP:D12_233:QEAE_7:NEG",
        "VAHTMP:D12_234:QEAE_7:NEG", "VALTMP:D04M045:QW5NLP:BL",
        "VALTMP:D04M046:QW5NLP:BL", "bogus:string", "",
    ]
    parsed = parse_all(samples)
    ok = True
    for d in parsed:
        print("  %-26s ring=%s sec=%s id=%-8s kind=%s idx=%s tag=%s"
              % (d["pv"], d["ring"], d["section"], d["sensor_id"],
                 d["kind"], d["idx"], d["tag"]))
    # 期待: 不正2件は除外
    ok &= (len(parsed) == 9)
    # section/ring 抽出
    d0 = parse_pv("VALTMP:D10M001:QDWNP_4:BL")
    ok &= (d0["ring"] == "LER" and d0["section"] == "D10" and d0["idx"] == 1)
    d1 = parse_pv("VAHTMP:D10_139:QD3E_11:BL")
    ok &= (d1["ring"] == "HER" and d1["section"] == "D10" and d1["idx"] == 139)
    ok &= (d0["natural_ring"] == "LER" and d1["natural_ring"] == "HER")  # 上書き無しは ring==natural_ring

    # VA形式でも ring_override が効くこと（HER温度計を一時的にLERへ物理移設した想定）。
    # natural_ring は変わらない（アーカイブ上のPV名・取得先ファイルは変わらないため）。
    reloc_pv = "VAHTMP:D10_139:QD3E_11:BL"
    d1_over = parse_pv(reloc_pv, ring_override={reloc_pv: "LER"})
    print("  一時移設テスト: %s → ring=%s (natural_ring=%s)"
          % (reloc_pv, d1_over["ring"], d1_over["natural_ring"]))
    ok &= (d1_over["ring"] == "LER" and d1_over["natural_ring"] == "HER")
    # ペア: D10M001/002, D10M003/004, D12_233/234, D04M045/046 = 4 ペア
    pairs = build_pairs(parsed)
    print("  pairs:")
    for p in pairs:
        print("    %s (top) / %s (bottom)  [%s %s %s]"
              % (p["top"], p["bottom"], p["ring"], p["section"], p["tag"]))
    ok &= (len(pairs) == 4)
    ok &= any(p["top"] == "D12_233" and p["bottom"] == "D12_234" for p in pairs)
    # グループ数
    groups = group_sensors(parsed)
    print("  groups:", {("%s/%s/%s" % k): len(v) for k, v in groups.items()})

    # (B) IR（FB_MOVE, BM/BMOthers）形式
    ir_samples = [
        "FB_MOVE:D01:QC1L:TEMP", "FB_MOVE:D01:QC1L:BWS:TEMP",
        "FB_MOVE:D01:QC1H:TEMP", "FB_MOVE:D01:QC1H:BWS:TEMP",
        "FB_MOVE:D02:QC1L:TEMP", "FB_MOVE:D02:QC1L:BWS:TEMP",
        "FB_MOVE:D02:QC1H:TEMP", "FB_MOVE:D02:QC1H:BWS:TEMP",
        "FB_MOVE:D01:QC1L2:TEMP", "FB_MOVE:D01:QC1H2:TEMP",
        "FB_MOVE:D01:QC1L:BWS2:TEMP", "FB_MOVE:D01:QC1H:BWS2:TEMP",
    ]
    ir_parsed = parse_all(ir_samples)
    print("  IR(FB_MOVE): %d/%d 件解析成功" % (len(ir_parsed), len(ir_samples)))
    for d in ir_parsed:
        print("    %-30s ring=%-3s family=%s sec=%s tag=%-8s suffix=%s"
              % (d["pv"], d["ring"], d["family"], d["section"], d["tag"], d["suffix"]))
    ok &= (len(ir_parsed) == len(ir_samples))              # 12本すべて解析できること
    ok &= all(d["family"] == "IR" for d in ir_parsed)      # IR センサとして識別される
    ok &= all(d["ring"] in ("LER", "HER") for d in ir_parsed)  # H/L からビームリングを推定
    di0 = parse_pv("FB_MOVE:D01:QC1L:TEMP")
    ok &= (di0["section"] == "D01" and di0["tag"] == "QC1L" and di0["suffix"] == "TEMP"
           and di0["ring"] == "LER")                        # QC1L → LER
    di1 = parse_pv("FB_MOVE:D01:QC1L:BWS:TEMP")
    ok &= (di1["tag"] == "QC1L" and di1["suffix"] == "BWS")   # 付帯(BWS)は suffix に区別して入る
    di2 = parse_pv("FB_MOVE:D01:QC1L2:TEMP")
    ok &= (di2["tag"] == "QC1L2" and di2["ring"] == "LER")     # 連番でも QC1L→LER は維持
    di3 = parse_pv("FB_MOVE:D01:QC1H:TEMP")
    ok &= (di3["ring"] == "HER")                                # QC1H → HER

    # 配線間違い等の既知の例外を ring_override で上書きできること（後から編集可能な仕組み）
    exc_pv = "FB_MOVE:D02:QC1H:BWS:TEMP"
    di4_default = parse_pv(exc_pv)
    ok &= (di4_default["ring"] == "HER")                        # 上書き無しなら規則通り HER
    di4_over = parse_pv(exc_pv, ring_override={exc_pv: "LER"})
    ok &= (di4_over["ring"] == "LER")                           # 上書きありなら例外反映

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
