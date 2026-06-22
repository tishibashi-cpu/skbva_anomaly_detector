#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_state.py — イオンポンプ放電電流をダッシュボード用の構造に変換する（拡張：開発中）

ip_fetch.fetch_history() が返す生データ（PVごとの時系列）を、ダッシュボードが扱いやすい
コンパクトな構造にまとめる。判定はまだ無く、トレンド表示が目的。

設計（おすすめ案：セクション一覧＋クリックで個別PVトレンド）:
  - リングごと・セクション（D01〜D12）ごとに集約。
  - セクションは代表値（有効値の最新・最大）と本数、電源種別を持つ。
  - 各 PV は最新値とダウンサンプルしたトレンド（点数を抑える）を持つ。
  - NODATA(=1e-10) は「データ無し」として欠測(None)扱いし、統計から除く。

検知本体（末次プログラム）には一切触れない。CCG パイプラインと並列の独立処理。
"""

import datetime
import json
import os

import ip_fetch
import ip_pv

NODATA = ip_fetch.NODATA
MAXPTS = 40          # 1 PV あたりのトレンド点数の上限（ダウンサンプル）
DEVICE_TYPE = "IonPump"

_HERE = os.path.dirname(os.path.abspath(__file__))
IP_DATA_FILE = os.path.join(_HERE, "ip_data.json")   # collect が書き、state_builder が読む


def _valid(v):
    return v is not None and v > NODATA


def _downsample(seq, maxpts=MAXPTS):
    n = len(seq)
    if n <= maxpts:
        return list(seq)
    step = n / float(maxpts)
    return [seq[int(i * step)] for i in range(maxpts)]


def _latest_valid(vals):
    for v in reversed(vals):
        if _valid(v):
            return v
    return None


def summarize(fetched_by_ring, updated=None, interval_sec=60):
    """fetched_by_ring = {"LER": fetch_history(...), "HER": ...} を
    ダッシュボード用の ion_pumps 構造に変換する。

    返り値:
      {device_type, updated, interval_sec,
       rings: {LER: {t:[...], sections:[{section, supply, n, n_active,
                                         max_latest, pvs:[{pv,supply,latest,trend}]}]}}}
    """
    out = {"device_type": DEVICE_TYPE, "updated": updated,
           "interval_sec": interval_sec, "rings": {}}
    for ring, data in (fetched_by_ring or {}).items():
        if not data:
            continue
        # 時間軸は全 PV 共通（同じ -t で取得）。最初の PV から作ってダウンサンプル。
        any_pv = next(iter(data.values()))
        full_t = [ts for ts, _ in any_pv["series"]]
        t = _downsample(full_t)

        secmap = {}
        for pv, rec in data.items():
            sec = rec["section"]
            sup = rec["supply"]
            vals = [v for _, v in rec["series"]]
            disp = [v if _valid(v) else None for v in vals]   # NODATA→None
            s = secmap.setdefault(sec, {"section": sec, "n": 0,
                                        "pvs": [], "_supplies": set()})
            s["n"] += 1
            s["_supplies"].add(sup)
            s["pvs"].append({
                "pv": pv,
                "supply": sup,
                "latest": _latest_valid(vals),
                "trend": _downsample(disp),
            })

        sections = []
        for sec in sorted(secmap):
            s = secmap[sec]
            latests = [p["latest"] for p in s["pvs"] if p["latest"] is not None]
            s["max_latest"] = max(latests) if latests else None
            s["n_active"] = len(latests)
            sup = s.pop("_supplies")
            s["supply"] = list(sup)[0] if len(sup) == 1 else "mixed"
            # PV はセクション内で名前順に
            s["pvs"].sort(key=lambda p: p["pv"])
            sections.append(s)
        out["rings"][ring] = {"t": t, "sections": sections}
    return out


def collect_and_save(start, end, interval_sec=60, rings=("LER", "HER"),
                     out_path=IP_DATA_FILE, **fetch_kw):
    """実機で各リングの履歴を取得し、要約して JSON に保存する（kblogrd 必要）。
    state_builder はこの JSON を読んでダッシュボードに載せる。"""
    fetched = {}
    for ring in rings:
        fetched[ring] = ip_fetch.fetch_history(ring, start, end,
                                               interval_sec=interval_sec, **fetch_kw)
    state = summarize(fetched, updated=_fmt(end), interval_sec=interval_sec)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    return out_path, state


def load_saved(path=IP_DATA_FILE):
    """保存済みの ion_pumps 構造を読む。無ければ None。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(yyyymmddhhmmss):
    """'20260610010000' → '2026-06-10 01:00:00'（表示用）。失敗時はそのまま。"""
    s = str(yyyymmddhhmmss)
    try:
        dt = datetime.datetime.strptime(s, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return s


if __name__ == "__main__":
    # 実機での収集例:
    #   python ip_state.py 20260610000000 20260611000000
    import sys
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
        interval = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
        path, state = collect_and_save(start, end, interval_sec=interval)
        nsec = sum(len(r["sections"]) for r in state["rings"].values())
        print("保存: %s（%d リング / 計 %d セクション）" %
              (path, len(state["rings"]), nsec))
        for ring, r in state["rings"].items():
            print(" [%s] %d セクション、時間軸 %d 点" % (ring, len(r["sections"]), len(r["t"])))
    else:
        print("usage: python ip_state.py <start yyyymmddhhmmss> <end> [interval_sec]")
