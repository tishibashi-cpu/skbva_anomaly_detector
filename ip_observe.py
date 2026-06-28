#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_observe.py — 同一地点ペアの「CCG圧力 vs イオンポンプ放電電流」を観察する（開発中）

イオンポンプ放電電流の異常判定（同一地点の CCG 圧力との整合性）が成り立つかを、
実データで確かめるための観察ツール。判定は行わない。

指定した期間・地点ペアについて、CCG 圧力（ccg_fetch, VA/CCG）と
イオンポンプ放電電流（ip_fetch, VA/IPump）を取得し、1 ペア 1 枚の PNG に
  上段: 時系列（圧力＝左軸・放電電流＝右軸、ともに対数）
  下段: 散布図（横=圧力, 縦=放電電流, 対数-対数）
        ＋ 圧力フロア（電源種別ごと）で足切りし、残った点で I=a·P^b をフィット
        （傾き b≈1 なら I∝P）。フロア未満の点はグレーで対象外。
を描いて保存する。ヘッドレス（Agg）。手元に持ってきて目で確認する用途。

確認したいこと:
  - 正常時、圧力と放電電流がきれいに相関するか（フィットの傾き b と相関 r）
  - その相関が KEK 電源と Agilent 4U で違うか
  - 圧力フロア（KEK 1e-7 / 4U 1e-8 Pa）の妥当性、フィットが場所ごとに安定か

使い方:
  python ip_observe.py LER 20260610000000 20260611000000
  python ip_observe.py LER 20260610000000 20260611000000 --section D05
  python ip_observe.py LER 20260610000000 20260611000000 --pairs 8 --out observe
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")          # ヘッドレス（画面に出さず PNG 保存）
import matplotlib.pyplot as plt
import numpy as np

import ip_pv
import ip_fetch
import ccg_fetch

NODATA = ip_fetch.NODATA

# ── 圧力フロア（電源種別ごと）──────────────────────────────────
# これ未満の圧力では放電電流が測定下限に張り付き、I-P 関係が見えないので
# フィットから除外する。経験値・仕様にもとづく初期値（後で場所ごとに調整可）。
#   KEK 電源 : 経験的に ~1e-7 Pa 以上でないと反応しない（2e-6 A 程度から）。個体差あり。
#   Agilent 4U: 電流測定下限 10 nA（マニュアル仕様）。1e-7 Pa 以下でも放電があれば測れる。
PRES_FLOOR = {
    ip_pv.SUPPLY_KEK: 1e-7,
    ip_pv.SUPPLY_4U:  1e-8,
}
PRES_FLOOR_DEFAULT = 1e-7

# Agilent 4U の電流測定下限（マニュアル仕様: 10 nA）。散布図に参考線として引く。
CUR_FLOOR_4U = 10e-9    # 10 nA = 1e-8 A


def floor_for(supply):
    return PRES_FLOOR.get(supply, PRES_FLOOR_DEFAULT)


def powerlaw_fit(pres, cur, p_floor):
    """圧力フロア以上の点だけで I = a * P^b を両対数直線フィットする。

    返り値: dict or None
      {a, b, r, n_used, p_floor} … b=傾き(log-log), a=係数, r=フィット相関, n_used=採用点数
    点が 3 未満、または圧力に変化が無い（log の分散ゼロ）場合は None。
    """
    if len(pres) == 0:
        return None
    mask = pres >= p_floor
    pf, cf = pres[mask], cur[mask]
    if len(pf) < 3:
        return None
    lp, lc = np.log10(pf), np.log10(cf)
    if lp.std() == 0 or lc.std() == 0:
        return None
    b, log_a = np.polyfit(lp, lc, 1)      # lc = b*lp + log_a
    a = 10.0 ** log_a
    r = float(np.corrcoef(lp, lc)[0, 1])
    return {"a": float(a), "b": float(b), "r": r,
            "n_used": int(len(pf)), "p_floor": float(p_floor)}


def _fetch_pvs(pvs, ttime, log_group):
    """指定 PV 群だけを kblogrd で取得（ip_fetch の低レベルを再利用）。
    返り値: {pv: [(ts, value_or_None), ...]}。存在しない PV は頑健に落とす。"""
    out = {}
    for chunk in ip_fetch._chunks(pvs, ip_fetch.CHUNK):
        out.update(ip_fetch._fetch_chunk(chunk, ttime, log_group, ip_fetch.KBLOGRD))
    return out


def _valid(v):
    return v is not None and v > NODATA


def _align(ccg_series, ip_series):
    """同時刻で両方が有効な点だけを取り出す。
    返り値: (timestamps[list], pressure[np.array], current[np.array])"""
    ip_map = dict(ip_series)
    ts, pres, cur = [], [], []
    for t, p in ccg_series:
        c = ip_map.get(t)
        if _valid(p) and _valid(c):
            ts.append(t); pres.append(p); cur.append(c)
    return ts, np.array(pres), np.array(cur)


def _logcorr(pres, cur):
    """log-log の相関係数（圧力∝電流なら 1 に近い）。点が少なければ None。"""
    if len(pres) < 3:
        return None
    lp, lc = np.log10(pres), np.log10(cur)
    if lp.std() == 0 or lc.std() == 0:
        return None
    return float(np.corrcoef(lp, lc)[0, 1])


def select_pairs(ring, n=6, section=None):
    """観察するペアを選ぶ。section 指定が無ければ KEK と 4U を混ぜて選ぶ。
    返り値: [(ip_pv, ccg_pv, supply), ...]"""
    ip_recs = ip_pv.load(ip_pv.csv_path(ring))
    ccg_set = set()
    import ccg_pv
    with open(ccg_pv.csv_path(ring), encoding="utf-8-sig", newline="") as f:
        import csv
        ccg_set = set(r[0].strip() for r in csv.reader(f) if r and r[0].strip())
    pairs = []
    for r in ip_recs:
        c = ip_pv.paired_ccg(r["pv"])
        if c and c in ccg_set:
            pairs.append((r["pv"], c, r["supply"]))
    if section:
        pairs = [p for p in pairs if ip_pv.section_of(p[0]) == section]
        return pairs[:n]
    # KEK と 4U を混ぜて選ぶ（電源種別の違いを見たいので）
    kek = [p for p in pairs if p[2] == ip_pv.SUPPLY_KEK]
    a4u = [p for p in pairs if p[2] == ip_pv.SUPPLY_4U]
    half = max(1, n // 2)
    chosen = kek[:n - min(half, len(a4u))] + a4u[:half]
    return chosen[:n]


def plot_pair(ip_pv_name, ccg_pv_name, supply, ccg_series, ip_series, out_dir):
    """1 ペアの時系列＋散布図を PNG に保存。返り値: (path, n_points, corr)。"""
    ts, pres, cur = _align(ccg_series, ip_series)
    corr = _logcorr(pres, cur)
    sec = ip_pv.section_of(ip_pv_name)
    sup_label = "Agilent 4U" if supply == ip_pv.SUPPLY_4U else "KEK"

    fig, (ax_ts, ax_sc) = plt.subplots(2, 1, figsize=(7, 7), tight_layout=True)

    # 上段: 時系列（2軸）。PNG 内ラベルは英語（日本語フォント非依存）
    x = np.arange(len(ts))
    ax_ts.set_title("%s  [%s, %s]\n%s  /  %s" %
                    (sec, sup_label, ring_of(ip_pv_name), ip_pv_name, ccg_pv_name),
                    fontsize=9)
    if len(ts):
        ax_ts.semilogy(x, pres, color="#c0392b", lw=1.2, label="CCG pressure [Pa]")
        ax_ts.set_ylabel("Pressure [Pa]", color="#c0392b", fontsize=9)
        ax_ts.tick_params(axis="y", labelcolor="#c0392b")
        ax2 = ax_ts.twinx()
        ax2.semilogy(x, cur, color="#2c6291", lw=1.2, label="IP current [A]")
        ax2.set_ylabel("Discharge current [A]", color="#2c6291", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#2c6291")
        ax_ts.set_xlabel("sample (time order)", fontsize=9)
    else:
        ax_ts.text(0.5, 0.5, "no valid data", ha="center", va="center")

    # 下段: 散布図（log-log）＋ 圧力フロア足切り＋ べき乗フィット
    fit = powerlaw_fit(pres, cur, floor_for(supply)) if len(ts) else None
    p_floor = floor_for(supply)
    if len(ts) >= 1:
        c = "#7a3fa0" if supply == ip_pv.SUPPLY_4U else "#2c6291"
        below = pres < p_floor
        # フロア未満（フィット対象外）は淡いグレー、以上は本来の色
        if below.any():
            ax_sc.loglog(pres[below], cur[below], "o", ms=3, alpha=0.25,
                         color="#999999", label="below floor (excluded)")
        if (~below).any():
            ax_sc.loglog(pres[~below], cur[~below], "o", ms=3, alpha=0.55,
                         color=c, label="used for fit")
        # 圧力フロアの縦線
        ax_sc.axvline(p_floor, color="#e67e22", ls="--", lw=1,
                      label="pressure floor %.0e Pa" % p_floor)
        # Agilent 4U の電流測定下限（10 nA）
        if supply == ip_pv.SUPPLY_4U:
            ax_sc.axhline(CUR_FLOOR_4U, color="#16a085", ls=":", lw=1,
                          label="4U current floor 10 nA")
        # べき乗フィット線 I = a*P^b（フロア以上の圧力範囲で描く）
        if fit:
            pf = pres[~below]
            xs = np.array([pf.min(), pf.max()])
            ys = fit["a"] * xs ** fit["b"]
            ax_sc.loglog(xs, ys, "-", color="#c0392b", lw=1.6,
                         label="fit  I=a·P^b")

    if fit:
        fit_tx = "fit: b=%.2f  a=%.2e  r=%.3f  (n=%d, P\u2265%.0e)" % (
            fit["b"], fit["a"], fit["r"], fit["n_used"], fit["p_floor"])
    else:
        fit_tx = "fit: N/A (フロア以上の点が不足/圧力一定)"
    corr_tx = (u"all-pts log-log r = %.3f" % corr) if corr is not None else "corr: too few points"
    ax_sc.set_title("Pressure vs Discharge current  (%s / %d pts)\n%s\n%s" %
                    (sup_label, len(ts), corr_tx, fit_tx), fontsize=8)
    ax_sc.set_xlabel("CCG pressure [Pa]", fontsize=9)
    ax_sc.set_ylabel("IP discharge current [A]", fontsize=9)
    ax_sc.grid(True, which="both", alpha=0.3)
    ax_sc.legend(fontsize=6, loc="best")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    safe = ip_pv_name.replace(":", "_").replace("/", "_")
    path = os.path.join(out_dir, "%s_%s_%s.png" % (ring_of(ip_pv_name), sec, safe))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, len(ts), corr, fit


def ring_of(pv):
    return "LER" if "VAL" in pv else "HER"


def observe(ring, start, end, n=6, section=None, interval_sec=60, out_dir="observe"):
    pairs = select_pairs(ring, n=n, section=section)
    if not pairs:
        print("ペアが見つかりません（section=%s）" % section)
        return
    ttime = ip_fetch.make_ttime(start, end, interval_sec)
    ip_pvs = [p[0] for p in pairs]
    ccg_pvs = [p[1] for p in pairs]
    print("取得中: %d ペア（IP=%s, CCG=%s）..." % (len(pairs), ip_fetch.LOG_GROUP, ccg_fetch.LOG_GROUP))
    ip_data = _fetch_pvs(ip_pvs, ttime, ip_fetch.LOG_GROUP)
    ccg_data = _fetch_pvs(ccg_pvs, ttime, ccg_fetch.LOG_GROUP)

    print("=== 観察結果（%s, %s〜%s）===" % (ring, start, end))
    for ip_name, ccg_name, supply in pairs:
        path, npt, corr, fit = plot_pair(ip_name, ccg_name, supply,
                                         ccg_data.get(ccg_name, []), ip_data.get(ip_name, []),
                                         out_dir)
        cs = ("r=%.3f" % corr) if corr is not None else "r=—"
        if fit:
            fs = "fit b=%.2f r=%.3f n=%d" % (fit["b"], fit["r"], fit["n_used"])
        else:
            fs = "fit —（フロア以上の点不足）"
        print("  [%-10s] %-28s 有効%4d点 %s  %s  → %s" %
              (supply, ip_name, npt, cs, fs, path))
    print("\nPNG は %s/ に保存しました。手元に持ってきて確認してください。" % out_dir)
    print("見方: 橙破線=圧力フロア（KEK 1e-7 / 4U 1e-8 Pa）。これ以上の点（色つき）だけで")
    print("      I=a·P^b をフィット。b≈1 なら I∝P。グレー点はフロア未満で対象外。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print("usage: python ip_observe.py <LER|HER> <start> <end> [--pairs N] [--section DXX] [--out DIR]")
        sys.exit(0)
    ring, start, end = args[0], args[1], args[2]
    n = 6
    section = None
    out_dir = "observe"
    if "--pairs" in args:
        n = int(args[args.index("--pairs") + 1])
    if "--section" in args:
        section = args[args.index("--section") + 1]
    if "--out" in args:
        out_dir = args[args.index("--out") + 1]
    observe(ring, start, end, n=n, section=section, out_dir=out_dir)
