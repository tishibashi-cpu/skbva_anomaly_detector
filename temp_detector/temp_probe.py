#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_probe.py — 特定の温度計だけを引いて波形と judge 結果を見る較正用ツール。

全台バッチ（数千本）を回す前に、既知故障（例 D01M095 = 6/17 に室温→ -14℃ の短絡）を
1本だけ細かい間隔で取得し、波形・統計・各層の判定・「自己中央値からの下振れ量」を表示する。
学習は系列の前半（健全部）、判定は後半（故障部）に分けて、現閾値での見え方を確認する。

使い方（実機・kblogrd 必要）:
  python temp_probe.py HER D01M095 20260615000000 20260618000000 600
  python temp_probe.py HER VAHTMP:D01M095:QLC3LE:BL 20260610000000 20260618000000 600 --learn-frac 0.6

  第2引数は完全な PV でも、PV に含まれる部分文字列（例 D01M095）でもよい（部分一致を全部対象）。
  既定間隔 600s。--learn-frac は学習に使う先頭割合（既定 0.5）。
"""

import sys
import numpy as np

import temp_fetch
import temp_judge
import temp_pv


def _resolve_pvs(ring, key):
    """ring の CSV から、key（完全PV or 部分文字列）に一致する PV を返す。"""
    recs = temp_fetch.load_pv_list(ring)
    pvs = [r["pv"] for r in recs]
    if key in pvs:
        return [key]
    hit = [p for p in pvs if key in p]
    return hit


def _sketch(T, width=60):
    """有限値だけで簡易 ASCII スパークライン（min..max を width 文字に量子化）。"""
    f = T[np.isfinite(T)]
    if len(f) < 2:
        return "(点が少なすぎ)"
    lo, hi = float(np.min(f)), float(np.max(f))
    if hi - lo < 1e-9:
        return "─" * min(width, len(T)) + "  (ほぼ一定)"
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(T) // width)
    out = []
    for i in range(0, len(T), step):
        v = T[i]
        if not np.isfinite(v):
            out.append(" ")
        else:
            q = int((v - lo) / (hi - lo) * (len(blocks) - 1))
            out.append(blocks[q])
    return "".join(out) + ("  [%.1f‥%.1f℃]" % (lo, hi))


def probe(ring, key, start, end, interval=600, learn_frac=0.5):
    pvs = _resolve_pvs(ring, key)
    if not pvs:
        print("一致する PV がありません: %r（ring=%s）" % (key, ring))
        return False
    print("対象 %d 本: %s" % (len(pvs), ", ".join(pvs)))
    data = temp_fetch.fetch_history(ring, start, end, interval_sec=interval, pvs=pvs)
    # ビーム電流も同じグリッドで取得（温度との相関・無ビーム時温度の確認用）
    try:
        beam_series = temp_fetch.fetch_beam(ring, start, end, interval_sec=interval)
    except Exception as ex:
        sys.stderr.write("ビーム取得に失敗（温度のみで継続）: %s\n" % ex)
        beam_series = []
    bmap = {t: (np.nan if val is None else float(val)) for t, val in beam_series}

    for pv in pvs:
        v = data.get(pv, {})
        ts, T = temp_fetch.series_to_arrays(v.get("series", []))
        B = np.array([bmap.get(t, np.nan) for t in ts], dtype=float)   # 温度と同時刻のビーム[mA]
        fin = T[np.isfinite(T)]
        print("\n" + "=" * 72)
        print("%s  [%s %s %s]" % (pv, v.get("section"), v.get("tag"), v.get("suffix")))
        if len(fin) == 0:
            print("  有効点なし（この期間に値が無い）")
            continue
        print("  点数 %d（有効 %d, 欠測 %d） t_med=%.2f  t_min=%.2f  t_max=%.2f  末尾=%.2f℃"
              % (len(T), len(fin), int(np.sum(~np.isfinite(T))),
                 float(np.median(fin)), float(np.min(fin)), float(np.max(fin)),
                 float(fin[-1])))
        print("  温度: " + _sketch(T))
        if np.any(np.isfinite(B)):
            print("  ビーム: " + _sketch(B))
            mb = np.isfinite(T) & np.isfinite(B)
            if int(np.sum(mb)) >= 5 and np.std(T[mb]) > 1e-6 and np.std(B[mb]) > 1e-6:
                r_tb = float(np.corrcoef(T[mb], B[mb])[0, 1])
                print("  温度-ビーム相関 r=%.2f（健全部は正に出やすい。窓内に故障が混じると乱れる）" % r_tb)

        # 学習=前半（健全部想定）/ 判定=後半（故障部想定）に分割
        k = max(temp_judge.CONFIG["noise_min_pts"], int(len(T) * learn_frac))
        T_learn, T_judge = T[:k], T[k:]
        model = temp_judge.learn_sensor(T_learn)
        med_learn = float(np.nanmedian(T_learn)) if np.any(np.isfinite(T_learn)) else float("nan")

        # 直近の尾（最近発症の故障は窓全体だと薄まるので、尾を別に判定して S 層の発火も見る）
        tail_frac = 0.20
        kt = max(12, int(len(T) * tail_frac))
        T_tail = T[-kt:]

        for label, seg, segb in (("全体", T, B), ("後半(判定対象)", T_judge, B[k:]),
                                 ("尾(末尾%d)" % kt, T_tail, B[-kt:])):
            r = temp_judge.judge_sensor(seg, model=model, beam=segb)
            h0, nz, g = r["layers"]["H0_range"], r["layers"]["N_noise"], r["layers"]["G_glitch"]
            sshort, bbeam = r["layers"].get("S_short", {}), r["layers"].get("B_beam", {})
            ohot = r["layers"].get("O_hot_noheat", {})
            iint = r["layers"].get("I_intermittent", {})
            print("  judge[%-12s] sev=%d %-26s  t_min=%.1f | N.exc=%s | G.n=%d | S.frac=%s | r_beam=%s | O.hot=%s | I.ev=%s"
                  % (label, r["severity"], r["reason"],
                     (h0["t_min"] if h0["t_min"] is not None else float("nan")),
                     (("%.2f" % nz["excess_dex"]) if nz["excess_dex"] is not None else "-"),
                     g["n_glitch"],
                     (("%.2f" % sshort["frac_low"]) if sshort.get("frac_low") is not None else "-"),
                     (("%+.2f" % bbeam["r_beam"]) if bbeam.get("r_beam") is not None else "-"),
                     (("%.2f" % ohot["frac_hot"]) if ohot.get("frac_hot") is not None else "-"),
                     (iint.get("n_events") if iint.get("n_events") is not None else "-")))

        # ▶ 「直近の尾」を学習中央値と比べる。ビームも併記して、温度低下が①ビーム減（正常な冷却）か
        #   ②センサ故障かを切り分ける。
        tail = T_tail
        tail_fin = tail[np.isfinite(tail)]
        med_tail = float(np.median(tail_fin)) if len(tail_fin) else float("nan")
        drop = med_learn - med_tail
        below = (float(np.mean(tail_fin < (med_learn - 10.0))) * 100.0) if len(tail_fin) else 0.0
        hrs = kt * interval / 3600.0
        print("  ▶ 直近尾(末尾%d点≈%.1fh)中央値=%.2f℃  学習中央値=%.2f℃  → drop=%.1f℃ / "
              "尾の%.0f%%が学習-10℃未満（持続短絡の指標）" % (kt, hrs, med_tail, med_learn, drop, below))
        Btail = B[-kt:]
        Btail_fin = Btail[np.isfinite(Btail)]
        if len(Btail_fin):
            bmed = float(np.median(Btail_fin))
            # 尾の温度-ビーム相関（反相関の確認）
            mbt = np.isfinite(tail) & np.isfinite(Btail)
            rb = None
            if int(np.sum(mbt)) >= 5 and np.std(tail[mbt]) > 1e-6 and np.std(Btail[mbt]) > 1e-6:
                rb = float(np.corrcoef(tail[mbt], Btail[mbt])[0, 1])
            if med_tail < 10.0 and bmed > 50.0:
                verdict = "⇒ ビームありなのに常温以下＝センサ故障濃厚（短絡）"
            elif med_tail < 10.0:
                verdict = "⇒ 尾は無ビームだが %.1f℃ は常温以下＝冷却で説明不可（短絡寄り）" % med_tail
            elif bmed < 50.0 and med_tail > 35.0:
                verdict = "⇒ 無ビームなのに %.1f℃＝発熱源なしに高温（near-open/高抵抗の疑い）" % med_tail
            elif rb is not None and rb <= -0.5:
                verdict = "⇒ ビーム反相関（r=%+.2f）＝無ビームで温度↑の非物理パターン（センサ異常）" % rb
            else:
                verdict = "⇒ 物理レンジ内・サブ常温でない（短絡ではない）"
            print("  ▶ 直近尾のビーム中央値=%.0f mA  %s" % (bmed, verdict))
    return True


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 5:
        print("usage: python temp_probe.py <LER|HER> <PV or 部分文字列> <start> <end> [interval=600] [--learn-frac F]")
        print("例:    python temp_probe.py HER D01M095 20260615000000 20260618000000 600")
        sys.exit(0)
    ring, key, start, end = a[1], a[2], a[3], a[4]
    interval = 600
    lf = 0.5
    rest = a[5:]
    if rest and rest[0].isdigit():
        interval = int(rest[0]); rest = rest[1:]
    if "--learn-frac" in rest:
        lf = float(rest[rest.index("--learn-frac") + 1])
    sys.exit(0 if probe(ring, key, start, end, interval, lf) else 1)
