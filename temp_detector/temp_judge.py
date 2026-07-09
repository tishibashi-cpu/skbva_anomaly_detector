#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_judge.py — ビームパイプ温度計の「センサ故障予兆」検知核（numpy のみ）。

目的は機器の温度上昇ではなく【測温抵抗体(RTD)センサ自身の故障の予兆】を、アラームが
出る前に拾うこと。SuperKEKB では機器故障による温度上昇はほぼ起きておらず、異常の大半は
センサ故障（断線→非現実的高温、短絡→低温/マイナス、接触不良→ノイズ/グリッチ増大）。

主軸（全台に効く・単独センサ）:
  H0 レンジ:   物理的にありえない高温/低温（断線・短絡の既発症を手前で拾う）
  H1 張り付き: 値がほぼ完全に固定（更新停止・通信断）。RTD は本来微小ノイズがあるはず
  N  ノイズ:   短時間の分散増大（接触抵抗ゆらぎ＝断線しかけの予兆。これが本命）
  G  グリッチ: 物理的にありえない急変（|ΔT/Δt| 大）の頻発＝間欠接触不良
補助（一部のみ）:
  P  ペア乖離: 上下ペア（ウィグラー等）の ΔT が普段の関係から乖離 → 片側センサ異常

機械学習は使わない（signature が単純・故障例が少ない・説明可能性重視）。学習(learn)は
各センサの「平常の短時間ノイズ幅」を記録するだけの軽い統計。場所ごとにノイズ水準が違う
ので、ノイズ判定は絶対閾値だけでなく自己平常との比較も併用する。
"""

import os

# numpy(OpenBLAS)は既定で「使えるだけのコア数」を毎回の計算にフル動員しようとし、共用サーバーで
# 無駄にCPUを奪い合う（detector_headless.py --watch で実機のCPU使用率1000%超として確認済み。
# 詳細はdetector_headless.py冒頭のコメント参照）。このファイルは selftest 等で単独実行される
# ことも多く、他のimportより前に対策を入れる。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

CONFIG = {
    # --- H0 レンジ（物理的にありえない値）---
    #   実機の正常温度はウィグラー部で ~280℃ まで上がりうる。一方センサ故障は
    #   -3000℃ や +9999℃ のような明らかに非現実値を返す。ハードな外れだけ拾う。
    "t_hard_high":   400.0,   # これ超は明らかにセンサ異常[℃]
    "t_hard_low":    -40.0,   # これ未満は明らかにセンサ異常[℃]（短絡でマイナス）
    "t_warn_high":   300.0,   # ここからは要観察（正常上限に近い）[℃]
    "t_warn_low":     -5.0,   # 環境温度を下回る → 要観察[℃]

    # --- H1 張り付き（更新停止/通信断）---
    "stuck_range":    0.005,  # 窓内の値域がこれ未満[℃]＝動かなさすぎ（RTD は微小ノイズが正常）
    "stuck_min_pts":  20,     # 張り付き判定に必要な点数

    # --- N ノイズ（短時間分散の増大）---
    "noise_win":      11,     # 短時間ノイズを測る移動窓の点数（奇数）
    "noise_abs_hi":   0.8,    # 移動標準偏差の中央値がこれ超で異常[℃]（モデル無し時のフォールバック）
    "noise_rel_dex":  0.5,    # 学習バンド p95 をこれ[dex]超で異常（自己平常比, ~3倍）
    "noise_min_pts":  30,

    # --- G グリッチ（物理的にありえない急変）---
    "glitch_jump":    5.0,    # 隣接点の差がこれ超[℃/点]＝熱質量的にありえない急変
    "glitch_min_n":   3,      # この回数以上で異常
    "glitch_beam_steady_ma": 50.0,  # 隣接点のビーム変化がこれ未満[mA]の「ビーム定常時」のジャンプだけ
                                    #   グリッチに数える（フィル/アボートの温度ステップを誤計上しない）

    # --- P 上下ペア乖離 ---
    "pair_dev_warn":  2.0,    # |ΔT - 学習ΔT| がこれ超[℃]で要観察
    "pair_dev_hi":    5.0,    # これ超[℃]で異常
    "pair_min_pts":   20,

    # --- S 短絡（自己平常から「常温以下」への持続的低下。ビーム非依存・モデル不要）---
    #   実機 D01M095（HER, 6/17 短絡）: 学習中央値 ~24℃ → 持続 ~-13℃（drop ~37℃）。
    #   尾は無ビームだが、トンネル部品は無ビームでも ~20℃ 止まりで -13℃ には冷えない。
    #   → サブ常温は「冷却」で説明できないのでビーム非依存に短絡と判定できる。
    #   実機確認済み（2026-06-27, LER/HER 両方）: 常温以下が平常のセンサは存在しない。
    #   よって「窓内でサブ常温が持続」の一点だけで判定してよく、学習モデルへの依存は外す。
    #   ローリング基準（実行時に直近数日を学習）は、基準窓の時点で既にセンサが故障していると
    #   学習中央値ごとサブ常温に汚染されうる。以前は学習中央値でゲートしていたため、この汚染時に
    #   短絡が抑制される不具合があった（基準そのものが壊れているのに「平常」と誤認したため）。
    #   モデル依存を外すことでこの問題を解消する。単発グリッチは frac（持続割合）で自然に落ちる。
    "short_floor":    10.0,   # これ以下[℃]はトンネル部品としてサブ常温＝非現実
    "short_min_frac": 0.5,    # 窓のこの割合以上がサブ常温なら「持続短絡」

    # --- B ビーム反相関（無ビームで温度が上がる等、ビームと逆に動く＝センサ異常）---
    #   健全な部品温度はビーム発熱（SR/HOM）で正相関 or 無相関。強い負相関（ビーム↓で温度↑）は
    #   非物理＝センサ異常。実例 D12_136（HER）。短絡(S)は逆に正相関寄り（無ビームで低温）で別物。
    "beam_anticorr_r":   -0.5,   # 温度-ビーム相関がこれ以下なら反相関異常
    "beam_corr_min_pts": 30,     # 相関に必要な有効点数（温度・ビーム両方有限）
    "beam_min_range_ma": 200.0,  # ビームがこれ以上[mA]振れていないと相関は当てにならない

    # --- O 高温側（発熱源が無いのに高温＝near-open / 高抵抗・絶縁寄り。ビーム条件付き）---
    #   無ビームなら部品はアンビエント(~20-25℃)へ冷えるはず。十分な無ビーム点で温度が ceiling を
    #   超えて持続していたら「発熱源が無いのに高温」＝センサ故障。t_hard_high(400)/warn(300) は
    #   ウィグラー正常高温(~280℃)を避けるため高く、定常的にほどほど高い near-open は抜ける
    #   （実例 D02_250 ~125℃, D06_242 ~51℃ が現状 sev0 だった）。これを無ビーム条件で拾う。
    #   ※NEG ベーキング等の意図的加熱は別物なので運用で suffix=NEG 等を除外する想定。
    "open_hot_ceil":   35.0,   # 無ビームでこれ超[℃]＝アンビエント超過（発熱源なしに高温）
    "open_min_frac":   0.5,    # 無ビーム点のこの割合以上が ceil 超で発火
    "beam_off_ma":     50.0,   # これ未満[mA]を無ビームとみなす
    "open_min_nb_pts": 20,     # 無ビーム点がこれ以上必要（冷却途中の一過性で誤検知しない）

    # --- I 間欠逸脱（稀だが繰り返す warn 域への逸脱＝間欠的接触不良/断続故障）---
    #   実例 D11M006: 中央値 ~18℃ だが時々 -10℃ に落ちる（range_watch=sev1 止まりだった）。
    #   warn 域（≤t_warn_low or ≥t_warn_high）への「逸脱イベント（立ち上がり）」が複数回あれば
    #   間欠故障とみなし sev2 へ格上げ。単発グリッチ（1イベント）や正常ノイズと区別し、
    #   持続（割合大）は S/O/range 側に任せる。-5℃以下はビーム冷却では届かないので誤検知も少ない。
    "exc_min_events": 3,       # warn 域への逸脱イベントがこの数以上で間欠逸脱
    "exc_max_frac":   0.40,    # ただし窓のこの割合未満（以上は持続＝S/O/range 側）
}


def _diff_noise(x):
    """高周波ノイズの頑健推定: 隣接差分の MAD（緩い温度変動=トレンドは差分で消える）。
    返り値 σ 相当[℃]。グリッチ少数には MAD なので鈍感。"""
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    d = np.diff(x)
    mad = np.median(np.abs(d - np.median(d)))
    # diff の σ ≈ 1.4826*MAD、さらに 1点差分の √2 を戻して 1センサ相当に
    return float(1.4826 * mad / np.sqrt(2.0))


def _rolling_diffnoise(x, win):
    """窓ごとの _diff_noise を並べた配列（学習のp50/p95・判定の中央値に使う）。"""
    n = len(x)
    if n < win:
        return np.array([_diff_noise(x)]) if n >= 3 else np.array([])
    out = []
    for i in range(0, n - win + 1, max(1, win // 2)):
        out.append(_diff_noise(x[i:i + win]))
    return np.array([v for v in out if v >= 0])


def learn_sensor(T, cfg=CONFIG):
    """1センサの平常モデル: 短時間ノイズ(移動std)の p50/p95 と 温度中央値。
    返り値 {noise_log_p50, noise_log_p95, t_med, n} または None。"""
    T = np.asarray(T, float)
    T = T[np.isfinite(T)]
    if len(T) < cfg["noise_min_pts"]:
        return None
    rs = _rolling_diffnoise(T, cfg["noise_win"])
    rs = rs[rs > 0]
    if len(rs) < 5:
        return None
    lrs = np.log10(rs)
    return {"noise_log_p50": float(np.percentile(lrs, 50)),
            "noise_log_p95": float(np.percentile(lrs, 95)),
            "t_med": float(np.median(T)), "n": int(len(T))}


def judge_sensor(T, cfg=CONFIG, model=None, pair_T=None, pair_dT0=None, dt_sec=None, beam=None, skip_open=False, quick=False):
    """1センサの判定。T は温度時系列[℃]（numpy 配列か list）。
    model: learn_sensor の出力（任意）。pair_T: 対センサの温度時系列（任意）。
    pair_dT0: そのペアの平常 ΔT（任意）。beam: T と同時刻のビーム電流[mA]配列（任意・反相関判定用）。
    skip_open: True で O層（無ビーム高温）を評価しない（NEG ベーキング等の意図的加熱センサ用）。
    quick: True で重い層（N ノイズ・B 相関）を省く（安価な構造判定だけ）。バッチ段階フィルタ用。
    返り値 {severity(0-3), reason, layers{...}, metrics{...}}。
    """
    T = np.asarray(T, float)
    fin = np.isfinite(T)
    Tf = T[fin]
    layers = {}
    metrics = {"t_med": float(np.median(Tf)) if len(Tf) else None,
               "n_pts": int(len(Tf))}

    # ── H0 レンジ ──
    h0 = {"hard_high": False, "hard_low": False, "warn_high": False, "warn_low": False,
          "t_max": float(np.max(Tf)) if len(Tf) else None,
          "t_min": float(np.min(Tf)) if len(Tf) else None}
    if len(Tf):
        h0["hard_high"] = bool(h0["t_max"] >= cfg["t_hard_high"])
        h0["hard_low"] = bool(h0["t_min"] <= cfg["t_hard_low"])
        h0["warn_high"] = bool(h0["t_max"] >= cfg["t_warn_high"])
        h0["warn_low"] = bool(h0["t_min"] <= cfg["t_warn_low"])
    layers["H0_range"] = h0

    # ── H1 張り付き ──
    h1 = {"fired": False, "range": None}
    if len(Tf) >= cfg["stuck_min_pts"]:
        rng = float(np.max(Tf) - np.min(Tf))
        h1["range"] = rng
        h1["fired"] = bool(rng < cfg["stuck_range"])
    layers["H1_stuck"] = h1

    # ── N ノイズ（短時間分散の増大）──
    nz = {"fired": False, "med_rstd": None, "basis": None, "excess_dex": None}
    if not quick and len(Tf) >= cfg["noise_min_pts"]:
        rs = _rolling_diffnoise(Tf, cfg["noise_win"])
        med_rstd = float(np.median(rs[rs > 0])) if np.any(rs > 0) else 0.0
        nz["med_rstd"] = med_rstd
        if model and ("noise_log_p95" in model) and med_rstd > 0:
            ref = model["noise_log_p95"]
            excess = np.log10(med_rstd) - ref
            nz["excess_dex"] = float(excess)
            nz["basis"] = "learned_p95"
            nz["fired"] = bool(excess >= cfg["noise_rel_dex"])
        else:
            nz["basis"] = "abs"
            nz["fired"] = bool(med_rstd >= cfg["noise_abs_hi"])
    layers["N_noise"] = nz

    # ── G グリッチ（反転スパイクの頻発）──
    #   「1点だけ跳ねて戻る」反転スパイク（接触不良の跳ね）を数える。なめらかな単調ランプ
    #   （ビーム発熱/冷却による上下）は同符号の連続差分なので数えない＝熱遅れにも左右されない。
    #   さらにフィル/アボート（ビーム大変化）の近傍はビーム由来として除外（保険）。
    g = {"n_glitch": 0, "fired": False, "beam_aware": False}
    if len(T) >= 3:
        d = np.diff(T)                       # 隣接差分（長さ n-1）
        d1, d2 = d[:-1], d[1:]               # 内点 i(1..n-2) の前差分・後差分
        fin = np.isfinite(T[:-2]) & np.isfinite(T[1:-1]) & np.isfinite(T[2:])
        # 反転スパイク: 前後とも |差| ≥ jump で符号が逆（1点だけの突出）
        spike = fin & (np.abs(d1) >= cfg["glitch_jump"]) & (np.abs(d2) >= cfg["glitch_jump"]) & (d1 * d2 < 0)
        if beam is not None:
            Bg = np.asarray(beam, float)
            if Bg.shape == T.shape:
                dB = np.abs(np.diff(Bg))
                # 内点 i の前後どちらかでビームが大きく動いていれば、ビーム由来として除外
                beam_dist = ((np.isfinite(Bg[:-2]) & np.isfinite(Bg[1:-1]) & (dB[:-1] >= cfg["glitch_beam_steady_ma"]))
                             | (np.isfinite(Bg[1:-1]) & np.isfinite(Bg[2:]) & (dB[1:] >= cfg["glitch_beam_steady_ma"])))
                spike = spike & ~beam_dist
                g["beam_aware"] = True
        n_g = int(np.sum(spike))
        g["n_glitch"] = n_g
        g["fired"] = bool(n_g >= cfg["glitch_min_n"])
    layers["G_glitch"] = g

    # ── P 上下ペア乖離（任意）──
    p = {"have": False, "dev": None, "warn": False, "hi": False}
    if pair_T is not None:
        pT = np.asarray(pair_T, float)
        m = np.isfinite(T) & np.isfinite(pT)
        if int(np.sum(m)) >= cfg["pair_min_pts"]:
            dT = T[m] - pT[m]
            cur = float(np.median(dT))
            base = pair_dT0 if pair_dT0 is not None else 0.0
            dev = abs(cur - base)
            p.update(have=True, dev=dev, dT_med=cur, dT0=base)
            p["warn"] = bool(dev >= cfg["pair_dev_warn"])
            p["hi"] = bool(dev >= cfg["pair_dev_hi"])
    layers["P_pair"] = p

    # ── S 短絡（自己平常から「常温以下」への持続的低下。ビーム非依存・モデル不要）──
    #   窓内でサブ常温（≤short_floor）が一定割合以上続いていれば短絡とみなす（モデル無しでも判定可）。
    #   モデルがあれば t_med_learn（学習中央値）を診断情報として付与するのみで、発火条件には使わない
    #   （ローリング基準が汚染されていても検知を抑制しないため。詳細は CONFIG コメント参照）。
    s = {"fired": False, "frac_low": None, "t_med_learn": None}
    if len(Tf) >= cfg["stuck_min_pts"]:
        frac_low = float(np.mean(Tf <= cfg["short_floor"]))
        s["frac_low"] = frac_low
        if model and ("t_med" in model):
            s["t_med_learn"] = float(model["t_med"])
        s["fired"] = bool(frac_low >= cfg["short_min_frac"])
    layers["S_short"] = s

    # ── B ビーム反相関（無ビームで温度↑＝ビームと逆に動く＝センサ異常）──
    #   健全部品はビーム発熱で正相関 or 無相関。強い負相関は非物理。ビームが十分振れていて
    #   温度も変動しているときだけ評価する（無ビーム一定期間や平坦期は r が当てにならない）。
    b = {"have": False, "r_beam": None, "fired": False}
    if not quick and beam is not None:
        B = np.asarray(beam, float)
        if B.shape == T.shape:
            mb = np.isfinite(T) & np.isfinite(B)
            if int(np.sum(mb)) >= cfg["beam_corr_min_pts"]:
                Tm, Bm = T[mb], B[mb]
                if (float(np.max(Bm) - np.min(Bm)) >= cfg["beam_min_range_ma"]
                        and np.std(Tm) > 1e-6 and np.std(Bm) > 1e-6):
                    r = float(np.corrcoef(Tm, Bm)[0, 1])
                    b.update(have=True, r_beam=r)
                    b["fired"] = bool(r <= cfg["beam_anticorr_r"])
    layers["B_beam"] = b

    # ── O 高温側（無ビームなのに高温＝発熱源なし＝near-open/高抵抗）──
    #   無ビーム点（beam < beam_off_ma）が十分あり、その温度の一定割合が ceiling 超なら発火。
    #   ただし反相関(B)が発火している場合は「無ビームで高温」はその帰結なので、より具体的な
    #   B（反相関）を優先して O は抑制する。ビームが無いと評価不能（have=False）。
    o = {"have": False, "frac_hot": None, "n_nb": None, "fired": False}
    if beam is not None and not b["fired"] and not skip_open:
        Bo = np.asarray(beam, float)
        if Bo.shape == T.shape:
            nb = np.isfinite(T) & np.isfinite(Bo) & (Bo < cfg["beam_off_ma"])
            n_nb = int(np.sum(nb))
            if n_nb >= cfg["open_min_nb_pts"]:
                frac_hot = float(np.mean(T[nb] > cfg["open_hot_ceil"]))
                o.update(have=True, frac_hot=frac_hot, n_nb=n_nb)
                o["fired"] = bool(frac_hot >= cfg["open_min_frac"])
    layers["O_hot_noheat"] = o

    # ── I 間欠逸脱（warn 域への逸脱イベントが複数回＝間欠的接触不良/断続故障）──
    #   逸脱「イベント数」＝ warn 域へ入った立ち上がり回数。単発（1回）は glitch 扱いで除外。
    #   持続（割合大）は S/O/range が担当するので exc_max_frac 未満に限る。安価なので quick でも評価。
    ii = {"n_exc": None, "n_events": None, "frac": None, "fired": False}
    if len(Tf) >= cfg["stuck_min_pts"]:
        inwarn = (Tf <= cfg["t_warn_low"]) | (Tf >= cfg["t_warn_high"])
        n_exc = int(np.sum(inwarn))
        if n_exc > 0:
            entries = int(inwarn[0]) + int(np.sum(inwarn[1:] & ~inwarn[:-1]))
        else:
            entries = 0
        frac = float(n_exc) / len(Tf)
        ii.update(n_exc=n_exc, n_events=entries, frac=frac)
        ii["fired"] = bool(entries >= cfg["exc_min_events"] and frac < cfg["exc_max_frac"])
    layers["I_intermittent"] = ii

    sev, reason = _aggregate(h0, h1, nz, g, p, s, b, o, ii)
    return {"severity": sev, "reason": reason, "layers": layers, "metrics": metrics}


def _aggregate(h0, h1, nz, g, p, s=None, b=None, o=None, ii=None):
    """層 → severity(0-3) と理由。センサ故障の確度が高いものを優先。"""
    s = s or {"fired": False}
    b = b or {"fired": False}
    o = o or {"fired": False}
    ii = ii or {"fired": False}
    # sev3: 明らかなセンサ故障（既発症）/ 断線しかけの強い予兆
    if h0["hard_high"]:
        return 3, "range_high_open_suspect"     # 非現実高温＝断線疑い
    if h0["hard_low"] or s.get("fired"):
        return 3, "range_low_short_suspect"     # 低温/マイナス（持続サブ常温含む）＝短絡疑い
    if o.get("fired"):
        return 3, "range_high_noheat_suspect"   # 無ビームなのに高温＝発熱源なし＝near-open/高抵抗
    if nz["fired"] and g["fired"]:
        return 3, "noise_and_glitch"            # 分散増大＋グリッチ＝断線間近
    if p["hi"]:
        return 3, "pair_deviation_high"
    # sev2: 予兆（本命）— ビーム反相関 / 間欠逸脱 / ノイズ増大 / グリッチ頻発 / 張り付き
    if b.get("fired"):
        return 2, "beam_anticorrelated"         # ビーム↓で温度↑＝非物理なセンサ異常
    if ii.get("fired"):
        return 2, "intermittent_excursion"      # 稀だが繰り返す warn 域逸脱＝間欠故障
    if nz["fired"]:
        return 2, "noise_increase"
    if g["fired"]:
        return 2, "glitch"
    if h1["fired"]:
        return 2, "stuck"
    # sev1: 要観察 — レンジ接近 / ペア乖離（軽）
    if h0["warn_high"] or h0["warn_low"]:
        return 1, "range_watch"
    if p["warn"]:
        return 1, "pair_deviation_watch"
    return 0, "normal"


# ───────────────────────── selftest ─────────────────────────

def _synth(kind, n=240, seed=0):
    rng = np.random.RandomState(seed)
    base = 25.0 + 3.0 * np.sin(np.arange(n) / 40.0)   # 平常のゆるい変動
    if kind == "healthy":
        return base + rng.normal(0, 0.05, n)
    if kind == "noisy":                                # 断線しかけ＝分散増大
        return base + rng.normal(0, 0.6, n)
    if kind == "glitchy":                              # 間欠接触不良＝突発スパイク
        T = base + rng.normal(0, 0.05, n)
        for i in rng.choice(n, 6, replace=False):
            T[i] += rng.choice([-1, 1]) * rng.uniform(8, 20)
        return T
    if kind == "open_high":                            # 断線＝非現実高温
        T = base + rng.normal(0, 0.05, n); T[n // 2:] = 9999.0
        return T
    if kind == "short_low":                            # 短絡＝マイナス
        T = base + rng.normal(0, 0.05, n); T[n // 2:] = -3000.0
        return T
    if kind == "stuck":                                # 更新停止＝完全固定
        return np.full(n, 24.0)
    raise ValueError(kind)


def _selftest():
    print("=== temp_judge selftest（合成データ）===")
    cfg = CONFIG
    ok = True

    h = _synth("healthy", seed=1)
    hm = learn_sensor(h, cfg)
    r = judge_sensor(h, cfg, hm)
    print("[healthy]   sev=%d %-20s med_rstd=%.3f" % (r["severity"], r["reason"], r["layers"]["N_noise"]["med_rstd"]))
    ok &= (r["severity"] == 0)

    r = judge_sensor(_synth("noisy", seed=2), cfg, hm)
    print("[noisy]     sev=%d %-20s excess=%s" % (r["severity"], r["reason"], r["layers"]["N_noise"]["excess_dex"]))
    ok &= (r["severity"] == 2 and r["reason"] == "noise_increase")

    r = judge_sensor(_synth("glitchy", seed=3), cfg, hm)
    print("[glitchy]   sev=%d %-20s n_glitch=%d" % (r["severity"], r["reason"], r["layers"]["G_glitch"]["n_glitch"]))
    ok &= (r["severity"] in (2, 3) and "glitch" in r["reason"])

    r = judge_sensor(_synth("open_high", seed=4), cfg, hm)
    print("[open_high] sev=%d %-20s t_max=%.0f" % (r["severity"], r["reason"], r["layers"]["H0_range"]["t_max"]))
    ok &= (r["severity"] == 3 and r["reason"] == "range_high_open_suspect")

    r = judge_sensor(_synth("short_low", seed=5), cfg, hm)
    print("[short_low] sev=%d %-20s t_min=%.0f" % (r["severity"], r["reason"], r["layers"]["H0_range"]["t_min"]))
    ok &= (r["severity"] == 3 and r["reason"] == "range_low_short_suspect")

    # 持続サブ常温の短絡（実 D01M095 相当: 学習 ~24℃ → 判定窓は持続 -13℃）。-40 ハードには
    # 届かないが S 層（窓内でサブ常温が持続）で sev3 になること。
    short13 = -13.0 + np.random.RandomState(11).normal(0, 0.5, 120)
    r = judge_sensor(short13, cfg, hm)            # hm は健全 ~25℃ で学習済み
    print("[short_-13] sev=%d %-20s frac_low=%.2f"
          % (r["severity"], r["reason"], r["layers"]["S_short"]["frac_low"]))
    ok &= (r["severity"] == 3 and r["reason"] == "range_low_short_suspect")

    # 無ビーム冷却（健全部品が ~20℃ まで冷える）: フロア(10℃)未満ではないので S は非発火。
    cool20 = 20.0 + np.random.RandomState(12).normal(0, 0.3, 120)
    r = judge_sensor(cool20, cfg, hm)
    print("[cool_20]   sev=%d %-20s frac_low=%.2f（冷却は短絡にしない）"
          % (r["severity"], r["reason"], r["layers"]["S_short"]["frac_low"]))
    ok &= (r["reason"] != "range_low_short_suspect")

    # モデル無しでも S は単独で機能する（窓自体の持続割合だけで判定）。
    r = judge_sensor(short13, cfg, model=None)
    print("[short_nomodel] sev=%d %-20s frac_low=%.2f（モデル無しでも短絡検知）"
          % (r["severity"], r["reason"], r["layers"]["S_short"]["frac_low"]))
    ok &= (r["severity"] == 3 and r["reason"] == "range_low_short_suspect")

    # 【回帰テスト】ローリング基準の汚染: 基準窓の時点で既に故障していた場合、学習中央値ごと
    # サブ常温に汚染される（実機 D01M095 で発生: 基準学習 -25℃ → 旧実装は warm_ok=False で
    # 短絡を見逃していた）。S はモデル非依存なので、汚染されたモデルを渡しても正しく検知できること。
    contaminated_model = learn_sensor(short13, cfg)          # 汚染された基準（学習自体が -13℃台）
    r = judge_sensor(short13, cfg, contaminated_model)
    print("[short_contam_baseline] sev=%d %-20s t_med_learn=%.1f（汚染基準でも検知維持）"
          % (r["severity"], r["reason"], r["layers"]["S_short"]["t_med_learn"]))
    ok &= (r["severity"] == 3 and r["reason"] == "range_low_short_suspect")


    # ビーム反相関（D12_136 相当: 無ビームで温度↑）→ B が O より優先され beam_anticorrelated。
    brng = np.random.RandomState(14)
    beam = np.where(np.arange(200) % 40 < 25, 1000.0, 0.0)      # 高/無ビームの繰り返し
    T_anti = 58.0 + np.where(beam > 100, -8.0, 22.0) + brng.normal(0, 0.4, 200)  # 無ビームで高温(80℃)
    r = judge_sensor(T_anti, cfg, learn_sensor(T_anti, cfg), beam=beam)
    print("[beam_anti] sev=%d %-22s r_beam=%.2f（無ビーム高温でもBが優先）"
          % (r["severity"], r["reason"], r["layers"]["B_beam"]["r_beam"]))
    ok &= (r["severity"] == 2 and r["reason"] == "beam_anticorrelated")

    # 健全（ビーム発熱で正相関・無ビームはアンビエント）→ B も O も発火しない。
    T_pos = np.where(beam > 100, 30.0, 21.0) + brng.normal(0, 0.4, 200)   # 無ビーム21℃, ビーム時30℃
    r = judge_sensor(T_pos, cfg, learn_sensor(T_pos, cfg), beam=beam)
    print("[beam_pos]  sev=%d %-22s r_beam=%.2f（正相関・無ビームは常温→異常にしない）"
          % (r["severity"], r["reason"], r["layers"]["B_beam"]["r_beam"]))
    ok &= (r["reason"] not in ("beam_anticorrelated", "range_high_noheat_suspect"))

    # O 無ビームなのに高温（D02_250/D06_242 相当: ~60℃ で張り付き、ビーム=0）→ sev3。
    beam0 = np.zeros(120)
    T_hot = 60.0 + brng.normal(0, 0.6, 120)
    r = judge_sensor(T_hot, cfg, beam=beam0)
    print("[open_hot]  sev=%d %-22s frac_hot=%.2f n_nb=%d"
          % (r["severity"], r["reason"], r["layers"]["O_hot_noheat"]["frac_hot"], r["layers"]["O_hot_noheat"]["n_nb"]))
    ok &= (r["severity"] == 3 and r["reason"] == "range_high_noheat_suspect")

    # ビーム時だけ高温（正常な発熱）→ 無ビーム点はアンビエントなので O 非発火。
    T_bheat = np.where(beam > 100, 60.0, 22.0) + brng.normal(0, 0.4, 200)
    r = judge_sensor(T_bheat, cfg, beam=beam)
    print("[beam_heat] sev=%d %-22s frac_hot=%.2f（ビーム発熱は故障にしない）"
          % (r["severity"], r["reason"], r["layers"]["O_hot_noheat"]["frac_hot"]))
    ok &= (r["reason"] != "range_high_noheat_suspect")

    # 間欠逸脱（D11M006 相当: ~18℃ で時々 -10℃、複数イベント）→ sev2 intermittent_excursion。
    inter = 18 + np.random.RandomState(21).normal(0, 0.2, 200)
    for s0 in (30, 80, 130, 170):
        inter[s0:s0 + 2] = -10.0                 # 4 イベント・各2点
    r = judge_sensor(inter, cfg)
    print("[intermit]  sev=%d %-22s events=%d frac=%.2f"
          % (r["severity"], r["reason"], r["layers"]["I_intermittent"]["n_events"], r["layers"]["I_intermittent"]["frac"]))
    ok &= (r["severity"] == 2 and r["reason"] == "intermittent_excursion")

    # 単発の落ち込み（1 イベント）→ 間欠ではない（range_watch=sev1 のまま）。
    single = 18 + np.random.RandomState(22).normal(0, 0.2, 200); single[100:102] = -10.0
    r = judge_sensor(single, cfg)
    print("[single_dip]sev=%d %-22s events=%d（単発は間欠にしない）"
          % (r["severity"], r["reason"], r["layers"]["I_intermittent"]["n_events"]))
    ok &= (r["reason"] != "intermittent_excursion")

    # glitch=反転スパイク: なめらかなビーム連動スイング（台形・単調ランプ＋プラトー）は数えない。
    seg = np.concatenate([np.full(8, 24.0), [31.0, 38.0, 45.0], np.full(8, 45.0), [38.0, 31.0, 24.0]])  # 7℃/点の単調ランプ＋プラトー
    swing = (np.tile(seg, 10)[:200]) + np.random.RandomState(31).normal(0, 0.1, 200)
    r = judge_sensor(swing, cfg)
    print("[swing]     sev=%d %-22s n_glitch=%d（なめらかな上下は glitch にしない）"
          % (r["severity"], r["reason"], r["layers"]["G_glitch"]["n_glitch"]))
    ok &= (not r["layers"]["G_glitch"]["fired"])

    # 反転スパイク（1点だけ +15℃ 飛んで戻る）を複数 → glitch 発火。
    spk = 30 + np.random.RandomState(32).normal(0, 0.2, 200)
    for s0 in (40, 80, 120, 160):
        spk[s0] += 15.0
    r = judge_sensor(spk, cfg)
    print("[spikes]    sev=%d %-22s n_glitch=%d" % (r["severity"], r["reason"], r["layers"]["G_glitch"]["n_glitch"]))
    ok &= (r["severity"] == 2 and r["reason"] == "glitch")

    r = judge_sensor(_synth("stuck", seed=6), cfg, hm)
    print("[stuck]     sev=%d %-20s range=%s" % (r["severity"], r["reason"], r["layers"]["H1_stuck"]["range"]))
    ok &= (r["severity"] == 2 and r["reason"] == "stuck")

    # ペア乖離: 上が +6℃ 跳ねる
    top = _synth("healthy", seed=7) + 6.0
    bot = _synth("healthy", seed=8)
    r = judge_sensor(top, cfg, learn_sensor(top, cfg), pair_T=bot, pair_dT0=0.0)
    print("[pair_dev]  sev=%d %-20s dev=%.1f" % (r["severity"], r["reason"], r["layers"]["P_pair"]["dev"]))
    ok &= (r["severity"] == 3 and r["reason"] == "pair_deviation_high")

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
