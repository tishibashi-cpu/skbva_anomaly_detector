#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_equipment.py — 機器側の熱結合特性の劣化検知（センサ故障とは別の判定軸）。

これまでの temp_judge.py の9層（H0/H1/N/G/S/O/B/I/P）は「温度計センサ自体の故障」を見る。
本モジュールは逆に、**センサは正常な前提で、測定対象の機器側の異常**（放熱不良・断熱劣化・
接触不良による発熱増加等）を検知する。着眼点は単純：

    健全な機器なら、同じビーム電流に対する温度上昇はほぼ一定のはず。
    「電流1A当たりの温度上昇（dT/dI）」が、過去の基準期間と比べて有意に増えていたら、
    機器側の熱的な劣化を疑う。

実例（2022年4月 vs 2026年3月、FB_MOVE:D01:QC1L:BWS:TEMP, BMLDCCT:CURRENT）：
    2022: dT/dI ≈ 4.25 ℃/A (r=0.84)
    2026: dT/dI ≈ 7.26 ℃/A (r=0.70)  ← 同じ電流でも約1.7倍発熱するようになっている

温度計側の他モジュール（temp_judge等）とは独立に使える設計（判定対象は「機器」であって
「センサ」ではないため、severity の意味も異なる）。ただし PV パーサ・kblogrd取得は
temp_pv.py / temp_fetch.py を共用する。

比較対象の2期間は、判定のたびに直近だけを見る他レイヤー（ローリング基準）とは違い、
**運用者が明示的に選ぶ**（「昔の健全期」と「直近」）。何年も前の実データは kblogrd の
アーカイブ粒度指定 (-t ...d<interval>) で引けないことがあるため、raw な変化ログ形式の
CSV（Timestamp, <温度PV>, <ビームPV> の3列。値が変化した行だけ記録＝疎で時刻不規則）も
直接読み込めるようにしてある（アップロードされた実データがこの形式だったため）。
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np

import temp_fetch
import temp_pv

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_FILE = os.path.join(HERE, "temp_equipment_models.json")
# detector_headless.py の定期実行（run_periodic_judge）が書くダッシュボード用状態ファイル。
# temp_headless.py の temp_dashboard_state.json と対になる（dashboard.py の「機器劣化検知」タブが読む）。
EQUIPMENT_STATE_FILE = os.path.join(HERE, "temp_equipment_state.json")

CONFIG = {
    "beam_on_ma":      50.0,    # これ以上[mA]をビームありとみなす（フィル中のみをフィットに使う。
                                # CCG の Storage 解析＝フィル開始〜アボートの区間と同じ考え方）
    # --- フィット信頼度（IPの trust 判定と同じ思想：相関＋レンジの十分性）---
    "r_min":           0.5,     # |相関係数| がこれ未満ならフィット不信頼
    "min_i_range_ma":  200.0,   # ビーム電流の変動幅[mA]がこれ未満ならフィット不信頼
    "min_pts":         20,      # フィットに使う点数の下限
    "theilsen_max_pts": 1500,   # Theil-Sen のペア計算量対策。これを超えたら間引く

    # --- 機器劣化の判定しきい値（比較は dT/dI の比＋絶対差の両方を要求）---
    "ratio_sev3":      1.5,     # 傾きが基準の1.5倍以上 → sev3
    "ratio_sev2":      1.2,     # 1.2倍以上 → sev2
    "min_abs_delta":   0.001,   # ℃/mA。比が大きくても絶対差がこれ未満なら見送る（ノイズ対策）

    # --- HOM項モデル T(I,Nb) = w0 + w1*I + w2*(I^2/Nb)^2 （Suetsugu et al., PRAB 27, 063201
    #     (2024) 式(5) と同形。CCG圧力の式をそのまま温度に適用したもの。物理的な妥当性の注記は
    #     fit_t_vs_i_hom のdocstring参照）---
    "hom_robust_iters": 3,      # 頑健フィットの sigma-clip 反復回数
    "hom_sigma_clip":   3.0,    # 残差がこの標準偏差数を超える点を外れ値として除外
    "hom_r2_min":       0.5,    # 決定係数 R^2 がこれ未満ならフィット不信頼（線形版のr_minに相当）

    # --- ビーム電流が急変した直後の熱の過渡（サーマルラグ）除外 ---
    #   ビーム電流が大きく変わった（アボートで急落 / フィル開始・電流アップで急上昇）直後は、
    #   電流はすぐ新しい値になっても機器の温度は熱容量のせいですぐには追従しない。この過渡が
    #   残っている間の点は「今の電流に対する定常的な温度」ではないため、フィットを歪める
    #   （急落側は低電流域を、急上昇側は高電流域を、それぞれ実際より高く/低く歪める）。
    #   CCGの論文で Storage（定常）と Tail（アボート直後の過渡）を分けているのと同じ考え方を、
    #   方向を問わず一般化する：直近 settle_after_change_min 分間の電流の変動幅（最大−最小）が
    #   settle_change_ma 以上あれば、その点は定常状態ではないとみなして除外する。
    "settle_after_change_min": 20.0,   # 変動を調べる直近の時間幅[分]（要調整。機器の熱時定数次第）
    "settle_change_ma":        200.0,  # この変動幅[mA]以上なら「まだ過渡中」とみなす（要調整）
}


# ───────────────────────── フィット ─────────────────────────

def filter_beam_on(T, I, cfg=CONFIG):
    """ビームが入っている点だけ残す（無ビーム期間は温度-電流の関係が意味を持たないため）。
    CCG の Storage 解析（フィル開始〜アボートの区間だけを見る）と同じ考え方を、フィル境界の
    検出はせず「電流がしきい値以上」という単純な条件で代替している。"""
    T = np.asarray(T, float)
    I = np.asarray(I, float)
    m = np.isfinite(T) & np.isfinite(I) & (I >= cfg["beam_on_ma"])
    return T[m], I[m]


def filter_beam_on3(T, I, Nb, cfg=CONFIG):
    """filter_beam_on の3引数版（T, I, Nb を同じマスクで揃えて絞り込む。HOMモデル用）。"""
    T = np.asarray(T, float); I = np.asarray(I, float); Nb = np.asarray(Nb, float)
    m = np.isfinite(T) & np.isfinite(I) & (I >= cfg["beam_on_ma"])
    return T[m], I[m], Nb[m]


def _settle_exclude_mask(t_sec, I, cfg=CONFIG):
    """各点について、直近 settle_after_change_min 分間の電流の変動幅（最大−最小）が
    settle_change_ma 以上あれば True（＝除外すべき＝まだ熱的に定常でない）を返す。
    向きは問わない：アボートによる急落（Tail期）も、フィル開始・電流アップによる急上昇も、
    同じ基準で「直近に大きく変動した」として扱う（CONFIG参照）。

    t_sec: 各点の経過秒（昇順。等間隔グリッドでも不等間隔でもよい＝実機の regular grid
    取得にも、変化ログCSVの不規則な実時刻にも両対応）。NaN の I はウィンドウ内の
    最大/最小の計算から除く（欠測は判断材料にしない）。
    settle_after_change_min<=0 なら何も除外しない。
    """
    n = len(I)
    settle_min = cfg.get("settle_after_change_min", 0)
    if settle_min <= 0 or n == 0:
        return np.zeros(n, dtype=bool)
    t_sec = np.asarray(t_sec, float)
    I = np.asarray(I, float)
    settle_sec = settle_min * 60.0
    # 各点 i の窓 [t_sec[i]-settle_sec, t_sec[i]] の左端インデックス（t_secは昇順前提）
    lo_idx = np.searchsorted(t_sec, t_sec - settle_sec, side="left")
    exclude = np.zeros(n, dtype=bool)
    for i in range(n):
        window = I[lo_idx[i]:i + 1]
        finite = window[np.isfinite(window)]
        if len(finite) < 2:
            continue
        if (np.max(finite) - np.min(finite)) >= cfg["settle_change_ma"]:
            exclude[i] = True
    return exclude


def _theilsen_slope(I, T, max_pts=CONFIG["theilsen_max_pts"], rng=None):
    """Theil-Sen 推定量（全ペアの傾きの中央値）。外れ値に頑健で numpy のみで書ける。
    点数が多いときはランダム間引きしてから使う（O(n^2) 対策）。"""
    n = len(I)
    if n > max_pts:
        rng = rng or np.random.RandomState(0)
        idx = rng.choice(n, max_pts, replace=False)
        I, T = I[idx], T[idx]
        n = max_pts
    dI = I[:, None] - I[None, :]
    dT = T[:, None] - T[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1) & (np.abs(dI) > 1e-9)
    if not np.any(mask):
        return None
    slopes = dT[mask] / dI[mask]
    return float(np.median(slopes))


def fit_t_vs_i(T, I, cfg=CONFIG):
    """T = a + b*I を頑健フィット（Theil-Sen 傾き＋中央値切片）。

    返り値: {b(dT/dI, ℃/mA), a(切片), r(相関係数), n(点数), i_range(電流変動幅mA), trust(bool)}
    trust は |r|≥r_min かつ i_range≥min_i_range_ma かつ n≥min_pts のとき True。
    """
    T = np.asarray(T, float)
    I = np.asarray(I, float)
    m = np.isfinite(T) & np.isfinite(I)
    T, I = T[m], I[m]
    n = len(T)
    if n < 2:
        return {"b": None, "a": None, "r": None, "n": n, "i_range": None, "trust": False}

    i_range = float(np.max(I) - np.min(I))
    r = float(np.corrcoef(I, T)[0, 1]) if (n >= 2 and np.std(I) > 1e-9 and np.std(T) > 1e-9) else None
    b = _theilsen_slope(I, T)
    a = float(np.median(T - b * I)) if b is not None else None
    trust = bool(n >= cfg["min_pts"] and i_range >= cfg["min_i_range_ma"]
                and r is not None and abs(r) >= cfg["r_min"])
    return {"b": b, "a": a, "r": r, "n": n, "i_range": i_range, "trust": trust}


def fit_t_vs_i_hom(T, I, Nb, cfg=CONFIG):
    """T(I, Nb) = w0 + w1*I + w2*(I^2/Nb)^2 を頑健フィットする
    （Suetsugu et al., Phys. Rev. Accel. Beams 27, 063201 (2024) 式(5)と同形。
    元は CCG 圧力 P(I)=Pb+ΔPs+ΔPt の式で、w1*I はSR起因(光子数∝I)、
    w2*(I^2/Nb)^2 はHOM起因（発熱 ∝ I^2/Nb、熱脱離ΔPt∝(ΔT)^2 なので二乗）を表す。

    ※物理的な注記：この「二乗」は圧力側の熱脱離物理（Arrhenius近似でΔPt∝(ΔT)^2）に
    由来し、温度そのものに対して二乗が妥当とは限らない（定常状態ならHOM発熱量に対して
    温度上昇は線形という考え方もある＝T∝w0+w1*I+w2*I^2/Nb、二乗なし）。ここではご要望通り
    式(5)と同形（二乗あり）で実装するが、fit_t_vs_i_hom_linear（二乗無し版）も用意したので
    両方の当てはまり（R^2）を比較して判断できるようにしてある。

    X = [1, I, (I^2/Nb)^2] の3特徴量に対する線形回帰なので通常の最小二乗（正規方程式）で解け、
    外れ値（フィル遷移の過渡等）に対しては残差シグマクリップを繰り返す頑健化を行う。

    返り値: {w(=[w0,w1,w2]), r2(決定係数), n, i_range, trust}
    """
    T = np.asarray(T, float); I = np.asarray(I, float); Nb = np.asarray(Nb, float)
    m = np.isfinite(T) & np.isfinite(I) & np.isfinite(Nb) & (Nb > 0)
    T, I, Nb = T[m], I[m], Nb[m]
    n0 = len(T)
    if n0 < cfg["min_pts"]:
        return {"w": None, "r2": None, "n": n0, "i_range": None, "trust": False}

    hom = (I ** 2 / Nb) ** 2
    # 数値条件対策: [1, I, hom] は桁が大きく違う（I~1e3, hom~1e5-1e12）ので、素の最小二乗だと
    # 条件数が悪化し係数が暴れる。列ごとにスケーリングしてフィットし、後で単位を戻す。
    scale_I = np.std(I) or 1.0
    scale_hom = np.std(hom) or 1.0
    X = np.column_stack([np.ones(n0), I / scale_I, hom / scale_hom])
    keep = np.ones(n0, dtype=bool)
    w = None
    for _ in range(cfg["hom_robust_iters"] + 1):
        Xk, Tk = X[keep], T[keep]
        if len(Tk) < cfg["min_pts"]:
            break
        w, *_ = np.linalg.lstsq(Xk, Tk, rcond=None)
        resid = T - X @ w
        s = np.std(resid[keep])
        if s < 1e-9:
            break
        new_keep = np.abs(resid) <= cfg["hom_sigma_clip"] * s
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    n = int(np.sum(keep))
    if w is None or n < cfg["min_pts"]:
        return {"w": None, "r2": None, "n": n0, "i_range": None, "trust": False}
    w = np.array([w[0], w[1] / scale_I, w[2] / scale_hom])   # スケールを戻す

    Tk, pred = T[keep], (X @ [w[0], w[1] * scale_I, w[2] * scale_hom])[keep]
    ss_res = float(np.sum((Tk - pred) ** 2))
    ss_tot = float(np.sum((Tk - np.mean(Tk)) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else None
    i_range = float(np.max(I[keep]) - np.min(I[keep]))
    trust = bool(n >= cfg["min_pts"] and i_range >= cfg["min_i_range_ma"]
                and r2 is not None and r2 >= cfg["hom_r2_min"])
    return {"w": [float(x) for x in w], "r2": r2, "n": n, "i_range": i_range, "trust": trust}


def fit_t_vs_i_hom_linear(T, I, Nb, cfg=CONFIG):
    """T(I,Nb) = w0 + w1*I + w2*(I^2/Nb) を頑健フィットする（HOM項に二乗を付けない版）。
    fit_t_vs_i_hom のdocstring参照。二乗あり/無しどちらが実データに合うかの比較用。"""
    T = np.asarray(T, float); I = np.asarray(I, float); Nb = np.asarray(Nb, float)
    m = np.isfinite(T) & np.isfinite(I) & np.isfinite(Nb) & (Nb > 0)
    T, I, Nb = T[m], I[m], Nb[m]
    n0 = len(T)
    if n0 < cfg["min_pts"]:
        return {"w": None, "r2": None, "n": n0, "i_range": None, "trust": False}
    hom = (I ** 2 / Nb)
    scale_I = np.std(I) or 1.0
    scale_hom = np.std(hom) or 1.0
    X = np.column_stack([np.ones(n0), I / scale_I, hom / scale_hom])
    keep = np.ones(n0, dtype=bool)
    w = None
    for _ in range(cfg["hom_robust_iters"] + 1):
        Xk, Tk = X[keep], T[keep]
        if len(Tk) < cfg["min_pts"]:
            break
        w, *_ = np.linalg.lstsq(Xk, Tk, rcond=None)
        resid = T - X @ w
        s = np.std(resid[keep])
        if s < 1e-9:
            break
        new_keep = np.abs(resid) <= cfg["hom_sigma_clip"] * s
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    n = int(np.sum(keep))
    if w is None or n < cfg["min_pts"]:
        return {"w": None, "r2": None, "n": n0, "i_range": None, "trust": False}
    w = np.array([w[0], w[1] / scale_I, w[2] / scale_hom])
    Tk, pred = T[keep], (X @ [w[0], w[1] * scale_I, w[2] * scale_hom])[keep]
    ss_res = float(np.sum((Tk - pred) ** 2))
    ss_tot = float(np.sum((Tk - np.mean(Tk)) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else None
    i_range = float(np.max(I[keep]) - np.min(I[keep]))
    trust = bool(n >= cfg["min_pts"] and i_range >= cfg["min_i_range_ma"]
                and r2 is not None and r2 >= cfg["hom_r2_min"])
    return {"w": [float(x) for x in w], "r2": r2, "n": n, "i_range": i_range, "trust": trust}


def compare_periods(T_ref, I_ref, T_now, I_now, cfg=CONFIG):
    """基準期間と現在期間の dT/dI を比較し、機器劣化を判定する。

    返り値: {severity(0-3), reason, ref{fit}, now{fit}, ratio, delta_dex_like}
    両方の期間でフィットが信頼できないと判定不能（severity=None, reason='insufficient_data'）。
    """
    ref = fit_t_vs_i(T_ref, I_ref, cfg)
    now = fit_t_vs_i(T_now, I_now, cfg)
    out = {"ref": ref, "now": now, "ratio": None, "delta": None}

    if not (ref["trust"] and now["trust"]):
        out["severity"] = None
        out["reason"] = "insufficient_data"
        return out

    b_ref, b_now = ref["b"], now["b"]
    delta = b_now - b_ref
    out["delta"] = delta

    if b_ref <= 0:
        # 基準の傾きがゼロ/負（ほぼ無相関〜逆相関）だと比が定義できない。絶対差だけで見る。
        ratio = None
    else:
        ratio = b_now / b_ref
    out["ratio"] = ratio

    # 判定: 比（あれば）と絶対差の両方を要求（比だけだと基準がほぼ0のとき暴れるため）
    if delta < cfg["min_abs_delta"]:
        out["severity"], out["reason"] = 0, "normal"
    elif ratio is not None and ratio >= cfg["ratio_sev3"]:
        out["severity"], out["reason"] = 3, "heating_gain_increase_severe"
    elif ratio is not None and ratio >= cfg["ratio_sev2"]:
        out["severity"], out["reason"] = 2, "heating_gain_increase"
    elif ratio is None or ratio > 1.0:
        out["severity"], out["reason"] = 1, "heating_gain_increase_watch"
    else:
        out["severity"], out["reason"] = 0, "normal"
    return out


def compare_periods_hom(T_ref, I_ref, Nb_ref, T_now, I_now, Nb_now, cfg=CONFIG, hom_squared=True):
    """基準期間と現在期間を HOMモデル T=w0+w1*I+w2*(I^2/Nb)^n （n=2 hom_squared時、n=1で線形版）
    でフィットし、代表的な運転電流での「予測発熱量（=T−w0）」を比較する。

    w1（SR項）とw2（HOM項）は互いにトレードオフしうる（同じ曲線形状を別の配分で説明できる）ため、
    どちらか一方の係数だけで比較すると不安定。そこで両方の期間で共通に観測された電流レンジ内の
    代表点（プールしたビームあり点の中央値I・Nb）で予測温度上昇を評価し、その比で判定する
    （＝「同じ運転条件なら今どれだけ発熱するか」を直接比較）。

    代表点は必ず両期間それぞれの実データ範囲の内側（共通部分）に収める。単純にプールした
    中央値だけを使うと、片方の期間（点数が多い・電流域が広い方）に引っ張られて、もう片方の
    期間の実データ最大を超えて外挿してしまうことがある（実データで確認された不具合：現在期間
    の方がデータ量・電流域とも大きいと、代表点が基準期間の観測範囲を超えてしまい、基準期間の
    予測発熱量が根拠の薄い外挿値になっていた）。min(両期間の最大電流) を上限にクリップすることで、
    どちらの期間についても「実際に観測された範囲内」での予測になるようにする。

    返り値: {severity, reason, ref{fit}, now{fit}, eval_i, eval_nb, dT_ref, dT_now, ratio, delta}
    """
    fitfn = fit_t_vs_i_hom if hom_squared else fit_t_vs_i_hom_linear
    Tr, Ir, Nr = filter_beam_on3(T_ref, I_ref, Nb_ref, cfg)
    Tn, In, Nn = filter_beam_on3(T_now, I_now, Nb_now, cfg)
    ref = fitfn(Tr, Ir, Nr, cfg)
    now = fitfn(Tn, In, Nn, cfg)
    out = {"ref": ref, "now": now, "ratio": None, "delta": None,
          "eval_i": None, "eval_nb": None, "dT_ref": None, "dT_now": None}

    if not (ref["trust"] and now["trust"]):
        out["severity"], out["reason"] = None, "insufficient_data"
        return out

    # 代表点: 両期間プールしたビームあり点の中央値を求めたうえで、min(両期間の観測最大電流)を
    # 上限にクリップする（＝どちらの期間の実データ範囲も超えない代表点にする。外挿防止）。
    i_cap = min(np.max(Ir), np.max(In))
    eval_i = min(float(np.median(np.concatenate([Ir, In]))), i_cap)
    eval_nb = float(np.median(np.concatenate([Nr, Nn])))
    out["eval_i"], out["eval_nb"] = eval_i, eval_nb

    def _predict_rise(w):
        w0, w1, w2 = w
        hom = (eval_i ** 2 / eval_nb) ** (2 if hom_squared else 1)
        return w1 * eval_i + w2 * hom   # w0(切片)を除いた「発熱分」

    dT_ref = _predict_rise(ref["w"])
    dT_now = _predict_rise(now["w"])
    out["dT_ref"], out["dT_now"] = dT_ref, dT_now
    delta = dT_now - dT_ref
    out["delta"] = delta
    ratio = (dT_now / dT_ref) if dT_ref > 0 else None
    out["ratio"] = ratio

    if delta < cfg["min_abs_delta"] * eval_i:   # 絶対差のしきい値は代表電流に比例させる
        out["severity"], out["reason"] = 0, "normal"
    elif ratio is not None and ratio >= cfg["ratio_sev3"]:
        out["severity"], out["reason"] = 3, "heating_gain_increase_severe"
    elif ratio is not None and ratio >= cfg["ratio_sev2"]:
        out["severity"], out["reason"] = 2, "heating_gain_increase"
    elif ratio is None or ratio > 1.0:
        out["severity"], out["reason"] = 1, "heating_gain_increase_watch"
    else:
        out["severity"], out["reason"] = 0, "normal"
    return out


# ───────────────────────── データ読み込み ─────────────────────────

def load_raw_csv(path, pv=None, with_nb=False, cfg=CONFIG):
    """変化ログ形式CSV（Timestamp, <温度PV>, <ビームPV>[, <NbPV>...] の列。値が変化した行のみ
    記録＝疎で時刻不規則）を読み、asof（直近の前方値）で対応付けて返す。

    pv: 温度列を明示したい場合の列名（省略時は2列目=1個目のPV列を使う）。
    ビーム列は列名に 'DCCT' または 'CURRENT' を含む列を自動検出する。
    with_nb=True のとき、列名に 'NOB'・'BKSEL'・'BUNCH' のいずれかを含む列を Nb 列として
    自動検出し、
    (T, I, Nb, pv名) の4値を返す（無ければ Nb は全て NaN）。False なら従来通り (T, I, pv名)。
    アボート直後の熱の余韻の点は、Timestamp列をパースできれば自動で除外する
    （_settle_exclude_mask、CONFIG参照。パース不可なら除外せず警告のみ）。
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    temp_col = None
    beam_col = None
    nb_col = None
    for i, h in enumerate(header[1:], start=1):
        hu = h.upper()
        if beam_col is None and ("DCCT" in hu or "CURRENT" in hu):
            beam_col = i
        elif with_nb and nb_col is None and ("NOB" in hu or "BKSEL" in hu or "BUNCH" in hu):
            nb_col = i
        elif temp_col is None and (pv is None or h.strip() == pv):
            temp_col = i
    if temp_col is None or beam_col is None:
        raise ValueError("温度列/ビーム列が見つかりません（header=%r）" % header)

    def _parse_ts(s):
        for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                   "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    last_beam = None
    last_nb = None
    T, I, NB, TS = [], [], [], []
    for row in rows:
        maxcol = max(temp_col, beam_col, nb_col or 0)
        if len(row) <= maxcol:
            continue
        bcell = row[beam_col].strip()
        if bcell not in ("", "N/A"):
            try:
                last_beam = float(bcell)
            except ValueError:
                pass
        if nb_col is not None:
            ncell = row[nb_col].strip()
            if ncell not in ("", "N/A"):
                try:
                    last_nb = float(ncell)
                except ValueError:
                    pass
        tcell = row[temp_col].strip()
        if tcell not in ("", "N/A") and last_beam is not None:
            try:
                T.append(float(tcell))
                I.append(last_beam)
                NB.append(last_nb if last_nb is not None else np.nan)
                TS.append(row[0].strip())
            except ValueError:
                pass

    T = np.array(T, float); I = np.array(I, float); NB = np.array(NB, float)
    parsed = [_parse_ts(s) for s in TS]
    if all(p is not None for p in parsed) and len(parsed) > 0:
        t0 = parsed[0]
        t_sec = np.array([(p - t0).total_seconds() for p in parsed], dtype=float)
        settle_excl = _settle_exclude_mask(t_sec, I, cfg)
        T, I, NB = T[~settle_excl], I[~settle_excl], NB[~settle_excl]
    else:
        sys.stderr.write("[load_raw_csv] Timestamp列を解釈できなかったため、アボート直後の"
                         "熱の余韻の除外はスキップしました（%s）。\n" % path)

    if with_nb:
        return T, I, NB, header[temp_col]
    return T, I, header[temp_col]


def _beam_ring_for_pv(ring, pv):
    """ビーム取得に使う実効リング（LER/HER）を求める。
    LER/HER形式のPVはそのまま ring 引数（=ファイルバケット）でよいが、IR形式のPVは
    BM/BMOthers 側にビームPVが無いため、PV自身の実効リング（tag のH/L推定＋
    TEMP_RING_OVERRIDE.csv 適用後）を使う必要がある（temp_fetch.BEAM_PV は LER/HER のみ）。
    """
    if ring.upper() in ("LER", "HER"):
        return ring.upper()
    d = temp_pv.parse_pv(pv, ring_override=temp_fetch.load_ring_overrides())
    beam_ring = d.get("ring") if d else None
    if beam_ring not in ("LER", "HER"):
        return None   # H/L推定できない・上書きも無い＝ビーム相関は諦める
    return beam_ring


def load_live(ring, pv, start, end, interval_sec=temp_fetch.DEFAULT_INTERVAL, with_nb=False, cfg=CONFIG,
             beam_series=None, nb_series=None, temp_series=None):
    """kblogrd から実機取得（温度・ビーム同一グリッド。既存 temp_fetch/temp_probe と同じ経路）。
    温度は ring（ファイルバケット。LER/HER/IR）で取得するが、ビーム・Nbは PV 自身の実効リング
    （LER/HER）で取得する（IR には専用のビーム/NbPVが無いため。_beam_ring_for_pv 参照）。
    with_nb=True で (T, I, Nb) の3値（HOMモデル用）、False なら従来通り (T, I) の2値を返す。
    アボート直後の熱の余韻の点は自動で除外する（_settle_exclude_mask、CONFIG参照）。

    beam_series/nb_series: 既に取得済みのビーム/Nb列 [(ts,val),...] を渡すと、その分の
    kblogrd 呼び出しを省略する。ビーム・Nbは PV に依らずリング共通なので、多数の PV を
    ループする呼び出し側（learn/judge/scan）は一度だけ取得してここに渡すことで、
    本数分の重複取得を避けられる（temp_batch.py の judge_all と同じ最適化）。

    temp_series: 既に取得済みの温度列 [(ts,val),...] を渡すと、温度の kblogrd 呼び出しも
    省略する。temp_fetch.fetch_history は複数PVをまとめて1回のkblogrd呼び出しで取得できる
    （CHUNK=13本ずつ）ので、呼び出し側が複数PV分をまとめて取得しておき、ここには該当PV分
    だけ切り出して渡す、という使い方を想定（1本ずつ呼ぶより kblogrd 呼び出し回数を減らせる）。
    """
    if temp_series is not None:
        data = {pv: {"series": temp_series}}
    else:
        data = temp_fetch.fetch_history(ring, start, end, interval_sec=interval_sec, pvs=[pv])
    beam_ring = _beam_ring_for_pv(ring, pv)
    if beam_ring is None:
        sys.stderr.write("[%s] ビームリングを特定できません（IR形式でH/L推定不可・上書きも無し）。"
                         "ビーム無しとして扱います。\n" % pv)
        beam, nb = [], []
    else:
        beam = beam_series if beam_series is not None else \
            temp_fetch.fetch_beam(beam_ring, start, end, interval_sec=interval_sec)
        if with_nb:
            nb = nb_series if nb_series is not None else \
                temp_fetch.fetch_nb(beam_ring, start, end, interval_sec=interval_sec)
        else:
            nb = []
    v = data.get(pv, {})
    ts, T = temp_fetch.series_to_arrays(v.get("series", []))
    bmap = {t: (np.nan if val is None else float(val)) for t, val in beam}
    I = np.array([bmap.get(t, np.nan) for t in ts], dtype=float)
    # 等間隔グリッド（kblogrd -t ...d<interval>）なので、インデックス×interval_secが経過秒になる
    t_sec = np.arange(len(I), dtype=float) * interval_sec
    settle_excl = _settle_exclude_mask(t_sec, I, cfg)
    if not with_nb:
        m = np.isfinite(T) & np.isfinite(I) & ~settle_excl
        return T[m], I[m]
    nbmap = {t: (np.nan if val is None else float(val)) for t, val in nb}
    Nb = np.array([nbmap.get(t, np.nan) for t in ts], dtype=float)
    m = np.isfinite(T) & np.isfinite(I) & np.isfinite(Nb) & (Nb > 0) & ~settle_excl
    return T[m], I[m], Nb[m]


# ───────────────────────── CLI ─────────────────────────

def cmd_compare(args):
    if args.model == "hom":
        if args.ref_csv:
            T_ref, I_ref, Nb_ref, pv_name = load_raw_csv(args.ref_csv, pv=args.pv, with_nb=True)
            print("[基準期間] CSV %s から %d 点（PV=%s）" % (args.ref_csv, len(T_ref), pv_name))
        else:
            T_ref, I_ref, Nb_ref = load_live(args.ring, args.pv, args.ref_start, args.ref_end,
                                             args.interval, with_nb=True)
            print("[基準期間] kblogrd %s〜%s から %d 点" % (args.ref_start, args.ref_end, len(T_ref)))
        if args.now_csv:
            T_now, I_now, Nb_now, _ = load_raw_csv(args.now_csv, pv=args.pv, with_nb=True)
            print("[現在期間] CSV %s から %d 点" % (args.now_csv, len(T_now)))
        else:
            T_now, I_now, Nb_now = load_live(args.ring, args.pv, args.now_start, args.now_end,
                                             args.interval, with_nb=True)
            print("[現在期間] kblogrd %s〜%s から %d 点" % (args.now_start, args.now_end, len(T_now)))
        r = compare_periods_hom(T_ref, I_ref, Nb_ref, T_now, I_now, Nb_now)
        _print_result_hom(args.pv, r)
        return

    if args.ref_csv:
        T_ref, I_ref, pv_name = load_raw_csv(args.ref_csv, pv=args.pv)
        print("[基準期間] CSV %s から %d 点（PV=%s）" % (args.ref_csv, len(T_ref), pv_name))
    else:
        T_ref, I_ref = load_live(args.ring, args.pv, args.ref_start, args.ref_end, args.interval)
        print("[基準期間] kblogrd %s〜%s から %d 点" % (args.ref_start, args.ref_end, len(T_ref)))
    T_ref, I_ref = filter_beam_on(T_ref, I_ref)
    print("  → ビームあり(%.0fmA以上)のみ絞り込み: %d 点" % (CONFIG["beam_on_ma"], len(T_ref)))

    if args.now_csv:
        T_now, I_now, _ = load_raw_csv(args.now_csv, pv=args.pv)
        print("[現在期間] CSV %s から %d 点" % (args.now_csv, len(T_now)))
    else:
        T_now, I_now = load_live(args.ring, args.pv, args.now_start, args.now_end, args.interval)
        print("[現在期間] kblogrd %s〜%s から %d 点" % (args.now_start, args.now_end, len(T_now)))
    T_now, I_now = filter_beam_on(T_now, I_now)
    print("  → ビームあり(%.0fmA以上)のみ絞り込み: %d 点" % (CONFIG["beam_on_ma"], len(T_now)))

    r = compare_periods(T_ref, I_ref, T_now, I_now)
    _print_result(args.pv, r)


def _print_result(pv, r):
    ref, now = r["ref"], r["now"]
    print("\n=== %s ===" % pv)
    print("  基準: dT/dI=%s ℃/A  r=%s  n=%s  電流幅=%s mA  trust=%s"
          % (_f(ref["b"], 1000), _f(ref["r"], 1, 2), ref["n"], _f(ref["i_range"], 1, 0), ref["trust"]))
    print("  現在: dT/dI=%s ℃/A  r=%s  n=%s  電流幅=%s mA  trust=%s"
          % (_f(now["b"], 1000), _f(now["r"], 1, 2), now["n"], _f(now["i_range"], 1, 0), now["trust"]))
    if r["severity"] is None:
        print("  → 判定不能（%s）：フィット信頼度が不足（点数/電流幅/相関のいずれか不足）" % r["reason"])
    else:
        ratio_s = ("%.2fx" % r["ratio"]) if r["ratio"] is not None else "—"
        print("  → sev%d %s  比=%s  差=%s ℃/A"
              % (r["severity"], r["reason"], ratio_s, _f(r["delta"], 1000)))


def _print_result_hom(pv, r):
    ref, now = r["ref"], r["now"]
    print("\n=== %s （HOMモデル T=w0+w1*I+w2*(I^2/Nb)^2） ===" % pv)
    if ref["trust"]:
        print("  基準: w0=%.2f w1=%.5f w2=%.3e  R^2=%.3f  n=%d  電流幅=%.0fmA"
              % (ref["w"][0], ref["w"][1], ref["w"][2], ref["r2"], ref["n"], ref["i_range"]))
    else:
        print("  基準: 信頼不足（n=%s）" % ref["n"])
    if now["trust"]:
        print("  現在: w0=%.2f w1=%.5f w2=%.3e  R^2=%.3f  n=%d  電流幅=%.0fmA"
              % (now["w"][0], now["w"][1], now["w"][2], now["r2"], now["n"], now["i_range"]))
    else:
        print("  現在: 信頼不足（n=%s）" % now["n"])
    if r["severity"] is None:
        print("  → 判定不能（%s）：フィット信頼度が不足" % r["reason"])
    else:
        ratio_s = ("%.2fx" % r["ratio"]) if r["ratio"] is not None else "—"
        print("  → sev%d %s  比=%s  代表点(I=%.0fmA,Nb=%.0f)での予測発熱: 基準%.2f℃ → 現在%.2f℃"
              % (r["severity"], r["reason"], ratio_s, r["eval_i"], r["eval_nb"], r["dT_ref"], r["dT_now"]))


def _f(x, scale=1, ndig=3):
    if x is None:
        return "—"
    return ("%." + str(ndig) + "f") % (x * scale)


# ───────────────────────── learn / judge（永続モデル方式）─────────────────────────
#   IP judge と同じ思想: 「過去の健全期間」を一度 learn してモデルを保存し、以後の judge は
#   そのモデルと直近を比較する。equipment の熱結合特性は年〜数年スケールでしか動かないはずの
#   量なので、CCG/温度計センサ判定のような「毎回直近数日を学習し直す」ローリング基準は不要
#   （むしろ短い基準窓では変動を捉えられない）。ip_models.json と同じく低信頼時は前回の
#   良いモデルを保持するマージを行う（temp_detector の短絡層で得た「基準汚染」の教訓）。

def _load_models(path):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, OSError):
            sys.stderr.write("警告: %s を読めません。空のモデルとして続行。\n" % path)
    return {}


def _save_models(models, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _make_beam_nb_cache(ring, recs, start, end, interval_sec, with_nb):
    """PVごとに毎回ビーム/Nbを取得し直す無駄を無くすためのキャッシュを作る。
    ビーム/Nbは PV に依らずリング（beam_ring）共通の値なので、対象PV群に現れる
    beam_ring（LER/HERのどちらか、IRなら両方混在しうる）ごとに一度だけ取得し使い回す
    （temp_batch.py の judge_all で使っている最適化と同じ）。
    返り値: {beam_ring: {"beam": [...], "nb": [...] or None}}
    """
    beam_rings = sorted({_beam_ring_for_pv(ring, r["pv"]) for r in recs} - {None})
    cache = {}
    for br in beam_rings:
        try:
            beam = temp_fetch.fetch_beam(br, start, end, interval_sec=interval_sec)
        except Exception as ex:
            sys.stderr.write("[beam_cache] %s のビーム取得失敗: %s\n" % (br, ex))
            beam = []
        nb = None
        if with_nb:
            try:
                nb = temp_fetch.fetch_nb(br, start, end, interval_sec=interval_sec)
            except Exception as ex:
                sys.stderr.write("[beam_cache] %s のNb取得失敗: %s\n" % (br, ex))
                nb = []
        cache[br] = {"beam": beam, "nb": nb}
    return cache


def learn(ring, start, end, interval_sec=temp_fetch.DEFAULT_INTERVAL, out_path=MODELS_FILE,
         pvs=None, cfg=CONFIG, model="hom"):
    """過去の健全期間から機器ごとのモデルを学習し保存する（ビームありの点のみ使用）。
    model="linear": T=a+b*I（Theil-Sen）。model="hom": T=w0+w1*I+w2*(I^2/Nb)^2
    （Suetsugu et al. PRAB 27, 063201 (2024) 式(5)型。Nb取得が必要）。
    低信頼（電流レンジ不足・相関不足）のPVは、既存の保存済みモデルが良ければそれを保持する
    （このモデルは何年もそのまま使う想定なので、たまたま静穏な学習期間で潰さないため）。

    ビーム/Nbはリング共通なので PV 本数分は取得し直さない（_make_beam_nb_cache）。
    温度自体も、temp_fetch.fetch_history が元から持つ複数PV一括取得（CHUNK=13本/回）を使い、
    1本ずつ kblogrd を呼ばずチャンク単位でまとめて取得する（kblogrd 呼び出し1回あたりの
    接続・クエリオーバーヘッドがデータ量以上に効くため、本数が多いほど効果が大きい）。
    """
    recs = temp_fetch.load_pv_list(ring) if pvs is None else \
           [d for d in (temp_pv.parse_pv(p) for p in pvs) if d]
    with_nb = (model == "hom")
    cache = _make_beam_nb_cache(ring, recs, start, end, interval_sec, with_nb)
    store = _load_models(out_path)
    rd = store.setdefault(ring, {})
    n_trust = n_low = 0
    for chunk in temp_fetch._chunks(recs, temp_fetch.CHUNK):
        pv_names = [r["pv"] for r in chunk]
        try:
            temp_data = temp_fetch.fetch_history(ring, start, end, interval_sec=interval_sec, pvs=pv_names)
        except Exception as ex:
            sys.stderr.write("[learn] チャンク取得失敗（%d本スキップ）: %s\n" % (len(pv_names), ex))
            continue
        for rec in chunk:
            pv = rec["pv"]
            br = _beam_ring_for_pv(ring, pv)
            c = cache.get(br, {})
            tser = temp_data.get(pv, {}).get("series", [])
            try:
                if model == "hom":
                    T, I, Nb = load_live(ring, pv, start, end, interval_sec, with_nb=True,
                                         beam_series=c.get("beam"), nb_series=c.get("nb"),
                                         temp_series=tser)
                    T, I, Nb = filter_beam_on3(T, I, Nb, cfg)
                    fit = fit_t_vs_i_hom(T, I, Nb, cfg)
                    m = {"model": "hom", "w": fit["w"], "r2": fit["r2"], "n": fit["n"],
                        "i_range": fit["i_range"], "trust": fit["trust"],
                        "trained_start": start, "trained_end": end}
                else:
                    T, I = load_live(ring, pv, start, end, interval_sec, beam_series=c.get("beam"),
                                     temp_series=tser)
                    T, I = filter_beam_on(T, I, cfg)
                    fit = fit_t_vs_i(T, I, cfg)
                    m = {"model": "linear", "b": fit["b"], "a": fit["a"], "r": fit["r"], "n": fit["n"],
                        "i_range": fit["i_range"], "trust": fit["trust"],
                        "trained_start": start, "trained_end": end}
            except Exception as ex:
                sys.stderr.write("[learn %s] 取得失敗（スキップ）: %s\n" % (pv, ex))
                continue
            prev = rd.get(pv)
            if not m["trust"] and prev and prev.get("trust") and prev.get("model", "linear") == model:
                m = prev   # 低信頼なら前回の良いモデルを保持（IPと同じ耐性策。モデル種別が同じ場合のみ）
            rd[pv] = m
            if m["trust"]:
                n_trust += 1
            else:
                n_low += 1
    _save_models(store, out_path)
    print("[learn %s model=%s] trust=%d(実効・保持分含む)  low_trust=%d  → %s"
          % (ring, model, n_trust, n_low, out_path))
    return out_path, store


def _decimate_xy(x, y, max_pts=250):
    """散布図用に (x, y) を等間隔間引きして {"i": [...], "t": [...]} にする（時系列順のstride間引き。
    運転電流はフィル内で掃引されるため、strideでも電流レンジ全体を概ねカバーできる）。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n == 0:
        return {"i": [], "t": []}
    step = max(1, (n + max_pts - 1) // max_pts)
    return {"i": [float(v) for v in x[::step]], "t": [float(v) for v in y[::step]]}


def _fit_curve_hom(w, i_max, eval_nb, n_pts=60):
    """HOMモデル T=w0+w1*I+w2*(I^2/Nb)^2 のフィット曲線を [0, i_max] で計算する
    （Nbは代表値 eval_nb に固定。プロット用）。"""
    if not w or i_max is None or not eval_nb:
        return None
    ii = np.linspace(0.0, float(i_max), n_pts)
    tt = w[0] + w[1] * ii + w[2] * (ii ** 2 / eval_nb) ** 2
    return {"i": [float(v) for v in ii], "t": [float(v) for v in tt]}


def _fit_curve_linear(a, b, i_min, i_max, n_pts=2):
    """線形モデル T=a+b*I のフィット直線を [i_min, i_max] で計算する（プロット用）。"""
    if a is None or b is None or i_min is None or i_max is None:
        return None
    ii = np.linspace(float(i_min), float(i_max), n_pts)
    return {"i": [float(v) for v in ii], "t": [float(a + b * v) for v in ii]}


def judge(ring, start, end, interval_sec=temp_fetch.DEFAULT_INTERVAL, models_path=MODELS_FILE,
         pvs=None, cfg=CONFIG, attach_plot=False):
    """直近期間を取得し、保存済みモデル（learn で作成）と比較して機器劣化を判定する
    （ビームありの点のみ使用）。モデルが無い/低信頼のPVは判定不能として返す。
    モデルの種別（"linear"/"hom"）は learn 時に保存された値をそのまま使う。
    返り値: [{pv, ring, section, tag, severity, reason, ratio, delta, model, ...}, ...]

    ビーム/Nbはリング共通なので、(beam_ring, 期間) の組み合わせごとに一度だけ取得して使い回す。
    温度自体も、対象PV（学習済み・信頼できるモデルがあるもの）をチャンク単位（CHUNK本/回）で
    まとめて取得する。判定窓は全PV共通なので一括、基準窓（hom型のみ再取得）はPVごとに違いうる
    ため (基準開始,基準終了) でグルーピングしてからチャンク単位で取得する。

    attach_plot=True: 異常(sev>=1)のPVに、ダッシュボードのクリック展開プロット用データ
    （plot: {ref/now散布(温度 vs ビーム電流、ビームあり点・間引き済み), ref_fit/now_fit曲線}）を
    添付する。hom型は判定のために基準期間を再取得するので基準散布も入るが、linear型は
    保存済みフィット係数（a_ref, b_ref）しか持たないため基準はフィット直線のみになる。
    """
    store = _load_models(models_path)
    rd = store.get(ring, {})
    recs = temp_fetch.load_pv_list(ring) if pvs is None else \
           [d for d in (temp_pv.parse_pv(p) for p in pvs) if d]

    # 対象PV（学習済み・信頼できるモデルがあるものだけ）を先に絞る
    targets = []
    for rec in recs:
        saved = rd.get(rec["pv"])
        if saved and saved.get("trust"):
            targets.append((rec, saved, saved.get("model", "linear")))

    now_beam_cache = {}   # beam_ring -> {"beam","nb"}（判定窓は全PV共通）
    ref_beam_cache = {}   # (beam_ring, ref_start, ref_end) -> {"beam","nb"}（基準窓はPVごとに違いうる）

    def _fetch_cached_beam(cache, key, br, s, e, with_nb):
        if key not in cache:
            try:
                beam = temp_fetch.fetch_beam(br, s, e, interval_sec=interval_sec) if br else []
            except Exception as ex:
                sys.stderr.write("[judge cache] %s ビーム取得失敗: %s\n" % (br, ex)); beam = []
            nb = None
            if with_nb:
                try:
                    nb = temp_fetch.fetch_nb(br, s, e, interval_sec=interval_sec) if br else []
                except Exception as ex:
                    sys.stderr.write("[judge cache] %s Nb取得失敗: %s\n" % (br, ex)); nb = []
            cache[key] = {"beam": beam, "nb": nb}
        return cache[key]

    # 判定窓の温度をまとめて取得（linear/hom問わず対象PV全体で共通）。
    # fetch_history 自身が内部で CHUNK 単位に分けて kblogrd を呼ぶので、ここで事前に
    # 手動チャンク分割してから呼ぶ必要はない（以前はそうしており、fetch_history() の
    # 呼び出し回数が余計に増えて実機で無駄なオーバーヘッド・ログ出力の原因になっていた）。
    all_pv_names = [rec["pv"] for rec, saved, mt in targets]
    try:
        now_temp = temp_fetch.fetch_history(ring, start, end, interval_sec=interval_sec, pvs=all_pv_names)
    except Exception as ex:
        sys.stderr.write("[judge] 判定窓の取得失敗（%d本スキップ）: %s\n" % (len(all_pv_names), ex))
        now_temp = {}

    # 基準窓の温度（hom型のみ再取得が要る）を、基準期間ごとにグルーピングして取得
    # （基準期間はPVごとに違いうるため、同じ期間のPVはまとめて1回で取得する）。
    ref_temp = {}
    by_ref_window = {}
    for rec, saved, mt in targets:
        if mt == "hom":
            by_ref_window.setdefault((saved["trained_start"], saved["trained_end"]), []).append(rec["pv"])
    for (rs, re_), pv_names in by_ref_window.items():
        try:
            d = temp_fetch.fetch_history(ring, rs, re_, interval_sec=interval_sec, pvs=pv_names)
            for pv, v in d.items():
                ref_temp[(rs, re_, pv)] = v
        except Exception as ex:
            sys.stderr.write("[judge] 基準窓の取得失敗（%d本スキップ）: %s\n" % (len(pv_names), ex))

    results = []
    for rec, saved, model_type in targets:
        pv = rec["pv"]
        with_nb = (model_type == "hom")
        br = _beam_ring_for_pv(ring, pv)
        nc = _fetch_cached_beam(now_beam_cache, br, br, start, end, with_nb)
        tser_now = now_temp.get(pv, {}).get("series", [])
        try:
            if model_type == "hom":
                rc = _fetch_cached_beam(ref_beam_cache, (br, saved["trained_start"], saved["trained_end"]),
                                        br, saved["trained_start"], saved["trained_end"], True)
                tser_ref = ref_temp.get((saved["trained_start"], saved["trained_end"], pv), {}).get("series", [])
                T_ref, I_ref, Nb_ref = load_live(ring, pv, saved["trained_start"], saved["trained_end"],
                                                 interval_sec, with_nb=True,
                                                 beam_series=rc.get("beam"), nb_series=rc.get("nb"),
                                                 temp_series=tser_ref)
                T, I, Nb = load_live(ring, pv, start, end, interval_sec, with_nb=True,
                                     beam_series=nc.get("beam"), nb_series=nc.get("nb"),
                                     temp_series=tser_now)
            else:
                T, I = load_live(ring, pv, start, end, interval_sec, beam_series=nc.get("beam"),
                                 temp_series=tser_now)
        except Exception as ex:
            sys.stderr.write("[judge %s] 取得失敗（スキップ）: %s\n" % (pv, ex))
            continue

        r = {"pv": pv, "ring": ring, "section": rec.get("section"), "tag": rec.get("tag"),
            "model": model_type}
        if model_type == "hom":
            # HOM型は保存済みwを直接比較対象にできないため（Nb分布が期間で違うと予測点も
            # ずれるため）、基準期間も同じ関数で毎回再フィットし直す（compare_periods_hom と同じ方式）。
            cr = compare_periods_hom(T_ref, I_ref, Nb_ref, T, I, Nb, cfg)
            w0_ref = cr["ref"]["w"][0] if cr["ref"].get("w") else None
            w0_now = cr["now"]["w"][0] if cr["now"].get("w") else None
            r.update(severity=cr["severity"], reason=cr["reason"], ratio=cr["ratio"],
                    delta=cr["delta"], w_ref=cr["ref"].get("w"), r2_ref=cr["ref"].get("r2"),
                    now=cr["now"], eval_i=cr["eval_i"], eval_nb=cr["eval_nb"],
                    dT_ref=cr["dT_ref"], dT_now=cr["dT_now"], b_ref=None,
                    delta_a=(w0_now - w0_ref) if (w0_ref is not None and w0_now is not None) else None)
        else:
            T, I = filter_beam_on(T, I, cfg)
            now = fit_t_vs_i(T, I, cfg)
            b_ref = saved["b"]
            r.update(b_ref=b_ref, a_ref=saved.get("a"), now=now)
            if not now["trust"]:
                r["severity"], r["reason"] = None, "insufficient_data"
                r["ratio"] = r["delta"] = r["delta_a"] = None
            else:
                delta = now["b"] - b_ref
                ratio = (now["b"] / b_ref) if b_ref > 0 else None
                r["delta"], r["ratio"] = delta, ratio
                r["delta_a"] = (now["a"] - saved["a"]) if saved.get("a") is not None else None
                if delta < cfg["min_abs_delta"]:
                    r["severity"], r["reason"] = 0, "normal"
                elif ratio is not None and ratio >= cfg["ratio_sev3"]:
                    r["severity"], r["reason"] = 3, "heating_gain_increase_severe"
                elif ratio is not None and ratio >= cfg["ratio_sev2"]:
                    r["severity"], r["reason"] = 2, "heating_gain_increase"
                elif ratio is None or ratio > 1.0:
                    r["severity"], r["reason"] = 1, "heating_gain_increase_watch"
                else:
                    r["severity"], r["reason"] = 0, "normal"

        # ダッシュボードのクリック展開プロット用データ（異常PVのみ添付＝JSONサイズを抑える）。
        # hom型: 判定のために基準期間の生データも再取得済みなので基準/現在両方の散布を入れる。
        # linear型: 保存済みフィット係数しか持たないため、基準はフィット直線のみ（散布は現在期間だけ）。
        if attach_plot and r.get("severity") is not None and r["severity"] >= 1:
            try:
                if model_type == "hom":
                    Tr_on, Ir_on, _ = filter_beam_on3(T_ref, I_ref, Nb_ref, cfg)
                    Tn_on, In_on, _ = filter_beam_on3(T, I, Nb, cfg)
                    i_max = (min(float(np.max(Ir_on)), float(np.max(In_on)))
                             if len(Ir_on) and len(In_on) else None)
                    r["plot"] = {
                        "ref": _decimate_xy(Ir_on, Tr_on),
                        "now": _decimate_xy(In_on, Tn_on),
                        "ref_fit": _fit_curve_hom(r.get("w_ref"), i_max, r.get("eval_nb")),
                        "now_fit": _fit_curve_hom((r.get("now") or {}).get("w"), i_max, r.get("eval_nb")),
                    }
                else:
                    # linear分岐ではこの時点で T, I はビームありに絞り済み
                    i_min = float(np.min(I)) if len(I) else None
                    i_max = float(np.max(I)) if len(I) else None
                    r["plot"] = {
                        "ref": None,   # 基準期間の生データは保持していない（固定係数のみ）
                        "now": _decimate_xy(I, T),
                        "ref_fit": _fit_curve_linear(r.get("a_ref"), r.get("b_ref"), i_min, i_max),
                        "now_fit": _fit_curve_linear((r.get("now") or {}).get("a"),
                                                     (r.get("now") or {}).get("b"), i_min, i_max),
                    }
            except Exception as ex:
                sys.stderr.write("[judge %s] プロットデータ添付失敗（判定自体は有効）: %s\n" % (pv, ex))
        results.append(r)
    results.sort(key=lambda x: (-(x["severity"] if x["severity"] is not None else -1),
                                -(x["ratio"] or 0)))
    return results


def _resolve_pvs_arg(args):
    """--pv（完全一致1本）/ --match（部分一致でしぼる）から pvs リストを作る。
    どちらも無ければ None（＝リング全体が対象、従来通り）。"""
    pv = getattr(args, "pv", None)
    match = getattr(args, "match", None)
    if pv:
        return [pv]
    if match:
        recs = temp_fetch.load_pv_list(args.ring)
        pvs = [r["pv"] for r in recs if match in r["pv"]]
        if not pvs:
            sys.exit("エラー: --match %r に一致するPVが %s に見つかりません。" % (match, args.ring))
        return pvs
    return None


def cmd_learn(args):
    learn(args.ring, args.start, args.end, interval_sec=args.interval, out_path=args.out,
         model=args.model, pvs=_resolve_pvs_arg(args))


def section_warnings(anomalies, total):
    """同一セクションで多数のPVが同時に悪化していないか集計する（機器個別の故障ではなく、
    季節/空調等の環境要因や共通設備の可能性を疑う目安）。cmd_judge の表示と run_periodic_judge
    のダッシュボードJSON化の両方から使う共通ロジック（元は cmd_judge に直書きしていたものを
    切り出した）。

    切片差Δaが揃って大きい場合は「周囲温度が全体的にシフトした」可能性、傾き比だけ揃って
    大きい場合は「ビーム発熱への結合が変わった」可能性が高い（あくまで目安、要現地確認）。

    返り値: [{"section", "count", "total", "median_abs_delta_a"(or None), "likely_environmental"}, ...]
    """
    by_section = {}
    for r in anomalies:
        by_section.setdefault(r.get("section"), []).append(r)
    out = []
    for sec, rs in by_section.items():
        if total and len(rs) / total >= 0.5 and len(rs) >= 3:
            das = [r["delta_a"] for r in rs if r.get("delta_a") is not None]
            med = float(np.median(np.abs(das))) if das else None
            out.append({
                "section": sec, "count": len(rs), "total": total,
                "median_abs_delta_a": med,
                "likely_environmental": bool(med is not None and med > 1.0),
            })
    return out


def _json_safe(obj):
    """dict/listを再帰的に走査し、numpyスカラー型やNaN/InfなどそのままではJSON化できない値を
    Python ネイティブ値（NaN/Infはnull）に変換する。judge()の返り値には np.float64 等の
    numpy スカラーが混ざりうるため、ダッシュボード用JSON書き出し前に必ず通す。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def cmd_judge(args):
    results = judge(args.ring, args.start, args.end, interval_sec=args.interval,
                    models_path=args.models, pvs=_resolve_pvs_arg(args))
    anomalies = [r for r in results if r["severity"] is not None and r["severity"] >= 1]
    print("判定 %d 本（モデル有り・信頼できるもののみ対象） / 異常(sev≥1) %d 本"
          % (len(results), len(anomalies)))
    for r in anomalies[:args.top]:
        if r.get("model") == "hom":
            print("  %-3d %-30s %-6s  dT@代表点: 基準%6s → 現在%6s ℃ (I=%.0fmA,Nb=%.0f)  R2:基準%s/現在%s  Δw0(環境温度差,参考)=%s℃  %s"
                  % (r["severity"], r["pv"], ("%.2fx" % r["ratio"]) if r["ratio"] else "—",
                     _f(r["dT_ref"], 1, 2), _f(r["dT_now"], 1, 2), r["eval_i"], r["eval_nb"],
                     _f(r.get("r2_ref"), 1, 3), _f(r["now"].get("r2"), 1, 3),
                     _f(r.get("delta_a"), 1, 1), r["reason"]))
        else:
            print("  %-3d %-30s %-6s %10s %10s %9s  %s"
                  % (r["severity"], r["pv"], ("%.2fx" % r["ratio"]) if r["ratio"] else "—",
                     _f(r["b_ref"], 1000), _f(r["now"]["b"], 1000), _f(r.get("delta_a"), 1, 2), r["reason"]))

    # 同一セクションで多数が同時に悪化していないか（機器個別ではなく環境要因等の共通原因を疑う目安）。
    for w in section_warnings(anomalies, len(results)):
        print("\n  ⚠ section=%s で %d/%d 本が同時に悪化（同一エリアの共通要因の可能性）。"
              % (w["section"], w["count"], w["total"]))
        if w["likely_environmental"]:
            print("    切片差(Δa)の中央値も %.1f℃ と大きい → 季節/空調等の周囲温度シフトを疑う"
                  "（基準期間と判定期間の季節を揃えて再確認を推奨）。" % w["median_abs_delta_a"])
        else:
            print("    切片差(Δa)は小さめ → ビーム発熱への結合そのものが変わった可能性。"
                  "個別に現地確認を推奨。")


def run_periodic_judge(rings=("LER", "HER", "IR"), hours=24, interval_sec=temp_fetch.DEFAULT_INTERVAL,
                       models_path=MODELS_FILE, out_path=EQUIPMENT_STATE_FILE, top=60):
    """detector_headless.py の検知サイクルから定期的に呼ばれる版（CCG/IP/温度計センサと同じ
    相乗り方式）。learn は行わず、リングごとに保存済みモデル（temp_equipment_models.json、
    運用者が `python temp_equipment.py learn ...` で明示的に作成）との judge だけを回す。
    機器の熱結合特性は年〜数年スケールでしか動かない量なので、CCG/温度計センサ判定のような
    「毎回直近数日を学習し直す」ローリング基準はここでは使わない（IPの固定モデル方式と同じ思想）。

    まだ learn されていない（保存済みモデルが無い/信頼できるモデルが1本もない）リングは
    エラーではなく通常運用として自動的にスキップする（例: LER/HERはまだlearnしていないが
    IRだけ運用中、という状態でも安全にJSONを書ける）。

    書き出す out_path のJSONは temp_headless.py の temp_dashboard_state.json と対になる形
    （{"generated_at", "rings": {ring: {...}}}）にしてあり、dashboard.py の「機器劣化検知」
    タブがこれを読む。片リング失敗/未学習でも他のリングは続行する。
    """
    end = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    start = (datetime.datetime.now() - datetime.timedelta(hours=hours)).strftime("%Y%m%d%H%M%S")
    store = _load_models(models_path)
    rings_out = {}
    for ring in rings:
        rd = store.get(ring)
        if not rd or not any(m.get("trust") for m in rd.values()):
            rings_out[ring] = {"skipped": True, "reason": "not_learned"}
            continue
        try:
            results = judge(ring, start, end, interval_sec=interval_sec, models_path=models_path,
                            attach_plot=True)
        except Exception as ex:
            rings_out[ring] = {"error": str(ex)}
            continue
        anomalies = [r for r in results if r.get("severity") is not None and r["severity"] >= 1]
        anomalies.sort(key=lambda x: (-(x["severity"] if x["severity"] is not None else -1),
                                      -(x["ratio"] or 0)))
        # learn済み(=このリングは対象)なのに、judge時点で全PVがinsufficient_dataなら
        # 「学習済みモデルとの不一致」ではなく「アーカイバ自体がデータ取得を停止している」と
        # 判断できる（temp_headless.py/flow_headless.py と対の判定ロジック）。
        n_insufficient = sum(1 for r in results if r.get("reason") == "insufficient_data")
        archiver_stopped = bool(len(results) > 0 and n_insufficient == len(results))
        rings_out[ring] = {
            "window": {"start": start, "end": end, "hours": hours, "interval_sec": interval_sec},
            "stats": {"n_judged": len(results),
                     "n_anomalies_sev3": sum(1 for r in anomalies if r["severity"] >= 3)},
            "n_anomalies": len(anomalies),
            "anomalies": anomalies[:top],
            "section_warnings": section_warnings(anomalies, len(results)),
            "archiver_stopped": archiver_stopped,
        }
    out = _json_safe({"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "rings": rings_out})
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    return out


def cmd_judge_all(args):
    """detector_headless.py 向けの手動テスト用: 学習済み全リングをまとめて judge し、
    ダッシュボード用JSON（既定 temp_equipment_state.json）を書く（run_periodic_judge の直接呼び出し）。"""
    out = run_periodic_judge(hours=args.hours, interval_sec=args.interval,
                             models_path=args.models, out_path=args.out, top=args.top)
    for ring, rd in out["rings"].items():
        if rd.get("skipped"):
            print("[%s] スキップ（%s）" % (ring, rd["reason"]))
        elif rd.get("error"):
            print("[%s] 失敗: %s" % (ring, rd["error"]))
        else:
            print("[%s] 判定 %d 本 / 異常(sev≥1) %d 本 / sev3=%d"
                  % (ring, rd["stats"]["n_judged"], rd["n_anomalies"], rd["stats"]["n_anomalies_sev3"]))
    print("→ %s を更新" % os.path.basename(args.out))


def cmd_scan(args):
    """[簡易版] リング内の全センサを基準期間 vs 現在期間で直接比較（モデル保存なし）。
    本来は learn（過去の健全期間を保存）→ judge（直近と比較）の運用を推奨するが、
    保存無しで1回だけざっと洗い出したいときはこちらが手軽。"""
    recs = temp_fetch.load_pv_list(args.ring)
    if args.match:
        recs = [r for r in recs if args.match in r["pv"]]
    print("対象 %d 本を比較（基準 %s〜%s / 現在 %s〜%s、ビームあり%.0fmA以上のみ使用）..."
          % (len(recs), args.ref_start, args.ref_end, args.now_start, args.now_end, CONFIG["beam_on_ma"]))
    results = []
    for rec in recs:
        pv = rec["pv"]
        try:
            T_ref, I_ref = load_live(args.ring, pv, args.ref_start, args.ref_end, args.interval)
            T_now, I_now = load_live(args.ring, pv, args.now_start, args.now_end, args.interval)
            T_ref, I_ref = filter_beam_on(T_ref, I_ref)
            T_now, I_now = filter_beam_on(T_now, I_now)
            r = compare_periods(T_ref, I_ref, T_now, I_now)
        except Exception as ex:
            sys.stderr.write("[%s] 取得/判定失敗（スキップ）: %s\n" % (pv, ex))
            continue
        if r["severity"] is not None and r["severity"] >= 1:
            results.append((pv, rec, r))
    results.sort(key=lambda x: (-(x[2]["severity"] or 0), -(x[2]["ratio"] or 0)))
    print("\n異常(sev≥1) %d 本:" % len(results))
    print("  %-3s %-30s %-6s %8s %8s %6s" % ("sev", "PV", "比", "基準dT/dI", "現在dT/dI", "理由"))
    for pv, rec, r in results[:args.top]:
        print("  %-3d %-30s %-6s %8s %8s  %s"
              % (r["severity"], pv, ("%.2fx" % r["ratio"]) if r["ratio"] else "—",
                 _f(r["ref"]["b"], 1000), _f(r["now"]["b"], 1000), r["reason"]))


# ───────────────────────── selftest（kblogrd 不要）─────────────────────────

def _selftest():
    print("=== temp_equipment selftest（kblogrd 不要）===")
    ok = True
    rng = np.random.RandomState(0)

    # 1) 実データを模した設定: 基準 dT/dI≈4.25℃/A(0.00425℃/mA)、現在≈7.26℃/A(0.00726℃/mA)
    #    （実際の 2022.csv/2026.csv から得た値と同水準。約1.7倍への悪化 → sev3 期待）
    n = 300
    I_ref = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_ref = 19.6 + 0.00425 * I_ref + rng.normal(0, 0.4, n)
    I_now = np.clip(rng.normal(900, 250, n), 0, 1500)
    T_now = 22.8 + 0.00726 * I_now + rng.normal(0, 0.4, n)

    fref = fit_t_vs_i(T_ref, I_ref)
    fnow = fit_t_vs_i(T_now, I_now)
    print("  fit_ref: b=%.5f r=%.2f trust=%s" % (fref["b"], fref["r"], fref["trust"]))
    print("  fit_now: b=%.5f r=%.2f trust=%s" % (fnow["b"], fnow["r"], fnow["trust"]))
    ok &= fref["trust"] and fnow["trust"]
    ok &= abs(fref["b"] - 0.00425) < 0.001    # Theil-Sen が真の傾きに近いこと
    ok &= abs(fnow["b"] - 0.00726) < 0.001

    r = compare_periods(T_ref, I_ref, T_now, I_now)
    print("  compare: sev=%s reason=%s ratio=%.2f" % (r["severity"], r["reason"], r["ratio"]))
    ok &= (r["severity"] == 3 and r["reason"] == "heating_gain_increase_severe")

    # 2) 劣化なし（同じ傾き）→ sev0
    I_a = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_a = 20.0 + 0.005 * I_a + rng.normal(0, 0.3, n)
    I_b = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_b = 20.0 + 0.005 * I_b + rng.normal(0, 0.3, n)
    r2 = compare_periods(T_a, I_a, T_b, I_b)
    print("  no-change: sev=%s reason=%s" % (r2["severity"], r2["reason"]))
    ok &= (r2["severity"] == 0)

    # 【回帰テスト】linearモデルでも、環境温度（切片a）だけのシフトで誤検知しないこと。
    I_e1 = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_e1 = 15.0 + 0.005 * I_e1 + rng.normal(0, 0.3, n)     # 環境温度15℃
    I_e2 = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_e2 = 30.0 + 0.005 * I_e2 + rng.normal(0, 0.3, n)     # 環境温度30℃（傾きは同一＝劣化なし）
    re = compare_periods(T_e1, I_e1, T_e2, I_e2)
    print("  compare_periods(環境温度15→30℃・劣化なし): sev=%s ratio=%.2f Δa=%.1f℃"
          % (re["severity"], re["ratio"], re["now"]["a"] - re["ref"]["a"]))
    ok &= (re["severity"] == 0)

    # 3) 軽度の悪化（比1.3倍程度）→ sev2
    I_c = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_c = 20.0 + 0.005 * I_c + rng.normal(0, 0.3, n)
    I_d = np.clip(rng.normal(950, 200, n), 0, 1300)
    T_d = 20.0 + 0.0065 * I_d + rng.normal(0, 0.3, n)
    r3 = compare_periods(T_c, I_c, T_d, I_d)
    print("  mild: sev=%s reason=%s ratio=%.2f" % (r3["severity"], r3["reason"], r3["ratio"]))
    ok &= (r3["severity"] == 2)

    # 4) 電流レンジ不足（静穏期間）→ 判定不能
    I_e = np.clip(rng.normal(950, 30, n), 0, 1300)   # レンジが狭い
    T_e = 20.0 + 0.005 * I_e + rng.normal(0, 0.3, n)
    r4 = compare_periods(T_ref, I_ref, T_e, I_e)
    print("  narrow_range: sev=%s reason=%s (now.trust=%s)" % (r4["severity"], r4["reason"], r4["now"]["trust"]))
    ok &= (r4["severity"] is None and not r4["now"]["trust"])

    # 5) 実データ形式（変化ログCSV）の読み込みロジック検証（アップロード実データを模した合成CSV）
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tf:
        tf.write("Timestamp,FB_MOVE:D01:QC1L:BWS:TEMP,BMLDCCT:CURRENT\n")
        tf.write("2022/04/20 19:45:51.430,24,N/A\n")
        tf.write("2022/04/20 19:59:59.709,N/A,992.68\n")
        tf.write("2022/04/20 20:10:00.000,25,N/A\n")
        tf.write("2022/04/20 20:20:00.000,N/A,1002.03\n")
        tf.write("2022/04/20 20:30:00.000,26,N/A\n")
        path = tf.name
    try:
        T_csv, I_csv, pvname = load_raw_csv(path)
        print("  load_raw_csv: %d点 pv=%s T=%s I=%s" % (len(T_csv), pvname, T_csv, I_csv))
        ok &= (len(T_csv) == 2 and pvname == "FB_MOVE:D01:QC1L:BWS:TEMP")
        # 1つ目の温度行(24)はビーム未知のため除外。以降は直近(前方)のビーム値と対応する。
        ok &= (T_csv[0] == 25 and I_csv[0] == 992.68)
        ok &= (T_csv[1] == 26 and I_csv[1] == 1002.03)
    finally:
        os.remove(path)

    # 6) ビームフィルタ: 無ビーム点（I<beam_on_ma）を除外すること
    I_mix = np.array([0, 10, 40, 60, 500, 900, 30, 1000], float)
    T_mix = np.array([20, 20, 21, 25, 40, 60, 20.5, 65], float)
    Tf, If = filter_beam_on(T_mix, I_mix)
    print("  filter_beam_on: %d点中 %d点が残る（ビーム>=%.0fmA）" % (len(I_mix), len(If), CONFIG["beam_on_ma"]))
    ok &= (len(If) == 4 and np.all(If >= CONFIG["beam_on_ma"]))

    # 6a2) 電流急変直後の熱の過渡除外（_settle_exclude_mask）。方向を問わず、直近
    # settle_after_change_min 分間の変動幅が settle_change_ma 以上なら除外されること。
    cfg_s = dict(CONFIG); cfg_s["settle_after_change_min"] = 20.0; cfg_s["settle_change_ma"] = 200.0
    n_s = 40
    # ケースA: アボート（急落）→ その後フル電流(1000mA)へ回復。20点目で1000mA→0mAに落ち、
    # 25分間無ビーム、その後1000mAへ回復（swing=1000mA、閾値200mAを大きく超える）。
    I_drop = np.full(n_s, 1000.0)
    I_drop[20:26] = 0.0
    I_drop[26:] = 1000.0
    t_sec_s = np.arange(n_s) * 300.0   # 300s間隔（=20分は4点）
    excl_drop = _settle_exclude_mask(t_sec_s, I_drop, cfg_s)
    print("  settle_exclude(急落→回復): 回復直後 idx26-29(20分以内)=%s, idx30(20分超)=%s"
          % (bool(np.all(excl_drop[26:30])), bool(excl_drop[30])))
    ok &= np.all(excl_drop[26:30]) and not excl_drop[30] and not excl_drop[10]  # 定常域(idx10)は影響なし

    # ケースB: フィル開始・電流アップ（急上昇）。20点目で100mA→1000mAに上がり、以降1000mA定常。
    # 上昇方向でも同じ settle window だけ除外され、定常に戻れば除外が外れること。
    I_rise = np.full(n_s, 100.0)
    I_rise[20:] = 1000.0
    excl_rise = _settle_exclude_mask(t_sec_s, I_rise, cfg_s)
    print("  settle_exclude(急上昇): 上昇直後 idx20-23(20分以内)=%s, idx24(20分超)=%s, 上昇前idx10=%s"
          % (bool(np.all(excl_rise[20:24])), bool(excl_rise[24]), bool(excl_rise[10])))
    ok &= np.all(excl_rise[20:24]) and not excl_rise[24] and not excl_rise[10]



    # （またはTEMP_RING_OVERRIDE.csv）でビーム取得先を決める必要がある。ring="IR"のまま
    # fetch_beamに渡すと常に空（→全点ビームNaN→trust=0）になっていたのが実機で見つかった不具合。
    ok &= (_beam_ring_for_pv("LER", "VALTMP:D10M001:QDWNP_4:BL") == "LER")   # LER形式はそのまま
    ok &= (_beam_ring_for_pv("IR", "FB_MOVE:D01:QC1L:BWS:TEMP") == "LER")    # IR形式: QC1L→LER
    ok &= (_beam_ring_for_pv("IR", "FB_MOVE:D01:QC1H:TEMP") == "HER")        # IR形式: QC1H→HER
    print("  _beam_ring_for_pv: LER形式→LER、IR形式(QC1L)→LER、IR形式(QC1H)→HER  OK")

    # 6c) HOMモデル T=w0+w1*I+w2*(I^2/Nb)^2（Suetsugu et al. 式(5)型）の頑健フィット検証。
    #    w2 は実運転電流(I~1000mA)でw1項と同程度の寄与になる規模（1e-6桁）で選ぶこと。
    #    小さすぎる(例1e-9)とノイズに埋もれて検出できない（実際に較正時に確認済み）。
    n3 = 400
    I3r = np.clip(rng.normal(900, 300, n3), 0, 1500)
    Nb3r = np.clip(rng.normal(1576, 100, n3), 800, 1800)
    w0h, w1h, w2h = 20.0, 0.003, 5e-6
    T3r = w0h + w1h * I3r + w2h * (I3r ** 2 / Nb3r) ** 2 + rng.normal(0, 0.3, n3)
    fh = fit_t_vs_i_hom(T3r, I3r, Nb3r)
    print("  fit_t_vs_i_hom: w=%s r2=%.3f trust=%s（真値 w0=%.1f w1=%.4f w2=%.2e）"
          % ([round(x, 6) for x in fh["w"]], fh["r2"], fh["trust"], w0h, w1h, w2h))
    ok &= fh["trust"] and fh["r2"] > 0.9
    ok &= abs(fh["w"][0] - w0h) < 1.0 and abs(fh["w"][1] - w1h) < 0.001
    ok &= abs(fh["w"][2] - w2h) / w2h < 0.3   # w2はw0/w1より復元誤差が出やすいので相対30%許容

    # HOM結合が3倍に劣化 → compare_periods_hom で sev3 検知できること
    I3n = np.clip(rng.normal(900, 300, n3), 0, 1500)
    Nb3n = np.clip(rng.normal(1576, 100, n3), 800, 1800)
    T3n = w0h + w1h * I3n + (w2h * 3) * (I3n ** 2 / Nb3n) ** 2 + rng.normal(0, 0.3, n3)
    rh = compare_periods_hom(T3r, I3r, Nb3r, T3n, I3n, Nb3n)
    print("  compare_periods_hom: sev=%s reason=%s ratio=%.2f dT_ref=%.2f dT_now=%.2f"
          % (rh["severity"], rh["reason"], rh["ratio"], rh["dT_ref"], rh["dT_now"]))
    ok &= (rh["severity"] == 3 and rh["reason"] == "heating_gain_increase_severe")

    # 劣化なし（同じw2）→ sev0 であること
    I3n2 = np.clip(rng.normal(900, 300, n3), 0, 1500)
    Nb3n2 = np.clip(rng.normal(1576, 100, n3), 800, 1800)
    T3n2 = w0h + w1h * I3n2 + w2h * (I3n2 ** 2 / Nb3n2) ** 2 + rng.normal(0, 0.3, n3)
    rh2 = compare_periods_hom(T3r, I3r, Nb3r, T3n2, I3n2, Nb3n2)
    print("  compare_periods_hom(劣化無し): sev=%s ratio=%.2f" % (rh2["severity"], rh2["ratio"]))
    ok &= (rh2["severity"] == 0)

    # 【回帰テスト】環境温度（切片w0）だけが大きくシフトし、真の熱結合特性(w1,w2)は同一の場合、
    # sev0 のままであること（w0=ビーム電流ゼロの温度=環境温度は判定に使わない設計の裏付け）。
    I3e1 = np.clip(rng.normal(900, 300, n3), 0, 1500); Nb3e1 = np.full(n3, 1600.0)
    T3e1 = 15.0 + w1h * I3e1 + w2h * (I3e1 ** 2 / Nb3e1) ** 2 + rng.normal(0, 0.3, n3)   # 環境温度15℃
    I3e2 = np.clip(rng.normal(900, 300, n3), 0, 1500); Nb3e2 = np.full(n3, 1600.0)
    T3e2 = 30.0 + w1h * I3e2 + w2h * (I3e2 ** 2 / Nb3e2) ** 2 + rng.normal(0, 0.3, n3)   # 環境温度30℃（劣化なし）
    rhe = compare_periods_hom(T3e1, I3e1, Nb3e1, T3e2, I3e2, Nb3e2)
    print("  compare_periods_hom(環境温度15→30℃・劣化なし): sev=%s ratio=%.2f Δw0=%.1f℃"
          % (rhe["severity"], rhe["ratio"], rhe["now"]["w"][0] - rhe["ref"]["w"][0]))
    ok &= (rhe["severity"] == 0)

    # 7) learn/judge の永続モデル運用を end-to-end で検証（temp_fetch をモック）
    import tempfile as _tf
    orig_hist, orig_beam, orig_list = temp_fetch.fetch_history, temp_fetch.fetch_beam, temp_fetch.load_pv_list
    rng2 = np.random.RandomState(1)
    n2 = 200
    def fake_list(ring, **k):
        return [{"pv": "FB_MOVE:D01:QC1L:BWS:TEMP", "ring": "LER", "section": "D01",
                 "tag": "QC1L", "suffix": "BWS", "family": "IR"}]

    def _mock_I(start, end):
        # start/end から決定的に I を再現する（呼び出し順序に依存しない。fake_hist と fake_beam の
        # どちらが先に呼ばれても同じ結果になる。テストモック用の簡易実装）。
        seed = abs(hash((start, end))) % (2**31)
        r = np.random.RandomState(seed)
        return np.clip(900 + 250 * np.sin(np.linspace(0, 4 * np.pi, n2)) + r.normal(0, 15, n2), 0, 1500)

    def fake_hist(ring, start, end, interval_sec=300, pvs=None, **k):
        # 実際のビーム電流は滑らかに推移する（フィル内で緩やかに減衰）ため、独立乱数ノイズでは
        # なく、緩やかな正弦波+小ノイズで模す（設定ノイズ幅=250mAだと隣接点で数百mA跳んで
        # settle_exclude_maskが常時発火してしまうため）。
        I = _mock_I(start, end)
        b = 0.00726 if start == "20260301000000" else 0.00425   # 現在期間だけ悪化させる
        T = 20 + b * I + rng2.normal(0, 0.3, n2)
        ts = list(range(n2))
        return {pvs[0]: {"ring": ring, "section": "D01", "tag": "QC1L", "suffix": "BWS",
                         "series": list(zip(ts, [float(x) for x in T]))}}, I

    def fake_hist_wrap(ring, start, end, interval_sec=300, pvs=None, **k):
        d, I = fake_hist(ring, start, end, interval_sec, pvs)
        return d

    def fake_beam(ring, start, end, interval_sec=300, **k):
        I = _mock_I(start, end)   # fake_hist と同じ (start,end) から同じ I を再現（呼び出し順序に依存しない）
        ts = list(range(len(I)))
        return list(zip(ts, [float(x) for x in I]))

    temp_fetch.load_pv_list = fake_list
    temp_fetch.fetch_history = fake_hist_wrap
    temp_fetch.fetch_beam = fake_beam
    try:
        with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
            model_path = tf2.name
        os.remove(model_path)
        learn("LER", "20220401000000", "20220501000000", out_path=model_path, model="linear")
        m = _load_models(model_path)
        print("  learn: モデル保存 b_ref=%.5f trust=%s"
              % (m["LER"]["FB_MOVE:D01:QC1L:BWS:TEMP"]["b"], m["LER"]["FB_MOVE:D01:QC1L:BWS:TEMP"]["trust"]))
        ok &= m["LER"]["FB_MOVE:D01:QC1L:BWS:TEMP"]["trust"]

        results = judge("LER", "20260301000000", "20260401000000", models_path=model_path)
        print("  judge: %s" % ([(r["pv"], r["severity"], r["reason"]) for r in results]))
        ok &= (len(results) == 1 and results[0]["severity"] == 3
              and results[0]["reason"] == "heating_gain_increase_severe")
    finally:
        temp_fetch.load_pv_list = orig_list
        temp_fetch.fetch_history = orig_hist
        temp_fetch.fetch_beam = orig_beam
        if os.path.isfile(model_path):
            os.remove(model_path)

    # 7b) 【一般性の確認】IR(FB_MOVE)専用ではなく、LER/HER本体センサ（VA{L,H}TMP形式）でも
    # 同じ learn/judge が正しく動くこと（機器劣化検知を全リングに広げる際の裏付け）。
    def fake_list_va(ring, **k):
        return [{"pv": "VAHTMP:D10_139:QD3E_11:BL", "ring": "HER", "section": "D10",
                 "tag": "QD3E_11", "suffix": "BL", "family": None}]

    def fake_hist_va(ring, start, end, interval_sec=300, pvs=None, **k):
        I = _mock_I(start, end)
        b = 0.008 if start == "20260301000000" else 0.004   # 現在期間だけ悪化させる
        T = 22 + b * I + rng2.normal(0, 0.3, n2)
        ts = list(range(n2))
        return {pvs[0]: {"ring": ring, "section": "D10", "tag": "QD3E_11", "suffix": "BL",
                         "series": list(zip(ts, [float(x) for x in T]))}}

    def fake_beam_va(ring, start, end, interval_sec=300, **k):
        I = _mock_I(start, end)
        return list(zip(range(len(I)), [float(x) for x in I]))

    temp_fetch.load_pv_list = fake_list_va
    temp_fetch.fetch_history = fake_hist_va
    temp_fetch.fetch_beam = fake_beam_va
    try:
        with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf2b:
            model_path_va = tf2b.name
        os.remove(model_path_va)
        learn("HER", "20220401000000", "20220501000000", out_path=model_path_va, model="linear")
        results_va = judge("HER", "20260301000000", "20260401000000", models_path=model_path_va)
        print("  judge(VA形式・HER本体センサ VAHTMP:D10_139:...): %s"
              % ([(r["pv"], r["severity"], r["reason"]) for r in results_va]))
        ok &= (len(results_va) == 1 and results_va[0]["severity"] == 3
              and results_va[0]["reason"] == "heating_gain_increase_severe")
    finally:
        temp_fetch.load_pv_list = orig_list
        temp_fetch.fetch_history = orig_hist
        temp_fetch.fetch_beam = orig_beam
        if os.path.isfile(model_path_va):
            os.remove(model_path_va)

    # 8) HOMモデル（model="hom"）の learn/judge も end-to-end で検証（fetch_nb も含めてモック）
    orig_nb = temp_fetch.fetch_nb
    rng3 = np.random.RandomState(2)
    n3b = 250
    w0e, w1e = 20.0, 0.003
    w2_ref_e, w2_now_e = 5e-6, 5e-6 * 3   # 現在期間だけ HOM 結合を3倍に劣化させる

    def _mock_I_Nb(start, end):
        seed = abs(hash((start, end))) % (2**31)
        r = np.random.RandomState(seed)
        I = np.clip(900 + 300 * np.sin(np.linspace(0, 5 * np.pi, n3b)) + r.normal(0, 15, n3b), 0, 1500)
        Nb = np.clip(1576 + 100 * np.sin(np.linspace(0, 3 * np.pi, n3b)) + r.normal(0, 10, n3b), 800, 1800)
        return I, Nb

    def fake_hist2(ring, start, end, interval_sec=300, pvs=None, **k):
        I, Nb = _mock_I_Nb(start, end)
        w2u = w2_now_e if start == "20260301000000" else w2_ref_e
        T = w0e + w1e * I + w2u * (I ** 2 / Nb) ** 2 + rng3.normal(0, 0.3, n3b)
        ts = list(range(n3b))
        return {pvs[0]: {"ring": ring, "section": "D01", "tag": "QC1L", "suffix": "BWS",
                         "series": list(zip(ts, [float(x) for x in T]))}}

    def fake_beam2(ring, start, end, interval_sec=300, **k):
        I, _ = _mock_I_Nb(start, end)
        return list(zip(range(len(I)), [float(x) for x in I]))

    def fake_nb2(ring, start, end, interval_sec=300, **k):
        _, Nb = _mock_I_Nb(start, end)
        return list(zip(range(len(Nb)), [float(x) for x in Nb]))

    temp_fetch.load_pv_list = fake_list
    temp_fetch.fetch_history = fake_hist2
    temp_fetch.fetch_beam = fake_beam2
    temp_fetch.fetch_nb = fake_nb2
    try:
        with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf3:
            model_path2 = tf3.name
        os.remove(model_path2)
        learn("LER", "20220401000000", "20220501000000", out_path=model_path2, model="hom")
        m2 = _load_models(model_path2)
        saved = m2["LER"]["FB_MOVE:D01:QC1L:BWS:TEMP"]
        print("  learn(hom): w=%s trust=%s" % ([round(x, 6) for x in saved["w"]], saved["trust"]))
        ok &= saved["trust"] and saved["model"] == "hom"

        results2 = judge("LER", "20260301000000", "20260401000000", models_path=model_path2)
        print("  judge(hom): %s" % ([(r["pv"], r["severity"], r["reason"], round(r["ratio"], 2))
                                     for r in results2]))
        ok &= (len(results2) == 1 and results2[0]["severity"] == 3
              and results2[0]["reason"] == "heating_gain_increase_severe")
    finally:
        temp_fetch.load_pv_list = orig_list
        temp_fetch.fetch_history = orig_hist
        temp_fetch.fetch_beam = orig_beam
        temp_fetch.fetch_nb = orig_nb
        if os.path.isfile(model_path2):
            os.remove(model_path2)

    # 8b) run_periodic_judge（detector_headless.py 相乗り版）を end-to-end 検証。
    # IR だけ learn 済み、LER/HER は未learnという実運用に近い状態を作り、
    # 「未learnリングは自動スキップ」「learn済みリングだけ判定＋JSON化」を確認する。
    temp_fetch.load_pv_list = fake_list
    temp_fetch.fetch_history = fake_hist2
    temp_fetch.fetch_beam = fake_beam2
    temp_fetch.fetch_nb = fake_nb2
    with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf4:
        model_path3 = tf4.name
    os.remove(model_path3)
    with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf5:
        state_path3 = tf5.name
    os.remove(state_path3)
    try:
        # 学習期間として「基準」側 (w2_ref_e) を使う（このモック関数群は start=="20260301000000"
        # のときだけ劣化後の値を返すため、学習はそれ以外の期間で行えば基準相当になる）。
        learn("IR", "20220401000000", "20220501000000", out_path=model_path3, model="hom")

        # datetime.now() を判定窓の終端に使うため、fake_hist2/fake_beam2/fake_nb2 が
        # 「現在期間」と認識する開始時刻 "20260301000000" になるよう、hours 分だけ
        # ずらして呼ぶのではなく、run_periodic_judge 内部で作る start 文字列がこの値と
        # 一致するとは限らない（now() 基準のため）。ここでは reason/severity の値ではなく
        # 「未learnリングのスキップ」と「JSON書き出しが壊れないこと」を主眼に検証する。
        out3 = run_periodic_judge(rings=("LER", "HER", "IR"), hours=24, models_path=model_path3,
                                  out_path=state_path3)
        print("  run_periodic_judge: LER=%s HER=%s IR=%s"
              % (out3["rings"]["LER"].get("skipped"), out3["rings"]["HER"].get("skipped"),
                 "skipped" if out3["rings"]["IR"].get("skipped") else "judged"))
        ok &= out3["rings"]["LER"].get("skipped") is True and out3["rings"]["LER"].get("reason") == "not_learned"
        ok &= out3["rings"]["HER"].get("skipped") is True
        ok &= "anomalies" in out3["rings"]["IR"] and "n_anomalies" in out3["rings"]["IR"]
        ok &= os.path.isfile(state_path3)
        with open(state_path3, encoding="utf-8") as f:
            reloaded = json.load(f)   # 書き出したJSONがそのまま読めること（_json_safeの検証を兼ねる）
        ok &= reloaded["rings"]["IR"]["n_anomalies"] == out3["rings"]["IR"]["n_anomalies"]

        # モデルファイル自体が全く無い場合でも、全リングskippedの有効なJSONを書けること
        # （実機に一度もlearnしていない初期状態でも detector_headless.py の相乗りがクラッシュしない）。
        out4 = run_periodic_judge(rings=("LER", "HER", "IR"), models_path=model_path3 + ".none",
                                  out_path=state_path3)
        ok &= all(rd.get("skipped") for rd in out4["rings"].values())
        print("  run_periodic_judge(モデル無し): 全リングskipped=%s"
              % all(rd.get("skipped") for rd in out4["rings"].values()))
    finally:
        temp_fetch.load_pv_list = orig_list
        temp_fetch.fetch_history = orig_hist
        temp_fetch.fetch_beam = orig_beam
        temp_fetch.fetch_nb = orig_nb
        for p in (model_path3, state_path3):
            if os.path.isfile(p):
                os.remove(p)

    # 9) 【回帰テスト】温度取得がPV単位ではなくチャンク単位（CHUNK本/回）でまとめて
    # kblogrd 呼び出しされること（1本ずつ呼ぶと呼び出し回数がPV本数分に膨らむ問題の再発防止）。
    # PV本数・期待するチャンク割りは temp_fetch.CHUNK の現在値から動的に決める
    # （CHUNKの既定値を変えてもこのテストが自動追従し、壊れないようにするため）。
    n9 = 50
    rng9 = np.random.RandomState(5)
    call_counts = {"hist": 0, "beam": 0}
    chunk_sizes = []
    n_pv9 = temp_fetch.CHUNK * 2 + 5   # CHUNKを2回跨ぐ本数（例: CHUNK=26なら57本）
    expected_sizes = [min(temp_fetch.CHUNK, n_pv9 - i) for i in range(0, n_pv9, temp_fetch.CHUNK)]

    def fake_list9(ring, **k):
        return [{"pv": "VAHTMP:D%02d_%03d:X:BL" % (i % 12 + 1, i), "ring": "HER", "section": "D%02d" % (i % 12 + 1),
                 "tag": "X", "suffix": "BL", "family": None} for i in range(n_pv9)]

    def fake_hist9(ring, start, end, interval_sec=300, pvs=None, **k):
        call_counts["hist"] += 1
        chunk_sizes.append(len(pvs))
        out = {}
        for pv in pvs:
            I = np.clip(900 + 250 * np.sin(np.linspace(0, 4 * np.pi, n9)) + rng9.normal(0, 15, n9), 0, 1500)
            T = 20 + 0.004 * I + rng9.normal(0, 0.3, n9)
            out[pv] = {"ring": ring, "section": "D01", "tag": "X", "suffix": "BL",
                      "series": list(zip(range(n9), [float(x) for x in T]))}
        return out

    def fake_beam9(ring, start, end, interval_sec=300, **k):
        call_counts["beam"] += 1
        I = np.clip(900 + 250 * np.sin(np.linspace(0, 4 * np.pi, n9)) + rng9.normal(0, 15, n9), 0, 1500)
        return list(zip(range(n9), [float(x) for x in I]))

    temp_fetch.load_pv_list = fake_list9
    temp_fetch.fetch_history = fake_hist9
    temp_fetch.fetch_beam = fake_beam9
    try:
        with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf9:
            model_path9 = tf9.name
        os.remove(model_path9)
        learn("HER", "20220401000000", "20220501000000", out_path=model_path9, model="linear")
        print("  チャンク一括取得: %d本(CHUNK=%d)処理で hist呼び出し=%d回(期待%d) チャンクサイズ=%s beam呼び出し=%d回(期待1)"
              % (n_pv9, temp_fetch.CHUNK, call_counts["hist"], len(expected_sizes), chunk_sizes, call_counts["beam"]))
        ok &= (call_counts["hist"] == len(expected_sizes) and chunk_sizes == expected_sizes
              and call_counts["beam"] == 1)
    finally:
        temp_fetch.load_pv_list = orig_list
        temp_fetch.fetch_history = orig_hist
        temp_fetch.fetch_beam = orig_beam
        if os.path.isfile(model_path9):
            os.remove(model_path9)

    # 10)【回帰テスト】judge() も learn() と同じく、対象PVを手動で26本ずつに事前分割してから
    # fetch_history を何度も呼ぶのではなく、fetch_history 自身の内部チャンク分割に任せて
    # 1回（基準期間はグループごとに1回）だけ呼ぶこと（実機でCPU高負荷・ログ大量出力の一因に
    # なっていた無駄な呼び出し回数増加の再発防止）。
    n10 = 50
    rng10 = np.random.RandomState(6)
    hist_calls10 = []

    def fake_list10(ring, **k):
        return [{"pv": "VAHTMP:D%02d_%03d:X:BL" % (i % 12 + 1, i), "ring": "HER", "section": "D%02d" % (i % 12 + 1),
                 "tag": "X", "suffix": "BL", "family": None} for i in range(temp_fetch.CHUNK * 2 + 5)]

    def fake_hist10(ring, start, end, interval_sec=300, pvs=None, **k):
        hist_calls10.append(len(pvs))
        out = {}
        for pv in pvs:
            I = np.clip(900 + 250 * np.sin(np.linspace(0, 4 * np.pi, n10)) + rng10.normal(0, 15, n10), 0, 1500)
            T = 20 + 0.004 * I + rng10.normal(0, 0.3, n10)
            out[pv] = {"ring": ring, "section": "D01", "tag": "X", "suffix": "BL",
                      "series": list(zip(range(n10), [float(x) for x in T]))}
        return out

    def fake_beam10(ring, start, end, interval_sec=300, **k):
        I = np.clip(900 + 250 * np.sin(np.linspace(0, 4 * np.pi, n10)) + rng10.normal(0, 15, n10), 0, 1500)
        return list(zip(range(n10), [float(x) for x in I]))

    temp_fetch.load_pv_list = fake_list10
    temp_fetch.fetch_history = fake_hist10
    temp_fetch.fetch_beam = fake_beam10
    try:
        with _tf.NamedTemporaryFile(suffix=".json", delete=False) as tf10:
            model_path10 = tf10.name
        os.remove(model_path10)
        learn("HER", "20220401000000", "20220501000000", out_path=model_path10, model="linear")
        hist_calls10.clear()   # learn 分の呼び出しは対象外。judge 分だけを数える
        judge("HER", "20260601000000", "20260602000000", models_path=model_path10)
        n_targets10 = temp_fetch.CHUNK * 2 + 5
        print("  judge()一括取得: %d本処理で fetch_history呼び出し=%d回（期待1回。以前は%d回近くに膨らんでいた）"
              % (n_targets10, len(hist_calls10), -(-n_targets10 // temp_fetch.CHUNK)))
        # linear型は基準期間を再取得しないので、判定窓分の1回だけになるはず
        ok &= (len(hist_calls10) == 1 and hist_calls10[0] == n_targets10)
    finally:
        temp_fetch.load_pv_list = orig_list
        temp_fetch.fetch_history = orig_hist
        temp_fetch.fetch_beam = orig_beam
        if os.path.isfile(model_path10):
            os.remove(model_path10)

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


def main():
    ap = argparse.ArgumentParser(description="機器側の熱結合特性の劣化検知（センサではなく測定対象の異常）")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("learn", help="過去の健全期間からモデルを学習・保存（推奨ワークフロー①）")
    p.add_argument("ring", choices=["LER", "HER", "IR"])
    p.add_argument("start"); p.add_argument("end")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.add_argument("--out", default=MODELS_FILE)
    p.add_argument("--model", choices=["linear", "hom"], default="hom",
                   help="hom: T=w0+w1*I+w2*(I^2/Nb)^2（既定。Suetsugu et al. PRAB 27,063201(2024) "
                        "式(5)型。Nb取得が必要）。linear: T=a+b*I（Theil-Sen。Nb取得不要の簡易版）")
    p.add_argument("--pv", default=None, help="1本だけ学習（完全なPV名）。LER/HERは1550/1260本と"
                   "多いので、まず1本で試したいときに使う")
    p.add_argument("--match", default=None, help="PV名の部分一致でしぼる（例: BWS）")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("judge", help="直近期間を保存済みモデルと比較（推奨ワークフロー②）")
    p.add_argument("ring", choices=["LER", "HER", "IR"])
    p.add_argument("start"); p.add_argument("end")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.add_argument("--models", default=MODELS_FILE)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--pv", default=None, help="1本だけ判定（完全なPV名）")
    p.add_argument("--match", default=None, help="PV名の部分一致でしぼる（例: BWS）")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("compare", help="[簡易] 1本のPVについて基準期間と現在期間のdT/dIを直接比較")
    p.add_argument("ring", choices=["LER", "HER", "IR"])
    p.add_argument("pv", help="完全なPV名")
    p.add_argument("--ref-start"); p.add_argument("--ref-end")
    p.add_argument("--now-start"); p.add_argument("--now-end")
    p.add_argument("--ref-csv", default=None, help="基準期間を実機取得の代わりに変化ログCSVから読む")
    p.add_argument("--now-csv", default=None, help="現在期間を実機取得の代わりに変化ログCSVから読む")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.add_argument("--model", choices=["linear", "hom"], default="hom",
                   help="hom: T=w0+w1*I+w2*(I^2/Nb)^2（既定）。linear: T=a+b*I（Theil-Sen）")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("scan", help="[簡易] リング内全PV（または部分一致）を比較しランキング表示（モデル保存なし）")
    p.add_argument("ring", choices=["LER", "HER", "IR"])
    p.add_argument("ref_start"); p.add_argument("ref_end")
    p.add_argument("now_start"); p.add_argument("now_end")
    p.add_argument("--match", default=None, help="PV名の部分一致フィルタ（例: BWS）")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("judge-all", help="[detector_headless.py向け] 学習済み全リングを一括judgeし"
                       "ダッシュボード用JSONを書く（手動テスト用。実運用は detector_headless.py "
                       "--equipment-judge か --watch/--once/定期ループへの相乗り経由）")
    p.add_argument("--hours", type=float, default=24, help="判定窓の長さ[h]（既定24）")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.add_argument("--models", default=MODELS_FILE)
    p.add_argument("--out", default=EQUIPMENT_STATE_FILE)
    p.add_argument("--top", type=int, default=60)
    p.set_defaults(func=cmd_judge_all)

    sub.add_parser("selftest", help="合成データで検証（kblogrd不要）")

    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(0 if _selftest() else 1)
    if not args.cmd:
        ap.print_help(); sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
