#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_judge.py — イオンポンプ放電電流の異常判定（学習＋判定）

主目的: HV フィードスルーでの放電／破損の「事前検知」。
実データ（D12_IP_L23 破損例 / D07_4U_H06 劣化疑い）から、破損ポンプの signature は
「放電電流が圧力・ビーム電流から切り離されて高値に張り付く（デカップリング）。主に持続的」
であることを確認した。これを 3 層＋ビーム軸で捉える。

────────────────────────────────────────────────────────────────────────
判定の 3 層（＋ビーム軸）
  Layer 0  無ビーム放電チェック（モデル不要・最も頑健）:
           ビーム電流が低い/無い期間はガス負荷が無いので健全ポンプの電流はフロア近傍に
           落ちるはず。そこで電流が高ければ放電で確定。CCG にも I-P フィットにも依存しない。
  Layer 1  I-P 整合（z スコア）:
           pred = max(a·P^b, I_floor) でクランプし z = log10(I/pred)/σ。
           上振れ(z≫0)=放電、下振れ(z≪0, フロア以上のみ)=排気劣化。主に 4U。
  Layer 2  デカップリング検出:
           窓内 r(logI, logP)（圧力相関）と r(logI, log I_beam)（ビーム相関）の崩壊。
           健全 ~0.8–1.0 → 破損で →0。散布図の「水平張り付き」を定量化したもの。
ビーム軸:  健全ポンプは I がビーム電流に追従、破損ポンプは追従しない。CCG とは独立なので
           CCG 故障に強い。Layer 0/2 で使用。

学習フェーズ（learn）: 4U/KEK ポンプごとに、直近の正常窓のフロア以上の点で I=a·P^b を
  ロバスト推定（Theil–Sen 傾き＋切片中央値、σ=残差の MAD×1.4826）。窓内の健全性指標
  （r_pi, r_beam, b, logP レンジ, 採用点数）も保存。相関が崩れている窓は「正常としては
  学習しない（low trust）」=直前の良モデルを保持し、そのポンプ自体を異常側に倒す材料に。
判定フェーズ（judge）: 直近窓の各ポンプを 3 層で評価し、per-pump の severity と理由を出す。

設計判断（このセッションで確定）:
  - フロアの使い分け: 上振れ(放電)=全圧力域・予測フロアクランプ／下振れ(劣化)=フロア以上のみ／
    学習=フロア以上のきれいな点。
  - b は自由フィット（[B_MIN,B_MAX] でクランプ、外れたら b=1 フォールバック）。
    b の 1 からのずれ・r の低下そのものも健全性指標として監視。
  - ロバスト推定は numpy のみ（scipy 非依存。実機に scipy が無くても動く）。

レガシー検知（Anomaly_Detection_112p.py）には一切触れない。CCG パイプラインと並列の独立処理。

使い方:
  # 学習（直近の正常期間でモデルを作る。粗い間隔で十分）
  python ip_judge.py learn  LER 20260601000000 20260615000000 [--interval 300] [--out ip_models.json]
  # 判定（直近窓を評価）
  python ip_judge.py judge  LER 20260615000000 20260615060000 [--interval 60] [--models ip_models.json]
  # 合成データで層・フィットを検証（kblogrd 不要・実機でも実行可）
  python ip_judge.py selftest
"""

import json
import os
import sys

import numpy as np

import ip_pv
import ip_fetch
import ccg_fetch
import ccg_pv
import beam_fetch
import ip_observe   # フロア定義を一元化（PRES_FLOOR / floor_for / CUR_FLOOR_4U）

NODATA = ip_fetch.NODATA
_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_FILE = os.path.join(_HERE, "ip_models.json")

# ── 設定（しきい値は初期値。場所/運用で調整可。すべてここに集約）──────────────
CONFIG = {
    # --- 学習（ロバストフィット）---
    "min_fit_pts":     8,      # フィットに必要な最小点数（フロア以上）
    "min_logp_range":  0.30,   # logP の最小レンジ[dex]。狭いと b 不定 → b=1 フォールバック
    "b_min":           0.5,    # b のクランプ下限
    "b_max":           1.5,    # b のクランプ上限
    "sigma_floor":     0.08,   # σ の下限[dex]（残差が小さすぎる時の z 暴走を防ぐ, ~20%）
    "ts_max_pts":      1200,   # Theil–Sen 前のダウンサンプル上限（O(n^2) 抑制）
    "r_healthy_min":   0.60,   # 学習窓が「健全」と認める最小 r_pi（未満は low trust）

    # --- Layer 0a: 無ビーム放電 ---
    #   グローバル定数だけだと、平常から moderate な電流のポンプを片端から拾う（偽陽性多）。
    #   実データ較正: 「学習 p95 から l0a_margin_dex 以上の上振れ」OR「絶対ハードシーリング
    #   abs_hard 超過」のときだけ発火。前者は各ポンプ自身の平常との比較（D12_L23/D01_H14 の
    #   ような自己平常からの急上昇を捕捉）、後者は学習窓で既に壊れていたポンプ（D01_L02 など）
    #   も絶対水準で拾うバックストップ。
    "beam_low_ma":     10.0,   # これ未満を「無/低ビーム」とみなす[mA]
    "l0a_margin_dex":  0.7,    # 無ビーム電流が学習 p95 をこれ[dex]超えたら発火（~5倍）
    "abs_hard": {              # 履歴に依らずこの絶対電流[A]超過なら発火（壊れたまま学習対策）
        ip_pv.SUPPLY_KEK: 1e-5,
        ip_pv.SUPPLY_4U:  1e-5,
    },
    "cur_high": {              # モデル未学習時のフォールバック閾値[A]
        ip_pv.SUPPLY_KEK: 5e-6,
        ip_pv.SUPPLY_4U:  1e-6,
    },
    "l0_min_pts":      5,      # Layer 0a 判定に必要な無ビーム点数
    "l0_frac":         0.5,    # 無ビーム点のうち高電流の割合がこれ以上で放電（持続型）
    "l0_spike_min_pts": 5,     # ハードシーリング超過点がこの数以上で放電（過渡スパイク型,
                               # D01_H14 のようなスパイク→HVオフを中央値に薄められず捕捉）

    # --- Layer 0b: 学習した正常電流バンドからの上振れ（KEK の 2 桁ジャンプ用）---
    # 学習窓の log10(I) パーセンタイル p95 を「正常上端」とし、判定窓の中央電流が
    # そこから何 dex 超えるかで判定。電源固有定数でなくポンプごとに自動較正される。
    "over_ceiling_dex":    0.7,   # p95 をこれ[dex]超えたら上振れ（~5倍）→ sev2
    "over_ceiling_dex_hi": 1.3,   # これ[dex]超えたら強い放電（~20倍）→ sev3
    "l0b_min_pts":     8,

    # --- Layer 1: z スコア ---
    "z_hi":            4.0,    # z_med がこれ以上で上振れ（放電疑い）
    "z_lo":           -4.0,    # z_med がこれ以下で下振れ（排気劣化）
    "l1_min_pts":      5,

    # --- Layer 2: デカップリング ---
    #   decoupled_pi は I-P 相関が本来安定な 4U のみ（KEK は相関が偶発的で誤検知源）。
    #   decoupled_beam は判定窓のビーム動的範囲が十分なときのみ（低変動窓のノイズ除去）。
    #   デカップリング単独は弱い証拠なので集約では sev1 止まり（電流上振れと併発で sev3）。
    "r_decouple":      0.30,   # 窓内 r がこれ未満で「相関崩壊」
    "decouple_min_pts": 8,     # デカップリング判定に必要な点数
    "min_beam_logrange": 0.30, # decoupled_beam に必要なビーム電流の最小 log レンジ[dex]

    # --- 急性/慢性の仕分け ---
    #   検知された異常を「最近 自分の平常から逸脱した(acute)」と「絶対値は高いが前から
    #   その水準(chronic)」に分類する。逸脱量は無ビーム電流の中央値逸脱(持続シフト)と
    #   95%点逸脱(過渡スパイク)の大きい方を、学習 p95 基準で測る。
    "acute_margin_dex": 0.7,   # 学習 p95 からこれ[dex]以上の逸脱で acute（最近の異常）

    # --- learn --robust（反復頑健化）: 中央値+MAD sigma-clip で学習窓の異常エポック除去 ---
    "robust_kmad":     4.0,    # 中央値から K·MAD を超える点を異常として除外
    "robust_min_dex":  0.7,    # ただし最低この[dex]より内側は決して刈らない（健全変動を守る）

    # --- ビーム軸 格下げ（無ビーム挙動で「ビーム由来＝正常寄り」を sev3 から落とす）---
    #   相関の正負では正常/故障を切れないため、無ビーム時の電流の振る舞いで切り分ける。
    #   健全＝電流はガス負荷(ほぼビーム由来)に追従 → ビームを落とすと電流も落ちる(drop 大)、
    #          かつ無ビーム電流は学習した無ビーム平常を超えない(nb_excess ≤ 0)。
    #   放電破損＝ガス負荷と無関係に高い → 無ビームでも下がらない(drop 小)、平常超過(nb_excess 大)。
    #   実機調査(ip_beam_survey, LER 2026-06)で D10_IP_L09 が典型的ビーム由来偽陽性、
    #   D03_IP_L09/D10_IP_L04 は無ビームでも高い本物の放電と確認。前者だけを落とす保守設計。
    #   安全網: nb_excess ≤ 0 を必須にして「無ビームでも平常超過」の本物は決して落とさない。
    #           さらに med(電流) が abs_hard 超なら高すぎるので格下げしない。
    "beam_downgrade_enable":  True,   # ビーム由来 sev3/2 の格下げを行うか
    "beam_drop_min_dex":      0.30,   # 無ビーム中央値がビーム有り中央値からこれ[dex]以上 下がる
    "beam_nb_excess_max_dex": 0.0,    # かつ無ビーム電流が学習無ビーム平常をこれ[dex]以下（超えない）
    "beam_r_min":             0.50,   # かつビーム-電流相関(log)がこれ以上（ビーム追従の裏付け）
    "beam_downgrade_to":      1,      # 条件成立時に下げる severity（1=要観察。表示ゲート未満）
}


# ───────────────────────── ロバスト推定（numpy のみ）─────────────────────────

def _theil_sen(x, y, max_pts):
    """Theil–Sen 推定。返り値 (slope, intercept) or (None, None)。
    点が多いと O(n^2) なので等間隔ダウンサンプルしてから全ペア中央値を取る。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return None, None
    if n > max_pts:
        idx = np.linspace(0, n - 1, max_pts).astype(int)
        x, y = x[idx], y[idx]
        n = len(x)
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return None, None
    slopes = (y[j][ok] - y[i][ok]) / dx[ok]
    b = float(np.median(slopes))
    a = float(np.median(y - b * x))      # 切片（log 空間の log_a）
    return b, a


def _mad_sigma(resid):
    """残差の MAD×1.4826（正規分布で σ に一致）。"""
    resid = np.asarray(resid, dtype=float)
    if resid.size == 0:
        return 0.0
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return float(1.4826 * mad)


def _logcorr(a, b):
    """log は呼び出し側で取る前提。a,b の相関係数。点が少/分散ゼロなら None。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def robust_powerlaw_fit(pres, cur, p_floor, cfg=CONFIG):
    """フロア以上の点で I=a·P^b をロバスト推定する。

    返り値 dict or None:
      {a, b, b_raw, log_a, sigma, r, n_used, p_floor, logp_range,
       fallback_b1, trust}
    trust: 学習窓が健全（r≥r_healthy_min かつレンジ十分）か。False なら正常モデルとして
           信用しない（＝そのポンプ自体が異常の可能性）。
    """
    pres = np.asarray(pres, dtype=float)
    cur = np.asarray(cur, dtype=float)
    mask = pres >= p_floor
    pf, cf = pres[mask], cur[mask]
    if len(pf) < cfg["min_fit_pts"]:
        return None
    lp, lc = np.log10(pf), np.log10(cf)
    logp_range = float(lp.max() - lp.min())
    r = _logcorr(lp, lc)

    b_raw, log_a = _theil_sen(lp, lc, cfg["ts_max_pts"])
    if b_raw is None:
        return None

    # b のクランプ／フォールバック判定
    fallback = False
    narrow = logp_range < cfg["min_logp_range"]
    if narrow or b_raw < cfg["b_min"] or b_raw > cfg["b_max"]:
        fallback = True
        b = 1.0
        log_a = float(np.median(lc - b * lp))   # b=1 で切片を取り直す
    else:
        b = b_raw

    a = 10.0 ** log_a
    resid = lc - (b * lp + log_a)
    sigma = max(_mad_sigma(resid), cfg["sigma_floor"])

    trust = (r is not None and r >= cfg["r_healthy_min"]
             and not narrow and len(pf) >= cfg["min_fit_pts"])

    return {"a": float(a), "b": float(b), "b_raw": float(b_raw),
            "log_a": float(log_a), "sigma": float(sigma),
            "r": (float(r) if r is not None else None),
            "n_used": int(len(pf)), "p_floor": float(p_floor),
            "logp_range": logp_range, "fallback_b1": bool(fallback),
            "trust": bool(trust)}


# ───────────────────────── データ整列（P, I, beam）─────────────────────────

def _valid(v):
    return v is not None and v > NODATA


def _valid_beam(v):
    # beam は 0 mA（無ビーム）が正当な値。圧力用の NODATA フロアは適用せず、
    # 有限かつ 0 以上なら有効とする（欠測は None / 負 / nan のみ）。
    return v is not None and np.isfinite(v) and v >= 0.0


def align_pib(ccg_series, ip_series, beam_map):
    """同時刻で IP 電流が有効な点を取り出し、P（無ければ None）と beam（無ければ None）を添える。
    返り値: dict of np.array {t, I, P, beam}（I は必ず有効。P/beam は欠測 np.nan）。"""
    ccg_map = dict(ccg_series or [])
    ts, I, P, B = [], [], [], []
    for t, c in ip_series or []:
        if not _valid(c):
            continue
        p = ccg_map.get(t)
        bm = beam_map.get(t)
        ts.append(t)
        I.append(c)
        P.append(p if _valid(p) else np.nan)
        B.append(bm if _valid_beam(bm) else np.nan)
    return {"t": ts,
            "I": np.array(I, dtype=float),
            "P": np.array(P, dtype=float),
            "beam": np.array(B, dtype=float)}


# ───────────────────────── 学習 ─────────────────────────

def _build_pairs(ring):
    """ring の全イオンポンプについて (ip_pv, ccg_pv or None, supply, section) を返す。"""
    ip_recs = ip_pv.load(ip_pv.csv_path(ring))
    import csv
    with open(ccg_pv.csv_path(ring), encoding="utf-8-sig", newline="") as f:
        ccg_set = set(r[0].strip() for r in csv.reader(f) if r and r[0].strip())
    out = []
    for r in ip_recs:
        c = ip_pv.paired_ccg(r["pv"])
        c = c if (c and c in ccg_set) else None
        out.append((r["pv"], c, r["supply"], r["section"]))
    return out


def _fetch_all(ring, start, end, interval_sec):
    """IP / CCG / beam をまとめて取得（kblogrd 必要）。"""
    ip_data = ip_fetch.fetch_history(ring, start, end, interval_sec=interval_sec)
    ccg_data = ccg_fetch.fetch_history(ring, start, end, interval_sec=interval_sec)
    beam = beam_fetch.fetch_series(ring, start, end, interval_sec=interval_sec)
    beam_map = {t: v for t, v in beam}
    return ip_data, ccg_data, beam_map


def _ip_series_of(ip_data, ip_name):
    v = ip_data.get(ip_name)
    return v.get("series", []) if isinstance(v, dict) else (v or [])


def _collect_d(fetched_list, ip_name, ccg_name):
    """複数窓ぶんの (ip,ccg,beam) から、1ポンプの P/I/beam を時刻ごとに整列して連結。
    無ビーム窓・ビームあり窓を合算でき、nb バンドと cur バンドを別窓から学べる。"""
    ts = []; Is = []; Ps = []; Bs = []
    for ip_data, ccg_data, beam_map in fetched_list:
        ip_series = _ip_series_of(ip_data, ip_name)
        ccg_series = (ccg_data.get(ccg_name, {}).get("series", []) if ccg_name else [])
        d = align_pib(ccg_series, ip_series, beam_map)
        if len(d["t"]):
            ts += list(d["t"]); Is.append(d["I"]); Ps.append(d["P"]); Bs.append(d["beam"])
    if not ts:
        return {"t": [], "I": np.array([]), "P": np.array([]), "beam": np.array([])}
    return {"t": ts, "I": np.concatenate(Is),
            "P": np.concatenate(Ps), "beam": np.concatenate(Bs)}


def learn(ring, start, end, interval_sec=300, out_path=MODELS_FILE, cfg=CONFIG,
          robust=0, windows=None, _fetched=None):
    """正常窓から各ポンプの I=a·P^b モデルと健全性指標を学習し、JSON に保存（更新）。
    robust>0 で反復頑健化（学習窓に紛れた異常エポックを落として再学習）。
    電流バンドはビーム有り(L0b)/無し(L0a)で分けて学習する。
    windows=[(s,e),...] を渡すと**複数期間を合算**して1つのモデルにする（無ビーム窓と
    ビームあり窓を合わせて学習する用途。start/end は windows[0] として扱う）。
    _fetched=(ip_data, ccg_data, beam_map) を渡すとそれを使う（テスト用・単一窓）。
    既存 JSON は読み込んでマージし、low-trust のポンプは前の良モデルを保持する。"""
    pairs = _build_pairs(ring)
    if _fetched is not None:
        fetched_list = [_fetched]
        win_desc = [[start, end]]
    else:
        if not windows:
            windows = [(start, end)]
        fetched_list = [_fetch_all(ring, s, e, interval_sec) for (s, e) in windows]
        win_desc = [[s, e] for (s, e) in windows]

    store = _load_models(out_path)
    rd = store.setdefault(ring, {})
    n_trust = n_low = n_skip = 0

    for ip_name, ccg_name, supply, section in pairs:
        d = _collect_d(fetched_list, ip_name, ccg_name)
        if len(d["t"]) == 0:
            n_skip += 1
            continue

        # 正常電流バンドは常に学習する（I-P が相関しない KEK でも電流の絶対水準は使える）
        band = _learn_bands(d, cfg, robust_iters=robust)
        if band is None:
            n_skip += 1
            continue

        model = {"supply": supply, "section": section,
                 "trained_start": win_desc[0][0], "trained_end": win_desc[-1][1],
                 "trained_windows": win_desc,
                 "r_beam": _window_beam_corr(d, cfg)}
        model.update(band)

        # I-P モデルは CCG ペアがあってフロア以上の点が足りるときだけ付与
        fit = None
        if ccg_name is not None:
            p_floor = ip_observe.floor_for(supply)
            has_p = ~np.isnan(d["P"])
            fit = robust_powerlaw_fit(d["P"][has_p], d["I"][has_p], p_floor, cfg)
        if fit is not None:
            model.update({k: fit[k] for k in
                          ("a", "b", "b_raw", "log_a", "sigma", "r", "n_used",
                           "p_floor", "logp_range", "fallback_b1")})
            model["ip_trust"] = fit["trust"]    # I-P モデルが信用できるか
        else:
            model["ip_trust"] = False

        prev = rd.get(ip_name)
        ip_trusted = model["ip_trust"]

        # 学習窓汚染ガード:
        #  (1) I-P モデル: low-trust なら前の良 I-P モデルを保持（放電を正常学習しない）
        #  (2) 電流バンド: 前回より p95 が大きく跳ねたら、その跳ね自体が異常 → 前の band を保持
        if prev:
            if (not ip_trusted) and prev.get("ip_trust"):
                for k in ("a", "b", "b_raw", "log_a", "sigma", "r", "n_used",
                          "p_floor", "logp_range", "fallback_b1", "ip_trust"):
                    if k in prev:
                        model[k] = prev[k]
            if "cur_log_p95" in prev and \
               model["cur_log_p95"] - prev["cur_log_p95"] > cfg["over_ceiling_dex"]:
                model["cur_log_p50"] = prev["cur_log_p50"]
                model["cur_log_p95"] = prev["cur_log_p95"]
                model["band_jump_held"] = True

        rd[ip_name] = model
        if ip_trusted:
            n_trust += 1
        else:
            n_low += 1

    store["_meta"] = {"cfg": cfg}
    _save_models(store, out_path)
    print("[learn %s] ip_trust=%d  ip_low_trust=%d  skip=%d  → %s"
          % (ring, n_trust, n_low, n_skip, out_path))
    return out_path, store


def _band_from(I, cfg, robust_iters=0):
    """電流 I の log10 パーセンタイル帯（p50/p95）。robust_iters>0 で反復頑健化:
    中央値＋MAD 基準の sigma-clip（中央値から K·MAD かつ最低 robust_min_dex を超える点＝
    学習窓に紛れた異常エポックを落として再計算）。中央値・MAD は最大50%汚染まで頑健なので、
    p95 が既に汚染に乗っているケースでも除去できる。健全なら何も落ちずバンドは不変。"""
    I = np.asarray(I, dtype=float)
    li = np.log10(I[I > 0])
    if len(li) < cfg["l0b_min_pts"]:
        return None
    used = 0
    for _ in range(int(robust_iters)):
        med = float(np.median(li))
        mad = float(np.median(np.abs(li - med))) * 1.4826
        margin = max(cfg["robust_kmad"] * mad, cfg["robust_min_dex"])
        keep = li <= med + margin
        if keep.all() or int(keep.sum()) < cfg["l0b_min_pts"]:
            break
        li = li[keep]; used += 1
    p50 = float(np.percentile(li, 50)); p95 = float(np.percentile(li, 95))
    return {"p50": p50, "p95": p95, "n": int(len(li)), "robust_iters_used": used}


def _learn_bands(d, cfg, robust_iters=0):
    """電流バンドをビーム有り/無しで分けて学習する。
      cur_log_p50/p95 … L0b 用（運転中＝ビーム有りの平常電流）。ビーム有り点で作る。
                        ビーム有り点が足りなければ全点で作る（band_basis で区別）。
      nb_log_p50/p95  … L0a 用（無ビームの平常電流）。無ビーム点が足りるときだけ付与。
    点不足（cur が作れない）なら None。"""
    beam = d["beam"]; I = d["I"]
    on = (~np.isnan(beam)) & (beam > cfg["beam_low_ma"]) & (I > 0)
    off = (~np.isnan(beam)) & (beam <= cfg["beam_low_ma"]) & (I > 0)

    if int(on.sum()) >= cfg["l0b_min_pts"]:
        cur = _band_from(I[on], cfg, robust_iters); basis = "beam_on"
    else:
        cur = _band_from(I[I > 0], cfg, robust_iters); basis = "all"   # ビーム少なめのポンプ
    if cur is None:
        return None
    out = {"cur_log_p50": cur["p50"], "cur_log_p95": cur["p95"],
           "cur_n": cur["n"], "band_basis": basis}
    if robust_iters:
        out["band_robust_iters"] = int(robust_iters)
        out["band_robust_used"] = cur["robust_iters_used"]

    nb = _band_from(I[off], cfg, robust_iters) if int(off.sum()) >= cfg["l0_min_pts"] else None
    if nb is not None:
        out["nb_log_p50"] = nb["p50"]; out["nb_log_p95"] = nb["p95"]; out["nb_n"] = nb["n"]
    return out


def _window_beam_corr(d, cfg):
    """窓内の r(logI, log beam)（beam>low かつ I,beam 有効な点）。点不足なら None。"""
    m = (~np.isnan(d["beam"])) & (d["beam"] > cfg["beam_low_ma"]) & (d["I"] > 0)
    if m.sum() < cfg["decouple_min_pts"]:
        return None
    return _logcorr(np.log10(d["beam"][m]), np.log10(d["I"][m]))


# ───────────────────────── 判定 ─────────────────────────

def judge(ring, start, end, interval_sec=60, models_path=MODELS_FILE, cfg=CONFIG,
          _fetched=None):
    """直近窓の各ポンプを 3 層で評価。返り値: judge state dict。"""
    pairs = _build_pairs(ring)
    models = _load_models(models_path).get(ring, {})
    if _fetched is not None:
        ip_data, ccg_data, beam_map = _fetched
    else:
        ip_data, ccg_data, beam_map = _fetch_all(ring, start, end, interval_sec)

    pumps = []
    for ip_name, ccg_name, supply, section in pairs:
        rec = ip_data.get(ip_name)
        ip_series = rec.get("series", []) if isinstance(rec, dict) else (rec or [])
        ccg_series = (ccg_data.get(ccg_name, {}).get("series", []) if ccg_name else [])
        d = align_pib(ccg_series, ip_series, beam_map)
        if len(d["t"]) == 0:
            continue
        model = models.get(ip_name)
        res = judge_pump(d, supply, model, cfg)
        res.update({"pv": ip_name, "ccg": ccg_name, "supply": supply,
                    "section": section})
        pumps.append(res)

    _krank = {"acute": 0, "unknown": 1, "chronic": 2, None: 3}
    pumps.sort(key=lambda p: (-p["severity"], _krank.get(p.get("kind"), 3), p["pv"]))
    summary = {"n": len(pumps),
               "sev3": sum(1 for p in pumps if p["severity"] == 3),
               "acute": sum(1 for p in pumps if p.get("kind") == "acute"),
               "chronic": sum(1 for p in pumps if p.get("kind") == "chronic"),
               "sev2": sum(1 for p in pumps if p["severity"] == 2),
               "sev1": sum(1 for p in pumps if p["severity"] == 1)}
    return {"device_type": "IonPumpJudge", "ring": ring,
            "window": {"start": start, "end": end},
            "summary": summary, "pumps": pumps}


def judge_pump(d, supply, model, cfg):
    """1 ポンプの 3 層判定。返り値: {severity, reason, layers{...}, metrics{...}}。"""
    I, P, beam = d["I"], d["P"], d["beam"]
    cur_high = cfg["cur_high"].get(supply, cfg["cur_high"][ip_pv.SUPPLY_KEK])
    p_floor = ip_observe.floor_for(supply)

    layers = {}

    # ── Layer 0a: 無ビーム放電（ポンプ相対＋絶対ハードシーリング）──
    nb = (~np.isnan(beam)) & (beam <= cfg["beam_low_ma"])
    l0 = {"n_nobeam": int(nb.sum()), "fired": False, "frac_high": None}
    if nb.sum() >= cfg["l0_min_pts"]:
        # 相対閾値: 無ビーム平常(nb_log_p95)を優先、無ければ L0b バンド(cur_log_p95)
        ref_p95 = None
        if model:
            ref_p95 = model.get("nb_log_p95", model.get("cur_log_p95"))
        if ref_p95 is not None:
            thr_rel = 10.0 ** (ref_p95 + cfg["l0a_margin_dex"])
        else:
            thr_rel = float("inf")
        thr_abs = cfg["abs_hard"].get(supply, cfg["abs_hard"][ip_pv.SUPPLY_KEK])
        # モデルが無いときは旧フォールバック(cur_high)も下限として併用
        if not (model and ("cur_log_p95" in model)):
            thr_abs = min(thr_abs, cur_high)
        thr = min(thr_rel, thr_abs)      # 相対 OR 絶対 で発火 ⇔ min を超える
        hi = I[nb] >= thr
        frac = float(hi.mean())
        l0["frac_high"] = frac
        l0["med_cur_nobeam"] = float(np.median(I[nb]))
        l0["thr"] = float(thr)
        # 過渡スパイク: ハードシーリング(1e-5級)超えの点数。中央値/割合では薄まる
        # スパイク→HVオフ型(D01_H14)を、超過点の「数」で捕捉する。
        hard = cfg["abs_hard"].get(supply, cfg["abs_hard"][ip_pv.SUPPLY_KEK])
        n_exc = int((I[nb] >= hard).sum())
        l0["n_excursion"] = n_exc
        spike = n_exc >= cfg["l0_spike_min_pts"]
        l0["fired"] = (frac >= cfg["l0_frac"]) or spike
    layers["L0a_nobeam"] = l0

    # ── Layer 0b: 学習正常バンドからの上振れ（KEK のビーム連続時の 2 桁ジャンプ用）──
    #   ビームの有無を問わず、判定窓の中央電流が学習 p95 を何 dex 超えるかで見る。
    #   モデルが無ければ電源固有定数 cur_high にフォールバック。
    l0b = {"fired_over": False, "strong": False, "excess_dex": None,
           "basis": None}
    pos = I[I > 0]
    if len(pos) >= cfg["l0b_min_pts"]:
        med_log = float(np.median(np.log10(pos)))
        if model and ("cur_log_p95" in model):
            ref = model["cur_log_p95"]; l0b["basis"] = "learned_p95"
        else:
            ref = np.log10(cur_high); l0b["basis"] = "global_cur_high"
        excess = med_log - ref
        l0b["excess_dex"] = float(excess)
        l0b["fired_over"] = bool(excess >= cfg["over_ceiling_dex"])
        l0b["strong"] = bool(excess >= cfg["over_ceiling_dex_hi"])
    layers["L0b_overband"] = l0b

    # ── Layer 1: I-P 整合（z）。I-P モデルが要る ──
    l1 = {"fired_over": False, "fired_under": False, "z_med": None,
          "z_med_floor": None, "have_model": bool(model and model.get("a") is not None)}
    if model and model.get("a") is not None:
        a, b, sigma = model["a"], model["b"], model["sigma"]
        # 予測ベースラインのクランプ用 電流フロア。4U はマニュアル仕様 10 nA。
        # KEK は明確な仕様が無いので保守的に 1 nA（ほぼ無クランプ＝予測そのまま）。
        i_floor = ip_observe.CUR_FLOOR_4U if supply == ip_pv.SUPPLY_4U else 1e-9
        have_p = (~np.isnan(P)) & (P > 0) & (I > 0)
        if have_p.sum() >= cfg["l1_min_pts"]:
            Pv, Iv = P[have_p], I[have_p]
            pred = np.maximum(a * Pv ** b, i_floor)        # 予測フロアクランプ
            z = (np.log10(Iv) - np.log10(pred)) / sigma
            l1["z_med"] = float(np.median(z))
            l1["fired_over"] = l1["z_med"] >= cfg["z_hi"]   # 上振れ=放電（全圧力域）
            # 下振れ（劣化）はフロア以上のみで・I-P モデルが信用できるときのみ
            fl = Pv >= p_floor
            if model.get("ip_trust") and fl.sum() >= cfg["l1_min_pts"]:
                zf = z[fl]
                l1["z_med_floor"] = float(np.median(zf))
                l1["fired_under"] = l1["z_med_floor"] <= cfg["z_lo"]
    layers["L1_zscore"] = l1

    # ── Layer 2: デカップリング（「以前は相関していた」ものの崩壊のみ）──
    #   健全 KEK は元々 I-P 相関が無いので、trust I-P モデルが無ければ圧力相関崩壊は
    #   主張しない（誤検知防止）。ビーム相関も、健全時に相関していた証拠があるときのみ。
    l2 = {"r_pi": None, "r_beam": None, "decoupled_pi": False,
          "decoupled_beam": False}
    mp = (~np.isnan(P)) & (P > 0) & (I > 0)
    if mp.sum() >= cfg["decouple_min_pts"]:
        lp = np.log10(P[mp]); li = np.log10(I[mp])
        if (lp.max() - lp.min()) >= cfg["min_logp_range"]:
            r_pi = _logcorr(lp, li)
            l2["r_pi"] = r_pi
            # I-P 相関が本来安定なのは 4U のみ。KEK は偶発的相関で誤検知するため除外。
            had_pi = bool(model and model.get("ip_trust") and supply == ip_pv.SUPPLY_4U)
            if r_pi is not None and had_pi and r_pi < cfg["r_decouple"]:
                l2["decoupled_pi"] = True
    mb = (~np.isnan(beam)) & (beam > cfg["beam_low_ma"]) & (I > 0)
    if mb.sum() >= cfg["decouple_min_pts"]:
        lb = np.log10(beam[mb]); li = np.log10(I[mb])
        # ビーム動的範囲が十分なときのみ（低変動窓だと相関が不安定でノイズ発火する）
        if (lb.max() - lb.min()) >= cfg["min_beam_logrange"]:
            r_b = _logcorr(lb, li)
            l2["r_beam"] = r_b
            had_beam = bool(model and model.get("r_beam") is not None
                            and model["r_beam"] >= cfg["r_healthy_min"])
            if r_b is not None and had_beam and r_b < cfg["r_decouple"]:
                l2["decoupled_beam"] = True
    layers["L2_decouple"] = l2

    # ── 集約 ──
    sev, reason = _aggregate(l0, l0b, l1, l2)

    # ── ビーム軸 格下げ（無ビーム挙動で「ビーム由来＝正常寄り」を落とす）──
    #   相関ではなく直接の挙動で切り分ける（ip_beam_survey と同じ指標）:
    #     drop_dex     = log10(med_on) - log10(med_nb)   … ビームを落として電流が下がる量
    #     nb_excess_dex= log10(med_nb) - nb_log_p95       … 無ビーム電流の平常超過量
    #   健全(ビーム由来): drop 大 & nb_excess ≤ 0 & ビーム追従 → sev を beam_downgrade_to へ。
    #   放電破損: 無ビームでも高い(nb_excess > 0) ので必須条件 nb_excess ≤ 0 で必ず保護される。
    def _medpos(x):
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.median(x)) if len(x) else None
    on = (~np.isnan(beam)) & (beam > cfg["beam_low_ma"])
    med_on = _medpos(I[on]); med_nb = _medpos(I[nb])
    nb_p95 = model.get("nb_log_p95", model.get("cur_log_p95")) if model else None
    drop_dex = (np.log10(med_on) - np.log10(med_nb)) if (med_on and med_nb) else None
    nb_excess_dex = (np.log10(med_nb) - nb_p95) if (med_nb and nb_p95 is not None) else None
    r_beam_axis = l2["r_beam"]          # ビーム有り点の log(I)-log(beam) 相関（L2 で算出済）
    abs_hard = cfg["abs_hard"].get(supply, cfg["abs_hard"][ip_pv.SUPPLY_KEK])
    beam_driven = bool(
        cfg.get("beam_downgrade_enable", True)
        and sev >= 2
        and drop_dex is not None and drop_dex >= cfg["beam_drop_min_dex"]
        and nb_excess_dex is not None and nb_excess_dex <= cfg["beam_nb_excess_max_dex"]
        and r_beam_axis is not None and r_beam_axis >= cfg["beam_r_min"]
        and (med_on is None or med_on < abs_hard)     # 絶対値が高すぎる物は落とさない
        and (med_nb is None or med_nb < abs_hard)
        and int(on.sum()) >= cfg["l0_min_pts"]        # ビーム有/無とも十分な点数があるとき
        and int(nb.sum()) >= cfg["l0_min_pts"]
    )
    sev_raw, reason_raw = sev, reason
    if beam_driven:
        sev = min(sev, int(cfg.get("beam_downgrade_to", 1)))
        reason = "beam_driven_current"

    # ── 急性/慢性の仕分け（学習 p95 からの逸脱量で判定）──
    #   acute  = 最近 自分の平常から逸脱（中央値シフト or 過渡スパイク）→ 緊急
    #   chronic= 絶対値は高いが自分の平常並み（前からその水準）→ 要観察
    #   unknown= 学習モデルが無く平常が不明（学習窓を別途用意すると分類できる）
    kind, dev = None, None
    p95 = model.get("cur_log_p95") if model else None
    if p95 is not None:
        base = I[nb] if int(nb.sum()) >= cfg["l0_min_pts"] else I
        base = base[base > 0]
        if len(base):
            dev_med = float(np.log10(np.median(base)) - p95)
            dev_peak = float(np.log10(np.percentile(base, 95)) - p95)
            dev = max(dev_med, dev_peak)
    if sev >= 2:
        if dev is None:
            kind = "unknown"
        elif dev >= cfg["acute_margin_dex"]:
            kind = "acute"
        else:
            kind = "chronic"

    metrics = {"med_cur": float(np.median(I)) if len(I) else None,
               "n_pts": int(len(I)),
               "drop_dex": (float(drop_dex) if drop_dex is not None else None),
               "nb_excess_dex": (float(nb_excess_dex) if nb_excess_dex is not None else None),
               "r_beam_axis": (float(r_beam_axis) if r_beam_axis is not None else None),
               "med_on": med_on, "med_nb": med_nb}
    return {"severity": sev, "reason": reason, "kind": kind,
            "deviation_dex": dev, "layers": layers,
            "beam_driven": beam_driven, "severity_raw": int(sev_raw),
            "reason_raw": reason_raw, "metrics": metrics}


def _aggregate(l0, l0b, l1, l2):
    """層の結果を severity(0-3) と理由に集約。放電（フィードスルー疑い）を最優先。"""
    over = l1["fired_over"] or l0b["fired_over"]
    decoup = l2["decoupled_pi"] or l2["decoupled_beam"]
    # severity 3: 放電が強く支持される（最重要・事前検知の主目的）
    #   無ビームで高電流 / 上振れ＋デカップリング / 学習バンドを大幅超過
    if l0["fired"] or (over and decoup) or l0b["strong"]:
        return 3, "feedthrough_discharge_suspect"
    # severity 2: 電流の上振れ（放電寄り）
    if over:
        return 2, "over_current"
    # severity 1: 弱い証拠（デカップリング単独 / 排気劣化）は要観察止まり
    if decoup:
        return 1, "decoupled"
    if l1["fired_under"]:
        return 1, "pumping_degradation"
    return 0, "normal"


# ───────────────────────── 永続化 ─────────────────────────

def _load_models(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _json_default(o):
    """json が書けない numpy スカラーを Python 型へ（np.bool_ は bool のサブクラスでない）。"""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError("Object of type %s is not JSON serializable" % type(o).__name__)


def _save_models(store, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1, default=_json_default)


# ───────────────────────── 合成データ自己検証 ─────────────────────────

def _synth(kind, n=300, seed=0):
    """合成 (P, I, beam) を作る。
    kind: healthy / discharge / degrade / discharge_beam / healthy_kek。
    返り値は align_pib 互換の dict（t は連番文字列）。"""
    rng = np.random.RandomState(seed)
    if kind == "discharge_beam":
        beam = np.linspace(200, 1000, n)            # ビーム連続（無ビーム区間なし）
    else:
        beam = np.concatenate([np.zeros(n // 6),    # 無ビーム区間あり
                               np.linspace(50, 1000, n - n // 6)])
    beam = np.clip(beam + rng.normal(0, 5, n), 0, None)
    base = 1.5e-8 + (beam / 1000.0) * 3e-7          # 圧力はビームに追従
    P = base * 10 ** rng.normal(0, 0.05, n)
    a_true, b_true = 0.5, 1.0
    if kind == "healthy":
        I = a_true * P ** b_true * 10 ** rng.normal(0, 0.06, n)
    elif kind == "healthy_kek":
        # 健全 KEK: 低圧で電流は ~1e-6 A に張り付き、P/beam と相関しない（r≈0）
        I = 1e-6 * 10 ** rng.normal(0, 0.15, n)
    elif kind in ("discharge", "discharge_beam"):
        I = 1e-3 * 10 ** rng.normal(0, 0.1, n)      # 高値張り付き（D12/D07 signature）
    elif kind == "degrade":
        I = (a_true / 10.0) * P ** b_true * 10 ** rng.normal(0, 0.06, n)
    elif kind == "spike_off":
        # 過渡型: 平常→短いスパイク(1e-4)→HVオフ(低値)。無ビーム全域。D01_H14 相当。
        beam = np.zeros(n)
        base = np.full(n, 1.5e-8)
        P = base * 10 ** rng.normal(0, 0.05, n)
        I = np.full(n, 8e-7) * 10 ** rng.normal(0, 0.1, n)      # 平常 ~1µA
        s0 = int(n * 0.45); s1 = int(n * 0.60)
        I[s0:s1] = 1e-4 * 10 ** rng.normal(0, 0.1, s1 - s0)     # スパイク
        I[s1:] = 1e-8 * 10 ** rng.normal(0, 0.1, n - s1)        # HV オフ後
        t = ["%06d" % i for i in range(n)]
        return {"t": t, "I": I, "P": P, "beam": beam}
    else:
        raise ValueError(kind)
    t = ["%06d" % i for i in range(n)]
    return {"t": t, "I": I, "P": P, "beam": beam}


def _model_from(d, supply, cfg):
    """学習と同じ手順で d から model dict を作る（selftest 用の簡易版）。"""
    band = _learn_bands(d, cfg, robust_iters=0)
    m = {"supply": supply, "r_beam": _window_beam_corr(d, cfg)}
    m.update(band or {})
    has_p = ~np.isnan(d["P"])
    fit = robust_powerlaw_fit(d["P"][has_p], d["I"][has_p],
                              ip_observe.floor_for(supply), cfg)
    if fit:
        m.update({k: fit[k] for k in ("a", "b", "log_a", "sigma", "r",
                                      "logp_range", "fallback_b1")})
        m["ip_trust"] = fit["trust"]
    else:
        m["ip_trust"] = False
    return m


def selftest():
    """kblogrd 無しでフィット＋3層を検証する。実機でも実行可。"""
    cfg = CONFIG
    print("=== ip_judge selftest（合成データ）===")
    ok = True

    # 1) 健全 4U: ip_trust=True, b≈1, severity 0
    d = _synth("healthy", seed=1)
    hm = _model_from(d, ip_pv.SUPPLY_4U, cfg)
    res = judge_pump(d, ip_pv.SUPPLY_4U, hm, cfg)
    print("[healthy 4U] ip_trust=%s b=%.2f r=%s  → sev=%d %s (z_med=%s r_pi=%s)"
          % (hm["ip_trust"], hm["b"], _f(hm["r"]), res["severity"], res["reason"],
             _f(res["layers"]["L1_zscore"]["z_med"]),
             _f(res["layers"]["L2_decouple"]["r_pi"])))
    ok &= (hm["ip_trust"] and abs(hm["b"] - 1.0) < 0.25 and res["severity"] == 0)

    # 1b) 健全 KEK（I-P 無相関）: 誤検知しないこと（severity 0）
    dk = _synth("healthy_kek", seed=11)
    km = _model_from(dk, ip_pv.SUPPLY_KEK, cfg)
    res = judge_pump(dk, ip_pv.SUPPLY_KEK, km, cfg)
    print("[healthy KEK] ip_trust=%s  → sev=%d %s  (相関無くても誤検知しない)"
          % (km["ip_trust"], res["severity"], res["reason"]))
    ok &= (res["severity"] == 0)

    # 2) 放電（無ビーム区間あり）: 健全モデルに対して severity 3 / acute（平常から逸脱）
    dd = _synth("discharge", seed=2)
    res = judge_pump(dd, ip_pv.SUPPLY_4U, hm, cfg)
    L = res["layers"]
    print("[discharge 4U] sev=%d %s kind=%s dev=%s  L0a=%s L0b.strong=%s"
          % (res["severity"], res["reason"], res["kind"], _f(res["deviation_dex"]),
             L["L0a_nobeam"]["fired"], L["L0b_overband"]["strong"]))
    ok &= (res["severity"] == 3 and res["kind"] == "acute")

    # 2-chronic) 既に壊れた状態で学習したモデルに同じ放電 → chronic（前から高い）
    chronic_model = _model_from(_synth("discharge", seed=21), ip_pv.SUPPLY_4U, cfg)
    res = judge_pump(_synth("discharge", seed=22), ip_pv.SUPPLY_4U, chronic_model, cfg)
    print("[discharge/壊れた状態で学習] sev=%d kind=%s dev=%s（chronic 期待）"
          % (res["severity"], res["kind"], _f(res["deviation_dex"])))
    ok &= (res["severity"] == 3 and res["kind"] == "chronic")

    # 2b) モデル未学習の破損でも L0a で拾えること
    res0 = judge_pump(dd, ip_pv.SUPPLY_4U, None, cfg)
    print("[discharge/no-model] sev=%d %s" % (res0["severity"], res0["reason"]))
    ok &= (res0["severity"] == 3)

    # 2c) KEK 破損・ビーム連続（D12 相当）: L0a 不発でも L0b で severity 3
    dcb = _synth("discharge_beam", seed=5)
    km_h = _model_from(_synth("healthy_kek", seed=12), ip_pv.SUPPLY_KEK, cfg)
    res = judge_pump(dcb, ip_pv.SUPPLY_KEK, km_h, cfg)
    L = res["layers"]
    print("[KEK discharge/beam-on] sev=%d %s  L0a=%s L0b.strong=%s(excess=%s dex)"
          % (res["severity"], res["reason"], L["L0a_nobeam"]["fired"],
             L["L0b_overband"]["strong"], _f(L["L0b_overband"]["excess_dex"])))
    ok &= (res["severity"] == 3 and not L["L0a_nobeam"]["fired"])

    # 2d) 同上・モデル未学習でも global cur_high で拾えること
    res = judge_pump(dcb, ip_pv.SUPPLY_KEK, None, cfg)
    print("[KEK discharge/beam-on/no-model] sev=%d %s" % (res["severity"], res["reason"]))
    ok &= (res["severity"] == 3)

    # 3) 排気劣化（下振れ）: severity 1
    dg = _synth("degrade", seed=3)
    res = judge_pump(dg, ip_pv.SUPPLY_4U, hm, cfg)
    print("[degrade 4U] sev=%d %s  under=%s z_med_floor=%s"
          % (res["severity"], res["reason"], res["layers"]["L1_zscore"]["fired_under"],
             _f(res["layers"]["L1_zscore"]["z_med_floor"])))
    ok &= (res["severity"] == 1 and res["reason"] == "pumping_degradation")

    # 4) 過渡スパイク→HVオフ（D01_H14 相当）: 中央値は平常でも超過点数で sev3
    sp = _synth("spike_off", seed=6)
    km_sp = _model_from(_synth("healthy_kek", seed=13), ip_pv.SUPPLY_KEK, cfg)
    res = judge_pump(sp, ip_pv.SUPPLY_KEK, km_sp, cfg)
    L = res["layers"]["L0a_nobeam"]
    print("[spike→off (D01_H14型)] sev=%d %s  frac_high=%.2f med_nb=%s n_exc=%d"
          % (res["severity"], res["reason"], L["frac_high"],
             _f(L["med_cur_nobeam"]), L["n_excursion"]))
    ok &= (res["severity"] == 3 and L["frac_high"] < 0.5)   # 割合では出ないが超過点数で出る

    # 5) ビーム由来の上振れ（無ビームで電流が下がり、無ビーム平常を超えない）→ 格下げ。
    #    D10_IP_L09 型: drop 大・nb_excess≤0・r_beam>0・絶対値は abs_hard(1e-5) 未満。
    #    raw は L0b で over_current(sev2) になるが、ビーム軸ゲートで sev1 へ落ちること。
    nb_n = 360; rbd = np.random.RandomState(7)
    beam_bd = np.concatenate([np.zeros(nb_n // 3),
                              np.linspace(100, 1000, nb_n - nb_n // 3)])  # 無ビーム+ramp
    onb = beam_bd > cfg["beam_low_ma"]
    I_bd = np.empty(nb_n)
    I_bd[~onb] = 8e-7 * 10 ** rbd.normal(0, 0.08, int((~onb).sum()))      # 無ビーム床 ~0.8µA
    I_bd[onb] = (9e-9 * beam_bd[onb]) * 10 ** rbd.normal(0, 0.08, int(onb.sum()))  # ビーム追従 →最大~9µA
    P_bd = (1.5e-8 + beam_bd / 1000.0 * 3e-7) * 10 ** rbd.normal(0, 0.05, nb_n)
    d_bd = {"t": ["%06d" % i for i in range(nb_n)], "I": I_bd, "P": P_bd, "beam": beam_bd}
    # 静かな期間で学習したモデル: cur_log_p95 低め（→判定窓は L0b 上振れ sev2）、
    # nb_log_p95 は無ビーム床相当（→無ビームは平常内 nb_excess≤0）。
    model_bd = {"supply": ip_pv.SUPPLY_KEK,
                "cur_log_p95": float(np.log10(3e-7)), "cur_log_p50": float(np.log10(2e-7)),
                "nb_log_p95": float(np.log10(1.0e-6)), "nb_log_p50": float(np.log10(6e-7)),
                "r_beam": 0.9, "ip_trust": False, "a": None}
    res = judge_pump(d_bd, ip_pv.SUPPLY_KEK, model_bd, cfg)
    m = res["metrics"]
    print("[beam-driven 格下げ] sev_raw=%d(%s)→sev=%d beam_driven=%s "
          "drop=%s nb_exc=%s r_beam=%s"
          % (res["severity_raw"], res["reason_raw"], res["severity"], res["beam_driven"],
             _f(m["drop_dex"]), _f(m["nb_excess_dex"]), _f(m["r_beam_axis"])))
    ok &= (res["severity_raw"] >= 2 and res["beam_driven"] and res["severity"] <= 1)

    # 5b) 同じ「ビーム有り側」でも無ビームで電流が高いまま（nb_excess>0）→ 格下げしない。
    #     D03_IP_L09 型の保護: 必須条件 nb_excess≤0 と abs_hard 未満で必ず守られること。
    I_hi = I_bd.copy()
    I_hi[~onb] = 3e-5 * 10 ** rbd.normal(0, 0.08, int((~onb).sum()))      # 無ビームでも高い
    d_hi = {"t": d_bd["t"], "I": I_hi, "P": P_bd, "beam": beam_bd}
    res = judge_pump(d_hi, ip_pv.SUPPLY_KEK, model_bd, cfg)
    print("[no-beam高で保護] sev=%d beam_driven=%s nb_exc=%s（残すべき）"
          % (res["severity"], res["beam_driven"], _f(res["metrics"]["nb_excess_dex"])))
    ok &= (res["severity"] == 3 and not res["beam_driven"])

    # 9) robust 学習: 汚染(5%の放電エポック)を中央値+MAD clip で除去、健全は不変、ビーム分離
    rng = np.random.RandomState(0); n = 400
    beam = np.concatenate([np.zeros(120), 800 + 50 * np.sin(np.arange(n - 120) / 30.0)])
    Ic = np.where(beam < 10, 1e-7, 1e-6) * 10 ** rng.normal(0, 0.1, n)
    bad = rng.choice(np.arange(120, n), size=int(0.05 * n), replace=False)
    Ic[bad] = 1e-4 * 10 ** rng.normal(0, 0.1, len(bad))
    dc = {"t": list(range(n)), "P": np.full(n, np.nan), "I": Ic, "beam": beam}
    b0 = _learn_bands(dc, cfg, 0); b2 = _learn_bands(dc, cfg, 2)
    print("[robust learn] cur_p95: 汚染込み %.1e → robust %.1e / nb %.1e（ビーム分離）"
          % (10 ** b0["cur_log_p95"], 10 ** b2["cur_log_p95"], 10 ** b2["nb_log_p95"]))
    ok &= (10 ** b2["cur_log_p95"] < 3e-6)                       # 汚染除去
    ok &= (abs(b2["cur_log_p95"] - b2["nb_log_p95"]) > 0.5)      # ビーム有/無で分離

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


def _f(x):
    return "None" if x is None else ("%.2f" % x)


# ───────────────────────── CLI ─────────────────────────

def _opt(args, name, default=None, cast=str):
    if name in args:
        return cast(args[args.index(name) + 1])
    return default


def main(argv):
    if not argv:
        print(__doc__)
        return
    cmd = argv[0]
    if cmd == "selftest":
        ok = selftest()
        sys.exit(0 if ok else 1)
    if cmd in ("learn", "judge"):
        if len(argv) < 4:
            print("usage: python ip_judge.py %s <LER|HER> <start> <end> [opts]" % cmd)
            if cmd == "learn":
                print("  opts: --interval N --robust [N] --out path "
                      "--window <start> <end>（複数期間を合算。繰り返し可）")
            return
        ring, start, end = argv[1], argv[2], argv[3]
        if cmd == "learn":
            interval = _opt(argv, "--interval", 300, int)
            out = _opt(argv, "--out", MODELS_FILE)
            robust = 0
            if "--robust" in argv:
                robust = 2                       # 既定 2 回
                j = argv.index("--robust") + 1
                if j < len(argv) and argv[j].isdigit():
                    robust = int(argv[j])
            # 複数期間の合算: 既定窓(start,end)に加え、--window S E を繰り返し指定可
            windows = [(start, end)]
            k = 1
            while True:
                try:
                    wi = argv.index("--window", k)
                except ValueError:
                    break
                windows.append((argv[wi + 1], argv[wi + 2]))
                k = wi + 3
            learn(ring, start, end, interval_sec=interval, out_path=out,
                  robust=robust, windows=windows)
        else:
            interval = _opt(argv, "--interval", 60, int)
            models = _opt(argv, "--models", MODELS_FILE)
            state = judge(ring, start, end, interval_sec=interval, models_path=models)
            s = state["summary"]
            print("=== judge %s（%s〜%s）  sev3=%d sev2=%d sev1=%d "
                  "[acute=%d chronic=%d] / %d pumps ==="
                  % (ring, start, end, s["sev3"], s["sev2"], s["sev1"],
                     s.get("acute", 0), s.get("chronic", 0), s["n"]))
            for p in state["pumps"]:
                if p["severity"] == 0:
                    continue
                kind = p.get("kind") or "-"
                dev = p.get("deviation_dex")
                devs = ("%+.1fdex" % dev) if dev is not None else "  -   "
                print("  sev%d %-8s %-30s [%-10s %s]  %s (%s)"
                      % (p["severity"], kind, p["pv"], p["supply"], p["section"],
                         p["reason"], devs))
            out = _opt(argv, "--out-json")
            if out:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=1, default=_json_default)
                print("判定結果を保存:", out)
        return
    print("unknown command:", cmd)
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
