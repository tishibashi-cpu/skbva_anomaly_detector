#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flow_judge.py — 冷却水流量計の判定コア（センサ自身の異常検知。ビーム電流とは無関係）。

検知したいのは「流量計**自身**の異常」（分解清掃で指示値が元に戻る個体故障）であり、
実際の流量低下は別のアラームシステムが検知するため対象外（ユーザ確認済み）。

CCG/温度計/IPと違い、本判定は**単一の直近窓（既定24h）だけを見る**（基準期間の別窓取得や
固定モデルの学習は不要）。これは以下の実データ分析（アップロードされた実測CSV。正常参照7本
vs 異常確認済み4本）で得た知見に基づく：

  - 正常な流量計は日次の変動係数(CV=std/mean)が **0.5〜2.2%** ときわめて安定（7本で最大2.18%）。
    異常4本はCVが **8.2〜44.2%** と、正常の3倍以上。しかも異常個体は記録期間の最初から既に
    CVが高いことが多く（例: 初日からCV≈15%）、**絶対閾値だけで初期から検知できる**
    （自分自身の過去との比較=ローリング基準を必要としない）。
  - 100%は「ある時点で取った基準流量と同じ流量が流れている」ことを意味する固定の校正基準であり
    （ユーザ確認済み）、正常個体でも87〜151%の個体差はあるが、実測で流量が本当にゼロ近くまで
    落ちることは無い、という運用上の事実がある。よって「校正基準100%に対する絶対的な落差」
    （例: 15%未満）はセンサ側の異常（詰まり・接点不良等）の強い証拠になる
    （実データ: D10_02_010 が間欠的に0.27〜0.31%まで落ちる張り付きを示した）。

このため、CCG式のローリング基準窓（何日か前を再取得して比較）を持たず、直近窓の統計量だけを
固定閾値と比較する、より単純な設計にしている（流量計はビーム電流と無関係なので、温度計/IP/
機器劣化検知のような「基準期間の学習」という考え方自体が本質的に不要）。
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
    # 直近窓の最小有効点数（これ未満は判定を保留=insufficient_data）
    "min_points": 30,

    # ── excess_noise（不安定化）: 窓のCV(=std/|mean|, %) ──
    # 正常参照7本の実測CV最大値は2.18%。3倍以上の安全マージンを見て以下の閾値にした。
    "cv_sev1": 4.0,
    "cv_sev2": 8.0,
    "cv_sev3": 15.0,

    # ── stuck_low（校正基準100%に対する落差）: 窓のロバスト中央値[%] ──
    # 正常参照7本の日次平均の実測最小値は約87%。実運用では流量が本当に大きく落ちることは
    # 無い（ユーザ確認済み）ので、大きな落差はほぼ確実にセンサ側異常。
    "low_sev1_below": 75.0,
    "low_sev2_below": 40.0,
    "low_sev3_below": 15.0,

    # ── frozen（値の張り付き・完全停止）──
    # 窓内のrange(max-min)がこの絶対値未満なら「変化が無い」と判定（%単位）。
    "frozen_range_abs": 0.05,

    # ── glitch（単発の外れ値）: 記録のみ。severityには使わない（678本規模では単発の外れ値は
    # 統計的に一定数出るのが自然で、それ単独では故障の証拠にならないと実データ検証で判明した）。
    "glitch_sigma": 8.0,        # ロバストσ(=1.4826*MAD)の何倍を外れ値とするか
}

REASON_JP = {
    "insufficient_data": "判定不能(データ不足)",
    "normal": "正常",
    "frozen_low": "値の固着(校正基準比も低い・センサ異常濃厚)",
    "frozen_watch": "値の固着(校正基準比は正常範囲・要注視)",
    "stuck_low_severe": "校正基準比で大幅低下(重度・センサ異常濃厚)",
    "stuck_low": "校正基準比で低下(中)",
    "stuck_low_watch": "校正基準比で低下(軽度・要注視)",
    "excess_noise_severe": "指示値不安定(重度)",
    "excess_noise": "指示値不安定(中)",
    "excess_noise_watch": "指示値不安定(軽度・要注視)",
}


def _robust_sigma(x):
    """MAD(絶対偏差の中央値)からロバストσを推定（1.4826倍。正規分布下でstdと一致するスケール）。"""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad * 1.4826)


def judge_series(vals, cfg=CONFIG):
    """1本分の直近窓（%値の配列。欠測はNaNで渡す）を判定する。

    返り値: {severity(0-3 or None), reason, layers:{...}, n, n_valid,
             median, cv_pct, frac_of_ref}
    """
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    n_valid = len(x)
    if n_valid < cfg["min_points"]:
        return {"severity": None, "reason": "insufficient_data", "layers": {},
                "n": len(vals), "n_valid": n_valid, "median": None, "cv_pct": None,
                "frac_of_ref": None}

    med = float(np.median(x))
    mean = float(np.mean(x))
    rng = float(np.max(x) - np.min(x))
    sigma = _robust_sigma(x)
    cv_pct = (abs(np.std(x)) / abs(mean) * 100.0) if mean != 0 else float("inf")
    frac_of_ref = med  # 100%基準に対する割合そのもの（PVの値自体が%表現のため）

    layers = {}
    sev_candidates = []

    # ── frozen（値の固着）。実データ検証で判明した点：678本規模で見ると、正常な校正基準比の
    # レベル（例: 100〜130%）で1日中まったく値が変化しないPVが一定数存在する（アーカイバの
    # 記録間隔が粗い、または本当に安定しているだけの可能性が高く、必ずしも故障ではない）。
    # 一方、実際の故障（D10_02_010）は「近ゼロ値に張り付く」という**レベルの異常**が本質であり、
    # 「変化が無いこと」自体は補助的な追加証拠にすぎない。そのため frozen 単独では重度にせず、
    # 校正基準比が低い場合にのみ重度とする（stuck_low層との組み合わせ判定）。
    is_frozen = rng < cfg["frozen_range_abs"]
    layers["frozen"] = {"range": rng, "is_frozen": is_frozen}
    if is_frozen:
        if med < cfg["low_sev2_below"]:
            sev_candidates.append((3, "frozen_low"))      # 固着 かつ 校正基準比も低い→センサ異常濃厚
        else:
            sev_candidates.append((1, "frozen_watch"))    # 固着だが校正基準比は正常範囲→要注視のみ

    # ── stuck_low（校正基準100%に対する落差）──
    if med < cfg["low_sev3_below"]:
        sev_candidates.append((3, "stuck_low_severe"))
    elif med < cfg["low_sev2_below"]:
        sev_candidates.append((2, "stuck_low"))
    elif med < cfg["low_sev1_below"]:
        sev_candidates.append((1, "stuck_low_watch"))
    layers["stuck_low"] = {"median_pct": med}

    # ── excess_noise（絶対CV） ──
    if not is_frozen:   # frozen中はCVの意味が無い（rangeで既に検知済み）
        if cv_pct > cfg["cv_sev3"]:
            sev_candidates.append((3, "excess_noise_severe"))
        elif cv_pct > cfg["cv_sev2"]:
            sev_candidates.append((2, "excess_noise"))
        elif cv_pct > cfg["cv_sev1"]:
            sev_candidates.append((1, "excess_noise_watch"))
    layers["excess_noise"] = {"cv_pct": cv_pct}

    # ── glitch（単発の外れ値）。あくまで情報提供的な内部メトリクスとして記録するのみで、
    # severityは動かさない（678本規模では単発の外れ値は統計的に一定数出るのが自然で、
    # それ単独では故障の証拠にならないと実データ検証で判明したため）。
    if sigma > 0:
        n_glitch = int(np.sum(np.abs(x - med) > cfg["glitch_sigma"] * sigma))
    else:
        n_glitch = 0
    glitch_frac = n_glitch / n_valid
    layers["glitch"] = {"n_glitch": n_glitch, "frac": glitch_frac}

    if not sev_candidates:
        severity, reason = 0, "normal"
    else:
        severity, reason = max(sev_candidates, key=lambda t: t[0])

    return {"severity": severity, "reason": reason, "layers": layers,
            "n": len(vals), "n_valid": n_valid, "median": med, "cv_pct": cv_pct,
            "frac_of_ref": frac_of_ref}


# ───────────────────────── selftest（合成データ＋実データ検証）─────────────────────────

def _selftest():
    print("=== flow_judge selftest ===")
    ok = True
    rng = np.random.RandomState(0)

    # 1) 正常データ（CV~1.5%、100%近辺）→ sev0
    normal = 100 + rng.normal(0, 1.5, 2000)
    r = judge_series(normal)
    print("  normal: sev=%s reason=%s cv=%.2f%%" % (r["severity"], r["reason"], r["cv_pct"]))
    ok &= (r["severity"] == 0 and r["reason"] == "normal")

    # 2) 不安定化（CV~12%）→ sev2(excess_noise)
    noisy = 90 + rng.normal(0, 11, 2000)
    r = judge_series(noisy)
    print("  noisy: sev=%s reason=%s cv=%.2f%%" % (r["severity"], r["reason"], r["cv_pct"]))
    ok &= (r["severity"] == 2 and r["reason"] == "excess_noise")

    # 3) 校正基準比で大幅低下 → sev3(stuck_low_severe)
    low = 3 + rng.normal(0, 0.5, 2000)
    r = judge_series(low)
    print("  low: sev=%s reason=%s median=%.2f" % (r["severity"], r["reason"], r["median"]))
    ok &= (r["severity"] == 3 and r["reason"] == "stuck_low_severe")

    # 4) 完全な値の固着（分散ほぼ0）かつ校正基準比も低い → sev3(frozen_low)。実際の故障
    # （D10_02_010）と同じ「近ゼロに張り付く」パターン。
    frozen_low = np.full(2000, 0.29) + rng.normal(0, 0.005, 2000)
    r = judge_series(frozen_low)
    print("  frozen_low: sev=%s reason=%s range=%.4f" % (r["severity"], r["reason"], r["layers"]["frozen"]["range"]))
    ok &= (r["severity"] == 3 and r["reason"] == "frozen_low")

    # 4b)【回帰テスト】完全な値の固着だが校正基準比は正常範囲 → sev1(frozen_watch)止まり。
    # 実データ検証で、678本規模だと正常なレベル（100%前後）で1日中値が変化しないPVが多数存在
    # すると判明した（アーカイバの記録間隔が粗い等の可能性が高く、必ずしも故障ではない）ため、
    # frozenだけでは重度にしない（このケースが以前は誤ってsev3になっていた＝実際の修正理由）。
    frozen_plausible = np.full(2000, 100.5)
    r = judge_series(frozen_plausible)
    print("  frozen_watch: sev=%s reason=%s median=%.2f" % (r["severity"], r["reason"], r["median"]))
    ok &= (r["severity"] == 1 and r["reason"] == "frozen_watch")

    # 4c)【回帰テスト】単発の外れ値だけ（他は完全に正常）→ severityには影響しない（sev0のまま）。
    # 678本規模では単発の外れ値は統計的に一定数出るのが自然で、それ単独では故障の証拠にならない
    # と実データ検証で判明したため（以前は誤ってsev1 "glitch" にしていた＝修正理由）。
    with_glitch = 100 + rng.normal(0, 1.5, 2000)
    with_glitch[500] = 250.0   # 単発の大きな外れ値
    r = judge_series(with_glitch)
    print("  glitch単独: sev=%s reason=%s n_glitch=%d" % (r["severity"], r["reason"], r["layers"]["glitch"]["n_glitch"]))
    ok &= (r["severity"] == 0 and r["reason"] == "normal" and r["layers"]["glitch"]["n_glitch"] >= 1)

    # 5) データ不足 → insufficient_data
    r = judge_series([100.0] * 5)
    print("  短い列: sev=%s reason=%s" % (r["severity"], r["reason"]))
    ok &= (r["severity"] is None and r["reason"] == "insufficient_data")

    # 6) NaN混入（欠測）は無視されて有効点だけで判定されること
    with_nan = np.r_[normal, [np.nan] * 500]
    r = judge_series(with_nan)
    print("  欠測混入: n=%d n_valid=%d sev=%s" % (r["n"], r["n_valid"], r["severity"]))
    ok &= (r["n"] == 2500 and r["n_valid"] == 2000 and r["severity"] == 0)

    # ── 7) 実データ検証（アップロードされたCSVがあれば。無ければスキップ） ──
    import os
    import csv as _csv
    data_dir = os.environ.get("FLOW_JUDGE_SELFTEST_DATA")
    if data_dir and os.path.isdir(data_dir):
        def _load(path):
            vals = []
            with open(path, encoding="utf-8-sig") as f:
                for row in _csv.reader(f):
                    if len(row) < 2 or row[0] == "Timestamp":
                        continue
                    try:
                        vals.append(float(row[1]))
                    except ValueError:
                        pass
            return np.array(vals)

        print("  --- 実データ検証（%s） ---" % data_dir)
        for fn in sorted(os.listdir(data_dir)):
            if not fn.endswith(".csv"):
                continue
            vals = _load(os.path.join(data_dir, fn))
            if len(vals) < 100:
                continue
            r = judge_series(vals)
            print("    %-24s n=%-7d sev=%s reason=%-22s median=%6.2f cv=%6.2f%%"
                  % (fn, len(vals), r["severity"], r["reason"], r["median"], r["cv_pct"]))

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
