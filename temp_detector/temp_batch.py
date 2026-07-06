#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_batch.py — 全温度計（LER 1550 / HER 1260）を一括判定するバッチ CLI。

サブコマンド:
  learn  <ring> <start> <end> [--interval S]      健全期間でモデル(temp_models.json)を作成
  run    <ring> [--hours H] [--end YYYYMMDDhhmmss] [--start YYYYMMDDhhmmss] [--interval S] [--top N]
                 [--rolling [--ref-last-days 5] [--ref-days 3]]
                                                  直近窓（既定）or --start/--end 明示指定で全センサ判定・ランキング
                 --start: 指定時は --hours を無視し、start〜end をそのまま判定窓にする
                 --rolling: CCG 式に実行時に基準窓（既定 8〜5日前）を学習（固定モデル不要・陳腐化なし）
  list-low <ring> [--below ℃]                     学習中央値が低いセンサ一覧（極低温系の確認）

設計の要点:
  ・直近窓で判定（S 短絡・O 無ビーム高温は「窓内の割合」で見るので、最近発症の故障を
    薄めないため。IP と同じ思想）。
  ・ビームは各リング一度だけ取得し、時刻で各センサに突き合わせて渡す（B 反相関・O 無ビーム高温用）。
  ・段階フィルタ: まず quick 判定（安価な H0/H1/S/O/G）で全件を捌き、sev3 が確定したものは
    重い層（N ノイズ・B 相関）を省く。残りだけ full 判定。最終 severity は全件 full と一致
    （sev3 は上限なので構造的 sev3 に重い層を足しても上がらない）。
  ・NEG 等の意図的加熱センサは O 層（無ビーム高温）を除外（OPEN_EXCLUDE_SUFFIX）。
  ・出力: severity 降順ランキングを temp_anomalies.json ＋ テキスト。

取得は実機 kblogrd 必須。判定部（judge_all）は kblogrd 不要で `selftest` で確認できる。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import temp_fetch
import temp_judge

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_FILE = os.path.join(HERE, "temp_models.json")
ANOM_FILE = os.path.join(HERE, "temp_anomalies.json")

DEFAULT_HOURS = 24
# O 層（無ビーム高温）を適用しない suffix（意図的に無ビームでも高温になりうる種別）
OPEN_EXCLUDE_SUFFIX = {"NEG"}

# severity 表示色付け用の理由→短い和文
REASON_JA = {
    "range_high_open_suspect": "断線疑い(非現実高温)",
    "range_high_noheat_suspect": "無ビーム高温(near-open/高抵抗)",
    "range_low_short_suspect": "短絡疑い(低温/サブ常温)",
    "noise_and_glitch": "断線間近(ノイズ+グリッチ)",
    "beam_anticorrelated": "ビーム反相関",
    "intermittent_excursion": "間欠逸脱(稀な極端値)",
    "noise_increase": "ノイズ増大",
    "glitch": "グリッチ頻発",
    "stuck": "張り付き",
    "range_watch": "レンジ接近/軽度",
    "pair_deviation_high": "ペア乖離(大)",
    "pair_deviation_watch": "ペア乖離(軽)",
    "normal": "正常",
}


# ───────────────────────── 時刻 ─────────────────────────

def now_str():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def hours_before(end_str, hours):
    """'yyyymmddhhmmss' から hours 時間前の同形式文字列。"""
    t = _to_epoch(end_str)
    return time.strftime("%Y%m%d%H%M%S", time.localtime(t - hours * 3600))


def _to_epoch(s):
    """'yyyymmddhhmmss'（14桁）を epoch 秒に変換。不正な形式は分かりやすいエラーにする。"""
    s2 = (s or "").strip()
    if len(s2) != 14 or not s2.isdigit():
        raise SystemExit(
            "エラー: 日時は 'YYYYMMDDhhmmss' の14桁で指定してください（例 20260630000000）。"
            " 指定値: %r（%d桁）。よくある間違い: 0 の付け過ぎ/足りない、月日の桁数ミス。"
            % (s, len(s2)))
    try:
        return time.mktime(time.strptime(s2, "%Y%m%d%H%M%S"))
    except ValueError as ex:
        raise SystemExit("エラー: 日時 %r が不正です（%s）。実在する年月日時分秒か確認してください。"
                         % (s, ex))


# ───────────────────────── モデル ─────────────────────────

def load_models():
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE) as f:
                return json.load(f)
        except (ValueError, OSError):
            sys.stderr.write("警告: %s を読めません。モデル無しで続行。\n" % MODELS_FILE)
    return {}


def save_models(models):
    tmp = MODELS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(models, f, ensure_ascii=False, indent=0)
    os.replace(tmp, MODELS_FILE)   # 原子的置換（NFS 多重書き対策は IP/CCG と同様、置換で担保）


# ───────────────────────── 判定（kblogrd 不要・テスト可能）─────────────────────────

def _skip_open_for(suffix):
    return suffix in OPEN_EXCLUDE_SUFFIX


def judge_all(data, beam_series, models, cfg=temp_judge.CONFIG):
    """取得済みデータを一括判定。fetch から分離してあるので kblogrd 不要でテストできる。

    data: {pv: {"ring","section","tag","suffix","series":[(ts,val_or_None),...]}}
    beam_series: [(ts, mA_or_None), ...]（そのリングのビーム。空でも可）
    models: {pv: learn_sensor出力}
    返り値: per-sensor 結果 dict のリスト（severity 降順ソート済み）と段階フィルタの統計。
    """
    bmap = {ts: (np.nan if v is None else float(v)) for ts, v in (beam_series or [])}
    have_beam = len(bmap) > 0

    results = []
    n_quick3 = 0
    for pv, v in data.items():
        ts, T = temp_fetch.series_to_arrays(v.get("series", []))
        if len(T) == 0:
            continue
        B = np.array([bmap.get(t, np.nan) for t in ts], dtype=float) if have_beam else None
        model = models.get(pv)
        skip_open = _skip_open_for(v.get("suffix"))

        # 段階1: quick（安価層のみ）。sev3 が確定したら重い層は省く。
        rq = temp_judge.judge_sensor(T, cfg, model=model, beam=B, skip_open=skip_open, quick=True)
        if rq["severity"] >= 3:
            r = rq
            n_quick3 += 1
        else:
            # 段階2: full（N ノイズ・B 相関も評価）
            r = temp_judge.judge_sensor(T, cfg, model=model, beam=B, skip_open=skip_open)

        m, h0 = r["metrics"], r["layers"]["H0_range"]
        results.append({
            "pv": pv, "ring": v.get("ring"), "section": v.get("section"),
            "tag": v.get("tag"), "suffix": v.get("suffix"),
            "severity": r["severity"], "reason": r["reason"],
            "n_pts": m.get("n_pts"), "t_med": m.get("t_med"),
            "t_min": h0.get("t_min"), "t_max": h0.get("t_max"),
            "frac_low": r["layers"]["S_short"].get("frac_low"),
            "frac_hot": r["layers"]["O_hot_noheat"].get("frac_hot"),
            "r_beam": r["layers"]["B_beam"].get("r_beam"),
            "n_events": r["layers"]["I_intermittent"].get("n_events"),
            "has_model": model is not None,
        })

    results.sort(key=lambda d: (-d["severity"], d["pv"]))
    stats = {"n_judged": len(results), "n_quick_sev3": n_quick3, "have_beam": have_beam}
    return results, stats


# ───────────────────────── サブコマンド ─────────────────────────

def learn_from_window(ring, start, end, interval):
    """指定窓を取得して各センサのモデルを作り {pv: model} を返す（保存はしない）。
    learn サブコマンドと run --rolling の両方から使う。"""
    data = temp_fetch.fetch_history(ring, start, end, interval_sec=interval)
    models = {}
    for pv, v in data.items():
        _, T = temp_fetch.series_to_arrays(v.get("series", []))
        m = temp_judge.learn_sensor(T)
        if m:
            models[pv] = m
    return models, len(data)


def cmd_learn(args):
    new, n_total = learn_from_window(args.ring, args.start, args.end, args.interval)
    models = load_models()
    models.update(new)
    save_models(models)
    print("[%s] 学習: %d 本にモデル付与（全 %d 本中）。保存先 %s"
          % (args.ring, len(new), n_total, os.path.basename(MODELS_FILE)))


def cmd_run(args):
    end = args.end or now_str()
    if args.start:
        start = args.start                     # 明示指定: start〜end をそのまま使う
        hours_label = round((_to_epoch(end) - _to_epoch(start)) / 3600.0, 1)
    else:
        start = hours_before(end, args.hours)   # 既定: end から hours 時間前
        hours_label = args.hours

    ref_win = None
    if args.rolling:
        # CCG 式ローリング基準: 判定窓の終端から ref_last_days 日前を基準窓の終端とし、ref_days 日さかのぼる。
        # 既定 (5,3) なら 8〜5日前。基準と判定で interval を揃える（ノイズ比較の整合のため必須）。
        ref_end = hours_before(end, args.ref_last_days * 24)
        ref_start = hours_before(ref_end, args.ref_days * 24)
        ref_win = {"start": ref_start, "end": ref_end,
                   "last_days": args.ref_last_days, "days": args.ref_days}
        sys.stderr.write("[%s] ローリング基準 %s〜%s（%g日前まで×%g日, %ds間隔）学習中...\n"
                         % (args.ring, ref_start, ref_end, args.ref_last_days, args.ref_days, args.interval))
        models, _ = learn_from_window(args.ring, ref_start, ref_end, args.interval)
        sys.stderr.write("[%s] 基準学習 %d 本\n" % (args.ring, len(models)))
    else:
        models = load_models()
        if not models:
            sys.stderr.write("注意: モデル未作成（先に learn、または --rolling）。S 短絡・N ノイズが弱くなります。\n")

    sys.stderr.write("[%s] 判定窓 %s〜%s（%sh, %ds間隔）取得中...\n" % (args.ring, start, end, hours_label, args.interval))
    data = temp_fetch.fetch_history(args.ring, start, end, interval_sec=args.interval)
    try:
        beam = temp_fetch.fetch_beam(args.ring, start, end, interval_sec=args.interval)
    except Exception as ex:
        sys.stderr.write("ビーム取得失敗（B/O 層は無効で続行）: %s\n" % ex)
        beam = []

    results, stats = judge_all(data, beam, models)
    anomalies = [r for r in results if r["severity"] >= 1]

    # 出力（JSON）
    out = {"ring": args.ring, "window": {"start": start, "end": end, "hours": hours_label,
           "interval_sec": args.interval}, "baseline": ref_win, "stats": stats,
           "n_anomalies": len(anomalies), "anomalies": anomalies}
    with open(ANOM_FILE, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 出力（テキスト）
    by_sev = {3: 0, 2: 0, 1: 0, 0: 0}
    for r in results:
        by_sev[r["severity"]] += 1
    print("\n=== [%s] 判定結果 窓 %s〜%s ===" % (args.ring, start, end))
    if ref_win:
        print("基準: ローリング %s〜%s（%g日前まで×%g日）" % (ref_win["start"], ref_win["end"], ref_win["last_days"], ref_win["days"]))
    else:
        print("基準: 固定モデル %s" % os.path.basename(MODELS_FILE))
    print("判定 %d 本 / sev3=%d sev2=%d sev1=%d sev0=%d（うち quick で sev3 確定 %d / ビーム %s）"
          % (stats["n_judged"], by_sev[3], by_sev[2], by_sev[1], by_sev[0],
             stats["n_quick_sev3"], "有" if stats["have_beam"] else "無"))
    print("異常(sev≥1) %d 本。上位 %d 本:" % (len(anomalies), min(args.top, len(anomalies))))
    print("  %-3s %-26s %-5s %6s %7s %7s  %s" % ("sev", "PV", "type", "t_med", "t_min", "t_max", "理由"))
    for r in anomalies[:args.top]:
        print("  %-3d %-26s %-5s %6s %7s %7s  %s"
              % (r["severity"], r["pv"].split(":", 1)[1] if ":" in r["pv"] else r["pv"],
                 r["suffix"] or "-",
                 _fmt(r["t_med"]), _fmt(r["t_min"]), _fmt(r["t_max"]),
                 REASON_JA.get(r["reason"], r["reason"])))
    print("詳細 JSON: %s" % os.path.basename(ANOM_FILE))


def cmd_list_low(args):
    models = load_models()
    rows = [(pv, m.get("t_med")) for pv, m in models.items()
            if pv.startswith("VAL" if args.ring.upper() == "LER" else "VAH")
            and m.get("t_med") is not None and m["t_med"] < args.below]
    rows.sort(key=lambda x: x[1])
    print("[%s] 学習中央値 < %g℃ のセンサ %d 本（極低温系/常時低温の確認用）:"
          % (args.ring, args.below, len(rows)))
    for pv, tm in rows:
        print("  %6.1f℃  %s" % (tm, pv))
    if not rows:
        print("  （該当なし＝常温以下が平常のセンサは無い）")


def _fmt(x):
    return "%.1f" % x if isinstance(x, (int, float)) else "-"


# ───────────────────────── selftest（kblogrd 不要）─────────────────────────

def _selftest():
    print("=== temp_batch selftest（judge_all を合成データで・kblogrd 不要）===")
    ok = True
    rng = np.random.RandomState(7)
    ts = ["06/27/2026 %02d:%02d:00" % ((i // 6) % 24, (i * 10) % 60) for i in range(200)]
    beam = np.where(np.arange(200) % 40 < 25, 1000.0, 0.0)
    bser = list(zip(ts, [float(x) for x in beam]))

    def mk(series, suffix="BL"): return {"ring": "HER", "section": "D01", "tag": "X", "suffix": suffix,
                                         "series": list(zip(ts, [float(x) for x in series]))}
    data = {
        "VAHTMP:D01_normal:X:BL": mk(24 + rng.normal(0, .1, 200)),                        # 正常
        "VAHTMP:D01_short:X:BL":  mk(np.r_[24 + rng.normal(0, .1, 100), -13 + rng.normal(0, .5, 100)]),  # 短絡
        "VAHTMP:D01_open:X:BL":   mk(np.r_[24 + rng.normal(0, .1, 100), np.full(100, 9999.)]),           # 断線
        "VAHTMP:D01_hot:X:BL":    mk(60 + rng.normal(0, .6, 200)),                        # 無ビーム高温
        "VAHTMP:D01_negbake:X:NEG": mk(60 + rng.normal(0, .6, 200), suffix="NEG"),        # NEG=O除外
    }
    hm = temp_judge.learn_sensor(24 + rng.normal(0, 0.15, 120))   # 健全~24℃ の有効モデル
    models = {pv: hm for pv in data}

    results, stats = judge_all(data, bser, models)
    by = {r["pv"].split(":")[1]: (r["severity"], r["reason"]) for r in results}
    for name, (sev, reason) in by.items():
        print("  %-12s sev=%d %s" % (name, sev, reason))
    ok &= by["D01_normal"][0] == 0
    ok &= by["D01_short"][1] == "range_low_short_suspect"
    ok &= by["D01_open"][1] == "range_high_open_suspect"
    ok &= by["D01_hot"][1] == "range_high_noheat_suspect"
    ok &= by["D01_negbake"][1] != "range_high_noheat_suspect"   # NEG は O 除外
    ok &= stats["n_quick_sev3"] >= 2          # short/open/hot は quick で sev3 確定
    # ランキングが severity 降順
    sevs = [r["severity"] for r in results]
    ok &= sevs == sorted(sevs, reverse=True)
    print("  段階フィルタ: quickでsev3確定 %d / 判定 %d" % (stats["n_quick_sev3"], stats["n_judged"]))
    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="温度計バッチ判定 CLI")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("learn", help="健全期間でモデル作成")
    p.add_argument("ring", choices=["LER", "HER"])
    p.add_argument("start"); p.add_argument("end")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("run", help="直近窓（既定）または --start/--end 明示指定で全センサ判定・ランキング")
    p.add_argument("ring", choices=["LER", "HER"])
    p.add_argument("--start", default=None, help="YYYYMMDDhhmmss。指定時は --hours を無視し start〜end をそのまま使う")
    p.add_argument("--hours", type=int, default=DEFAULT_HOURS, help="--start 未指定時: end から何時間前を窓開始にするか")
    p.add_argument("--end", default=None, help="YYYYMMDDhhmmss（既定: 今）")
    p.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--rolling", action="store_true",
                   help="CCG式: 実行時に基準窓を学習（temp_models.json 不要）")
    p.add_argument("--ref-last-days", type=float, default=5.0, dest="ref_last_days",
                   help="基準窓の最終日＝何日前か（既定 5）")
    p.add_argument("--ref-days", type=float, default=3.0, dest="ref_days",
                   help="基準窓の長さ[日]（既定 3）→ 既定は 8〜5日前")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("list-low", help="学習中央値が低いセンサ一覧")
    p.add_argument("ring", choices=["LER", "HER"])
    p.add_argument("--below", type=float, default=18.0)
    p.set_defaults(func=cmd_list_low)

    sub.add_parser("selftest", help="judge_all を合成データで検証（kblogrd 不要）")

    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(0 if _selftest() else 1)
    if not args.cmd:
        ap.print_help(); sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
