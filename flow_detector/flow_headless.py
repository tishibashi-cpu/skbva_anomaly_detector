#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flow_headless.py — 冷却水流量計異常検知の定期実行。

temp_headless.py と同じ位置づけの軽量ループ（同パッケージを壊さないため独立に動く）。
使い方は2通り:

  1) cron から1回だけ実行（推奨・シンプル）:
       0 */4 * * *  cd ~/skbva_anomaly_detector/flow_detector && python flow_headless.py --once >> flow_headless.log 2>&1

  2) 常駐ループとして起動（tmux/screen 等で）:
       python flow_headless.py --interval-hours 4

流量計はビーム電流と無関係のため、CCG/温度計と違い**リング判定も基準期間の別窓取得も無い**
（flow_judge.py 参照：直近窓だけを固定閾値で判定する設計）。ダッシュボード用の
flow_dashboard_state.json を書く（dashboard.py が読む）。
"""

import argparse
import datetime
import json
import math
import os
import time

# numpy(OpenBLAS)は既定で「使えるだけのコア数」を毎回の計算にフル動員しようとし、共用サーバーで
# 無駄にCPUを奪い合う（detector_headless.py --watch で実機のCPU使用率1000%超として確認済み。
# 詳細はdetector_headless.py冒頭のコメント参照）。このファイルはcronから単独実行されることも
# 多く（README参照）、--watchと同時に動くと同じ問題が起きうるため、他のimportより前に対策を入れる。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import flow_fetch
import flow_judge

HERE = os.path.dirname(os.path.abspath(__file__))
DASH_STATE_FILE = os.path.join(HERE, "flow_dashboard_state.json")

DEFAULT_INTERVAL_HOURS = 4     # 判定サイクル間隔（CCG/温度計と同じ既定4h）
DEFAULT_HOURS = 24             # 判定窓の長さ


def now_str():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def hours_before(end, hours):
    dt = datetime.datetime.strptime(end, "%Y%m%d%H%M%S") - datetime.timedelta(hours=hours)
    return dt.strftime("%Y%m%d%H%M%S")


def _json_safe(obj):
    """numpyスカラー/NaN/InfをJSON安全な値に変換する（temp_equipment.py と同じヘルパー）。"""
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


def _decimate_series(series, max_pts=400):
    """[(ts, v_or_None), ...] を等間隔間引きして {"t": [...], "v": [...]} にする（プロット用）。
    max_pts 点以下ならそのまま（プロット幅は~580pxなので400点で視覚的な損失は無い）。値の None(欠測) は JSON では null になり、プロット側で線を切る。"""
    n = len(series)
    if n == 0:
        return {"t": [], "v": []}
    step = max(1, (n + max_pts - 1) // max_pts)
    sel = series[::step]
    return {"t": [t for t, _ in sel], "v": [v for _, v in sel]}


def judge_all(data):
    """flow_fetch.fetch_history() の返り値を全PV分 flow_judge にかけ、severity 降順で返す。
    リング概念が無いので temp_batch.judge_all のような beam/models 引数は無い（指示値だけで判定）。
    返す anomalies は全件（切り詰めは呼び出し側の責務。temp_headless.py 等と同じ流儀）。
    異常(sev>=1)のPVには、ダッシュボードのクリック展開プロット用に間引いた時系列
    （plot: {"t","v"}）を添付する（正常PVには付けない＝JSONサイズを抑えるため）。"""
    results = []
    for pv, v in data.items():
        _, V = flow_fetch.series_to_arrays(v.get("series", []))
        r = flow_judge.judge_series(V)
        rec = {"pv": pv, "section": v["section"], "tag": v["tag"], "sensor_id": v["sensor_id"],
               "severity": r["severity"], "reason": r["reason"],
               "median_pct": r["median"], "cv_pct": r["cv_pct"],
               "n": r["n"], "n_valid": r["n_valid"], "layers": r["layers"]}
        if r["severity"] is not None and r["severity"] >= 1:
            rec["plot"] = _decimate_series(v.get("series", []))
        results.append(rec)
    n_insufficient = sum(1 for r in results if r["severity"] is None)
    anomalies = [r for r in results if r["severity"] is not None and r["severity"] >= 1]
    anomalies.sort(key=lambda d: (-d["severity"], d["pv"]))
    stats = {"n_judged": len(results), "n_insufficient": n_insufficient,
             "n_anomalies_sev3": sum(1 for r in anomalies if r["severity"] == 3),
             "n_anomalies_sev2": sum(1 for r in anomalies if r["severity"] == 2),
             "n_anomalies_sev1": sum(1 for r in anomalies if r["severity"] == 1)}
    return anomalies, stats


def run_once(hours=DEFAULT_HOURS, interval_sec=flow_fetch.DEFAULT_INTERVAL, top=80,
            out_path=DASH_STATE_FILE, end=None):
    """指定窓（既定は直近hours時間）を取得・全PV判定し flow_dashboard_state.json を書く。
    end を指定すると「直近」ではなくその時刻を終端とした過去の窓で判定する
    （アーカイバ停止中の動作確認や、過去の既知の期間で閾値を検証したいときに使う）。"""
    try:
        end = end or now_str()
        start = hours_before(end, hours)   # try内: 不正な--end指定でもクラッシュせずerror JSONを書く
        data = flow_fetch.fetch_history(start, end, interval_sec=interval_sec)
        anomalies, stats = judge_all(data)
        # PVリスト自体は読めているのに、全PVが insufficient_data（=有効なサンプルが1点も
        # 取れていない）なら「個々のセンサ故障」ではなく「アーカイバ自体がデータ取得を停止して
        # いる」と判断できる（temp_headless.py と対の判定ロジック。dashboard.py 側が表示する）。
        archiver_stopped = bool(stats["n_judged"] > 0 and stats["n_insufficient"] == stats["n_judged"])
        block = {"window": {"start": start, "end": end, "hours": hours, "interval_sec": interval_sec},
                "stats": stats, "n_anomalies": len(anomalies), "anomalies": anomalies[:top],
                "archiver_stopped": archiver_stopped}
        if archiver_stopped:
            print("[flow_headless] 注意: PVリスト%d本に対し有効データが1点も取得できていない"
                  "（アーカイバがデータ取得を停止している可能性）" % stats["n_judged"], flush=True)
        print("[flow_headless] 判定 %d本 sev3=%d sev2=%d sev1=%d（データ不足 %d 本）"
              % (stats["n_judged"], stats["n_anomalies_sev3"], stats["n_anomalies_sev2"],
                 stats["n_anomalies_sev1"], stats["n_insufficient"]), flush=True)
    except Exception as ex:
        print("[flow_headless] 判定失敗: %s" % ex, flush=True)
        block = {"error": str(ex)}

    out = _json_safe({"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **block})
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)   # 原子的置換（ダッシュボードが読み取り中に壊れた内容を見せない）
    print("[flow_headless] 保存: %s" % os.path.basename(out_path), flush=True)
    return out


def loop(interval_hours=DEFAULT_INTERVAL_HOURS, **kw):
    print("[flow_headless] 常駐ループ開始（%gh 間隔）。Ctrl-C で終了。" % interval_hours, flush=True)
    while True:
        try:
            run_once(**kw)
        except Exception as ex:
            print("[flow_headless] サイクル失敗（次回まで待機）: %s" % ex, flush=True)
        time.sleep(interval_hours * 3600)


def main():
    ap = argparse.ArgumentParser(description="冷却水流量計異常検知の定期実行")
    ap.add_argument("--once", action="store_true", help="1回だけ実行して終了（cron 向け）")
    ap.add_argument("--interval-hours", type=float, default=DEFAULT_INTERVAL_HOURS, dest="interval_hours",
                    help="常駐ループの判定サイクル間隔[h]（既定 %g）" % DEFAULT_INTERVAL_HOURS)
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="判定窓の長さ[h]（既定 %g）" % DEFAULT_HOURS)
    ap.add_argument("--interval", type=int, default=flow_fetch.DEFAULT_INTERVAL, help="取得サンプリング間隔[s]")
    ap.add_argument("--top", type=int, default=80, help="保存する上位異常件数")
    ap.add_argument("--out", default=DASH_STATE_FILE)
    ap.add_argument("--end", default=None,
                    help="判定窓の終端を明示指定（YYYYMMDDhhmmss）。既定は「今」。"
                         "アーカイバ停止中の動作確認や過去期間の検証に使う（--onceと併用）")
    args = ap.parse_args()

    kw = dict(hours=args.hours, interval_sec=args.interval, top=args.top, out_path=args.out,
             end=args.end)
    if args.once:
        run_once(**kw)
    else:
        if args.end:
            print("[flow_headless] 注意: --end は --once と併用してください"
                  "（常駐ループでは無視され、毎回「今」を使います）。", flush=True)
        loop(interval_hours=args.interval_hours, **{k: v for k, v in kw.items() if k != "end"})


if __name__ == "__main__":
    main()
