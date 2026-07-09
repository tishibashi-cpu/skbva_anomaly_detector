#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_headless.py — 温度計異常検知の定期実行。

CCG/イオンポンプの detector_headless.py と独立に動く軽量ループ（同パッケージを壊さないため）。
使い方は2通り:

  1) cron から1回だけ実行（推奨・シンプル）:
       0 */4 * * *  cd ~/skbva_temp_detector && python temp_headless.py --once >> temp_headless.log 2>&1

  2) 常駐ループとして起動（tmux/screen 等で）:
       python temp_headless.py --interval-hours 4

いずれも両リング（LER/HER）を CCG 式ローリング基準（実行時に 8〜5日前を学習）で判定し、
ダッシュボード用の統合ファイル temp_dashboard_state.json を書く（dashboard.py が読む）。
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
# 多く（README参照）、--watchと同時に動くと同じ問題が起きうるため、temp_batch(→numpy)を
# importするより前に同じ対策を入れる。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import temp_batch
import temp_fetch

HERE = os.path.dirname(os.path.abspath(__file__))
DASH_STATE_FILE = os.path.join(HERE, "temp_dashboard_state.json")

DEFAULT_INTERVAL_HOURS = 4     # 判定サイクル間隔（CCG セーフティネットと同じ 4h）
DEFAULT_HOURS = 24             # 判定窓の長さ
DEFAULT_REF_LAST_DAYS = 5.0    # ローリング基準: 何日前まで
DEFAULT_REF_DAYS = 3.0         # ローリング基準: 期間の長さ（既定 8〜5日前）


def run_once(hours=DEFAULT_HOURS, interval_sec=temp_fetch.DEFAULT_INTERVAL,
             ref_last_days=DEFAULT_REF_LAST_DAYS, ref_days=DEFAULT_REF_DAYS, top=60,
             stagger_sec=0):
    """両リングを判定し temp_dashboard_state.json を書く。片リング失敗でも他方は続行。
    stagger_sec>0: LER の判定が終わってから HER に取り掛かるまで待つ（kblogrd/EPICS への
    同時アクセス負荷を分散するため。CCG/IP と時間をずらして呼ぶのと同じ狙い）。"""
    end = temp_batch.now_str()
    start = temp_batch.hours_before(end, hours)
    rings_out = {}
    ring_list = ("LER", "HER")
    for i, ring in enumerate(ring_list):
        if i > 0 and stagger_sec > 0:
            print("[temp_headless] %s へ移る前に %ds 待機（負荷分散）..." % (ring, stagger_sec), flush=True)
            time.sleep(stagger_sec)
        try:
            ref_end = temp_batch.hours_before(end, ref_last_days * 24)
            ref_start = temp_batch.hours_before(ref_end, ref_days * 24)
            models, _ = temp_batch.learn_from_window(ring, ref_start, ref_end, interval_sec)
            data = temp_fetch.fetch_history(ring, start, end, interval_sec=interval_sec)
            try:
                beam = temp_fetch.fetch_beam(ring, start, end, interval_sec=interval_sec)
            except Exception as ex:
                print("[temp_headless] %s ビーム取得失敗（B/O層無効で続行）: %s" % (ring, ex), flush=True)
                beam = []
            results, stats = temp_batch.judge_all(data, beam, models)
            anomalies = [r for r in results if r["severity"] >= 1]
            anomalies.sort(key=lambda d: (-d["severity"], d["pv"]))
            # ダッシュボードのクリック展開プロット用に、異常PV（保存対象=上位topのみ）へ
            # 間引いた時系列（温度＋同時刻グリッドのビーム電流）を添付する。judge時点で
            # メモリ上にあるデータをそのまま使うため、クリック時のkblogrd再取得は不要。
            bmap = dict(beam) if beam else {}

            def _fin(v):
                # NaN/Inf は JSON にすると JS 側の JSON.parse が失敗するため null にする
                return v if (v is not None and math.isfinite(v)) else None
            for a in anomalies[:top]:
                series = data.get(a["pv"], {}).get("series", [])
                n = len(series)
                if not n:
                    continue
                step = max(1, (n + 399) // 400)   # 400点以下に間引き
                sel = series[::step]
                a["plot"] = {"t": [t for t, _ in sel],
                             "temp": [_fin(v) for _, v in sel],
                             "beam": [_fin(bmap.get(t)) for t, _ in sel] if bmap else None}
            # judge_all はデータが空(len(T)==0)のPVを黙って除外するので、取得できたPV総数
            # (len(data)) に対して判定できたPV数(n_judged)が0なら「アーカイバ自体が停止して
            # いてデータが1本も取れなかった」と判断できる（個々のセンサ故障とは別の状態として
            # ダッシュボードに知らせる。dashboard.py 側の判定ロジックと対）。
            n_total = len(data)
            archiver_stopped = bool(n_total > 0 and stats["n_judged"] == 0)
            rings_out[ring] = {
                "window": {"start": start, "end": end, "hours": hours, "interval_sec": interval_sec},
                "baseline": {"start": ref_start, "end": ref_end,
                            "last_days": ref_last_days, "days": ref_days},
                "stats": dict(stats, n_total=n_total), "n_anomalies": len(anomalies),
                "anomalies": anomalies[:top], "archiver_stopped": archiver_stopped,
            }
            if archiver_stopped:
                print("[temp_headless] %s 注意: PVリスト%d本に対し判定できたのは0本"
                      "（アーカイバがデータ取得を停止している可能性）" % (ring, n_total), flush=True)
            print("[temp_headless] %s 判定 %d本 sev3=%d sev2=%d sev1=%d"
                  % (ring, stats["n_judged"],
                     sum(1 for r in results if r["severity"] == 3),
                     sum(1 for r in results if r["severity"] == 2),
                     sum(1 for r in results if r["severity"] == 1)), flush=True)
        except Exception as ex:
            print("[temp_headless] %s 判定失敗（スキップして続行）: %s" % (ring, ex), flush=True)
            rings_out[ring] = {"error": str(ex)}

    out = {"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "rings": rings_out}
    tmp = DASH_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, DASH_STATE_FILE)   # 原子的置換（ダッシュボードが読み取り中に壊れた内容を見せない）
    print("[temp_headless] 保存: %s" % os.path.basename(DASH_STATE_FILE), flush=True)
    return out


def loop(interval_hours=DEFAULT_INTERVAL_HOURS, **kw):
    print("[temp_headless] 常駐ループ開始（%gh 間隔）。Ctrl-C で終了。" % interval_hours, flush=True)
    while True:
        try:
            run_once(**kw)
        except Exception as ex:
            print("[temp_headless] サイクル失敗（次回まで待機）: %s" % ex, flush=True)
        time.sleep(interval_hours * 3600)


def main():
    ap = argparse.ArgumentParser(description="温度計異常検知の定期実行")
    ap.add_argument("--once", action="store_true", help="1回だけ実行して終了（cron 向け）")
    ap.add_argument("--interval-hours", type=float, default=DEFAULT_INTERVAL_HOURS, dest="interval_hours",
                    help="常駐ループの判定サイクル間隔[h]（既定 %g）" % DEFAULT_INTERVAL_HOURS)
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="判定窓の長さ[h]（既定 %g）" % DEFAULT_HOURS)
    ap.add_argument("--ref-last-days", type=float, default=DEFAULT_REF_LAST_DAYS, dest="ref_last_days")
    ap.add_argument("--ref-days", type=float, default=DEFAULT_REF_DAYS, dest="ref_days")
    ap.add_argument("--interval", type=int, default=temp_fetch.DEFAULT_INTERVAL, help="取得サンプリング間隔[s]")
    ap.add_argument("--top", type=int, default=60, help="保存する上位異常件数（リングごと）")
    ap.add_argument("--stagger-sec", type=int, default=0, dest="stagger_sec",
                    help="LER→HER の間に待機する秒数（既定0=待たない。負荷分散用）")
    args = ap.parse_args()

    kw = dict(hours=args.hours, interval_sec=args.interval, stagger_sec=args.stagger_sec,
              ref_last_days=args.ref_last_days, ref_days=args.ref_days, top=args.top)
    if args.once:
        run_once(**kw)
    else:
        loop(interval_hours=args.interval_hours, **kw)


if __name__ == "__main__":
    main()
