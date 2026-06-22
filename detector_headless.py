#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector_headless.py — 末次さんの検知プログラムを GUI なしで回す雛形（v0.1）

方針: Anomaly_Detection_112p.py を【一行も書き換えず】import し、判定中核だけを呼ぶ。
      tkinter / matplotlib の GUI 呼び出しはスタブ（無害な空処理）で抑える。
      判定関数が結果を *_Class2_Result_*.txt 等に保存するので、そのあと
      state_builder で dashboard_state.json を更新する。

┌─ 重要 ───────────────────────────────────────────────────────┐
│ このファイルは EPICS / kblogrd のある実機 (kekb-co-userNN) でしか動きません。  │
│ 手元では検証できないため、実機で動かしながら詰めてください。特に:               │
│   ・CCG_N（CCG総数）… 実機の 112p で `grep LER_CCG_n` で確認して設定          │
│   ・No_STD_Abort / Cn_Beam の細かい分岐 … Main_Command_LER の該当行を参照     │
│   ・EPICS アボートトリガ … 今は「間隔ポーリング」。トリガ版は TODO（次段階）    │
└──────────────────────────────────────────────────────────────┘

対応関係（112p の Main_Command_LER の判定パイプラインを移植）:
  Define_Date_Range → No_Beam 判定
  Get_Fit_STD_*  （基準データ取得＋回帰）
  Get_Fit_CHK_*  （調査データ取得＋回帰）
  Get_DIF_*      （差分）
  Find_Abnormal_*（判定＋結果ファイル保存） ← ここまでが self 不要
  ※ Record_Freq_plot / Find_Possible_Cause は self(GUI)依存なのでスキップ。
     カウントは state_builder で再現済み。原因は後段で接続。
"""

import os
import sys
import time
import threading
import datetime

# ── 1. GUI を無害化してから 112p を import する ───────────────────────────

# matplotlib を画面に出さない（ヘッドレス）
import matplotlib
matplotlib.use("Agg")

# tkinter.messagebox を no-op に。判定関数がエラー時に呼ぶが、表示は不要。
# 抑制した内容はログに残してあとで追えるようにする。
import tkinter.messagebox as _mb
def _suppress(title="", msg="", *a, **k):
    print("[messagebox抑制] %s: %s" % (title, msg))
    return "ok"
_mb.showinfo = _suppress
_mb.showwarning = _suppress
_mb.showerror = _suppress

# ── ディレクトリ構成 ──────────────────────────────────────────────────
#   HERE/                    ← このスクリプト・state_builder・dashboard・CSV
#   HERE/legacy/             ← 末次プログラム本体 + 実行時データ(モデル/.sh/結果ファイル)
# 末次プログラムは I/O をすべて CWD 相対で行う（'sh LERD01CCG.sh ...'、
# load_model('model_*.h5')、np.save('*_Result_*') など）。そのため判定を回す前に
# CWD を legacy/ へ移す。一方、CSV と dashboard_state.json はトップに置くので、
# それらは __file__ 基準の絶対パスで参照し、chdir の影響を受けないようにする。
HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.join(HERE, "legacy")
# 現在の検知状態（リング別の検知時刻・ビームON/OFF）をトップに残す。
# state_builder がヘッダ（最終チェック・ビーム状態）と異常の実時刻フィルタに使う。
STATUS_FILE = os.path.join(HERE, "detector_status.json")


def _record_status(ring, no_beam, check_time):
    """detector_status.json にこのリングの検知時刻・ビーム有無を記録する。"""
    import json
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {}
    st[ring] = {"check_time": check_time, "beam_on": (no_beam == 0)}
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        print("[%s] detector_status 書き込み失敗: %s" % (ring, ex))

# 112p を import（`if __name__ == "__main__"` ブロックは実行されない＝Tk()もmainloopも起動しない）
# 末次さんの本体はファイル名・中身とも無改変のまま legacy/ に置く。
DETECTOR_MODULE = "Anomaly_Detection_112p"
sys.path.insert(0, LEGACY_DIR)   # legacy/ から 112p 本体を import
sys.path.insert(0, HERE)         # トップから state_builder / ccg_pv を import
det = __import__(DETECTOR_MODULE)

# state_builder / ccg_pv（トップに置く）
import state_builder
import ccg_pv

# 監視 CCG リストは CSV（最新の監視対象）から読む。CSV はトップに置く。
# chdir で CWD が legacy/ に移っても読めるよう、絶対パスにしておく。
LER_CSV = os.path.join(HERE, "pv_info", "LER_CCG_PV.csv")
HER_CSV = os.path.join(HERE, "pv_info", "HER_CCG_PV.csv")
_LER_PV = ccg_pv.load_side_rooms(LER_CSV)
_HER_PV = ccg_pv.load_side_rooms(HER_CSV)

# 112p が内部（GUI選択リスト・CCG_n 算出）で使う PVリストも CSV ベースに統一する。
# 112p 本体は無改変のまま、import 後に Make_*_Record を CSV 版へ差し替える（モンキーパッチ）。
det.Make_LER_Record = lambda _lst: ccg_pv.load_side_rooms(LER_CSV)
det.Make_HER_Record = lambda _lst: ccg_pv.load_side_rooms(HER_CSV)

# 末次プログラムの CWD 相対 I/O を legacy/ に向ける（モデル・.sh・結果ファイル一式）。
os.chdir(LEGACY_DIR)

# ── 2. 設定（GUI 入力の代わり）─────────────────────────────────────────

CONFIG = {
    "Method":           "FNN",      # 回帰モデル種別（112p では 'FNN' 固定運用）
    "Class_Method":     "Keras",    # 判定器: "Keras"（学習済み.h5）or "SDM"（自前重み）
    "Last_Refd":        2.0,        # 基準データ最終日 = 2 日前
    "Ref_Pd":           1.0,        # 基準期間 = 1 日
    "File_Save_Para":   "Save",     # 結果を蓄積ファイルに保存する
    "Plot_Para":        0,          # 0 = プロットしない（ヘッドレス）
    "Run_Mode":         "Auto",     # 自動モード
    "Auto_Mode":        "Abt Trg",  # "Abt Trg"=アボートトリガ / "Cns Intr"=一定間隔
    "Cns_Mode":         0,          # LER_trg_cns 相当（0/1）… 実機で確認
    "Arun":             0,          # LER_auto_run 相当（初回 0）… 実機で確認
    "Hadv":             4.0,        # 一定間隔の時間 [h]
    "Check_Record_Name": "none",    # 自動検知では個別指定なし
}

# 各リングの側室リスト（.sh 名と一致。112p の self.{LER,HER}_CCG_List 相当）
CCG_LIST = {
    "LER": ["LERD%02dCCG" % i for i in range(1, 13)],
    "HER": ["HERD%02dCCG" % i for i in range(1, 13)],
}

# CCG 数（Get_Fit_* / Get_DIF_* に渡る）。112p の self.{LER,HER}_CCG_n と同じく
# 「側室ごと(D01〜D12)のCCG数の配列」。PVリストの各側室の長さ('none'込み)から作る。
CCG_N = {
    "LER": [len(_LER_PV[i]) for i in range(12)],
    "HER": [len(_HER_PV[i]) for i in range(12)],
}

# ── 3. 判定パイプライン（Main_Command_LER/HER の self 非依存部分を移植）──────

def run_check(ring, abort_timing):
    """1 リング分の判定を実行し、結果を蓄積ファイルに保存する。"""
    cfg = CONFIG
    List_Para = CCG_LIST[ring]
    CCG_n = CCG_N[ring]
    if not CCG_n or len(CCG_n) != 12:
        raise RuntimeError("CCG_N['%s'] が取得できていません（PVリスト取得を確認）。" % ring)

    # (a) 日付範囲とビーム有無
    #   No_Beam=1: ビームなし(シャットダウン中等) / 0: ビームあり
    #   Define_Date_Range が 'No_STD_Abort' を返す＝基準期間にアボートが無く Tail 基準が
    #   作れない。これはエラーではなく No_STD_Abort=1 として扱い、Tail 解析を省いて続行する。
    No_Beam, err = det.Define_Date_Range(
        abort_timing, List_Para, cfg["Run_Mode"], cfg["Auto_Mode"],
        cfg["Cns_Mode"], cfg["Arun"], cfg["Hadv"], cfg["Last_Refd"], cfg["Ref_Pd"])
    No_STD_Abort = 0
    if err == "No_STD_Abort":
        No_STD_Abort = 1
    elif err != "none":
        print("[%s] Define_Date_Range: %s" % (ring, err))
        return
    # 現在状態（検知時刻・ビーム有無）を記録 → state_builder がヘッダ/実時刻フィルタに使う
    _record_status(ring, No_Beam, abort_timing)

    if No_Beam == 0:
        # ── ビームあり ─────────────────────────────────────────
        # (b) 基準データ Storage
        _, e = det.Get_Fit_STD_Strg(List_Para, cfg["Method"], cfg["Ref_Pd"], CCG_n)
        if e != "none":
            print("[%s] STD_Strg: %s" % (ring, e)); return
        # 基準にアボートがあったときだけ Tail 基準を取る（No_STD_Abort==0）
        if No_STD_Abort == 0:
            det.Get_Fit_STD_Tail(List_Para, cfg["Method"])

        # (c) 調査データ Storage（エラーでも return せず、後段で Error_fcs によりゲート）
        _, Error_fcs = det.Get_Fit_CHK_Strg(cfg["Method"], cfg["Check_Record_Name"],
                                            "CHK_Strg", List_Para, abort_timing, CCG_n)
        # 調査データ Tail（基準にアボートがあったときだけ）。'No Tail Data' なら Cn_Beam=1
        Cn_Beam = 0
        Error_fct = "none"
        if No_STD_Abort == 0:
            _, Error_fct = det.Get_Fit_CHK_Tail(cfg["Method"], cfg["Check_Record_Name"],
                                                "CHK_Tail", List_Para, abort_timing)
            if Error_fct == "No Tail Data":
                Cn_Beam = 1

        # (d) 差分＋判定（結果はここで蓄積ファイルに保存される）
        if Error_fcs == "none" and Error_fct == "none":
            # Storage も Tail も有り → 両方判定
            _, e = det.Get_DIF_Strg(List_Para, CCG_n)
            if e == "none":
                n = det.Find_Abnormal_Strg(cfg["Method"], "DIF_Strg", List_Para,
                        cfg["File_Save_Para"], cfg["Plot_Para"], abort_timing,
                        cfg["Class_Method"], cfg["Check_Record_Name"])
                print("[%s] Storage 異常 %s 件" % (ring, n))
            if No_STD_Abort == 0:
                _, e = det.Get_DIF_Tail(List_Para)
                if e == "none":
                    n = det.Find_Abnormal_Tail(cfg["Method"], "DIF_Tail", List_Para,
                            cfg["File_Save_Para"], cfg["Plot_Para"], abort_timing,
                            cfg["Class_Method"], cfg["Check_Record_Name"])
                    print("[%s] Tail 異常 %s 件" % (ring, n))
        elif Error_fcs == "none" and (Cn_Beam == 1 or No_STD_Abort == 1):
            # Storage のみ（今フィルに Tail データ無し or 基準にアボート無し）
            _, e = det.Get_DIF_Strg(List_Para, CCG_n)
            if e == "none":
                n = det.Find_Abnormal_Strg(cfg["Method"], "DIF_Strg", List_Para,
                        cfg["File_Save_Para"], cfg["Plot_Para"], abort_timing,
                        cfg["Class_Method"], cfg["Check_Record_Name"])
                print("[%s] Storage 異常 %s 件（Tailなし: Cn_Beam=%d No_STD_Abort=%d）"
                      % (ring, n, Cn_Beam, No_STD_Abort))
        else:
            print("[%s] CHK_Strg エラーのため判定スキップ: %s" % (ring, Error_fcs))
    else:
        # ── ビームなし（No-beam）─────────────────────────────────
        _, e = det.Get_Fit_STD_Strg_NB(List_Para, cfg["Method"], cfg["Ref_Pd"], CCG_n)
        if e != "none":
            print("[%s] STD_Strg_NB: %s（kblog データが無いかも）" % (ring, e)); return
        # CHK_NB はエラーでも return せず、Error_fcs でゲート（legacy と同じ）
        _, Error_fcs = det.Get_Fit_CHK_Strg_NB(cfg["Method"], cfg["Check_Record_Name"],
                                               "CHK_Strg", List_Para, abort_timing, CCG_n)
        if Error_fcs == "none":
            _, e = det.Get_DIF_Strg_NB(List_Para, CCG_n)
            if e == "none":
                det.Find_Abnormal_Strg_NB(cfg["Method"], "DIF_Strg", List_Para,
                    cfg["File_Save_Para"], cfg["Plot_Para"], abort_timing,
                    cfg["Class_Method"], cfg["Check_Record_Name"])
                det.Find_Abnormal_Tail_NB(cfg["Method"], "DIF_Tail", List_Para,
                    cfg["File_Save_Para"], cfg["Plot_Para"], abort_timing)
        else:
            print("[%s] CHK_Strg_NB エラー: %s" % (ring, Error_fcs))
        print("[%s] No-beam チェック完了" % ring)


def rebuild_json():
    """判定結果（蓄積ファイル）から dashboard_state.json を更新する。"""
    state = state_builder.build_state()
    import json
    with open(state_builder.OUT, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print("dashboard_state.json 更新（異常 %d 件）" % len(state["anomalies"]))


# ── 4. 制御ループ ─────────────────────────────────────────────────────
# v0.1 は「間隔ポーリング」。EPICS アボートトリガ版（112p の LERtriggerEvent 相当、
# epics.camonitor でアボート信号を待つ）は次段階で追加する。

def _run_ring(ring, now):
    """1リング分を実行し、所要時間を表示して JSON を更新する。"""
    t0 = time.time()
    print("[%s] チェック開始 %s ..." % (ring, now), flush=True)
    try:
        run_check(ring, now)
    except Exception as ex:
        print("[%s] run_check 例外: %s" % (ring, ex), flush=True)
    rebuild_json()  # このリングまでの結果で dashboard_state.json を更新
    print("[%s] 完了（%.0f 秒）" % (ring, time.time() - t0), flush=True)


# アボートPV（実機の Channel Access）。アボート時にこれが非0になる。
ABORT_PV = {"LER": "CGLSAFE:MR:ABORT", "HER": "CGHSAFE:MR:ABORT"}


def watch_aborts(delay_min=3, debounce_min=30, interval_h=4):
    """EPICS のアボートPVを監視し、アボート時に delay_min 分待ってから検知を走らせる。
    あわせて interval_h ごとの定期チェックも行う（アボートが無くても最低 interval_h に1回）。
    末次プログラムの Abt Trg 自動モードのヘッドレス版（3分待ち・30分デバウンス）。"""
    try:
        import epics
    except Exception as ex:
        print("epics モジュールが必要です（実機のみ）: %s" % ex)
        return

    st = {ring: {"first": True, "last_run": datetime.datetime(2000, 1, 1)}
          for ring in ABORT_PV}
    run_lock = threading.Lock()   # LER/HER の検知を直列化（競合防止）

    def do_check(ring, when):
        with run_lock:
            now = when.strftime("%Y-%m-%d %H:%M:%S")
            _run_ring(ring, now)

    def make_cb(ring):
        def cb(pvname=None, value=None, **kw):
            s = st[ring]
            if s["first"]:                 # 接続時の初回コールは無視（legacy と同じ）
                s["first"] = False
                return
            if not value:                  # 0 に戻った等は無視
                return
            now = datetime.datetime.now()
            if (now - s["last_run"]).total_seconds() < debounce_min * 60:
                print("[%s] アボート受信（前回から %d 分未満のため抑制）"
                      % (ring, debounce_min), flush=True)
                return
            s["last_run"] = now
            print("[%s] アボート受信 → %d 分後に解析（Tailデータ待ち）"
                  % (ring, delay_min), flush=True)
            t = threading.Timer(delay_min * 60, do_check, args=(ring, now))
            t.daemon = True
            t.start()
        return cb

    pvs = {ring: epics.PV(pv, callback=make_cb(ring))
           for ring, pv in ABORT_PV.items()}   # noqa: F841 (参照保持のため束ねる)

    print("アボートトリガ監視を開始", flush=True)
    print("  LER: %s / HER: %s" % (ABORT_PV["LER"], ABORT_PV["HER"]), flush=True)
    print("  アボート後 %d 分で解析、前回から %d 分未満は抑制、最低 %d 時間ごとに定期チェック。"
          % (delay_min, debounce_min, interval_h), flush=True)
    print("  Ctrl-C で停止。", flush=True)
    try:
        while True:
            time.sleep(60)
            # 定期セーフティネット: アボートが無くても interval_h ごとに走らせる
            now = datetime.datetime.now()
            for ring in ABORT_PV:
                if (now - st[ring]["last_run"]).total_seconds() >= interval_h * 3600:
                    st[ring]["last_run"] = now
                    threading.Thread(target=do_check, args=(ring, now),
                                     daemon=True).start()
    except KeyboardInterrupt:
        print("\n監視を停止しました")


def loop_interval(interval_sec=4 * 3600):
    print("detector_headless 起動（間隔 %d 秒ごとにチェック）" % interval_sec)
    while True:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ring in ("LER", "HER"):
            _run_ring(ring, now)   # リングごとに JSON も更新
        time.sleep(interval_sec)


if __name__ == "__main__":
    # 二重起動防止（共用サーバーで検知を同時に複数走らせない）
    try:
        import singleton
        _lock = os.path.join(HERE, ".detector.lock")
        if not singleton.guard(_lock, "detector_headless.py", "検知プログラム"):
            sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        pass

    # 実行モード:
    #   --once   1回だけ検知して終了
    #   --watch  アボートPVを監視し、アボート時＋定期に検知（常駐・実機向け推奨）
    #   引数なし  単純な定期ループ（loop_interval）
    if "--once" in sys.argv:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_all = time.time()
        for ring in ("LER", "HER"):
            _run_ring(ring, now)   # リングごとに JSON 更新（片方終わればすぐ表示に反映）
        print("=== --once 完了（合計 %.0f 秒）dashboard_state.json を更新しました ==="
              % (time.time() - t_all), flush=True)
    elif "--watch" in sys.argv:
        watch_aborts()
    else:
        loop_interval()
