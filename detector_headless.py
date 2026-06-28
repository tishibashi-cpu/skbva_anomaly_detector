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
    "File_Save_Para":   1,          # 結果を蓄積ファイルに保存（レガシーは `File_Save_Para == 1` で判定。文字列だと保存されない）
    "Plot_Para":        0,          # 0 = プロットしない（ヘッドレス）
    "Run_Mode":         "Auto",     # 自動モード
    "Auto_Mode":        "Abt Trg",  # "Abt Trg"=アボートトリガ / "Cns Intr"=一定間隔
    "Cns_Mode":         0,          # LER_trg_cns 相当（0/1）… 実機で確認
    "Arun":             0,          # LER_auto_run 相当（初回 0）… 実機で確認
    "Hadv":             4.0,        # 一定間隔の時間 [h]
    "Check_Record_Name": "none",    # 自動検知では個別指定なし
}

# 定期チェック/--once で渡す「現在時刻」を何分過去にずらすか。
# レガシーは調査期間の終端を Abort_Timing + minit_advance(≈2分) で決めるため、
# 「今」を渡すと終端が現在時刻に張り付き、側室(D01〜D12)を順番に取得する間に
# 時刻が進んで側室ごとに時刻点数(行数)がずれ、Get_Fit_CHK_Strg の連結(axis=1)で
# 落ちる（特に長時間連続フィル時）。終端を数分過去に固定すれば全側室が完成済みの
# 同一データを取得し行数が揃う。データ鮮度を数分犠牲にするだけで監視には影響しない。
# アボートで走る本来の検知（make_cb が実アボート時刻を渡す経路）には適用しない。
CHK_LAG_MIN = 10

# ── イオンポンプ放電電流の judge を検知サイクルに相乗りさせる設定 ──
#   モデル(ip_models.json)は別途 `python ip_judge.py learn …` で一度作って固定する方針
#   （CCG の .h5 と同じ思想）。ここでは judge だけを定期実行し ip_judge_state.json を更新。
#   モデルが無ければスキップ（操作者に learn を促すメッセージを出す）。
IP_JUDGE_ENABLE     = True
IP_JUDGE_EVERY_SEC  = 4 * 3600      # judge を回す間隔 [s]（CCG の定期セーフティネットと同じ 4h。
                                    # 放電は持続性で即時性は不要。アボート連動はしない＝純粋に周期実行）
IP_JUDGE_WINDOW_H   = 24            # 判定窓 = 直近この時間 [h]
IP_JUDGE_INTERVAL   = 300           # judge の取得間隔 [s]
IP_MODELS_FILE      = os.path.join(HERE, "ip_models.json")
IP_JUDGE_STATE_FILE = os.path.join(HERE, "ip_judge_state.json")
IP_JUDGE_COUNTS_FILE = os.path.join(HERE, "ip_judge_counts.json")  # sev3 の累積カウント（持続性）
IP_JUDGE_HISTORY_FILE = os.path.join(HERE, "ip_judge_history.json")  # カードのカウント推移プロット用（PVごと履歴）
IP_HISTORY_MAX      = 30            # 履歴に残す judge サイクル数（カウントプロットの横軸長）
_ip_judge_state = {"last_run": datetime.datetime(2000, 1, 1), "warned_no_model": False}
_ip_judge_lock = threading.Lock()

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


def run_ip_judge(window_h=IP_JUDGE_WINDOW_H):
    """イオンポンプ放電電流 judge を両リング実行し ip_judge_state.json を書く。
    モデル(ip_models.json)が無ければスキップ。各リングは独立に try（片方失敗でも続行）。
    結果はリング結果 dict のリスト [{LER}, {HER}] として保存（state_builder が読む）。
    """
    if not IP_JUDGE_ENABLE:
        return
    import json
    if not os.path.isfile(IP_MODELS_FILE):
        if not _ip_judge_state["warned_no_model"]:
            print("[ip_judge] %s が無いためスキップ。先に一度 learn してください:\n"
                  "  python ip_judge.py learn LER <健全開始> <健全終了> --interval 300 --robust --out ip_models.json\n"
                  "  python ip_judge.py learn HER <健全開始> <健全終了> --interval 300 --robust --out ip_models.json"
                  % os.path.basename(IP_MODELS_FILE), flush=True)
            _ip_judge_state["warned_no_model"] = True
        return
    try:
        import ip_judge
    except Exception as ex:
        print("[ip_judge] import 失敗（実機で確認）: %s" % ex, flush=True)
        return

    end = datetime.datetime.now() - datetime.timedelta(minutes=CHK_LAG_MIN)
    start = end - datetime.timedelta(hours=window_h)
    s = start.strftime("%Y%m%d%H%M%S")
    e = end.strftime("%Y%m%d%H%M%S")
    results = []
    for ring in ("LER", "HER"):
        try:
            res = ip_judge.judge(ring, s, e, interval_sec=IP_JUDGE_INTERVAL,
                                 models_path=IP_MODELS_FILE)
            results.append(res)
            sm = res.get("summary", {})
            print("[ip_judge] %s sev3=%s sev2=%s sev1=%s [acute=%s chronic=%s]"
                  % (ring, sm.get("sev3"), sm.get("sev2"), sm.get("sev1"),
                     sm.get("acute"), sm.get("chronic")), flush=True)
        except Exception as ex:
            print("[ip_judge] %s 判定失敗: %s" % (ring, ex), flush=True)

    # sev3 の累積カウント（CCG 同様、持続して出るものだけを表示するため）。
    # sev3 のサイクルで +1、そうでなければ -1（下限0）。ヒステリシスを持たせる。
    try:
        counts = json.load(open(IP_JUDGE_COUNTS_FILE, encoding="utf-8")) \
                 if os.path.isfile(IP_JUDGE_COUNTS_FILE) else {}
    except Exception:
        counts = {}
    for res in results:
        ring = res.get("ring")
        rc = counts.setdefault(ring, {})
        sev3_pvs = {p["pv"] for p in res.get("pumps", []) if p.get("severity", 0) >= 3}
        for pv in set(rc) | sev3_pvs:
            rc[pv] = min(99, rc.get(pv, 0) + 1) if pv in sev3_pvs else max(0, rc.get(pv, 0) - 1)
            if rc[pv] == 0:
                del rc[pv]
        for p in res.get("pumps", []):
            p["count"] = rc.get(p["pv"], 0)   # 表示側の累積ゲートに使う
    try:
        with open(IP_JUDGE_COUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump(counts, f, ensure_ascii=False, indent=1)
    except Exception as ex:
        print("[ip_judge] カウント保存失敗: %s" % ex, flush=True)

    # カウント推移プロット用の履歴（PVごとに各 judge サイクルの累積カウントを追記）。
    # 表示は state_builder 側でゲート(sev3+count)した PV のみ使う。ここでは現在カウント>0 の
    # PV をすべて IP_HISTORY_MAX 件まで保持し、0 に戻った PV は履歴ごと忘れる（肥大化防止）。
    try:
        hist = json.load(open(IP_JUDGE_HISTORY_FILE, encoding="utf-8")) \
               if os.path.isfile(IP_JUDGE_HISTORY_FILE) else {}
    except Exception:
        hist = {}
    for ring in (r.get("ring") for r in results):
        rc = counts.get(ring, {})
        rh = hist.setdefault(ring, {})
        for pv in set(rh) | set(rc):                  # 既存履歴 ∪ 現在カウント有り
            seq = rh.get(pv, [])
            seq.append(int(rc.get(pv, 0)))
            rh[pv] = seq[-IP_HISTORY_MAX:]
            if rc.get(pv, 0) == 0 and not any(rh[pv]):  # ずっと 0 なら忘れる
                del rh[pv]
    try:
        with open(IP_JUDGE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=1)
    except Exception as ex:
        print("[ip_judge] 履歴保存失敗: %s" % ex, flush=True)

    if results:
        try:
            with open(IP_JUDGE_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            print("[ip_judge] %s を更新（窓 %s〜%s）"
                  % (os.path.basename(IP_JUDGE_STATE_FILE), s, e), flush=True)
        except Exception as ex:
            print("[ip_judge] 保存失敗: %s" % ex, flush=True)


def _maybe_run_ip_judge(force=False):
    """前回から IP_JUDGE_EVERY_SEC 以上経っていれば judge を実行（多重起動はロックで防止）。
    検知ループから 60s ごと等に呼ばれる想定。重い取得を別スレッドで回しても安全。"""
    if not IP_JUDGE_ENABLE:
        return
    now = datetime.datetime.now()
    due = (now - _ip_judge_state["last_run"]).total_seconds() >= IP_JUDGE_EVERY_SEC
    if not (force or due):
        return
    if not _ip_judge_lock.acquire(blocking=False):
        return   # 前回の judge がまだ走っている
    _ip_judge_state["last_run"] = now
    try:
        run_ip_judge()
        rebuild_json()   # ip_judge_state.json を dashboard_state.json に反映
    finally:
        _ip_judge_lock.release()


# ── 4. 制御ループ ─────────────────────────────────────────────────────
# v0.1 は「間隔ポーリング」。EPICS アボートトリガ版（112p の LERtriggerEvent 相当、
# epics.camonitor でアボート信号を待つ）は次段階で追加する。

def _check_now(lag_min=CHK_LAG_MIN):
    """定期/--once チェック用の『現在時刻』文字列を返す。

    レガシーは調査期間の終端を (渡した時刻 + minit_advance) で決めるため、
    lag_min 分だけ過去にずらして終端を過去に固定する。これで全側室が完成済みの
    同一データを取得し、行数(時刻点数)が揃って Get_Fit_CHK_Strg の連結が成立する。
    """
    t = datetime.datetime.now() - datetime.timedelta(minutes=lag_min)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _run_ring(ring, now):
    """1リング分を実行し、所要時間を表示して JSON を更新する。"""
    t0 = time.time()
    print("[%s] チェック開始 %s ..." % (ring, now), flush=True)
    try:
        run_check(ring, now)
    except Exception as ex:
        # 完全なトレースバックを出す（chr() 例外などの発生箇所を特定するため）
        import traceback
        print("[%s] run_check 例外: %s" % (ring, ex), flush=True)
        print("---- トレースバック（ここから）----", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        print("---- トレースバック（ここまで）----", flush=True)
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
                    # 終端を過去に固定して側室間の行数ずれを防ぐ（アボート経路は対象外）
                    when = now - datetime.timedelta(minutes=CHK_LAG_MIN)
                    threading.Thread(target=do_check, args=(ring, when),
                                     daemon=True).start()
            # イオンポンプ judge を別スレッドで（CCG ループを止めない）相乗り実行
            threading.Thread(target=_maybe_run_ip_judge, daemon=True).start()
    except KeyboardInterrupt:
        print("\n監視を停止しました")


def loop_interval(interval_sec=4 * 3600):
    print("detector_headless 起動（間隔 %d 秒ごとにチェック）" % interval_sec)
    while True:
        now = _check_now()
        for ring in ("LER", "HER"):
            _run_ring(ring, now)   # リングごとに JSON も更新
        _maybe_run_ip_judge()      # イオンポンプ judge も相乗り（間隔は IP_JUDGE_EVERY_SEC）
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

    #   --once     1回だけ検知して終了
    #   --ip-judge イオンポンプ judge だけ1回実行（ip_judge_state.json 更新・初回投入/テスト用）
    #   --watch    アボートPVを監視し、アボート時＋定期に検知（常駐・実機向け推奨）
    #   引数なし    単純な定期ループ（loop_interval）
    if "--ip-judge" in sys.argv:
        _maybe_run_ip_judge(force=True)
        print("=== --ip-judge 完了 ===", flush=True)
    elif "--once" in sys.argv:
        now = _check_now()
        t_all = time.time()
        for ring in ("LER", "HER"):
            _run_ring(ring, now)   # リングごとに JSON 更新（片方終わればすぐ表示に反映）
        _maybe_run_ip_judge(force=True)   # イオンポンプ judge も1回（モデルがあれば）
        print("=== --once 完了（合計 %.0f 秒）dashboard_state.json を更新しました ==="
              % (time.time() - t_all), flush=True)
    elif "--watch" in sys.argv:
        watch_aborts()
    else:
        loop_interval()
