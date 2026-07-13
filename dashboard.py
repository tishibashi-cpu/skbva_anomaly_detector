#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperKEKB 圧力異常検知 — 監視ダッシュボード（プロトタイプ）

依存ゼロ（Python 標準ライブラリのみ）。外部 CDN も使いません。
制御ネット上の kekb-co-user0X で:
    /cont/python/x86_64-AlmaLinux9/3.9.17/bin/python3 dashboard.py
で起動し、自席 PC からは SSH トンネル経由でブラウザ http://localhost:18050 を開きます。

データの渡し方:
  - 同じディレクトリに dashboard_state.json があればそれを表示します（本番モード）。
  - 無ければ内蔵のダミーデータを表示します（プロトタイプ確認用）。
  実運用では、既存の検知プログラムに dashboard_state.json を書き出すフックを
  足すだけで、このダッシュボードはそのまま使えます（判定ロジックには触りません）。
"""

import json
import os
import sys

_DASH_DIR = os.path.dirname(os.path.abspath(__file__))
import random
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 共用サーバーの負荷表示（標準ライブラリのみ）。無くてもダッシュボードは動く。
try:
    import sysload
    _cpu_sampler = sysload.CpuSampler()
    _cpu_sampler.percent()   # 基準サンプルを1回取っておく
    _proc_sampler = sysload.ProcSampler()
    _proc_sampler.sample()   # 自プロセスも基準取り
except Exception:
    sysload = None
    _cpu_sampler = None
    _proc_sampler = None

# detector_headless.py（別プロセス。親子関係が無いため ps の子プロセス探索では見つからない）の
# CPU/メモリも表示する。singleton.py が書くロックファイルからPIDを教えてもらう方式
# （実機でdetector_headless.py --watchがCPU使用率1000%超に達した際、dashboard.py自身の
# 「本プログラムCPU」表示は無関係の値しか出ておらず気づけなかったための追加）。
_DETECTOR_LOCK = os.path.join(_DASH_DIR, ".detector.lock")
try:
    import singleton
    _detector_proc_sampler = sysload.ProcSampler(pid=os.getpid()) if sysload else None
except Exception:
    singleton = None
    _detector_proc_sampler = None

# 蓄積電流（リアルタイム）取得。EPICS が無くてもダッシュボードは動く。
try:
    import beamcurrent
except Exception:
    beamcurrent = None

# 詳細ビュー用の生データ取得（legacy 準拠の窓で圧力/ビームを引き直す）。
# numpy / kblogrd が無くてもダッシュボード自体は起動できるよう任意依存にする。
try:
    import record_raw
except Exception:
    record_raw = None

PORT = 18050
# dashboard_state.json はこのスクリプトと同じ場所（トップ）に置かれる。
# CWD に依存しないよう __file__ 基準で解決する。
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_state.json")
LABEL_QUEUE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_queue.jsonl")
# 温度計異常検知（temp_detector/ サブパッケージ、または環境変数で明示指定）が生成する状態ファイル。
# 新レイアウト: skbva_anomaly_detector/temp_detector/temp_dashboard_state.json（既定）。
# 旧レイアウト（temp_detector が兄弟ディレクトリ skbva_temp_detector/ だった配置）にも
# 後方互換でフォールバックする。環境変数が最優先。
_TEMP_STATE_CANDIDATES = [
    os.path.join(_DASH_DIR, "temp_detector", "temp_dashboard_state.json"),          # 新レイアウト
    os.path.join(os.path.dirname(_DASH_DIR), "skbva_temp_detector", "temp_dashboard_state.json"),  # 旧レイアウト
]
TEMP_STATE_FILE = os.environ.get("TEMP_DASHBOARD_STATE") or next(
    (p for p in _TEMP_STATE_CANDIDATES if os.path.isfile(p)), _TEMP_STATE_CANDIDATES[0])

# 機器劣化検知（temp_equipment.py の run_periodic_judge が書く。センサ故障ではなく測定対象
# 機器側の熱結合劣化を見る、温度計異常検知とは別の判定軸）。パス解決の考え方はTEMP_STATE_FILEと同じ。
_EQUIPMENT_STATE_CANDIDATES = [
    os.path.join(_DASH_DIR, "temp_detector", "temp_equipment_state.json"),
    os.path.join(os.path.dirname(_DASH_DIR), "skbva_temp_detector", "temp_equipment_state.json"),
]
EQUIPMENT_STATE_FILE = os.environ.get("EQUIPMENT_DASHBOARD_STATE") or next(
    (p for p in _EQUIPMENT_STATE_CANDIDATES if os.path.isfile(p)), _EQUIPMENT_STATE_CANDIDATES[0])

# 冷却水流量計異常検知（flow_detector/flow_headless.py が書く。ビーム電流と無関係の機器で、
# センサ自身の異常だけを見る判定軸。リング概念が無いので rings ラッパーは持たずフラットなJSON）。
_FLOW_STATE_CANDIDATES = [
    os.path.join(_DASH_DIR, "flow_detector", "flow_dashboard_state.json"),
    os.path.join(os.path.dirname(_DASH_DIR), "skbva_flow_detector", "flow_dashboard_state.json"),
]
FLOW_STATE_FILE = os.environ.get("FLOW_DASHBOARD_STATE") or next(
    (p for p in _FLOW_STATE_CANDIDATES if os.path.isfile(p)), _FLOW_STATE_CANDIDATES[0])

# ----------------------------------------------------------------------------
# ダミーデータ生成（実運用 JSON と同じスキーマ）
# ----------------------------------------------------------------------------
def _series_rising(n=60, base=6e-8, slope=2.2e-9, noise=3e-9):
    """加熱・放電型: 圧力が時間とともに上昇。"""
    t = list(range(n))
    pres = [base + slope * i + random.uniform(-noise, noise) for i in range(n)]
    beam = [800 + random.uniform(-15, 15) for _ in range(n)]
    return t, pres, beam

def _series_burst(n=60, base=8e-8, noise=4e-9):
    """Tail バースト型: ところどころ突発的なスパイク。"""
    t = list(range(n))
    pres = []
    for i in range(n):
        spike = random.choice([0, 0, 0, 0, 1.2e-7]) if i % 7 == 0 else 0
        pres.append(base + spike + random.uniform(-noise, noise))
    beam = [700 + random.uniform(-20, 20) for _ in range(n)]
    return t, pres, beam

def _series_gradual(n=60, base=5e-8, slope=9e-10, noise=2e-9):
    """軌道異常・リーク型: 緩やかな上昇。"""
    t = list(range(n))
    pres = [base + slope * i + random.uniform(-noise, noise) for i in range(n)]
    beam = [802 + random.uniform(-10, 10) for _ in range(n)]
    return t, pres, beam

def make_dummy_state():
    t1, p1, b1 = _series_rising()
    t2, p2, b2 = _series_burst()
    t3, p3, b3 = _series_gradual()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    anomalies = [
        {
            "id": "her-d01-h16",
            "record": "VAHCCG:D01_H16:PRES",
            "ring": "HER", "section": "D01", "place": "大穂",
            "period": "Storage", "count": 8, "max_count": 8,
            "cause": "異常加熱 or 放電", "severity": "danger",
            "series": {"t": t1, "pressure": p1, "beam": b1,
                       "abnormal": [1 if i >= len(t1)-8 else 0 for i in range(len(t1))]},
        },
        {
            "id": "ler-d04-l07",
            "record": "VALCCG:D04_L07:PRES",
            "ring": "LER", "section": "D04", "place": "富士",
            "period": "Tail", "count": 6, "max_count": 6,
            "cause": "圧力バースト or リーク", "severity": "danger",
            "series": {"t": t2, "pressure": p2, "beam": b2,
                       "abnormal": [1 if i >= len(t2)-6 else 0 for i in range(len(t2))]},
        },
        {
            "id": "ler-d10-l09",
            "record": "VALCCG:D10_L09:PRES",
            "ring": "LER", "section": "D10", "place": "日光 (Wiggler)",
            "period": "Storage", "count": 4, "max_count": 4,
            "cause": "軌道異常 or リーク", "severity": "warning",
            "series": {"t": t3, "pressure": p3, "beam": b3,
                       "abnormal": [1 if i >= len(t3)-4 else 0 for i in range(len(t3))]},
        },
    ]

    sections = {"LER": {}, "HER": {}}
    for ring in ("LER", "HER"):
        for i in range(1, 13):
            sections[ring]["D%02d" % i] = None
    for a in anomalies:
        sections[a["ring"]][a["section"]] = a["severity"]

    ip_anoms = [
        {"id": "ip-demo-1", "pv": "VALIP:D12_IP_L23:CUR", "ring": "LER", "section": "D12",
         "supply": "KEK", "severity": "danger", "severity_n": 3, "count": 6,
         "series_count": [0, 1, 2, 3, 2, 3, 4, 5, 6],
         "kind": "acute", "deviation_dex": 2.1, "reason": "feedthrough_discharge_suspect"},
        {"id": "ip-demo-2", "pv": "VALIP:D01_IP_H14:CUR", "ring": "HER", "section": "D01",
         "supply": "KEK", "severity": "danger", "severity_n": 3, "count": 4,
         "series_count": [0, 0, 1, 2, 3, 4, 3, 4],
         "kind": "acute", "deviation_dex": 1.3, "reason": "feedthrough_discharge_suspect"},
        {"id": "ip-demo-3", "pv": "VALIP:D07_4U_H06_A02C4:CUR", "ring": "HER", "section": "D07",
         "supply": "Agilent_4U", "severity": "danger", "severity_n": 3, "count": 9,
         "series_count": [3, 4, 5, 6, 7, 8, 9, 9, 9],
         "kind": "chronic", "deviation_dex": 0.1, "reason": "feedthrough_discharge_suspect"},
        {"id": "ip-demo-4", "pv": "VALIP:D04_IP_L06:CUR", "ring": "LER", "section": "D04",
         "supply": "KEK", "severity": "warning", "severity_n": 2,
         "kind": "chronic", "deviation_dex": 0.3, "reason": "over_current"},
        {"id": "ip-demo-5", "pv": "VALIP:D06_IP_L11:CUR", "ring": "LER", "section": "D06",
         "supply": "KEK", "severity": "watch", "severity_n": 1,
         "kind": "chronic", "deviation_dex": 0.0, "reason": "decoupled"},
    ]
    ip_sections = {r: {"D%02d" % i: None for i in range(1, 13)} for r in ("LER", "HER")}
    for a in ip_anoms:
        ip_sections[a["ring"]][a["section"]] = a["severity"]

    return {
        "updated": now,
        "classifier": "Keras",
        "rings": {
            "LER": {"beam_on": True, "current_mA": 802},
            "HER": {"beam_on": True, "current_mA": 712},
        },
        "summary": {"monitored": {"total": 605, "LER": 308, "HER": 297}, "critical": 2, "warning": 1},
        "anomalies": anomalies,
        "sections": sections,
        "ion_pump_anomalies": ip_anoms,
        "ip_sections": ip_sections,
    }

def _dummy_ts_plot(base, amp, n=120, beam=True, seed=1):
    """デモ用の間引き時系列（temp/flow の plot と同じ形）を合成する。"""
    import math as _m
    import random as _rnd
    r = _rnd.Random(seed)
    t = ["07/04/2026 %02d:%02d:00" % ((i * 12) // 60, (i * 12) % 60) for i in range(n)]
    vals = [base + amp * _m.sin(i / 14.0) + r.gauss(0, abs(amp) * 0.08 + 0.05) for i in range(n)]
    bm = [800 - 2.2 * i + r.gauss(0, 6) for i in range(n)] if beam else None
    return t, vals, bm

def make_dummy_temp_state():
    """温度計タブのデモ用ダミー（temp_headless.py が書く temp_dashboard_state.json と同じ形）。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ler_anoms = [
        {"pv": "VALTMP:D01M095:QLC3LE:BL", "ring": "LER", "section": "D01", "tag": "QLC3LE", "suffix": "BL",
         "severity": 3, "reason": "range_low_short_suspect", "n_pts": 288,
         "t_med": -25.6, "t_min": -30.0, "t_max": -22.2, "frac_low": 1.0, "frac_hot": None,
         "r_beam": None, "n_events": 1, "has_model": True},
        {"pv": "VALTMP:D02_221:QKARE:BL", "ring": "LER", "section": "D02", "tag": "QKARE", "suffix": "BL",
         "severity": 2, "reason": "intermittent_excursion", "n_pts": 288,
         "t_med": 14.9, "t_min": -6.8, "t_max": 18.2, "frac_low": None, "frac_hot": None,
         "r_beam": None, "n_events": 4, "has_model": True},
    ]
    her_anoms = [
        {"pv": "VAHTMP:D01M106:QLY2LE_2:BL", "ring": "HER", "section": "D01", "tag": "QLY2LE_2", "suffix": "BL",
         "severity": 3, "reason": "range_high_noheat_suspect", "n_pts": 288,
         "t_med": 39.1, "t_min": 36.5, "t_max": 39.6, "frac_low": None, "frac_hot": 1.0,
         "r_beam": None, "n_events": None, "has_model": True},
        {"pv": "VAHTMP:D12_136:QD1E_4:BL", "ring": "HER", "section": "D12", "tag": "QD1E_4", "suffix": "BL",
         "severity": 2, "reason": "beam_anticorrelated", "n_pts": 288,
         "t_med": 62.0, "t_min": 55.4, "t_max": 82.8, "frac_low": None, "frac_hot": None,
         "r_beam": -0.71, "n_events": None, "has_model": True},
        {"pv": "VAHTMP:D09_119:B2E_42:BL", "ring": "HER", "section": "D09", "tag": "B2E_42", "suffix": "BL",
         "severity": 2, "reason": "glitch", "n_pts": 288,
         "t_med": 35.8, "t_min": 23.4, "t_max": 44.8, "frac_low": None, "frac_hot": None,
         "r_beam": None, "n_events": None, "has_model": True},
    ]
    def ring_block(anoms, n_judged):
        for k, a in enumerate(anoms):   # 各異常にクリック展開プロット用の合成時系列を付ける
            t, vals, bm = _dummy_ts_plot(a["t_med"], (a["t_max"] - a["t_min"]) / 3.0, seed=k + 1)
            a["plot"] = {"t": t, "temp": vals, "beam": bm}
        return {"window": {"start": "20260701000000", "end": "20260702000000", "hours": 24, "interval_sec": 300},
                "baseline": {"start": "20260624000000", "end": "20260627000000", "last_days": 5.0, "days": 3.0},
                "stats": {"n_judged": n_judged, "n_quick_sev3": sum(1 for a in anoms if a["severity"] >= 3),
                          "have_beam": True},
                "n_anomalies": len(anoms), "anomalies": anoms}
    return {"generated_at": now, "rings": {"LER": ring_block(ler_anoms, 1547), "HER": ring_block(her_anoms, 1258)}}

def make_dummy_equipment_state():
    """機器劣化検知タブのデモ用ダミー（temp_equipment.py の run_periodic_judge が書く
    temp_equipment_state.json と同じ形。linear/hom 両モデルの表示形式を1件ずつ含める）。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ler_anoms = [
        {"pv": "VALTMP:D10_139:QD3E_11:BL", "ring": "LER", "section": "D10", "tag": "QD3E_11",
         "model": "linear", "severity": 3, "reason": "heating_gain_increase_severe",
         "ratio": 1.68, "delta": 0.0030, "delta_a": 2.1,
         "b_ref": 0.00425, "a_ref": 19.6, "now": {"b": 0.00715, "a": 21.7, "r": 0.91, "n": 812, "trust": True}},
        {"pv": "VALTMP:D04_221:QKARE:BL", "ring": "LER", "section": "D04", "tag": "QKARE",
         "model": "linear", "severity": 2, "reason": "heating_gain_increase",
         "ratio": 1.27, "delta": 0.0011, "delta_a": 0.4,
         "b_ref": 0.00410, "a_ref": 18.2, "now": {"b": 0.00521, "a": 18.6, "r": 0.88, "n": 790, "trust": True}},
    ]
    ir_anoms = [
        {"pv": "FB_MOVE:D01:QC1L:BWS:TEMP", "ring": "IR", "section": "D01", "tag": "QC1L",
         "model": "hom", "severity": 3, "reason": "heating_gain_increase_severe",
         "ratio": 1.62, "delta": 2.63, "delta_a": -0.3,
         "eval_i": 980.0, "eval_nb": 1576.0, "dT_ref": 4.25, "dT_now": 6.88,
         "r2_ref": 0.97, "now": {"w": [20.1, 0.00286, 5e-06], "r2": 0.98, "n": 640, "trust": True}},
    ]
    def _mk_equip_plot(a):
        """散布＋フィット曲線の合成プロット（linear/hom両形式）。基準/調査の傾き差が見えるように作る。"""
        import random as _rnd
        r = _rnd.Random(hash(a["pv"]) & 0xffff)
        ii = [50 + k * (1400 / 79.0) for k in range(80)]
        if a["model"] == "hom":
            w_ref = [20.1, 0.00286, 5e-06 / a["ratio"]]
            w_now = a["now"]["w"]
            nb = a["eval_nb"]
            f = lambda w, i: w[0] + w[1] * i + w[2] * (i * i / nb) ** 2
            ref = {"i": ii, "t": [f(w_ref, i) + r.gauss(0, 0.15) for i in ii]}
            nowp = {"i": ii, "t": [f(w_now, i) + r.gauss(0, 0.15) for i in ii]}
            return {"ref": ref, "now": nowp,
                    "ref_fit": {"i": ii, "t": [f(w_ref, i) for i in ii]},
                    "now_fit": {"i": ii, "t": [f(w_now, i) for i in ii]}}
        else:
            f = lambda a_, b_, i: a_ + b_ * i
            nowp = {"i": ii, "t": [f(a["now"]["a"], a["now"]["b"], i) + r.gauss(0, 0.3) for i in ii]}
            return {"ref": None, "now": nowp,
                    "ref_fit": {"i": [ii[0], ii[-1]],
                                "t": [f(a["a_ref"], a["b_ref"], ii[0]), f(a["a_ref"], a["b_ref"], ii[-1])]},
                    "now_fit": {"i": [ii[0], ii[-1]],
                                "t": [f(a["now"]["a"], a["now"]["b"], ii[0]), f(a["now"]["a"], a["now"]["b"], ii[-1])]}}
    def ring_block(anoms, n_judged):
        for a in anoms:
            a["plot"] = _mk_equip_plot(a)
        return {"window": {"start": "20260703000000", "end": "20260704000000", "hours": 24, "interval_sec": 300},
                "stats": {"n_judged": n_judged, "n_anomalies_sev3": sum(1 for a in anoms if a["severity"] >= 3)},
                "n_anomalies": len(anoms), "anomalies": anoms, "section_warnings": []}
    return {"generated_at": now, "rings": {
        "LER": ring_block(ler_anoms, 1550),
        "HER": {"skipped": True, "reason": "not_learned"},
        "IR": ring_block(ir_anoms, 42),
    }}

def make_dummy_flow_state():
    """流量計異常検知タブのデモ用ダミー（flow_headless.py が書く flow_dashboard_state.json と
    同じ形。実際に確認された故障パターン3種＝不安定化/校正基準比の低下/値の固着を1件ずつ含める）。
    流量計はリング概念を持たないため、他タブと違い rings ラッパーが無いフラットな構造。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anomalies = [
        {"pv": "VA_FLS:D10_02_010:RATE", "section": "D10", "tag": "010", "sensor_id": "D10_02",
         "severity": 3, "reason": "frozen_low", "median_pct": 0.29, "cv_pct": 3.1, "n": 2880, "n_valid": 2880,
         "layers": {"frozen": {"range": 0.03, "is_frozen": True}}},
        {"pv": "VA_FLS:D05_14_018:RATE", "section": "D05", "tag": "018", "sensor_id": "D05_14",
         "severity": 2, "reason": "excess_noise", "median_pct": 99.6, "cv_pct": 12.8, "n": 2880, "n_valid": 2880,
         "layers": {"excess_noise": {"cv_pct": 12.8}}},
        {"pv": "VA_FLS:D04_15_084:RATE", "section": "D04", "tag": "084", "sensor_id": "D04_15",
         "severity": 1, "reason": "excess_noise_watch", "median_pct": 74.9, "cv_pct": 5.2, "n": 2880, "n_valid": 2880,
         "layers": {"excess_noise": {"cv_pct": 5.2}}},
    ]
    # 各異常に、その故障パターンらしい合成時系列を付ける（クリック展開プロット用）
    import random as _rnd
    _r = _rnd.Random(7)
    _t = ["07/04/2026 %02d:%02d:00" % ((k * 12) // 60, (k * 12) % 60) for k in range(120)]
    anomalies[0]["plot"] = {"t": _t, "v": [0.29 + _r.gauss(0, 0.01) for _ in range(120)]}          # 固着(近ゼロ)
    anomalies[1]["plot"] = {"t": _t, "v": [99.6 + _r.gauss(0, 12.8) for _ in range(120)]}          # 不安定化
    anomalies[2]["plot"] = {"t": _t, "v": [74.9 + _r.gauss(0, 3.9) for _ in range(120)]}           # 軽度不安定
    return {"generated_at": now,
            "window": {"start": "20260703000000", "end": "20260704000000", "hours": 24, "interval_sec": 30},
            "stats": {"n_judged": 678, "n_insufficient": 0, "n_anomalies_sev3": 1,
                     "n_anomalies_sev2": 1, "n_anomalies_sev1": 1},
            "n_anomalies": len(anomalies), "anomalies": anomalies}

def load_state():
    # デモモードでは dashboard_state.json があっても内蔵ダミーを使う
    # （ダミー series には abnormal が入っており、サイドバーのトレンドが正しく描ける）。
    if os.environ.get("RECORD_RAW_DEMO"):
        return make_dummy_state()
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": "state file read failed: %s" % e}
    return make_dummy_state()

def load_temp_state():
    if os.environ.get("RECORD_RAW_DEMO"):
        return make_dummy_temp_state()
    if os.path.isfile(TEMP_STATE_FILE):
        try:
            with open(TEMP_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": "temp state file read failed: %s" % e}
    return {"rings": {}, "error": "%s が見つかりません。temp_headless.py を実行してください。"
            % os.path.basename(TEMP_STATE_FILE)}

def load_equipment_state():
    if os.environ.get("RECORD_RAW_DEMO"):
        return make_dummy_equipment_state()
    if os.path.isfile(EQUIPMENT_STATE_FILE):
        try:
            with open(EQUIPMENT_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": "equipment state file read failed: %s" % e}
    return {"rings": {}, "error": "%s が見つかりません。detector_headless.py --equipment-judge "
            "（または temp_equipment.py judge-all）を実行してください。"
            % os.path.basename(EQUIPMENT_STATE_FILE)}

def load_flow_state():
    if os.environ.get("RECORD_RAW_DEMO"):
        return make_dummy_flow_state()
    if os.path.isfile(FLOW_STATE_FILE):
        try:
            with open(FLOW_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": "flow state file read failed: %s" % e}
    return {"anomalies": [], "error": "%s が見つかりません。detector_headless.py --flow-judge "
            "（または flow_detector/flow_headless.py --once）を実行してください。"
            % os.path.basename(FLOW_STATE_FILE)}

# ----------------------------------------------------------------------------
# フロントエンド（HTML/CSS/JS を 1 つにまとめて配信）
# ----------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SuperKEKB MR真空システム異常モニター</title>
<style>
  :root {
    --bg: #14161b; --surface: #1c1f27; --surface2: #232732;
    --line: #2c313d; --text: #e7e8ec; --muted: #969aa6; --faint: #6b7080;
    --danger: #e2574a; --danger-bg: #2a1a18; --danger-fg: #f0a097;
    --warning: #e6a02a; --warning-bg: #2a2316; --warning-fg: #f0c878;
    --ok: #4fc59a; --accent: #5dcaa5;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --sans: system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: var(--sans); font-size: 14px; line-height: 1.5; }
  a { color: inherit; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 18px 22px 60px; }

  .topbar { display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 12px; padding-bottom: 16px; }
  .brand { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  .brand h1 { font-size: 17px; font-weight: 600; margin: 0; letter-spacing: .2px; }
  .ip-title { font-size: 17px; font-weight: 600; letter-spacing: .2px; color: var(--text); }
  .section-title.ip-section { margin-top: 30px; }
  .ring-stat { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
  .dot.on { background: var(--ok); }
  .meta { display: flex; align-items: center; gap: 14px; font-size: 12.5px; color: var(--muted); }
  .pill { padding: 3px 10px; border-radius: 6px; background: var(--surface2); color: var(--muted); }
  .pill.load-ok   { color: var(--ok); }
  .pill.load-warn { color: #e6a02a; }
  .pill.load-high { color: var(--danger-fg); background: var(--danger-bg); }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 10px; margin: 4px 0 22px; }
  .stat { background: var(--surface); border: 1px solid var(--line);
          border-radius: 10px; padding: 13px 15px; }
  .stat .lbl { font-size: 12.5px; color: var(--muted); }
  .stat .num { font-size: 25px; font-weight: 600; margin-top: 2px; }
  .stat .sub { font-size: 11.5px; color: var(--muted); margin-top: 3px; }
  .num.danger { color: var(--danger-fg); }
  .num.warning { color: var(--warning-fg); }
  .num.ok { color: var(--ok); }

  .section-title { font-size: 13px; color: var(--muted); margin: 0 0 10px; font-weight: 500; }

  .alist { display: flex; flex-direction: column; gap: 9px; }
  .acard { background: var(--surface); border: 1px solid var(--line);
           border-left: 3px solid var(--faint); border-radius: 0 10px 10px 0;
           padding: 13px 16px; display: flex; align-items: center; gap: 16px;
           flex-wrap: wrap; cursor: pointer; transition: background .12s; }
  .acard:hover { background: var(--surface2); }
  .acard.danger { border-left-color: var(--danger); }
  .acard.warning { border-left-color: var(--warning); }
  .acard.watch { border-left-color: var(--faint); }
  .acard.sel { background: var(--surface2); border-color: var(--accent); }
  /* イオンポンプ異常カード（CCG とは別セクション）*/
  .ipcard { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
            background: var(--surface); border: 1px solid var(--line);
            border-left: 3px solid var(--faint); border-radius: 0 10px 10px 0;
            padding: 11px 15px; cursor: pointer; transition: background .12s; }
  .ipcard:hover { background: var(--surface2); }
  .ipcard.danger { border-left-color: var(--danger); }
  .ipcard.warning { border-left-color: var(--warning); }
  .ipcard.watch { border-left-color: var(--faint); }
  .ipcard.sel { background: var(--surface2); border-color: var(--accent); }
  .sevtag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
            min-width: 38px; text-align: center; }
  .sevtag.danger { background: var(--danger-bg); color: var(--danger-fg); }
  .sevtag.warning { background: var(--warning-bg); color: var(--warning-fg); }
  .sevtag.watch { background: var(--surface2); color: var(--muted); }
  .kindtag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; }
  .kindtag.acute { background: #4a1d1d; color: #ff8f8f; border: 1px solid #7a2a2a; }
  .kindtag.chronic { background: #2a2f1d; color: #d9d98f; border: 1px solid #4a4f2a; }
  .kindtag.unknown { background: var(--surface2); color: var(--faint); }
  .cnttag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
            background: var(--surface2); color: var(--muted); font-family: var(--mono); }
  .devtag2 { font-size: 11px; color: var(--muted); font-family: var(--mono); }
  .count { min-width: 60px; text-align: center; }
  .count .n { font-size: 25px; font-weight: 600; line-height: 1; }
  .count.danger .n { color: var(--danger-fg); }
  .count.warning .n { color: var(--warning-fg); }
  .count .of { font-size: 11px; color: var(--faint); }
  .ainfo { flex: 1; min-width: 190px; }
  .rec { font-family: var(--mono); font-size: 14px; }
  .loc { font-size: 12px; color: var(--muted); margin-top: 1px; }
  .cause { display: inline-block; margin-top: 5px; font-size: 12px;
           padding: 2px 9px; border-radius: 6px; }
  .cause.danger { background: var(--danger-bg); color: var(--danger-fg); }
  .cause.warning { background: var(--warning-bg); color: var(--warning-fg); }
  .spark { flex-shrink: 0; }
  .iptrend { flex-shrink: 0; }

  .detail { margin-top: 10px; background: var(--surface); border: 1px solid var(--accent);
            border-radius: 10px; padding: 16px 18px; }
  .detail h3 { margin: 0 0 2px; font-size: 15px; font-family: var(--mono); font-weight: 600; }
  .detail .sub { font-size: 12.5px; color: var(--muted); margin-bottom: 12px; }
  .plotrow { display: grid; grid-template-columns: 1fr; gap: 14px; }
  .plotbox { background: var(--surface2); border: 1px solid var(--line);
             border-radius: 8px; padding: 10px 12px; }
  .plotbox .cap { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .btns { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
  button { font-family: var(--sans); font-size: 13px; color: var(--text);
           background: var(--surface2); border: 1px solid var(--line);
           border-radius: 7px; padding: 7px 14px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.close { margin-left: auto; }

  .map { margin-top: 26px; }
  .maprow { display: flex; gap: 5px; align-items: center; margin-top: 6px; }
  .maprow .rl { width: 34px; font-size: 12px; color: var(--faint); }
  .cells { display: flex; gap: 5px; flex: 1; }
  .cell { flex: 1; min-width: 34px; text-align: center; font-size: 11px;
          padding: 7px 0; border-radius: 6px; background: var(--surface);
          border: 1px solid var(--line); color: var(--muted); cursor: default; }
  .cell.danger { background: var(--danger-bg); border-color: var(--danger);
                 color: var(--danger-fg); }
  .cell.warning { background: var(--warning-bg); border-color: var(--warning);
                  color: var(--warning-fg); }

  .legend { display: flex; gap: 16px; margin-top: 14px; font-size: 12px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .sw { width: 11px; height: 11px; border-radius: 3px; }
  .toast { position: relative; }
  .footnote { margin-top: 22px; font-size: 12px; color: var(--faint); }
  .empty { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
           padding: 34px 20px; text-align: center; }
  .empty .emoji { font-size: 26px; color: var(--ok); margin-bottom: 8px; }
  .empty .etxt { font-size: 15px; color: var(--text); }
  .empty .esub { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
  /* イオンポンプパネル */
  .devtag { font-size: 10.5px; color: #b06a1a; background: #f6e8d2; border-radius: 4px;
            padding: 1px 6px; margin-left: 8px; vertical-align: middle; }
  .ipmeta { font-size: 11.5px; color: var(--muted); margin: -4px 0 10px; }
  .iprow { display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start; }
  .iplabel { width: 38px; font-weight: 600; color: var(--muted); padding-top: 6px; flex-shrink: 0; }
  .ipgrid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 6px; flex: 1; }
  @media (max-width: 900px) { .ipgrid { grid-template-columns: repeat(6, 1fr); } }
  .ipcard { background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
            padding: 6px 7px; cursor: pointer; transition: border-color .12s; }
  .ipcard:hover { border-color: #4a90d9; }
  .ipcard.sel { border-color: #4a90d9; box-shadow: 0 0 0 1px #4a90d9 inset; }
  .ipsec { font-size: 12px; font-weight: 600; display: flex; justify-content: space-between;
           align-items: center; }
  .badge { font-size: 9px; border-radius: 3px; padding: 0 4px; font-weight: 600; }
  .bkek { background: #e3eef7; color: #2c6291; }
  .b4u { background: #f0e6f7; color: #7a3fa0; }
  .ipbar { height: 4px; background: var(--line); border-radius: 2px; margin: 5px 0 3px; overflow: hidden; }
  .ipbar span { display: block; height: 100%; background: linear-gradient(90deg,#5aa0e0,#e2574a); }
  .ipval { font-size: 11px; font-variant-numeric: tabular-nums; }
  .ipn { font-size: 10px; color: var(--muted); }
  .ippvs { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
           padding: 10px 12px; margin: 2px 0 12px 48px; }
  .ippvs-h { font-size: 12.5px; font-weight: 600; margin-bottom: 6px; }
  .ippv { display: flex; align-items: center; gap: 10px; padding: 3px 0;
          border-top: 1px solid var(--line); }
  .ippv:first-of-type { border-top: none; }
  .ippv-name { flex: 1; font-size: 11.5px; font-family: ui-monospace, monospace; color: var(--text); }
  .ippv-val { width: 92px; text-align: right; font-size: 11px; font-variant-numeric: tabular-nums;
              color: var(--muted); }
  .tabbar { display: flex; gap: 4px; margin-bottom: 18px; border-bottom: 1px solid var(--line); }
  .tabbtn { appearance: none; background: transparent; border: none; color: var(--muted);
            font: inherit; font-size: 13.5px; font-weight: 500; padding: 9px 16px 10px;
            cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .tabbtn:hover { color: var(--text); }
  .tabbtn.active { color: var(--text); border-bottom-color: var(--accent); }
  .tabbtn .badge { display: inline-block; margin-left: 7px; padding: 1px 7px; border-radius: 8px;
                    font-size: 11px; background: var(--surface2); color: var(--muted); }
  .tabbtn .badge.hot { background: var(--danger-bg); color: var(--danger-fg); }
  .tabpanel { display: none; }
  .tabpanel.active { display: block; }
  .temp-table { width: 100%; border-collapse: collapse; font-size: 14px; }
  .temp-table th { text-align: left; color: var(--muted); font-weight: 500; font-size: 13px;
                    padding: 6px 10px; border-bottom: 1px solid var(--line); }
  .temp-table td { padding: 8px 10px; border-bottom: 1px solid var(--line); }
  .temp-table tr:hover td { background: var(--surface2); }
  .sev-badge { display: inline-block; min-width: 18px; text-align: center; padding: 2px 8px;
               border-radius: 5px; font-weight: 600; font-size: 13px; }
  .sev-badge.sev3 { background: var(--danger-bg); color: var(--danger-fg); }
  .sev-badge.sev2 { background: var(--warning-bg); color: var(--warning-fg); }
  .sev-badge.sev1 { background: var(--surface2); color: var(--muted); }
  .temp-ring-title { font-weight: 600; margin: 18px 0 8px; color: var(--text); font-size: 15px; }
  .temp-ring-title:first-child { margin-top: 0; }
  .archiver-banner { background: var(--warning-bg); color: var(--warning-fg);
                      border: 1px solid var(--warning); border-radius: 8px;
                      padding: 11px 15px; margin: 0 0 14px; font-size: 13.5px; }
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <h1>SuperKEKB MR真空システム異常モニター</h1>
      <span class="ring-stat" id="ler-stat"><span class="dot"></span> LER —</span>
      <span class="ring-stat" id="her-stat"><span class="dot"></span> HER —</span>
    </div>
    <div class="meta">
      <span class="pill" id="sysload" title="共用サーバー全体の負荷">負荷 —</span>
      <span class="pill" id="selfload" title="このダッシュボード（＋子プロセス）自身の使用量">本プログラム —</span>
      <span class="pill" id="detectorload" title="detector_headless.py（別プロセス）の使用量。&#10;本プログラムの数字とは別物なので注意（親子関係が無い独立プロセス）。">検知プログラム —</span>
      <span id="updated">— 更新待ち</span>
      <span class="pill" id="classifier">判定: —</span>
    </div>
  </div>

  <div class="tabbar" id="tabbar">
    <button class="tabbtn active" data-tab="ccg" onclick="switchTab('ccg')">圧力異常検知<span class="badge" id="tab-badge-ccg"></span></button>
    <button class="tabbtn" data-tab="ip" onclick="switchTab('ip')">イオンポンプ異常放電<span class="badge" id="tab-badge-ip"></span></button>
    <button class="tabbtn" data-tab="temp" onclick="switchTab('temp')">温度計異常検知<span class="badge" id="tab-badge-temp"></span></button>
    <button class="tabbtn" data-tab="equip" onclick="switchTab('equip')">機器劣化検知<span class="badge" id="tab-badge-equip"></span></button>
    <button class="tabbtn" data-tab="flow" onclick="switchTab('flow')">流量計異常検知<span class="badge" id="tab-badge-flow"></span></button>
  </div>

  <div class="tabpanel active" id="tab-ccg">
    <div class="cards" id="summary"></div>

    <div class="section-title">検知された異常 — カウントの多い順</div>
    <div class="alist" id="alist"></div>
    <div id="detail-host"></div>
  </div>

  <div class="tabpanel" id="tab-ip">
    <div id="ip-anomalies"></div>
    <div id="ip-detail-host"></div>

    <div id="ionpumps"></div>
  </div>

  <div class="tabpanel" id="tab-temp">
    <div id="temp-anomalies"></div>
  </div>

  <div class="tabpanel" id="tab-equip">
    <div id="equip-anomalies"></div>
  </div>

  <div class="tabpanel" id="tab-flow">
    <div id="flow-anomalies"></div>
  </div>

  <div class="footnote" id="footnote"></div>
</div>

<script>
let STATE = null;
let TEMP_STATE = null;
let EQUIP_STATE = null;
let FLOW_STATE = null;
let SELECTED = null;
let TEMP_SEL = null;         // 展開中の温度異常 {ring, pv} または null
let EQUIP_SEL = null;        // 展開中の機器劣化異常 {ring, pv} または null
let FLOW_SEL = null;         // 展開中の流量計異常 pv または null（リング概念が無いのでキーはpvのみ）
let CUR_TAB = "ccg";
let DETAIL_RENDERED = null;   // 現在 詳細ビューに描画済みのレコード id（不要な再描画防止）

function switchTab(name) {
  CUR_TAB = name;
  document.querySelectorAll(".tabbtn").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tabpanel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
}

function fmtPres(v) {
  if (!isFinite(v) || v <= 0) return "0";
  const e = Math.floor(Math.log10(v));
  const m = v / Math.pow(10, e);
  return m.toFixed(2) + "e" + e;
}

// legacy「Abnormal Record Trend」相当のミニプロット。
//   横軸: Period（フィル順, 古い→新しい）
//   左軸: 異常数/P（トレーリング maxCount フィル内で異常と出た回数, 0〜maxCount）
//   右軸: Beam[mA]（各フィルの調査データ最大電流）
function trendPlot(a, color, w, h) {
  const ser = a.series || {};
  const beam = ser.beam || [];
  const n = beam.length;
  const maxCount = a.max_count || 8;
  if (n < 2) return spark(ser, color, w, h);
  if (!ser.abnormal) return spark(ser, color, w, h);
  const abn = ser.abnormal;
  const cnt = [];
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let k = Math.max(0, i - maxCount + 1); k <= i; k++) c += (abn[k] || 0);
    cnt.push(c);
  }
  const padL = 44, padR = 46, padT = 10, padB = 30;
  const bmax = Math.max.apply(null, beam) || 1;
  const X = i => padL + (i / (n - 1)) * (w - padL - padR);
  const YC = v => padT + (1 - v / maxCount) * (h - padT - padB);
  const YB = v => padT + (1 - v / bmax) * (h - padT - padB);
  const cy = (padT + (h - padB)) / 2;
  const FS = 11, FSL = 11;   // 目盛り / 軸ラベルのフォントサイズ
  let s = '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h +
          '" style="display:block">';
  // 横グリッド＋左右の目盛り（0 と max）
  for (let g = 0; g <= 1; g++) {
    const yy = padT + g * (h - padT - padB);
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 4) + '" y="' + (yy + 4) + '" fill="#9aa0ad" font-size="' + FS + '" text-anchor="end">' + (g === 0 ? maxCount : 0) + '</text>';
    s += '<text x="' + (w - padR + 4) + '" y="' + (yy + 4) + '" fill="#5dcaa5" font-size="' + FS + '">' + (g === 0 ? Math.round(bmax) : 0) + '</text>';
  }
  // 縦グリッド＋ x 目盛り（Period: 新しい→古い順に 0,-1,-2…。混む場合は間引く）
  const step = Math.max(1, Math.ceil(n / 7));
  for (let i = n - 1; i >= 0; i--) {
    const per = i - (n - 1);            // 右端=0, 左へ -1,-2,...
    if (per % step !== 0) continue;
    const xx = X(i);
    s += '<line x1="' + xx.toFixed(1) + '" y1="' + padT + '" x2="' + xx.toFixed(1) + '" y2="' + (h - padB) + '" stroke="#23272f"/>';
    s += '<text x="' + xx.toFixed(1) + '" y="' + (h - padB + 14) + '" fill="#9aa0ad" font-size="' + FS + '" text-anchor="middle">' + per + '</text>';
  }
  // 軸ラベル（縦書き：左 +90°, 右 -90°、目盛り数字とかぶらない外側に配置）
  s += '<text transform="rotate(-90 11 ' + cy.toFixed(1) + ')" x="11" y="' + cy.toFixed(1) + '" fill="' + color + '" font-size="' + FSL + '" text-anchor="middle">Anomaly Count</text>';
  s += '<text transform="rotate(-90 ' + (w - 9) + ' ' + cy.toFixed(1) + ')" x="' + (w - 9) + '" y="' + cy.toFixed(1) + '" fill="#5dcaa5" font-size="' + FSL + '" text-anchor="middle">Beam [mA]</text>';
  s += '<text x="' + ((padL + w - padR) / 2) + '" y="' + (h - 3) + '" fill="#9aa0ad" font-size="' + FS + '" text-anchor="middle">Period</text>';
  // ビーム電流（緑・点線, 右軸）
  s += '<polyline points="' + beam.map((v, i) => X(i).toFixed(1) + ',' + YB(v).toFixed(1)).join(' ') +
       '" fill="none" stroke="#5dcaa5" stroke-width="1.4" stroke-dasharray="3,2"/>';
  // 異常数（severity の色・実線, 左軸）
  s += '<polyline points="' + cnt.map((v, i) => X(i).toFixed(1) + ',' + YC(v).toFixed(1)).join(' ') +
       '" fill="none" stroke="' + color + '" stroke-width="2"/>';
  s += '</svg>';
  return s;
}

function spark(series, color, w, h) {
  const p = series.pressure;
  const min = Math.min.apply(null, p), max = Math.max.apply(null, p);
  const rng = (max - min) || 1;
  const pts = p.map((v, i) => {
    const x = (i / (p.length - 1)) * w;
    const y = h - ((v - min) / rng) * (h - 4) - 2;
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
  return '<svg class="spark" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">' +
         '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2"/></svg>';
}

function sevColor(sev) {
  return sev === "danger" ? "#e2574a" : sev === "warning" ? "#e6a02a" : "#6b7080";
}

function renderSummary(s) {
  const el = document.getElementById("summary");
  const m = s.summary.monitored;
  let mainNum, sub = "";
  if (m && typeof m === "object") {
    mainNum = m.total;
    sub = "LER " + m.LER + " / HER " + m.HER;
  } else {
    mainNum = "~" + m;   // 旧形式（数値）との後方互換
  }
  el.innerHTML =
    statCard("監視中の真空計", mainNum, "", sub) +
    statCard("要注意 (≥6)", s.summary.critical, "danger") +
    statCard("注意 (3–5)", s.summary.warning, "warning") +
    statCard("状態", s.summary.critical > 0 ? "ALERT" : "OK",
             s.summary.critical > 0 ? "danger" : "ok");
}
function statCard(lbl, num, cls, sub) {
  return '<div class="stat"><div class="lbl">' + lbl + '</div>' +
         '<div class="num ' + cls + '">' + num + '</div>' +
         (sub ? '<div class="sub">' + sub + '</div>' : '') + '</div>';
}

function renderTop(s) {
  document.getElementById("updated").textContent = "最終チェック " + (s.updated || "—");
  document.getElementById("classifier").textContent = "判定: " + (s.classifier || "—");
  renderRings();
}

let BEAM_RT = {LER: null, HER: null};   // リアルタイム蓄積電流（/api/beam）

function renderRings() {
  if (!STATE || !STATE.rings) return;
  setRing("ler-stat", "LER", STATE.rings.LER, BEAM_RT.LER);
  setRing("her-stat", "HER", STATE.rings.HER, BEAM_RT.HER);
}

function setRing(id, name, r, rt) {
  const el = document.getElementById(id);
  const hasRT = (rt !== null && rt !== undefined);
  const rtOn = hasRT && rt >= 1;          // 1 mA 以上を ON とみなす
  // ドットはリアルタイムがあればそれを、無ければ最終チェック由来を反映
  const dotOn = hasRT ? rtOn : (r && r.beam_on);
  let parts = [];
  if (hasRT) parts.push("現在 " + (rtOn ? (Math.round(rt) + " mA") : "OFF"));
  if (r) parts.push("最終チェック " + (r.beam_on ? ("ON " + Math.round(r.current_mA) + " mA") : "OFF"));
  if (!parts.length) parts.push("—");
  el.innerHTML = '<span class="dot ' + (dotOn ? "on" : "") + '"></span> ' +
                 name + " " + parts.join(" ⋅ ");
}

async function refreshBeam() {
  try {
    const r = await fetch("/api/beam", {cache: "no-store"});
    const d = await r.json();
    BEAM_RT.LER = d.LER; BEAM_RT.HER = d.HER;
    renderRings();
  } catch (e) { /* 取得できなくてもヘッダは最終チェック表示のまま */ }
}

function renderList(s) {
  const host = document.getElementById("alist");
  const items = s.anomalies.slice().sort((a, b) => b.count - a.count);
  if (items.length === 0) {
    const days = s.recent_window_days || 3;
    let msg = '<div class="empty"><div class="emoji">✓</div>' +
      '<div class="etxt">直近 ' + days + ' 日に検知された異常はありません</div>';
    if (s.last_anomaly)
      msg += '<div class="esub">最後に記録された異常: ' + s.last_anomaly + '</div>';
    host.innerHTML = msg + '</div>';
    return;
  }
  host.innerHTML = items.map(a => {
    const sev = a.severity;
    return '<div class="acard ' + sev + (SELECTED === a.id ? ' sel' : '') +
      '" onclick="select(\'' + a.id + '\')">' +
      '<div class="count ' + sev + '"><div class="n">' + a.count + '</div>' +
      '<div class="of">/ ' + a.max_count + ' checks</div></div>' +
      '<div class="ainfo"><div class="rec">' + a.record + '</div>' +
      '<div class="loc">' + a.ring + ' ' + a.section + ' · ' + a.period + '</div>' +
      '<span class="cause ' + sev + '">推定: ' + a.cause + '</span></div>' +
      trendPlot(a, sevColor(sev), 240, 112) + '</div>';
  }).join("");
}

function renderDetail() {
  const host = document.getElementById("detail-host");
  if (!SELECTED) {
    if (DETAIL_RENDERED !== null) { host.innerHTML = ""; DETAIL_RENDERED = null; }
    return;
  }
  // 既に同じレコードを表示中なら何もしない（5秒ごとの自動更新で再描画・再取得・
  // 強制スクロールが起きてプロットが点滅／スクロールが戻る問題を防ぐ）。
  if (SELECTED === DETAIL_RENDERED) return;
  const a = STATE.anomalies.find(x => x.id === SELECTED);
  if (!a) { host.innerHTML = ""; DETAIL_RENDERED = null; return; }
  DETAIL_RENDERED = SELECTED;
  host.innerHTML =
    '<div class="detail"><h3>' + a.record + '</h3>' +
    '<div class="sub">' + a.ring + ' ' + a.section + ' · ' +
      a.period + ' 部 · 推定原因: ' + a.cause + '</div>' +
    '<div id="raw-plots"><div class="plotbox" style="text-align:center;color:var(--muted);padding:22px">' +
      '生データを取得中…</div></div>' +
    '<div class="btns">' +
      '<button onclick="saveLabel(\'Normal\')">Normal として保存</button>' +
      '<button onclick="saveLabel(\'Abnormal\')">Abnormal として保存</button>' +
      '<button class="close" onclick="select(null)">閉じる</button>' +
    '</div></div>';
  host.scrollIntoView({behavior: "smooth", block: "nearest"});
  loadRaw(a);
}

async function loadRaw(a) {
  let v;
  try {
    const r = await fetch("/api/raw?ring=" + encodeURIComponent(a.ring) +
                          "&record=" + encodeURIComponent(a.record), {cache: "no-store"});
    v = await r.json();
  } catch (e) { v = {error: "取得に失敗しました: " + e}; }
  const box = document.getElementById("raw-plots");
  if (!box || SELECTED !== a.id) return;   // 別レコードに切替済みなら破棄
  const note = msg => '<div class="plotbox" style="text-align:center;color:var(--muted);padding:22px">' +
                      msg + '</div>';
  if (!v || v.error) { box.innerHTML = note(v && v.error ? v.error : "生データを取得できませんでした"); return; }
  const w = v.windows || {};
  let html = "";
  if (v.storage && v.storage.beam && v.storage.beam.length) {
    html += '<div class="plotbox"><div class="cap">圧力 vs ビーム電流（調査' +
            (w.chk_strg ? ' ' + fmtWin(w.chk_strg) : '') +
            '）　モデル: 圧力=w0·I+w1·(I²/Nb)²+w2　' +
            '<span style="color:#e2574a">●実測</span> ' +
            '<span style="color:#e2574a">━調査回帰</span> ' +
            '<span style="color:#9ec5fe">━基準回帰</span></div>' +
            scatterPlot(v.storage, v.reference) + '</div>';
    html += '<div class="plotbox"><div class="cap">圧力＆ビーム電流 vs 時刻（調査）　' +
            '<span style="color:#e2574a">━圧力[Pa]</span> ' +
            '<span style="color:#5dcaa5">┄ビーム電流[mA]</span></div>' +
            timeSeriesDual(v.storage) + '</div>';
  }
  box.innerHTML = html || note("表示できる生データがありません（窓ファイル未生成かレコード名不一致）");
}

// "YYYYMMDDhhmmss" または "start-end" を見やすく整形
function fmtWin(s) {
  const one = x => (x && x.length === 14)
    ? (x.slice(4,6)+"/"+x.slice(6,8)+" "+x.slice(8,10)+":"+x.slice(10,12)) : x;
  if (s && s.indexOf("-") > 0) { const p = s.split("-"); return one(p[0]) + "〜" + one(p[1]); }
  return one(s);
}
// "MM/DD/YYYY HH:MM:SS" → "HH:MM"
function hhmm(ts) { const m = /(\d{2}:\d{2}):\d{2}/.exec(ts || ""); return m ? m[1] : ""; }

function linePlot(t, y, color, w, h, fmt) {
  const pad = 38, padR = 10, padT = 10, padB = 22;
  const min = Math.min.apply(null, y), max = Math.max.apply(null, y);
  const rng = (max - min) || 1;
  const X = i => pad + (i / (t.length - 1)) * (w - pad - padR);
  const Y = v => padT + (1 - (v - min) / rng) * (h - padT - padB);
  const pts = y.map((v, i) => X(i).toFixed(1) + "," + Y(v).toFixed(1)).join(" ");
  let svg = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" style="display:block">';
  for (let g = 0; g <= 2; g++) {
    const yy = padT + g * (h - padT - padB) / 2;
    const val = max - g * rng / 2;
    svg += '<line x1="' + pad + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy +
           '" stroke="#2c313d" stroke-width="1"/>';
    svg += '<text x="4" y="' + (yy + 4) + '" fill="#6b7080" font-size="11">' + fmt(val) + '</text>';
  }
  svg += '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2"/>';
  svg += '</svg>';
  return svg;
}

// 圧力 vs ビーム電流の散布図（実測点＋調査回帰=青＋基準回帰=薄青）。legacy Make_Plot_Strg 相当。
function scatterPlot(storage, reference) {
  const w = 640, h = 240, padL = 78, padR = 14, padT = 12, padB = 36;
  let xs = storage.beam.slice(), ys = storage.pressure.slice();
  if (reference && reference.beam) { xs = xs.concat(reference.beam); ys = ys.concat(reference.pressure); }
  [storage.fit_chk, storage.fit_std].forEach(f => { if (f) { xs = xs.concat(f.beam); ys = ys.concat(f.pred); } });
  const xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
  const ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
  const xr = (xmax - xmin) || 1, yr = (ymax - ymin) || 1;
  const X = v => padL + ((v - xmin) / xr) * (w - padL - padR);
  const Y = v => padT + (1 - (v - ymin) / yr) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  for (let g = 0; g <= 3; g++) {
    const yy = padT + g * (h - padT - padB) / 3;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#6b7080" font-size="11" text-anchor="end">' + fmtPres(ymax - g * yr / 3) + '</text>';
  }
  for (let g = 0; g <= 3; g++) {
    const xx = padL + g * (w - padL - padR) / 3;
    s += '<text x="' + xx.toFixed(0) + '" y="' + (h - padB + 16) + '" fill="#6b7080" font-size="11" text-anchor="middle">' +
         Math.round(xmin + g * xr / 3) + '</text>';
  }
  s += '<text x="' + ((padL + w - padR) / 2) + '" y="' + (h - 5) + '" fill="#6b7080" font-size="11" text-anchor="middle">ビーム電流 [mA]</text>';
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#9aa0ad" font-size="11" text-anchor="middle">圧力 [Pa]</text>';
  if (reference && reference.beam) {
    for (let i = 0; i < reference.beam.length; i++)
      s += '<circle cx="' + X(reference.beam[i]).toFixed(1) + '" cy="' + Y(reference.pressure[i]).toFixed(1) + '" r="1.7" fill="#9ec5fe" opacity="0.45"/>';
  }
  for (let i = 0; i < storage.beam.length; i++)
    s += '<circle cx="' + X(storage.beam[i]).toFixed(1) + '" cy="' + Y(storage.pressure[i]).toFixed(1) + '" r="2.2" fill="#e2574a" opacity="0.75"/>';
  const fitCurve = (f, color, wd) => {
    if (!f || !f.beam || f.beam.length < 2) return '';
    const pts = f.beam.map((b, i) => X(b).toFixed(1) + ',' + Y(f.pred[i]).toFixed(1)).join(' ');
    return '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="' + wd + '"/>';
  };
  s += fitCurve(storage.fit_std, "#9ec5fe", 2);
  s += fitCurve(storage.fit_chk, "#e2574a", 2.4);
  s += '</svg>';
  return s;
}

// 圧力（左軸・赤）＆ビーム電流（右軸・緑点線）vs 時刻。legacy Make_Plot_Strg_Time 相当。
function timeSeriesDual(storage) {
  const t = storage.t, P = storage.pressure, B = storage.beam, n = t.length;
  const w = 640, h = 175, padL = 78, padR = 64, padT = 12, padB = 28;
  const pmin = Math.min.apply(null, P), pmax = Math.max.apply(null, P), pr = (pmax - pmin) || 1;
  const bmax = Math.max.apply(null, B) || 1, br = bmax || 1;
  const X = i => padL + (i / ((n - 1) || 1)) * (w - padL - padR);
  const YP = v => padT + (1 - (v - pmin) / pr) * (h - padT - padB);
  const YB = v => padT + (1 - v / br) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  for (let g = 0; g <= 2; g++) {
    const yy = padT + g * (h - padT - padB) / 2;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#e2574a" font-size="11" text-anchor="end">' + fmtPres(pmax - g * pr / 2) + '</text>';
    s += '<text x="' + (w - padR + 6) + '" y="' + (yy + 4) + '" fill="#5dcaa5" font-size="11">' + Math.round(bmax - g * br / 2) + '</text>';
  }
  [0, Math.floor((n - 1) / 2), n - 1].forEach(i => {
    if (i >= 0 && i < n) s += '<text x="' + X(i).toFixed(1) + '" y="' + (h - 8) + '" fill="#6b7080" font-size="10" text-anchor="middle">' + hhmm(t[i]) + '</text>';
  });
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#e2574a" font-size="11" text-anchor="middle">圧力 [Pa]</text>';
  s += '<text transform="rotate(-90 ' + (w - 9) + ' ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="' + (w - 9) + '" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#5dcaa5" font-size="11" text-anchor="middle">ビーム電流 [mA]</text>';
  s += '<polyline points="' + P.map((v, i) => X(i).toFixed(1) + ',' + YP(v).toFixed(1)).join(" ") + '" fill="none" stroke="#e2574a" stroke-width="2"/>';
  s += '<polyline points="' + B.map((v, i) => X(i).toFixed(1) + ',' + YB(v).toFixed(1)).join(" ") + '" fill="none" stroke="#5dcaa5" stroke-width="1.6" stroke-dasharray="3,2"/>';
  s += '</svg>';
  return s;
}


function renderMap(s) {
  const host = document.getElementById("map");
  host.innerHTML = mapRow("LER", s.sections.LER) + mapRow("HER", s.sections.HER);
}
function mapRow(ring, secs) {
  let cells = "";
  for (let i = 1; i <= 12; i++) {
    const key = "D" + String(i).padStart(2, "0");
    const sev = secs[key];
    // セクションは異常箇所のハイライト表示のみ（クリック不可）。
    // 1セクションに複数の異常があるとクリックで何が出るか曖昧なため、表示専用にする。
    cells += '<div class="cell ' + (sev || '') + '">' + key + '</div>';
  }
  return '<div class="maprow"><div class="rl">' + ring + '</div><div class="cells">' + cells + '</div></div>';
}

function select(id) { SELECTED = id; renderList(STATE); renderDetail(); }
function selectSection(ring, sec) {
  const a = STATE.anomalies.find(x => x.ring === ring && x.section === sec);
  if (a) select(a.id);
}
async function saveLabel(kind) {
  const a = STATE.anomalies.find(x => x.id === SELECTED);
  if (!a) return;
  const jp = (kind === "Normal") ? "正常(Normal)" : "異常(Abnormal)";
  if (!confirm(a.record + " を " + jp + " として教師ラベルに登録します。よろしいですか？\n" +
               "（この操作はラベルをキューに溜めるだけで、再学習は別途実行します）")) return;
  const payload = {
    ring: a.ring, record: a.record, period: a.period,
    abort_time: a.abort_time, beam_at_check: a.beam_at_check, klass: kind,
  };
  try {
    const r = await fetch("/api/label", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const res = await r.json();
    if (res && res.ok) toast(a.record + " を " + jp + " としてラベル登録しました（キュー件数 " + res.queued + "）");
    else toast("登録に失敗しました: " + (res && res.error ? res.error : "不明"), true);
  } catch (e) { toast("登録に失敗しました: " + e, true); }
}

function toast(msg, isErr) {
  let t = document.getElementById("toast");
  if (!t) {
    t = document.createElement("div"); t.id = "toast";
    t.style.cssText = "position:fixed;left:50%;bottom:26px;transform:translateX(-50%);" +
      "padding:11px 18px;border-radius:10px;font-size:13px;z-index:9999;" +
      "box-shadow:0 6px 24px rgba(0,0,0,.4);max-width:80vw;";
    document.body.appendChild(t);
  }
  t.style.background = isErr ? "#4a1d1d" : "#1d3a2a";
  t.style.color = isErr ? "#ff9f9f" : "#9fe0bf";
  t.style.border = "1px solid " + (isErr ? "#7a2a2a" : "#2a6a4a");
  t.textContent = msg; t.style.opacity = "1";
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => { t.style.transition = "opacity .5s"; t.style.opacity = "0"; }, 3600);
}

async function refreshLoad() {
  try {
    const r = await fetch("/api/sysload", {cache: "no-store"});
    const d = await r.json();
    const el = document.getElementById("sysload");
    el.className = "pill load-" + (d.level || "ok");
    if (!d.loadavg) { el.textContent = "負荷 取得不可"; el.title = ""; return; }
    const parts = ["負荷 " + d.loadavg[0]];
    if (d.cpu_percent !== null && d.cpu_percent !== undefined)
      parts.push("CPU " + d.cpu_percent + "%");
    if (d.mem_percent !== null && d.mem_percent !== undefined)
      parts.push("Mem " + d.mem_percent + "%");
    el.textContent = parts.join(" ⋅ ");
    el.title = "load avg " + d.loadavg.join(" / ") + "（1/5/15分）, " +
               d.ncpu + " コア, 比 " + d.load_ratio;
    // このプログラム自身（＋子プロセス）の使用量
    const se = document.getElementById("selfload");
    if (se && d.self_mem_mb !== undefined && d.self_mem_mb !== null) {
      const sc = (d.self_cpu_percent === null || d.self_cpu_percent === undefined)
                 ? "—" : (d.self_cpu_percent + "%");
      se.textContent = "本プログラム CPU " + sc + " ⋅ Mem " + d.self_mem_mb + " MB";
      se.title = "このダッシュボード（＋子プロセス " + d.self_nproc + " 個）自身の使用量。" +
                 "CPU% はサーバー全体に対する割合なので、左の全体CPU%と直接比べられる。";
    }
    // detector_headless.py（別プロセス。親子関係が無いのでps探索では見えない。
    // .detector.lock からPIDを教えてもらって計測）の使用量。
    const de = document.getElementById("detectorload");
    if (de) {
      if (d.detector_alive) {
        const dc = (d.detector_cpu_percent === null || d.detector_cpu_percent === undefined)
                   ? "—" : (d.detector_cpu_percent + "%");
        de.textContent = "検知プログラム CPU " + dc + " ⋅ Mem " + (d.detector_mem_mb ?? "—") + " MB";
        de.title = "detector_headless.py（＋子プロセス " + (d.detector_nproc ?? "—") + " 個）の使用量。" +
                   "CPU% はサーバー全体に対する割合。1プロセスでここが高い（目安: 20%以上）ときは" +
                   "--watch の負荷が高すぎる可能性があるので、実機で top 等も確認するとよい。";
        de.className = "pill" + (d.detector_cpu_percent >= 50 ? " load-high"
                                 : d.detector_cpu_percent >= 20 ? " load-warn" : "");
      } else {
        de.textContent = "検知プログラム: 停止中/未検出";
        de.title = "detector_headless.py --watch が動いていない、または .detector.lock が" +
                   "見つからない（ダッシュボードとは別プロセスなので起動していないと出ません）。";
        de.className = "pill";
      }
    }
  } catch (e) {
    const el = document.getElementById("sysload");
    el.className = "pill"; el.textContent = "負荷 —";
  }
}

// ── イオンポンプ異常（ip_judge 結果）専用セクション ─────────────────
let IP_SELECTED = null;
let IP_DETAIL_RENDERED = null;
const IP_REASON_JP = {feedthrough_discharge_suspect: "フィードスルー放電疑い",
  over_current: "電流上振れ", decoupled: "相関崩れ", pumping_degradation: "排気劣化"};
const IP_KIND_JP = {acute: "急性", chronic: "慢性", unknown: "不明"};

function renderIPAnomalies(s) {
  const host = document.getElementById("ip-anomalies");
  const list = (s.ion_pump_anomalies || []).slice().sort((a, b) => (b.severity_n || 0) - (a.severity_n || 0));
  if (!list.length) { host.innerHTML = ""; return; }
  let html = '<div class="section-title ip-section"><span class="ip-title">イオンポンプ 異常モニター</span>' +
             '（sev3=急性/継続、sev2=要観察、sev1=軽微。severity 降順で表示）</div>';
  list.forEach(a => {
    const sev = a.severity;
    const dev = (a.deviation_dex != null)
      ? ((a.deviation_dex >= 0 ? "+" : "") + a.deviation_dex.toFixed(1) + " dex") : "";
    const kindTag = a.kind
      ? '<span class="kindtag ' + a.kind + '">' + (IP_KIND_JP[a.kind] || a.kind) + '</span>' : "";
    const cntTag = (a.count != null)
      ? '<span class="cnttag">' + a.count + ' 回連続</span>' : "";
    html += '<div class="ipcard ' + sev + (IP_SELECTED === a.id ? ' sel' : '') +
      '" onclick="ipSelect(\'' + a.id + '\')">' +
      '<span class="sevtag ' + sev + '">sev' + a.severity_n + '</span>' + kindTag + cntTag +
      '<span class="ainfo"><span class="rec">' + a.pv + '</span>' +
      '<div class="sub">' + a.ring + ' ' + a.section + ' · ' + (a.supply || "") + ' · ' +
        (IP_REASON_JP[a.reason] || a.reason) + '</div></span>' +
      '<span class="devtag2">' + dev + '</span>' +
      ipTrendPlot(a.series_count || [a.count || 0], sevColor(sev), 210, 70) + '</div>';
  });
  host.innerHTML = html;
}

// イオンポンプ異常カウントの推移（各 judge サイクルの累積 sev3 カウント＝「N 回連続」の履歴）。
//   横軸: judge サイクル（古い→新しい・右端が最新）  縦軸: 連続カウント 0..max
function ipTrendPlot(counts, color, w, h) {
  const c = (counts && counts.length) ? counts : [0];
  const n = c.length;
  const padL = 34, padR = 8, padT = 8, padB = 16;
  const cmax = Math.max(Math.max.apply(null, c), 4);
  const X = i => padL + (n <= 1 ? 0.5 : i / (n - 1)) * (w - padL - padR);
  const Y = v => padT + (1 - v / cmax) * (h - padT - padB);
  let s = '<svg class="iptrend" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h +
          '" style="display:block">';
  for (let g = 0; g <= 1; g++) {            // 横グリッド＋ y 目盛り（0 と max）
    const yy = padT + g * (h - padT - padB);
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 4) + '" y="' + (yy + 4) + '" fill="#9aa0ad" font-size="10" text-anchor="end">' +
         (g === 0 ? cmax : 0) + '</text>';
  }
  const line = c.map((v, i) => X(i).toFixed(1) + ',' + Y(v).toFixed(1)).join(' ');
  const area = padL + ',' + Y(0).toFixed(1) + ' ' + line + ' ' + X(n - 1).toFixed(1) + ',' + Y(0).toFixed(1);
  s += '<polygon points="' + area + '" fill="' + color + '" opacity="0.13"/>';
  s += '<polyline points="' + line + '" fill="none" stroke="' + color + '" stroke-width="2"/>';
  s += '<circle cx="' + X(n - 1).toFixed(1) + '" cy="' + Y(c[n - 1]).toFixed(1) + '" r="2.6" fill="' + color + '"/>';
  const cy = (padT + (h - padB)) / 2;
  s += '<text transform="rotate(-90 10 ' + cy.toFixed(1) + ')" x="10" y="' + cy.toFixed(1) +
       '" fill="#9aa0ad" font-size="9" text-anchor="middle">Anomaly Count</text>';
  s += '<text x="' + ((padL + w - padR) / 2) + '" y="' + (h - 4) +
       '" fill="#6b7080" font-size="9" text-anchor="middle">Judge Cycle</text>';
  s += '</svg>';
  return s;
}

function ipSelect(id) {
  IP_SELECTED = (IP_SELECTED === id) ? null : id;
  renderIPAnomalies(STATE); renderIPDetail();
}

function renderIPDetail() {
  const host = document.getElementById("ip-detail-host");
  if (!IP_SELECTED) {
    if (IP_DETAIL_RENDERED !== null) { host.innerHTML = ""; IP_DETAIL_RENDERED = null; }
    return;
  }
  if (IP_SELECTED === IP_DETAIL_RENDERED) return;
  const a = (STATE.ion_pump_anomalies || []).find(x => x.id === IP_SELECTED);
  if (!a) { host.innerHTML = ""; IP_DETAIL_RENDERED = null; return; }
  IP_DETAIL_RENDERED = IP_SELECTED;
  host.innerHTML =
    '<div class="detail"><h3>' + a.pv + '</h3>' +
    '<div class="sub">' + a.ring + ' ' + a.section + ' · ' + (a.supply || "") + ' · ' +
      (IP_REASON_JP[a.reason] || a.reason) +
      (a.kind ? (' · ' + (IP_KIND_JP[a.kind] || a.kind)) : "") +
      (a.deviation_dex != null ? '（' + (a.deviation_dex >= 0 ? "+" : "") + a.deviation_dex.toFixed(1) + ' dex）' : "") +
      '</div>' +
    '<div id="ip-raw-plots"><div class="plotbox" style="text-align:center;color:var(--muted);padding:22px">' +
      '生データを取得中…</div></div>' +
    '<div class="btns"><button class="close" onclick="ipSelect(\'' + a.id + '\')">閉じる</button></div></div>';
  host.scrollIntoView({behavior: "smooth", block: "nearest"});
  loadIPRaw(a);
}

async function loadIPRaw(a) {
  let v;
  try {
    const r = await fetch("/api/ip_raw?ring=" + encodeURIComponent(a.ring) +
                          "&record=" + encodeURIComponent(a.pv), {cache: "no-store"});
    v = await r.json();
  } catch (e) { v = {error: "取得に失敗しました: " + e}; }
  const box = document.getElementById("ip-raw-plots");
  if (!box || IP_SELECTED !== a.id) return;
  const note = m => '<div class="plotbox" style="text-align:center;color:var(--muted);padding:22px">' + m + '</div>';
  if (!v || v.error) { box.innerHTML = note(v && v.error ? v.error : "取得できませんでした"); return; }
  let html = "";
  if (v.current && v.current.t && v.current.t.length) {
    html += '<div class="plotbox"><div class="cap">放電電流 vs 時刻（直近・無ビーム区間を網掛け）[A]　' +
            '<span style="color:#9ec5fe">▤ 学習バンド(p50–p95)</span></div>' +
            currentTimePlot(v.current, v.band) + '</div>';
  }
  if (v.scatter && v.scatter.P && v.scatter.P.length) {
    html += '<div class="plotbox"><div class="cap">電流 vs 圧力（I-P・両対数）　' +
            '<span style="color:#e2574a">●実測</span> ' +
            '<span style="color:#e2574a">━ I=a·P^b 回帰</span> ' +
            '<span style="color:#9ec5fe">┄ 学習 I=a·P^b</span></div>' +
            ipScatter(v.scatter, v.learned_fit) + '</div>';
  }
  box.innerHTML = html || note("表示できる生データがありません");
}

// 放電電流 vs 時刻（対数縦軸・無ビーム区間を網掛け・学習バンド p50-p95 を水平帯で重ねる）
function currentTimePlot(cur, band) {
  const I = cur.I, t = cur.t, beam = cur.beam, n = I.length;
  const w = 640, h = 200, padL = 78, padR = 14, padT = 12, padB = 28;
  const li = I.map(v => Math.log10(v > 0 ? v : 1e-12));
  let ay = li.slice();
  if (band) { ay = ay.concat([band.p50, band.p95]); }   // バンドも軸範囲に含める
  const ymin = Math.min.apply(null, ay), ymax = Math.max.apply(null, ay), yr = (ymax - ymin) || 1;
  const X = i => padL + (i / ((n - 1) || 1)) * (w - padL - padR);
  const Y = v => padT + (1 - (v - ymin) / yr) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  // 学習バンド（電流 p50–p95 の水平帯）
  if (band) {
    const yhi = Y(band.p95), ylo = Y(band.p50);
    s += '<rect x="' + padL + '" y="' + Math.min(yhi, ylo).toFixed(1) + '" width="' + (w - padL - padR) +
         '" height="' + Math.abs(ylo - yhi).toFixed(1) + '" fill="#9ec5fe" opacity="0.13"/>';
    s += '<line x1="' + padL + '" y1="' + yhi.toFixed(1) + '" x2="' + (w - padR) + '" y2="' + yhi.toFixed(1) +
         '" stroke="#9ec5fe" stroke-width="1" stroke-dasharray="4,3"/>';
    s += '<text x="' + (w - padR - 2) + '" y="' + (yhi - 3).toFixed(1) + '" fill="#9ec5fe" font-size="10" text-anchor="end">p95</text>';
  }
  // 無ビーム区間（beam<10 or null）の網掛け
  let i = 0;
  while (i < n) {
    if (beam[i] == null || beam[i] < 10) {
      let j = i; while (j < n && (beam[j] == null || beam[j] < 10)) j++;
      const x0 = X(i), x1 = X(Math.max(i, j - 1));
      s += '<rect x="' + x0.toFixed(1) + '" y="' + padT + '" width="' + Math.max(1, x1 - x0).toFixed(1) +
           '" height="' + (h - padT - padB) + '" fill="#ffffff" opacity="0.05"/>';
      i = j;
    } else i++;
  }
  for (let g = 0; g <= 3; g++) {
    const yy = padT + g * (h - padT - padB) / 3, val = ymax - g * yr / 3;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#6b7080" font-size="11" text-anchor="end">' + fmtPres(Math.pow(10, val)) + '</text>';
  }
  [0, Math.floor((n - 1) / 2), n - 1].forEach(k => {
    if (k >= 0 && k < n) s += '<text x="' + X(k).toFixed(1) + '" y="' + (h - 8) +
      '" fill="#6b7080" font-size="10" text-anchor="middle">' + hhmm(t[k]) + '</text>';
  });
  s += '<polyline points="' + li.map((v, k) => X(k).toFixed(1) + ',' + Y(v).toFixed(1)).join(' ') +
       '" fill="none" stroke="#e2574a" stroke-width="1.8"/>';
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#9aa0ad" font-size="11" text-anchor="middle">放電電流 [A]</text>';
  s += '</svg>';
  return s;
}

// I-P 散布図（両対数）＋ I=a·P^b 回帰線
function ipScatter(scat, learnedFit) {
  const P = scat.P, I = scat.I, n = P.length;
  const lx = P.map(v => Math.log10(v)), ly = I.map(v => Math.log10(v));
  let ax = lx.slice(), ay = ly.slice();
  if (scat.fit) { scat.fit.P.forEach(p => ax.push(Math.log10(p))); scat.fit.I.forEach(v => ay.push(Math.log10(v))); }
  const w = 640, h = 230, padL = 78, padR = 14, padT = 12, padB = 36;
  const xmin = Math.min.apply(null, ax), xmax = Math.max.apply(null, ax), xr = (xmax - xmin) || 1;
  const ymin = Math.min.apply(null, ay), ymax = Math.max.apply(null, ay), yr = (ymax - ymin) || 1;
  const X = v => padL + ((v - xmin) / xr) * (w - padL - padR);
  const Y = v => padT + (1 - (v - ymin) / yr) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  for (let g = 0; g <= 3; g++) {
    const yy = padT + g * (h - padT - padB) / 3, val = ymax - g * yr / 3;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#6b7080" font-size="11" text-anchor="end">' + fmtPres(Math.pow(10, val)) + '</text>';
  }
  for (let g = 0; g <= 3; g++) {
    const xx = padL + g * (w - padL - padR) / 3, val = xmin + g * xr / 3;
    s += '<text x="' + xx.toFixed(0) + '" y="' + (h - padB + 16) + '" fill="#6b7080" font-size="11" text-anchor="middle">' + fmtPres(Math.pow(10, val)) + '</text>';
  }
  s += '<text x="' + ((padL + w - padR) / 2) + '" y="' + (h - 5) + '" fill="#6b7080" font-size="11" text-anchor="middle">圧力 [Pa]</text>';
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#9aa0ad" font-size="11" text-anchor="middle">放電電流 [A]</text>';
  for (let k = 0; k < n; k++)
    s += '<circle cx="' + X(lx[k]).toFixed(1) + '" cy="' + Y(ly[k]).toFixed(1) + '" r="2.1" fill="#e2574a" opacity="0.7"/>';
  // 学習 I=a·P^b（薄青の参照線）
  if (learnedFit && learnedFit.a && learnedFit.b) {
    const px = [xmin, xmax];
    const pl = px.map(lp => X(lp).toFixed(1) + ',' + Y(Math.log10(learnedFit.a) + learnedFit.b * lp).toFixed(1)).join(' ');
    s += '<polyline points="' + pl + '" fill="none" stroke="#9ec5fe" stroke-width="1.8" stroke-dasharray="5,3"/>';
  }
  // 判定窓の I=a·P^b 回帰（青・実線）
  if (scat.fit) {
    const fl = scat.fit.P.map((p, k) => X(Math.log10(p)).toFixed(1) + ',' + Y(Math.log10(scat.fit.I[k])).toFixed(1)).join(' ');
    s += '<polyline points="' + fl + '" fill="none" stroke="#e2574a" stroke-width="2.2"/>';
  }
  s += '</svg>';
  return s;
}

// ── イオンポンプ放電電流（開発中・判定なし）──────────────────────
let IP_SEL = {};   // {ring: section} 展開中のセクション

function fmtCur(v) {
  if (v === null || v === undefined) return "—";
  return v.toExponential(2) + " A";
}

// 放電電流は桁が広い(1e-10〜1e-6)ので対数スケール。null は線を切る（欠測）。
function sparkIP(trend, color, w, h) {
  const logs = trend.map(v => (v === null || v <= 0) ? null : Math.log10(v));
  const valid = logs.filter(x => x !== null);
  if (!valid.length) return '<svg width="' + w + '" height="' + h + '"></svg>';
  let lo = Math.min.apply(null, valid), hi = Math.max.apply(null, valid);
  const rng = (hi - lo) || 1;
  let segs = [], cur = [];
  logs.forEach((x, i) => {
    if (x === null) { if (cur.length) { segs.push(cur); cur = []; } return; }
    const px = (i / (logs.length - 1)) * w;
    const py = h - ((x - lo) / rng) * (h - 4) - 2;
    cur.push(px.toFixed(1) + "," + py.toFixed(1));
  });
  if (cur.length) segs.push(cur);
  const lines = segs.map(s =>
    '<polyline points="' + s.join(" ") + '" fill="none" stroke="' + color +
    '" stroke-width="1.5"/>').join("");
  return '<svg class="spark" width="' + w + '" height="' + h +
         '" viewBox="0 0 ' + w + ' ' + h + '">' + lines + '</svg>';
}

function renderIonPumps(s) {
  // 旧「イオンポンプ 放電電流（開発中・判定なし）」観測パネルは非表示。
  // 判定済みの「イオンポンプ 異常モニター」(renderIPAnomalies) に一本化した。
  const host = document.getElementById("ionpumps");
  if (host) host.innerHTML = "";
  return;
}

function _renderIonPumps_disabled(s) {
  const host = document.getElementById("ionpumps");
  const ip = s.ion_pumps;
  if (!ip || !ip.rings) { host.innerHTML = ""; return; }   // データ無ければ非表示
  let html = '<div class="section-title">イオンポンプ 放電電流' +
             '<span class="devtag">開発中・判定なし</span></div>' +
             '<div class="ipmeta">取得: ' + (ip.updated || "—") +
             ' ／ 間隔 ' + (ip.interval_sec || "?") + ' 秒 ／ 値が大きい色ほど放電電流が大きい（相対）</div>';
  ["LER", "HER"].forEach(ring => {
    const r = ip.rings[ring];
    if (!r) return;
    html += '<div class="iprow"><div class="iplabel">' + ring + '</div><div class="ipgrid">';
    // 色スケール用: このリングの max_latest の最大
    const maxs = r.sections.map(x => x.max_latest || 0);
    const gmax = Math.max.apply(null, maxs.concat([1e-12]));
    r.sections.forEach(sec => {
      const frac = sec.max_latest ? Math.min(1, Math.log10(sec.max_latest / 1e-10) / 4) : 0;
      const sel = (IP_SEL[ring] === sec.section) ? " sel" : "";
      html += '<div class="ipcard' + sel + '" onclick="toggleIP(\'' + ring + '\',\'' +
              sec.section + '\')">' +
              '<div class="ipsec">' + sec.section +
              (sec.supply === "Agilent_4U" ? '<span class="badge b4u">4U</span>' : '') +
              '</div>' +
              '<div class="ipbar"><span style="width:' + (frac * 100).toFixed(0) + '%"></span></div>' +
              '<div class="ipval">' + fmtCur(sec.max_latest) + '</div>' +
              '<div class="ipn">' + sec.n_active + '/' + sec.n + ' 台</div>' +
              '</div>';
    });
    html += '</div></div>';
    // 展開: 選択セクションの PV 一覧＋トレンド
    if (IP_SEL[ring]) {
      const sec = r.sections.find(x => x.section === IP_SEL[ring]);
      if (sec) {
        html += '<div class="ippvs"><div class="ippvs-h">' + ring + ' ' + sec.section +
                '（' + (sec.supply === "Agilent_4U" ? "Agilent 4U" : "KEK") + '／' +
                sec.n + ' 台）</div>';
        sec.pvs.forEach(p => {
          html += '<div class="ippv"><span class="ippv-name">' + p.pv + '</span>' +
                  sparkIP(p.trend, "#4a90d9", 120, 26) +
                  '<span class="ippv-val">' + fmtCur(p.latest) + '</span></div>';
        });
        html += '</div>';
      }
    }
  });
  host.innerHTML = html;
}

// 温度計異常の理由コード → 日本語（temp_batch.py の REASON_JA と対応）
const TEMP_REASON_JP = {
  range_high_open_suspect: "断線疑い(非現実高温)",
  range_high_noheat_suspect: "無ビーム高温(near-open/高抵抗)",
  range_low_short_suspect: "短絡疑い(低温/サブ常温)",
  noise_and_glitch: "断線間近(ノイズ+グリッチ)",
  beam_anticorrelated: "ビーム反相関",
  intermittent_excursion: "間欠逸脱(稀な極端値)",
  noise_increase: "ノイズ増大",
  glitch: "グリッチ頻発",
  stuck: "張り付き",
  range_watch: "レンジ接近/軽度",
  pair_deviation_high: "ペア乖離(大)",
  pair_deviation_watch: "ペア乖離(軽)",
  normal: "正常",
};

function fmtT(v) { return (v == null || !isFinite(v)) ? "—" : v.toFixed(1); }

function renderTempAnomalies(s) {
  const host = document.getElementById("temp-anomalies");
  const badge = document.getElementById("tab-badge-temp");
  // s.error: 状態ファイルが無い/読めない（temp_headless.py 未実行など）。rings が空 {} でも
  // JSでは truthy なので、error を先に明示チェックしないと案内文が出ずタブが真っ白になる。
  if (!s || !s.rings || (s.error && !Object.keys(s.rings || {}).length)) {
    host.innerHTML = '<div class="footnote">' +
      ((s && s.error) || '温度計データがありません（temp_headless.py が未実行の可能性）') + '</div>';
    if (badge) { badge.textContent = ""; badge.classList.remove("hot"); }
    return;
  }
  let totalSev3 = 0, totalAnom = 0, html = "", anyArchiverStopped = false;
  ["LER", "HER"].forEach(ring => {
    const r = s.rings[ring];
    if (!r) return;
    if (r.error) {
      html += '<div class="temp-ring-title">' + ring + '</div><div class="footnote">判定失敗: ' + r.error + '</div>';
      return;
    }
    if (r.archiver_stopped) {
      anyArchiverStopped = true;
      html += '<div class="archiver-banner">⚠ ' + ring + ': 現在、アーカイバ(kblog)が温度計のデータ取得を'
            + '停止中です。PVリスト全' + ((r.stats && r.stats.n_total) || "?") + '本で有効なデータが取得'
            + 'できませんでした（判定は行われていません。センサの異常ではありません）。</div>';
    }
    const anomalies = r.anomalies || [];
    const nAnom = (r.n_anomalies != null) ? r.n_anomalies : anomalies.length;  // JSONの総数（表は上位topのみ）
    totalAnom += nAnom;
    totalSev3 += anomalies.filter(a => a.severity >= 3).length;
    const stats = r.stats || {};
    html += '<div class="temp-ring-title">' + ring + ' — 判定 ' + (stats.n_judged || 0) + ' 本 / 異常 ' +
            nAnom + ' 本' + (nAnom > anomalies.length ? '（表は上位' + anomalies.length + '本）' : '') +
            (r.baseline ? '　基準: ' + r.baseline.start + '〜' + r.baseline.end : '') + '</div>';
    if (!anomalies.length) { html += '<div class="footnote">異常なし</div>'; return; }
    html += '<table class="temp-table"><thead><tr>' +
            '<th>sev</th><th>PV</th><th>種別</th><th>t_med</th><th>t_min</th><th>t_max</th><th>理由</th>' +
            '</tr></thead><tbody>';
    anomalies.forEach(a => {
      const rid = ring + ":" + a.pv;
      html += '<tr onclick="toggleTemp(\'' + rid + '\')" style="cursor:pointer">' +
        '<td><span class="sev-badge sev' + a.severity + '">' + a.severity + '</span></td>' +
        '<td>' + a.pv.split(":").slice(1).join(":") + '</td>' +
        '<td>' + (a.suffix || "-") + '</td>' +
        '<td>' + fmtT(a.t_med) + '</td><td>' + fmtT(a.t_min) + '</td><td>' + fmtT(a.t_max) + '</td>' +
        '<td>' + (TEMP_REASON_JP[a.reason] || a.reason) + '</td></tr>';
      if (TEMP_SEL === rid) {
        html += '<tr><td colspan="7" style="background:var(--surface2)">' +
          '<div class="footnote">section=' + a.section + ' tag=' + a.tag +
          ' n_pts=' + a.n_pts +
          (a.frac_low != null ? '　S.frac_low=' + a.frac_low.toFixed(2) : '') +
          (a.frac_hot != null ? '　O.frac_hot=' + a.frac_hot.toFixed(2) : '') +
          (a.r_beam != null ? '　B.r_beam=' + a.r_beam.toFixed(2) : '') +
          (a.n_events != null ? '　I.events=' + a.n_events : '') +
          '　model=' + (a.has_model ? "有" : "無") + '</div>' +
          detailPlotBox('温度＆ビーム電流 vs 時刻（判定窓）　' +
                        '<span style="color:#e2574a">━温度[℃]</span> ' +
                        '<span style="color:#5dcaa5">┄ビーム電流[mA]</span>',
                        a.plot ? tempTimePlot(a.plot) : '', !!a.plot) +
          '</td></tr>';
      }
    });
    html += '</tbody></table>';
  });
  host.innerHTML = html;
  if (badge) {
    if (anyArchiverStopped) {
      badge.textContent = "停止中";
      badge.classList.remove("hot");
    } else {
      badge.textContent = totalAnom ? (totalSev3 + "/" + totalAnom) : "";
      badge.classList.toggle("hot", totalSev3 > 0);
    }
  }
}

function toggleTemp(rid) {
  TEMP_SEL = (TEMP_SEL === rid) ? null : rid;
  renderTempAnomalies(TEMP_STATE);
}

// 機器劣化検知の理由コード → 日本語（temp_equipment.py の compare_periods/compare_periods_hom と対応。
// linear/hom どちらのモデルも同じ理由コード体系を使う）
const EQUIP_REASON_JP = {
  insufficient_data: "判定不能(データ不足)",
  normal: "正常",
  heating_gain_increase_watch: "発熱増加(軽度・要注視)",
  heating_gain_increase: "発熱増加(中)",
  heating_gain_increase_severe: "発熱増加(重度)",
};

function renderEquipmentAnomalies(s) {
  const host = document.getElementById("equip-anomalies");
  const badge = document.getElementById("tab-badge-equip");
  // s.error: 状態ファイルが無い/読めない場合の案内（rings が空 {} でも truthy なため明示チェック）
  if (!s || !s.rings || (s.error && !Object.keys(s.rings || {}).length)) {
    host.innerHTML = '<div class="footnote">' +
      ((s && s.error) || '機器劣化データがありません（temp_equipment.py の learn/judge が未実行の可能性）') + '</div>';
    if (badge) { badge.textContent = ""; badge.classList.remove("hot"); }
    return;
  }
  let totalSev3 = 0, totalAnom = 0, html = "", anyArchiverStopped = false, allSkipped = true;
  ["LER", "HER", "IR"].forEach(ring => {
    const r = s.rings[ring];
    if (!r) return;
    if (r.skipped) {
      // 機器劣化検知は温度計異常検知と同じアーカイバ(温度)を使うため、そちらが停止中なら
      // ここでも参考情報として出す（learn前はこのタブ自身では判定を試みていないので、
      // 「未学習」と「アーカイバ停止中」は別の理由だが、判断材料として関連付けて示す）。
      const tRing = TEMP_STATE && TEMP_STATE.rings && TEMP_STATE.rings[ring];
      const tempDown = tRing && tRing.archiver_stopped;
      html += '<div class="temp-ring-title">' + ring + '</div><div class="footnote">未学習のためスキップ' +
              '（<code>python temp_equipment.py learn ' + ring + ' &lt;開始&gt; &lt;終了&gt;</code> で学習してください）' +
              (tempDown ? '　※温度計異常検知タブでも現在アーカイバ停止中と出ています。learn後もこの期間は同様に'
                        + '「アーカイバ停止中」表示になる見込みです' : '') +
              '</div>';
      return;
    }
    allSkipped = false;
    if (r.error) {
      html += '<div class="temp-ring-title">' + ring + '</div><div class="footnote">判定失敗: ' + r.error + '</div>';
      return;
    }
    if (r.archiver_stopped) {
      anyArchiverStopped = true;
      html += '<div class="archiver-banner">⚠ ' + ring + ': 現在、アーカイバ(kblog)が温度計のデータ取得を'
            + '停止中です。' + (r.archiver_stopped_borrowed
              ? '（温度計異常検知タブの直近の判定結果より。機器劣化検知側では無駄な再取得をせず'
                + 'スキップしました）'
              : '学習済みPVすべてで有効なデータが取得できませんでした')
            + '（判定は行われていません。機器の異常ではありません）。</div>';
    }
    const anomalies = r.anomalies || [];
    const nAnom = (r.n_anomalies != null) ? r.n_anomalies : anomalies.length;  // JSONの総数（表は上位topのみ）
    totalAnom += nAnom;
    totalSev3 += anomalies.filter(a => a.severity >= 3).length;
    const stats = r.stats || {};
    html += '<div class="temp-ring-title">' + ring + ' — 判定 ' + (stats.n_judged || 0) + ' 本 / 異常 ' +
            nAnom + ' 本' + (nAnom > anomalies.length ? '（表は上位' + anomalies.length + '本）' : '') +
            (r.window ? '　窓: ' + r.window.start + '〜' + r.window.end : '') + '</div>';
    if (!anomalies.length) { html += '<div class="footnote">異常なし</div>'; return; }
    html += '<table class="temp-table"><thead><tr>' +
            '<th>sev</th><th>PV</th><th>モデル</th><th>比</th><th>基準</th><th>現在</th><th>理由</th>' +
            '</tr></thead><tbody>';
    anomalies.forEach(a => {
      const rid = ring + ":" + a.pv;
      const isHom = a.model === "hom";
      const refCol = isHom ? fmtT(a.dT_ref) + "℃@" + (a.eval_i||0).toFixed(0) + "mA" : _f3(a.b_ref, 1000);
      const nowCol = isHom ? fmtT(a.dT_now) + "℃@" + (a.eval_i||0).toFixed(0) + "mA" : _f3(a.now && a.now.b, 1000);
      html += '<tr onclick="toggleEquip(\'' + rid + '\')" style="cursor:pointer">' +
        '<td><span class="sev-badge sev' + a.severity + '">' + a.severity + '</span></td>' +
        '<td>' + a.pv + '</td>' +
        '<td>' + (a.model || "-") + '</td>' +
        '<td>' + (a.ratio != null ? a.ratio.toFixed(2) + "x" : "—") + '</td>' +
        '<td>' + refCol + '</td><td>' + nowCol + '</td>' +
        '<td>' + (EQUIP_REASON_JP[a.reason] || a.reason) + '</td></tr>';
      if (EQUIP_SEL === rid) {
        html += '<tr><td colspan="7" style="background:var(--surface2)">' +
          '<div class="footnote">section=' + a.section + ' tag=' + a.tag +
          (a.delta_a != null ? '　Δa/Δw0(環境温度差,参考)=' + a.delta_a.toFixed(1) + '℃' : '') +
          (isHom && a.r2_ref != null ? '　R2:基準' + a.r2_ref.toFixed(3) + '/現在' + (a.now && a.now.r2 != null ? a.now.r2.toFixed(3) : "—") : '') +
          (!isHom && a.now ? '　n=' + a.now.n + ' r=' + (a.now.r != null ? a.now.r.toFixed(2) : "—") : '') +
          '</div>' +
          detailPlotBox('温度 vs ビーム電流（ビームあり点のみ）　' +
                        '<span style="color:#9ec5fe">●基準</span> ' +
                        '<span style="color:#e2574a">●調査</span> ' +
                        '<span style="color:#9ec5fe">━基準フィット</span> ' +
                        '<span style="color:#e2574a">━調査フィット</span>' +
                        (a.plot && !a.plot.ref ? '　※linear型は基準期間の生データを保持しないため基準はフィット直線のみ' : ''),
                        a.plot ? equipScatterPlot(a.plot) : '', !!a.plot) +
          '</td></tr>';
      }
    });
    html += '</tbody></table>';
  });
  host.innerHTML = html;
  if (badge) {
    if (anyArchiverStopped) {
      badge.textContent = "停止中";
      badge.classList.remove("hot");
    } else if (allSkipped) {
      badge.textContent = "未学習";
      badge.classList.remove("hot");
    } else {
      badge.textContent = totalAnom ? (totalSev3 + "/" + totalAnom) : "";
      badge.classList.toggle("hot", totalSev3 > 0);
    }
  }
}

function _f3(v, scale) {
  return (v == null || !isFinite(v)) ? "—" : (v * scale).toFixed(3);
}

function toggleEquip(rid) {
  EQUIP_SEL = (EQUIP_SEL === rid) ? null : rid;
  renderEquipmentAnomalies(EQUIP_STATE);
}

// 流量計異常検知の理由コード → 日本語（flow_judge.py の REASON_JP と対応）
const FLOW_REASON_JP = {
  insufficient_data: "判定不能(データ不足)",
  normal: "正常",
  frozen_low: "値の固着(校正基準比も低い・センサ異常濃厚)",
  frozen_watch: "値の固着(校正基準比は正常範囲・要注視)",
  stuck_low_severe: "校正基準比で大幅低下(重度・センサ異常濃厚)",
  stuck_low: "校正基準比で低下(中)",
  stuck_low_watch: "校正基準比で低下(軽度・要注視)",
  excess_noise_severe: "指示値不安定(重度)",
  excess_noise: "指示値不安定(中)",
  excess_noise_watch: "指示値不安定(軽度・要注視)",
};

function renderFlowAnomalies(s) {
  const host = document.getElementById("flow-anomalies");
  const badge = document.getElementById("tab-badge-flow");
  if (!s) {
    host.innerHTML = '<div class="section-title">流量計データがありません</div>';
    if (badge) { badge.textContent = ""; badge.classList.remove("hot"); }
    return;
  }
  // s.error: 状態ファイルが無い/読めない場合の案内。anomalies が空配列 [] でも JS では
  // truthy なので、error があり異常データが無いなら案内文を出す（以前は !s.anomalies 判定
  // だったため [] のとき案内が出ず「異常なし」に見えてしまっていた）。
  if (s.error && !(s.anomalies && s.anomalies.length)) {
    host.innerHTML = '<div class="footnote">' + s.error + '</div>';
    if (badge) { badge.textContent = ""; badge.classList.remove("hot"); }
    return;
  }
  const anomalies = s.anomalies || [];
  const nAnom = (s.n_anomalies != null) ? s.n_anomalies : anomalies.length;  // JSONの総数（表は上位topのみ）
  const stats = s.stats || {};
  let html = "";
  if (s.archiver_stopped) {
    html += '<div class="archiver-banner">⚠ 現在、アーカイバ(kblog)が流量計のデータ取得を停止中です。'
          + 'PV ' + (stats.n_judged || 0) + ' 本すべてで有効なデータが取得できませんでした'
          + '（判定は行われていません。センサの異常ではありません）。'
          + (s.generated_at ? '　最終確認: ' + s.generated_at : '') + '</div>';
  }
  html += '<div class="temp-ring-title">流量計（センサ自身の異常のみ検知） — '
        + '判定 ' + (stats.n_judged || 0) + ' 本 / 異常 ' + nAnom + ' 本'
        + (nAnom > anomalies.length ? '（表は上位' + anomalies.length + '本）' : '')
        + (stats.n_insufficient ? '　データ不足 ' + stats.n_insufficient + ' 本' : '')
        + (s.window ? '　窓: ' + s.window.start + '〜' + s.window.end : '') + '</div>';
  if (!anomalies.length) {
    html += '<div class="footnote">異常なし</div>';
  } else {
    html += '<table class="temp-table"><thead><tr>' +
            '<th>sev</th><th>PV</th><th>セクション</th><th>校正基準比[%]</th><th>CV[%]</th><th>理由</th>' +
            '</tr></thead><tbody>';
    anomalies.forEach(a => {
      html += '<tr onclick="toggleFlow(\'' + a.pv + '\')" style="cursor:pointer">' +
        '<td><span class="sev-badge sev' + a.severity + '">' + a.severity + '</span></td>' +
        '<td>' + a.pv + '</td>' +
        '<td>' + a.section + '</td>' +
        '<td>' + fmtT(a.median_pct) + '</td>' +
        '<td>' + fmtT(a.cv_pct) + '</td>' +
        '<td>' + (FLOW_REASON_JP[a.reason] || a.reason) + '</td></tr>';
      if (FLOW_SEL === a.pv) {
        html += '<tr><td colspan="6" style="background:var(--surface2)">' +
          '<div class="footnote">tag=' + a.tag + ' sensor_id=' + a.sensor_id +
          '　n=' + a.n + '(有効' + a.n_valid + ')' +
          (a.layers && a.layers.frozen ? '　range=' + fmtT(a.layers.frozen.range) : '') +
          (a.layers && a.layers.glitch ? '　外れ値' + a.layers.glitch.n_glitch + '本' : '') +
          '</div>' +
          detailPlotBox('流量 vs 時刻（判定窓。100%=校正基準流量）　' +
                        '<span style="color:#5b9bd5">━流量[%]</span>',
                        a.plot ? flowTimePlot(a.plot) : '', !!a.plot) +
          '</td></tr>';
      }
    });
    html += '</tbody></table>';
  }
  host.innerHTML = html;
  if (badge) {
    const sev3 = stats.n_anomalies_sev3 || 0;
    if (s.archiver_stopped) {
      badge.textContent = "停止中";
      badge.classList.remove("hot");
    } else {
      badge.textContent = nAnom ? (sev3 + "/" + nAnom) : "";
      badge.classList.toggle("hot", sev3 > 0);
    }
  }
}

function toggleFlow(pv) {
  FLOW_SEL = (FLOW_SEL === pv) ? null : pv;
  renderFlowAnomalies(FLOW_STATE);
}

// ── クリック展開プロット（温度計/機器劣化/流量計）───────────────────────
// データは各 judge が状態JSONに埋め込んだ間引き済み配列（クリック時の再取得なし）。
// 欠測(null)は線を切って描く。

// null を含む系列を <polyline> の連続セグメント群に分割して描く
function _segPolylines(xs, ys, color, width, dash) {
  let out = "", pts = [];
  const flush = () => {
    if (pts.length >= 2) {
      out += '<polyline points="' + pts.join(" ") + '" fill="none" stroke="' + color +
             '" stroke-width="' + width + '"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + '/>';
    } else if (pts.length === 1) {
      const [x, y] = pts[0].split(",");
      out += '<circle cx="' + x + '" cy="' + y + '" r="1.6" fill="' + color + '"/>';
    }
    pts = [];
  };
  for (let i = 0; i < ys.length; i++) {
    if (ys[i] == null || !isFinite(ys[i])) { flush(); continue; }
    pts.push(xs[i].toFixed(1) + "," + ys[i].toFixed(1));
  }
  flush();
  return out;
}

function _finite(arr) { return (arr || []).filter(v => v != null && isFinite(v)); }

// 温度計: 左軸=温度[℃](赤)、右軸=ビーム電流[mA](緑点線) の時系列（CCGの timeSeriesDual と同型）
function tempTimePlot(p) {
  const t = p.t, T = p.temp, B = p.beam, n = t.length;
  if (!n || !_finite(T).length) return '<div class="footnote">プロットできる有効データがありません</div>';
  const w = 640, h = 175, padL = 64, padR = 64, padT = 12, padB = 28;
  const tv = _finite(T);
  const tmin = Math.min.apply(null, tv), tmax = Math.max.apply(null, tv), tr = (tmax - tmin) || 1;
  const bv = _finite(B || []);
  const bmax = bv.length ? (Math.max.apply(null, bv) || 1) : 1;
  const X = i => padL + (i / ((n - 1) || 1)) * (w - padL - padR);
  const YT = v => padT + (1 - (v - tmin) / tr) * (h - padT - padB);
  const YB = v => padT + (1 - v / bmax) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  for (let g = 0; g <= 2; g++) {
    const yy = padT + g * (h - padT - padB) / 2;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#e2574a" font-size="11" text-anchor="end">' +
         (tmax - g * tr / 2).toFixed(1) + '</text>';
    if (bv.length) s += '<text x="' + (w - padR + 6) + '" y="' + (yy + 4) + '" fill="#5dcaa5" font-size="11">' +
                        Math.round(bmax - g * bmax / 2) + '</text>';
  }
  [0, Math.floor((n - 1) / 2), n - 1].forEach(i => {
    if (i >= 0 && i < n) s += '<text x="' + X(i).toFixed(1) + '" y="' + (h - 8) +
      '" fill="#6b7080" font-size="10" text-anchor="middle">' + hhmm(t[i]) + '</text>';
  });
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#e2574a" font-size="11" text-anchor="middle">温度 [℃]</text>';
  if (bv.length) s += '<text transform="rotate(-90 ' + (w - 9) + ' ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="' +
       (w - 9) + '" y="' + ((padT + h - padB) / 2).toFixed(1) +
       '" fill="#5dcaa5" font-size="11" text-anchor="middle">ビーム電流 [mA]</text>';
  s += _segPolylines(T.map((_, i) => X(i)), T.map(v => (v == null || !isFinite(v)) ? null : YT(v)), "#e2574a", 2);
  if (bv.length) s += _segPolylines(B.map((_, i) => X(i)),
      B.map(v => (v == null || !isFinite(v)) ? null : YB(v)), "#5dcaa5", 1.6, "3,2");
  s += '</svg>';
  return s;
}

// 機器劣化: 横軸=ビーム電流[mA]、縦軸=温度[℃]。基準散布(薄青)＋調査散布(赤)＋各フィット曲線
function equipScatterPlot(p) {
  const groups = [];
  if (p.ref && p.ref.i && p.ref.i.length) groups.push({d: p.ref, color: "#9ec5fe", kind: "pts"});
  if (p.now && p.now.i && p.now.i.length) groups.push({d: p.now, color: "#e2574a", kind: "pts"});
  if (p.ref_fit) groups.push({d: p.ref_fit, color: "#9ec5fe", kind: "line"});
  if (p.now_fit) groups.push({d: p.now_fit, color: "#e2574a", kind: "line"});
  if (!groups.length) return '<div class="footnote">プロットできる有効データがありません</div>';
  const w = 640, h = 220, padL = 64, padR = 14, padT = 12, padB = 34;
  let imin = Infinity, imax = -Infinity, tmin = Infinity, tmax = -Infinity;
  groups.forEach(g => {
    g.d.i.forEach(v => { if (v < imin) imin = v; if (v > imax) imax = v; });
    g.d.t.forEach(v => { if (v < tmin) tmin = v; if (v > tmax) tmax = v; });
  });
  const ir = (imax - imin) || 1, tr = (tmax - tmin) || 1;
  const X = v => padL + ((v - imin) / ir) * (w - padL - padR);
  const Y = v => padT + (1 - (v - tmin) / tr) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  for (let g = 0; g <= 2; g++) {
    const yy = padT + g * (h - padT - padB) / 2;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#6b7080" font-size="11" text-anchor="end">' +
         (tmax - g * tr / 2).toFixed(1) + '</text>';
  }
  [imin, (imin + imax) / 2, imax].forEach(v => {
    s += '<text x="' + X(v).toFixed(1) + '" y="' + (h - 8) +
         '" fill="#6b7080" font-size="10" text-anchor="middle">' + Math.round(v) + '</text>';
  });
  s += '<text x="' + ((padL + w - padR) / 2).toFixed(1) + '" y="' + (h - 20) +
       '" fill="#6b7080" font-size="11" text-anchor="middle">ビーム電流 [mA]</text>';
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#6b7080" font-size="11" text-anchor="middle">温度 [℃]</text>';
  groups.forEach(g => {
    if (g.kind === "pts") {
      g.d.i.forEach((iv, k) => {
        const tv = g.d.t[k];
        if (tv == null || !isFinite(tv) || iv == null || !isFinite(iv)) return;
        s += '<circle cx="' + X(iv).toFixed(1) + '" cy="' + Y(tv).toFixed(1) +
             '" r="1.7" fill="' + g.color + '" fill-opacity="0.55"/>';
      });
    } else {
      s += '<polyline points="' + g.d.i.map((iv, k) => X(iv).toFixed(1) + ',' + Y(g.d.t[k]).toFixed(1)).join(" ") +
           '" fill="none" stroke="' + g.color + '" stroke-width="2.2"/>';
    }
  });
  s += '</svg>';
  return s;
}

// 流量計: 縦軸=流量[%] の時系列（単軸）
function flowTimePlot(p) {
  const t = p.t, V = p.v, n = t.length;
  if (!n || !_finite(V).length) return '<div class="footnote">プロットできる有効データがありません</div>';
  const w = 640, h = 160, padL = 64, padR = 14, padT = 12, padB = 28;
  const vv = _finite(V);
  const vmin = Math.min.apply(null, vv), vmax = Math.max.apply(null, vv), vr = (vmax - vmin) || 1;
  const X = i => padL + (i / ((n - 1) || 1)) * (w - padL - padR);
  const Y = v => padT + (1 - (v - vmin) / vr) * (h - padT - padB);
  let s = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">';
  for (let g = 0; g <= 2; g++) {
    const yy = padT + g * (h - padT - padB) / 2;
    s += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy + '" stroke="#2c313d"/>';
    s += '<text x="' + (padL - 6) + '" y="' + (yy + 4) + '" fill="#5b9bd5" font-size="11" text-anchor="end">' +
         (vmax - g * vr / 2).toFixed(1) + '</text>';
  }
  [0, Math.floor((n - 1) / 2), n - 1].forEach(i => {
    if (i >= 0 && i < n) s += '<text x="' + X(i).toFixed(1) + '" y="' + (h - 8) +
      '" fill="#6b7080" font-size="10" text-anchor="middle">' + hhmm(t[i]) + '</text>';
  });
  s += '<text transform="rotate(-90 14 ' + ((padT + h - padB) / 2).toFixed(1) + ')" x="14" y="' +
       ((padT + h - padB) / 2).toFixed(1) + '" fill="#5b9bd5" font-size="11" text-anchor="middle">流量 [%]</text>';
  s += _segPolylines(V.map((_, i) => X(i)), V.map(v => (v == null || !isFinite(v)) ? null : Y(v)), "#5b9bd5", 2);
  s += '</svg>';
  return s;
}

// 展開行に入れるプロット枠（plot データが無い旧形式JSONへの案内も出す）
function detailPlotBox(cap, inner, hasData) {
  if (!hasData) return '<div class="footnote">プロットデータがありません（この判定JSONは旧形式です。' +
                       '次回の判定サイクルから表示されます）</div>';
  return '<div class="plotbox" style="margin-top:8px"><div class="cap">' + cap + '</div>' + inner + '</div>';
}

function toggleIP(ring, section) {
  IP_SEL[ring] = (IP_SEL[ring] === section) ? null : section;
  renderIPAnomalies(STATE); renderIPDetail();
  renderIonPumps(STATE);
}

async function refresh() {
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    STATE = await r.json();
    if (STATE.error) { document.getElementById("footnote").textContent = STATE.error; return; }
    renderTop(STATE); renderSummary(STATE); renderList(STATE);
    renderDetail(); renderIPAnomalies(STATE); renderIPDetail(); renderIonPumps(STATE);
    document.getElementById("footnote").textContent =
      "5 秒ごとに自動更新。データ源: " + (STATE._source || "dashboard_state.json / ダミー");
  } catch (e) {
    document.getElementById("footnote").textContent = "更新失敗: " + e;
  }
  try {
    const rt = await fetch("/api/temp_state", {cache: "no-store"});
    TEMP_STATE = await rt.json();
    renderTempAnomalies(TEMP_STATE);
  } catch (e) {
    // 温度計データ取得失敗は圧力/IP タブの動作を止めない
  }
  try {
    const re_ = await fetch("/api/equipment_state", {cache: "no-store"});
    EQUIP_STATE = await re_.json();
    renderEquipmentAnomalies(EQUIP_STATE);
  } catch (e) {
    // 機器劣化データ取得失敗は他タブの動作を止めない
  }
  try {
    const rf = await fetch("/api/flow_state", {cache: "no-store"});
    FLOW_STATE = await rf.json();
    renderFlowAnomalies(FLOW_STATE);
  } catch (e) {
    // 流量計データ取得失敗は他タブの動作を止めない
  }
}
refresh();
setInterval(refresh, 5000);
refreshLoad();
setInterval(refreshLoad, 5000);
refreshBeam();
setInterval(refreshBeam, 3000);
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------------
# HTTP サーバー
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith("/api/label"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            except Exception as e:
                self._send(400, json.dumps({"ok": False, "error": "bad request: %s" % e}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            klass = body.get("klass")
            record = body.get("record")
            ring = body.get("ring")
            if klass not in ("Normal", "Abnormal") or not record or ring not in ("LER", "HER"):
                self._send(400, json.dumps({"ok": False, "error": "klass/record/ring が不正です"}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            entry = {
                "ts": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "ring": ring, "record": record,
                "period": body.get("period"),
                "abort_time": body.get("abort_time"),
                "beam_at_check": body.get("beam_at_check"),
                "klass": klass, "status": "queued",
            }
            try:
                with open(LABEL_QUEUE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                queued = sum(1 for _ in open(LABEL_QUEUE, encoding="utf-8"))
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "error": "保存に失敗: %s" % e}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            self._send(200, json.dumps({"ok": True, "queued": queued}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b'{"error":"not found"}', "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/api/beam"):
            data = beamcurrent.read() if beamcurrent is not None else {"LER": None, "HER": None}
            self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/sysload"):
            if sysload is not None:
                extra = None
                if singleton is not None:
                    det_pid = singleton.read_alive_pid(_DETECTOR_LOCK, tag="detector_headless.py")
                    if det_pid is not None:
                        _detector_proc_sampler.set_pid(det_pid)
                    extra = {"detector": (_detector_proc_sampler, det_pid is not None)}
                body = json.dumps(sysload.snapshot(_cpu_sampler, _proc_sampler, extra_procs=extra),
                                  ensure_ascii=False)
            else:
                body = json.dumps({"level": "ok", "loadavg": None})
            self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
        elif self.path.startswith("/api/state"):
            state = load_state()
            if os.environ.get("RECORD_RAW_DEMO"):
                state["_source"] = "デモ（合成データ）"
            else:
                state["_source"] = "dashboard_state.json" if os.path.isfile(STATE_FILE) else "内蔵ダミーデータ"
            self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/temp_state"):
            state = load_temp_state()
            self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/equipment_state"):
            state = load_equipment_state()
            self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/flow_state"):
            state = load_flow_state()
            self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/ip_raw"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            ring = (q.get("ring", [""])[0] or "").upper()
            record = q.get("record", [""])[0]
            if record_raw is None:
                body = {"error": "record_raw を読み込めません（numpy 未導入の可能性）"}
            else:
                body = record_raw.build_ip_view(ring, record)
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/raw"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            ring = (q.get("ring", [""])[0] or "").upper()
            record = q.get("record", [""])[0]
            if record_raw is None:
                body = {"error": "record_raw を読み込めません（numpy 未導入の可能性）"}
            else:
                body = record_raw.build_record_view(ring, record)
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args):
        pass  # アクセスログを抑制

def main():
    # ポート: --port 引数 > DASHBOARD_PORT 環境変数 > 既定 PORT(18050)。
    # 本番を止めずに別ポートでデモ表示を確認できるようにするため。
    port = PORT
    if "--port" in sys.argv:
        i = sys.argv.index("--port")
        if i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                print("エラー: --port の値が不正です: %r" % sys.argv[i + 1]); return
    elif os.environ.get("DASHBOARD_PORT"):
        try:
            port = int(os.environ["DASHBOARD_PORT"])
        except ValueError:
            print("エラー: DASHBOARD_PORT が不正です: %r" % os.environ["DASHBOARD_PORT"]); return

    # 二重起動防止（共用サーバーで複数立ち上げを防ぐ）。ロックはポートごとに分ける。
    # ポートを変えて（例: デモ確認用に別ポートで）起動する場合は、本番インスタンスの
    # ロックと衝突しないようにするため（同じ dashboard.py プロセス名でも別ポートなら共存可）。
    try:
        import singleton
        lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dashboard.%d.lock" % port)
        if not singleton.guard(lock, "dashboard.py", "ダッシュボード（ポート %d）" % port):
            return
    except Exception:
        pass  # singleton が無くても起動はできる（ポート占有でも二重起動は実質防がれる）

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as ex:
        print("ポート %d を開けませんでした（既に使用中かもしれません）: %s" % (port, ex))
        return
    src = ("デモ（合成データ）" if os.environ.get("RECORD_RAW_DEMO")
           else ("dashboard_state.json" if os.path.isfile(STATE_FILE) else "内蔵ダミーデータ"))
    print("SuperKEKB MR真空システム異常モニター（プロトタイプ）起動")
    print("  データ源: %s" % src)
    print("  ローカル:  http://localhost:%d" % port)
    print("  停止: Ctrl-C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")
        srv.shutdown()

if __name__ == "__main__":
    main()
