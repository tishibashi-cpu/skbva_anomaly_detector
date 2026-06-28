#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_raw.py — 詳細ビュー用に「生データ」を legacy 準拠の窓で取得し、
末次GUI の Make_Plot_Strg / Make_Plot_Strg_Time 相当のプロット材料を組み立てる。

ダッシュボードの詳細ポップアップで1レコード(CCG)をクリックしたときに呼ぶ。
state_builder のフィル単位トレンドはサイドバーのスパークライン用に残し、詳細だけ
この生データ版（圧力 vs ビーム電流の散布図＋回帰線、圧力＆ビーム電流 vs 時刻、
アボート直後の圧力 vs 経過秒）に差し替える。

legacy 準拠の時刻範囲（検知実行時に legacy/ に書き出される窓ファイルをそのまま読む）:
  調査 Storage : {Ring}_Date_Range_CHK_Strg_File.txt （前アボート〜今アボート＝1フィル）
  調査 Tail    : {Ring}_Date_Range_CHK_Tail_File.txt  （アボート直後）
  基準 Reference: {Ring}_Date_Range_STD_File.txt       （基準期間）

回帰線について:
  legacy の回帰は 圧力 = w0·ビーム電流 + w1·HOM_2 + w2 の2入力線形で、係数は各実行の
  一時ファイル(*_Result_Data_*_Dict)にしか残らず後から安定して読めない。そこで取得した
  生データ上で 圧力 = a·ビーム電流 + b をロバスト回帰(Theil–Sen)し直して線を引く。
  実測点＋回帰線という legacy の見た目・意図はそのまま再現できる（違いは HOM_2 を含む
  2入力ではなくビーム電流のみの回帰になる点）。基準期間にも同じ回帰をかけて薄青線を出す。

依存は numpy のみ（ロバスト回帰）＋ ip_fetch / ccg_fetch / beam_fetch（kblogrd 取得）。
kblogrd が無い/失敗しても、呼び出し側に例外を投げず {"error": ...} を返す。
"""

import os
import json

import numpy as np

import ip_fetch
import ccg_fetch
import beam_fetch
import ip_pv
import ip_judge

_HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.join(_HERE, "legacy")   # 窓ファイルが書かれる場所（detector が chdir）

NODATA = ip_fetch.NODATA
CCG_LOG_GROUP = ccg_fetch.LOG_GROUP           # "VA/CCG"

# 取得間隔[秒]（Storage はフィル全体なので粗め、Tail は短時間なので細かめ）
INTERVAL_STRG = 60
INTERVAL_TAIL = 6


# ───────────────────── 窓ファイル ─────────────────────

def _range_path(ring, kind):
    # kind: "STD" / "CHK_Strg" / "CHK_Tail"
    return os.path.join(LEGACY_DIR, "%s_Date_Range_%s_File.txt" % (ring, kind))


def read_range(ring, kind):
    """legacy の窓ファイルを読み (start, end) を返す。無ければ None。
    内容は '20240701085610-20240701091200' のような1行。"""
    path = _range_path(ring, kind)
    try:
        with open(path, encoding="utf-8") as f:
            line = f.readline().strip()
    except Exception:
        return None
    if "-" not in line:
        return None
    start, end = line.split("-", 1)
    start, end = start.strip(), end.strip()
    if len(start) == 14 and len(end) == 14 and start.isdigit() and end.isdigit():
        return start, end
    return None


# ───────────────────── legacy の2入力モデル（verbatim 移植）─────────────────────
# モデル: 圧力 = w0·I + w1·HOM_2 + w2,  HOM_2 = (I*I/Nb)^2
#   I=ビーム電流[mA], Nb=バンチ数。w1·HOM_2 は HOM 加熱由来の圧力上昇項。
# 回帰は legacy Anomaly_Detection_112p.py の fit_plane → (負係数なら) fit_plane_num →
#   w2 下限クランプ、を選別点(_Sel)で実行する手順をそのまま再現する（判定と同じ式）。
# 注: legacy は圧力の読み値を3倍して係数を出すが、ここでは取得した生圧力でフィットする
#   （実測点も予測曲線も同じ圧力スケールなので曲線の当たり方は不変。表示は実圧力）。

# 選別条件（legacy 1314-1315）と予測下限（Make_Plot_Strg y3=max(y3_r,3e-8)）
_SEL_PRES_MIN = 1e-8
_SEL_BEAM_LO = 0.30
_SEL_BEAM_HI = 0.95
_SEL_NB_MIN = 10.0
_PRED_FLOOR = 3e-8


def _fit_plane(x0, x1, t):
    """2変数線形回帰の解析解（legacy fit_plane の verbatim）。返り値 [w0,w1,w2]。"""
    c_tx0 = np.mean(t * x0) - np.mean(t) * np.mean(x0)
    c_tx1 = np.mean(t * x1) - np.mean(t) * np.mean(x1)
    c_x0x1 = np.mean(x0 * x1) - np.mean(x0) * np.mean(x1)
    v_x0 = np.var(x0)
    v_x1 = np.var(x1)
    denom = c_x0x1 ** 2 - v_x0 * v_x1
    w0 = (c_tx1 * c_x0x1 - v_x1 * c_tx0) / denom
    w1 = (c_tx0 * c_x0x1 - v_x0 * c_tx1) / denom
    w2 = -w0 * np.mean(x0) - w1 * np.mean(x1) + np.mean(t)
    return np.array([w0, w1, w2])


def _dmse_plane(x0, x1, t, w):
    y = w[0] * x0 + w[1] * x1 + w[2]
    d_w0 = 2 * np.mean((y - t) * x0)
    d_w1 = 2 * np.mean((y - t) * x1)
    d_w2 = 2 * np.mean(y - t)
    return d_w0, d_w1, d_w2


def _fit_plane_num(w_1, x0, x1, t):
    """非負制約付き勾配法（legacy fit_plane_num の verbatim）。返り値 [w0,w1,w2]。"""
    if w_1[0] < 0.: w_1[0] = 0.
    if w_1[1] < 0.: w_1[1] = 0.
    if w_1[2] < 1.e-9: w_1[2] = 1.e-9
    alpha1, alpha2, alpha3 = 1.e-7, 1.e-12, 1.e-1
    tau_max, eps = 2000, 5.e-9
    dmse = (0., 0., 0.)
    for _ in range(1, tau_max):
        dmse = _dmse_plane(x0, x1, t, w_1)
        w_1[0] = w_1[0] - alpha1 * dmse[0]
        w_1[1] = w_1[1] - alpha2 * dmse[1]
        w_1[2] = w_1[2] - alpha3 * dmse[2]
        if w_1[0] < 0.: w_1[0] = 0.
        if w_1[1] < 0.: w_1[1] = 0.
        if w_1[2] < 1.e-9: w_1[2] = 1.e-9
        if max(np.absolute(dmse)) < eps:
            break
    return np.array([w_1[0], w_1[1], w_1[2]])


def fit_model(beam, hom2, pres):
    """選別点で legacy と同じ手順で w=[w0,w1,w2] を推定。点不足なら None。"""
    beam = np.asarray(beam, float); hom2 = np.asarray(hom2, float); pres = np.asarray(pres, float)
    if len(beam) < 3:
        return None
    bmax = float(np.max(beam))
    sel = (pres >= _SEL_PRES_MIN) & (beam >= _SEL_BEAM_LO * bmax) & \
          (beam <= _SEL_BEAM_HI * bmax)
    x0, x1, t = beam[sel], hom2[sel], pres[sel]
    if len(x0) < 3 or np.var(x0) == 0:
        return None
    W = _fit_plane(x0, x1, t)
    if (W[0] < 0.) or (W[1] < 0.) or (W[2] < 1.e-9):
        W = _fit_plane_num([float(W[0]), float(W[1]), float(W[2])], x0, x1, t)
    if W[2] < 1.e-9:
        W[2] = 1.e-9
    return {"w0": float(W[0]), "w1": float(W[1]), "w2": float(W[2]),
            "n_sel": int(len(x0))}


def _pred_curve(beam, hom2, w):
    """回帰曲線（散布図に重ねる）。全点で予測し、ビーム電流順にソートして返す。
    予測は legacy と同様 max(pred, 3e-8) でクランプ。"""
    beam = np.asarray(beam, float); hom2 = np.asarray(hom2, float)
    pred = np.maximum(w["w0"] * beam + w["w1"] * hom2 + w["w2"], _PRED_FLOOR)
    order = np.argsort(beam)
    return {"beam": beam[order].tolist(), "pred": pred[order].tolist(),
            "w0": w["w0"], "w1": w["w1"], "w2": w["w2"], "n_sel": w["n_sel"]}


# ───────────────────── 取得・整列 ─────────────────────

def _valid(v):
    return v is not None and v > NODATA


def _fetch_record_series(record, start, end, interval):
    """1 CCG レコードの圧力時系列を取得 [(ts, val_or_None), ...]。"""
    ttime = ip_fetch.make_ttime(start, end, interval)
    d = ip_fetch._fetch_chunk([record], ttime, CCG_LOG_GROUP, ip_fetch.KBLOGRD)
    return d.get(record, [])


def _aligned_pib(record, ring, start, end, interval):
    """同時刻で 圧力・ビーム電流・バンチ数 がそろう点を返し、HOM_2=(I*I/Nb)^2 を付ける。
    返り値 dict {t, pressure, beam, hom2}（時刻順、Nb>0 の点のみ）。"""
    pres = _fetch_record_series(record, start, end, interval)
    beam = beam_fetch.fetch_series(ring, start, end, interval_sec=interval)
    nb = beam_fetch.fetch_nb(ring, start, end, interval_sec=interval)
    bmap = {ts: v for ts, v in beam}
    nmap = {ts: v for ts, v in nb}
    t, P, B, H = [], [], [], []
    for ts, p in pres:
        if not _valid(p):
            continue
        b = bmap.get(ts)
        n = nmap.get(ts)
        if not _valid(b) or n is None or n <= 0:
            continue
        bb, nn = float(b), float(n)
        t.append(ts); P.append(float(p)); B.append(bb)
        H.append((bb * bb / nn) ** 2)            # HOM_2 = (I*I/Nb)^2
    return {"t": t, "pressure": P, "beam": B, "hom2": H}


def _ts_to_seconds(ts):
    """'MM/DD/YYYY HH:MM:SS' → epoch 秒（Tail の経過秒計算用）。失敗で None。"""
    import datetime
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(ts, fmt).timestamp()
        except Exception:
            pass
    return None


def _range_start_seconds(start_yyyymmddhhmmss):
    import datetime
    try:
        return datetime.datetime.strptime(start_yyyymmddhhmmss, "%Y%m%d%H%M%S").timestamp()
    except Exception:
        return None


# ───────────────────── 詳細ビュー組み立て ─────────────────────

def _demo_view(ring, record):
    """RECORD_RAW_DEMO 用の合成詳細ビュー（kblogrd 不要）。手元で #5/#6 の見た目確認用。
    2入力モデル 圧力=w0·I+w1·(I²/Nb)²+w2 の経路（散布図＋回帰曲線・時系列）を実際に通す。"""
    import datetime
    rng = np.random.RandomState(abs(hash(record)) % (2 ** 32))

    def synth(n, w0, w1, w2, base_dt, jitter=0.06):
        beam = np.sort(rng.uniform(50, 1250, n))
        nb = 2000.0
        hom2 = (beam * beam / nb) ** 2
        pres = (w0 * beam + w1 * hom2 + w2) * 10 ** rng.normal(0, jitter, n)
        ts = [(base_dt + datetime.timedelta(seconds=i * 60)).strftime("%m/%d/%Y %H:%M:%S")
              for i in range(n)]
        return {"t": ts, "beam": beam.tolist(), "pressure": pres.tolist(),
                "hom2": hom2.tolist()}

    # 調査（やや高め＝異常寄り）と基準（健全）。圧力が e-8〜e-6 に渡るよう、また HOM 項で
    # 高ビーム側が湾曲するよう係数を設定（#6 の指数表記と回帰曲線の見た目確認用）。
    chk = synth(80, 7.0e-10, 1.0e-12, 3.0e-8, datetime.datetime(2026, 6, 25, 9, 0, 0))
    std = synth(80, 6.0e-10, 8.0e-13, 1.8e-8, datetime.datetime(2026, 6, 20, 9, 0, 0))

    w_chk = fit_model(chk["beam"], chk["hom2"], chk["pressure"])
    w_std = fit_model(std["beam"], std["hom2"], std["pressure"])
    storage = dict(chk)
    storage["fit_chk"] = _pred_curve(chk["beam"], chk["hom2"], w_chk) if w_chk else None
    storage["fit_std"] = _pred_curve(std["beam"], std["hom2"], w_std) if w_std else None
    reference = {"beam": std["beam"], "pressure": std["pressure"],
                 "fit_std": storage["fit_std"]}
    return {"record": record, "ring": ring, "demo": True,
            "windows": {"chk_strg": "20260625090000-20260625092000",
                        "std": "20260620090000-20260620092000"},
            "storage": storage, "reference": reference, "tail": None}


def build_record_view(ring, record, interval_strg=INTERVAL_STRG, interval_tail=INTERVAL_TAIL):
    """1レコードの詳細ビュー材料を組み立てて返す。

    返り値 dict:
      record, ring, windows{chk_strg, chk_tail, std},
      storage{t, pressure, beam, fit_chk, fit_std},   # 散布図(beam-pres)＋時系列(t)
      reference{beam, pressure},                       # 薄青回帰の基データ
      tail{dt, pressure, abort}                        # アボート後経過秒 vs 圧力
    取得できなかった部分は None。全滅や例外時は {"error": ...}。
    """
    if ring not in ("LER", "HER") or not record:
        return {"error": "ring/record が不正です"}
    # デモモード: kblogrd を呼ばず合成データを返す（手元で見た目確認）。
    if os.environ.get("RECORD_RAW_DEMO"):
        try:
            return _demo_view(ring, record)
        except Exception as ex:
            return {"error": "demo 生成失敗: %s" % ex}
    try:
        out = {"record": record, "ring": ring, "windows": {}}

        # 調査 Storage（散布図＋時系列の主役）
        r = read_range(ring, "CHK_Strg")
        storage = None
        if r:
            out["windows"]["chk_strg"] = "%s-%s" % r
            d = _aligned_pib(record, ring, r[0], r[1], interval_strg)
            if d["t"]:
                storage = d
                w = fit_model(d["beam"], d["hom2"], d["pressure"])
                storage["fit_chk"] = _pred_curve(d["beam"], d["hom2"], w) if w else None

        # 基準 Reference（薄青回帰）
        r = read_range(ring, "STD")
        reference = None
        if r:
            out["windows"]["std"] = "%s-%s" % r
            d = _aligned_pib(record, ring, r[0], r[1], interval_strg)
            if d["t"]:
                w = fit_model(d["beam"], d["hom2"], d["pressure"])
                fit_std = _pred_curve(d["beam"], d["hom2"], w) if w else None
                reference = {"beam": d["beam"], "pressure": d["pressure"], "fit_std": fit_std}
                if storage is not None:
                    storage["fit_std"] = fit_std

        # 調査 Tail は詳細ビューで非表示にしたため取得しない（kblogrd 呼び出し節約）。
        # 必要になれば下のブロックを戻す。
        tail = None

        out["storage"] = storage
        out["reference"] = reference
        out["tail"] = tail
        if storage is None and tail is None:
            out["error"] = ("この期間の生データを取得できませんでした"
                            "（窓ファイル未生成か、レコード名不一致の可能性）")
        return out
    except RuntimeError as ex:
        # kblogrd 不在/失敗（実機以外など）
        return {"error": str(ex)}
    except Exception as ex:
        return {"error": "生データ取得に失敗: %s" % ex}


# ═════════════════ イオンポンプ詳細（電流 vs 時刻／I-P 散布図）═════════════════

IP_LOG_GROUP = "VA/IPump"
PRES_FLOOR = {ip_pv.SUPPLY_KEK: 1e-7, ip_pv.SUPPLY_4U: 1e-8}
IP_VIEW_HOURS = 24            # 詳細に出す直近時間[h]
IP_VIEW_INTERVAL = 300        # 取得間隔[s]


def _now_window(hours):
    import datetime
    end = datetime.datetime.now()
    start = end - datetime.timedelta(hours=hours)
    return start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S")


def _fetch_pv_lg(pv, start, end, interval, log_group):
    ttime = ip_fetch.make_ttime(start, end, interval)
    d = ip_fetch._fetch_chunk([pv], ttime, log_group, ip_fetch.KBLOGRD)
    return d.get(pv, [])


def _learned_band(ring, record):
    """ip_models.json から学習バンド(電流 p50/p95)と学習 I=a·P^b を読む。
    返り値 (band|None, learned_fit|None)。band={p50,p95}(log10電流), learned_fit={a,b}。"""
    path = os.environ.get("RECORD_RAW_MODELS",
                          os.path.join(_HERE, "ip_models.json"))
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            models = json.load(f)
    except Exception:
        return None, None
    m = (models.get(ring) or {}).get(record)
    if not m:
        return None, None
    band = None
    if ("cur_log_p50" in m) and ("cur_log_p95" in m):
        band = {"p50": float(m["cur_log_p50"]), "p95": float(m["cur_log_p95"])}
    lfit = None
    if m.get("a") and m.get("b"):
        lfit = {"a": float(m["a"]), "b": float(m["b"])}
    return band, lfit


def build_ip_view(ring, record, hours=IP_VIEW_HOURS, interval=IP_VIEW_INTERVAL):
    """イオンポンプ1台の詳細ビュー材料。直近 hours 時間を取得して返す。
      current{t, I, beam}        … 電流 vs 時刻（無ビーム区間ハイライト用に beam も）
      scatter{P, I, fit{a,b,P,I}}… I-P 散布図（I=a·P^b のロバスト回帰線つき）
    取得失敗/例外は {"error": ...}。デモ時は合成。"""
    if ring not in ("LER", "HER") or not record:
        return {"error": "ring/record が不正です"}
    if os.environ.get("RECORD_RAW_DEMO"):
        try:
            return _demo_ip_view(ring, record)
        except Exception as ex:
            return {"error": "demo 生成失敗: %s" % ex}
    try:
        supply = ip_pv.supply_of(record)
        start, end = _now_window(hours)
        cur = _fetch_pv_lg(record, start, end, interval, IP_LOG_GROUP)
        beam = beam_fetch.fetch_series(ring, start, end, interval_sec=interval)
        bmap = {ts: v for ts, v in beam}
        ccg = ip_pv.paired_ccg(record)
        pres = _fetch_pv_lg(ccg, start, end, interval, CCG_LOG_GROUP) if ccg else []
        pmap = {ts: v for ts, v in pres}

        t, I, B = [], [], []
        sP, sI = [], []
        for ts, iv in cur:
            if not _valid(iv):
                continue
            bv = bmap.get(ts)
            t.append(ts); I.append(float(iv))
            B.append(float(bv) if _valid(bv) else None)
            pv = pmap.get(ts)
            if _valid(pv) and iv > 0:
                sP.append(float(pv)); sI.append(float(iv))
        out = {"record": record, "ring": ring, "supply": supply,
               "window": "%s-%s" % (start, end), "current": None, "scatter": None}
        if t:
            out["current"] = {"t": t, "I": I, "beam": B}
        if len(sP) >= 5:
            floor = PRES_FLOOR.get(supply, 1e-7)
            fit = ip_judge.robust_powerlaw_fit(np.array(sP), np.array(sI), floor)
            scat = {"P": sP, "I": sI, "fit": None}
            if fit:
                a, b = fit["a"], fit["b"]
                pp = np.array(sorted(p for p in sP if p > 0))
                if len(pp):
                    scat["fit"] = {"a": a, "b": b,
                                   "P": pp.tolist(), "I": (a * pp ** b).tolist()}
            out["scatter"] = scat
        if out["current"] is None and out["scatter"] is None:
            out["error"] = "この期間のイオンポンプ生データを取得できませんでした"
        # 学習バンド（学習窓の電流 p50/p95）と学習 I=a·P^b を ip_models から重ねる
        band, lfit = _learned_band(ring, record)
        out["band"] = band
        out["learned_fit"] = lfit
        return out
    except RuntimeError as ex:
        return {"error": str(ex)}
    except Exception as ex:
        return {"error": "IP生データ取得に失敗: %s" % ex}


def _demo_ip_view(ring, record):
    """RECORD_RAW_DEMO 用の合成 IP 詳細（電流時系列＋I-P 散布図）。"""
    import datetime
    rng = np.random.RandomState((abs(hash(record)) + 7) % (2 ** 32))
    supply = ip_pv.supply_of(record)
    n = 120
    base = datetime.datetime(2026, 6, 25, 0, 0, 0)
    t = [(base + datetime.timedelta(seconds=i * 720)).strftime("%m/%d/%Y %H:%M:%S")
         for i in range(n)]
    # ビーム: 前半 蓄積(高)→ 中盤 無ビーム → 後半 蓄積。無ビーム区間ハイライト確認用。
    beam = []
    for i in range(n):
        if 45 <= i < 70:
            beam.append(0.0)                       # 無ビーム
        else:
            beam.append(700 + 250 * np.sin(i / 8.0) + rng.normal(0, 20))
    beam = [max(0.0, b) for b in beam]
    # 圧力: ビームに概ね比例（光脱離）＋床
    floor_p = PRES_FLOOR.get(supply, 1e-7)
    P = [floor_p * (1 + 0.0 if b < 10 else 1.0) + 8e-10 * b * 10 ** rng.normal(0, 0.05)
         for b in beam]
    # 電流: 放電気味（無ビームでも下がりきらず、スパイクあり）→ acute/放電らしさ
    I = []
    for i, b in enumerate(beam):
        baseI = 3e-5 + 4e-8 * b
        if 52 <= i < 58:
            baseI = 2e-4                            # スパイク
        I.append(baseI * 10 ** rng.normal(0, 0.08))
    out = {"record": record, "ring": ring, "supply": supply, "demo": True,
           "window": "demo", "current": {"t": t, "I": I, "beam": beam}, "scatter": None}
    sP = [p for p, iv in zip(P, I) if p > 0 and iv > 0]
    sI = [iv for p, iv in zip(P, I) if p > 0 and iv > 0]
    fit = ip_judge.robust_powerlaw_fit(np.array(sP), np.array(sI), floor_p)
    scat = {"P": sP, "I": sI, "fit": None}
    if fit:
        pp = np.array(sorted(p for p in sP if p > 0))
        scat["fit"] = {"a": fit["a"], "b": fit["b"],
                       "P": pp.tolist(), "I": (fit["a"] * pp ** fit["b"]).tolist()}
    out["scatter"] = scat
    # デモの学習バンド: 平常電流帯（放電点が上に外れて見えるよう低めに置く）
    out["band"] = {"p50": float(np.log10(1.8e-5)), "p95": float(np.log10(3.5e-5))}
    out["learned_fit"] = {"a": fit["a"] * 0.5, "b": fit["b"]} if fit else None
    return out


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) >= 3:
        print(json.dumps(build_record_view(sys.argv[1], sys.argv[2]),
                         ensure_ascii=False, indent=1))
    else:
        print("usage: python record_raw.py <LER|HER> <VALCCG:..:PRES>")
