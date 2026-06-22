#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
state_builder.py  ——  検知プログラムの蓄積ファイル → dashboard_state.json

末次さんの検知プログラム（Anomaly_Detection_112p.py）が追記する
  {RING}_{METHOD}_Abnormal_Class2_Result_{MODE}_WB.txt
を読み取り専用で解析し、ダッシュボード用の dashboard_state.json を書き出す。
検知本体には一切触らない（読むだけ）。

カウント集計は検知側 Record_Freq_plot と同じ定義:
  ・各レコードの「直近 NFILL フィル(=1 Period)で異常と出た回数」を数える
  ・回数の多い順に上位を採用

第一版で出すもの: カウント / ビーム電流 / 最終チェック時刻 / セクション / 深刻度
次段階で接続: 推定原因（このファイルには残らないため）/ 圧力トレンドの時系列
"""

import csv
import json
import os
import datetime
from collections import Counter

# 推定原因（ヘッドレス・純numpy再現）。h5py が無い環境でも壊れないよう任意依存にする。
try:
    import cause_infer
except Exception:
    cause_infer = None

# パスは __file__ 基準で解決する（detector_headless が legacy/ に chdir しても壊れないように）。
#   蓄積ファイル(*_Result_*) … legacy/（検知本体が CWD 相対で書く場所）
#   dashboard_state.json     … トップ（dashboard.py が読む場所）
_HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(_HERE, "legacy")          # 蓄積ファイルのあるディレクトリ
OUT = os.path.join(_HERE, "dashboard_state.json")
NFILL = 8                     # 1 Period = 8 フィル
TOPN = 5                      # 各モードで採用する上位レコード数
NTREND = 40                   # 圧力トレンドに載せる直近フィル数（詳細ビュー用）
METHOD = "FNN"               # ファイル名の Method 部（FNN または SDM）

# 列インデックス（ヘッダ準拠。Strg/Tail で Abort/Beam/Max_Pre 位置は共通）
COL_FILL = 0                  # Date_Range_DIF  例 "20240701085610-20240701091200"
COL_ABORT = 2                # Abort_Timing    例 "2024-07-01 09:12:00"
COL_RECORD = 3               # Record Name     例 "VALCCG:D05_L24:PRES"
COL_BEAM = 6                 # Beam_Max [mA]   例 "1.50e+03"
COL_MAXPRE = 8               # Max_Pre  [Pa]   Storage は生圧力、Tail は規格化圧力
COL_TAIL_RAWPRE = 12         # Max_Raw_Pre [Pa]  Tail の生圧力（トレンドはこちらを使う）


def ring_of(record):
    """VALCCG:... → LER, VAHCCG:... → HER"""
    return "LER" if record.startswith("VAL") else "HER"


def section_of(record):
    """VALCCG:D05_L24:PRES → D05"""
    try:
        return record.split(":")[1].split("_")[0]
    except Exception:
        return "?"


def _read_result(ring, mode, kind):
    """kind='Abnormal'/'Normal' の結果ファイル（WB）をヘッダ除いて行で返す。"""
    path = os.path.join(ML_DIR,
                        "%s_%s_%s_Class2_Result_%s_WB.txt" % (ring, METHOD, kind, mode))
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:]


def read_abnormal(ring, mode):
    return _read_result(ring, mode, "Abnormal")


def build_series(record, abn_rows, nrm_rows, pres_col=COL_MAXPRE):
    """指定レコードの「フィル単位の圧力・ビーム電流トレンド」を作る。

    Abnormal と Normal の両結果ファイルから当該レコードの行を集め、フィル(Date_Range_DIF)で
    重複を除き、アボート時刻でソートして直近 NTREND 件を返す。各点は
    (フィル順index, 圧力[Pa], Beam_Max[mA])。生の時系列(kblogrd)は永続化されないため、
    永続化されているフィル要約からトレンドを構成する（Fig.16b 相当：異常の進行が見える）。
    pres_col は圧力列（Storage=Max_Pre, Tail=Max_Raw_Pre）。
    """
    need = max(pres_col, COL_BEAM, COL_RECORD)
    by_fill = {}
    for r in abn_rows + nrm_rows:
        if len(r) <= need or r[COL_RECORD] != record:
            continue
        fill = r[COL_FILL]
        # 同一フィルが両方にある場合は後勝ち（実際はどちらか一方）
        by_fill[fill] = r

    pts = []
    for fill, r in by_fill.items():
        try:
            t = datetime.datetime.strptime(r[COL_ABORT], "%Y-%m-%d %H:%M:%S")
            pres = float(r[pres_col])
            beam = float(r[COL_BEAM])
        except Exception:
            continue
        pts.append((t, pres, beam))

    pts.sort(key=lambda x: x[0])
    pts = pts[-NTREND:]
    return {
        "t": list(range(len(pts))),
        "pressure": [p[1] for p in pts],
        "beam": [p[2] for p in pts],
        "t_abort": [p[0].strftime("%Y-%m-%d %H:%M:%S") for p in pts],
    }


def recent_period(data):
    """直近 NFILL フィルの行と、最新フィル識別子を返す。"""
    fills = sorted({r[COL_FILL][15:] for r in data}, reverse=True)
    if not fills:
        return [], None
    recent = set(fills[:NFILL])
    sub = [r for r in data if r[COL_FILL][15:] in recent]
    return sub, fills[0]


def severity(count):
    if count >= 6:
        return "danger"
    if count >= 3:
        return "warning"
    return None  # 0–2 はカードに出さない


RECENT_DAYS = 3                          # 「現在の異常」とみなす検知時刻からの日数
STATUS_FILE = os.path.join(_HERE, "detector_status.json")  # detector が書く現在状態
CHK_FILE = os.path.join(ML_DIR, "Date_Range_CHK_File.txt")  # 今回の検知時刻範囲


def _parse_abort(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def count_monitored():
    """監視CCG台数を CSV（最新の監視対象）から数える。(LER, HER, total) を返す。
    各 CSV はヘッダ1行＋1行=1ゲージなので、データ行数がそのまま台数。"""
    counts = {}
    for ring in ("LER", "HER"):
        path = os.path.join(_HERE, "pv_info", "%s_CCG_PV.csv" % ring)
        n = 0
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            # ヘッダを除き、PV名（1列目）が空でない行を数える
            n = sum(1 for r in rows[1:] if r and r[0].strip())
        except Exception:
            n = 0
        counts[ring] = n
    return counts["LER"], counts["HER"], counts["LER"] + counts["HER"]


def load_status():
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def check_time_for(ring, status, data):
    """このリングの「検知時刻」を datetime で返す（実時刻フィルタの基準）。
    優先: detector_status.json → Date_Range_CHK_File.txt の末尾 → データ中の最大時刻 → now。"""
    st = (status or {}).get(ring)
    if st and st.get("check_time"):
        dt = _parse_abort(st["check_time"])
        if dt:
            return dt
    try:
        with open(CHK_FILE, encoding="utf-8") as f:
            end = f.read().strip().split("-")[-1]
        return datetime.datetime.strptime(end, "%Y%m%d%H%M%S")
    except Exception:
        pass
    times = [t for t in (_parse_abort(r[COL_ABORT]) for r in data) if t]
    return max(times) if times else datetime.datetime.now()


def build_state():
    anomalies = []
    sections = {ring: {"D%02d" % i: None for i in range(1, 13)} for ring in ("LER", "HER")}

    # 推定原因の推論器（legacy/ のモデル・標準化を読む）。失敗しても致命的にしない。
    inferer = None
    if cause_infer is not None:
        try:
            inferer = cause_infer.CauseInferer(ML_DIR)
        except Exception:
            inferer = None

    def infer_cause(row, mode):
        """結果ファイル1行 → 原因テキスト。推論不可なら従来のプレースホルダ。"""
        if inferer is None:
            return "—（原因モデル未読込）"
        period_key = "strg" if mode == "Strg" else "tail"
        try:
            if not inferer.available(period_key, "wb"):
                return "—（原因モデルなし）"
            cause, _ci, _p = inferer.infer_row(row, period_key, "wb")
            return cause if cause else "—（推定不可）"
        except Exception:
            return "—（推定エラー）"

    status = load_status()
    last_anomaly = {"LER": None, "HER": None}   # 履歴上の最新異常（フィルタで消えても footnote に残す）
    ring_check = {"LER": None, "HER": None}      # 検知時刻（datetime）
    ring_beam_recent = {"LER": None, "HER": None}

    for ring in ("LER", "HER"):
        check_dt = check_time_for(ring, status, [])  # 状態/CHKファイルから先に決める
        for mode, period in (("Strg", "Storage"), ("Tail", "Tail")):
            data = read_abnormal(ring, mode)
            if not data:
                continue

            # 履歴上の最新異常（実時刻フィルタの前に全データから拾う）
            for r in data:
                t = _parse_abort(r[COL_ABORT])
                if t and (last_anomaly[ring] is None or t > last_anomaly[ring]["dt"]):
                    last_anomaly[ring] = {"dt": t, "record": r[COL_RECORD],
                                          "abort_time": r[COL_ABORT], "period": period}

            # データが無い時の検知時刻はデータからも補えるよう再評価
            if ring_check[ring] is None:
                ring_check[ring] = check_time_for(ring, status, data)
                check_dt = ring_check[ring]

            # ── 実時刻フィルタ: 検知時刻から RECENT_DAYS 以内の異常だけを「現在」とする ──
            cutoff = check_dt - datetime.timedelta(days=RECENT_DAYS)
            recent_data = [r for r in data
                           if (_parse_abort(r[COL_ABORT]) or datetime.datetime.min) >= cutoff]
            if not recent_data:
                continue  # 直近に異常なし → このリング/モードは出さない（静かな期間は OK 表示に）

            normal_rows = _read_result(ring, mode, "Normal")  # トレンドを密にするため
            sub, latest_fill = recent_period(recent_data)
            if not sub:
                continue

            # 各レコードの最新行（後勝ち = より新しいフィル）
            seen = {}
            for r in sub:
                seen[r[COL_RECORD]] = r

            cnt = Counter(r[COL_RECORD] for r in sub)
            for rec, c in cnt.most_common(TOPN):
                sev = severity(c)
                if sev is None:
                    continue
                sec = section_of(rec)
                row = seen[rec]
                try:
                    beam_val = float(row[COL_BEAM])
                except Exception:
                    beam_val = None
                anomalies.append({
                    "id": "%s-%s-%s" % (ring.lower(), sec.lower(), mode.lower()),
                    "device_type": "CCG",
                    "record": rec,
                    "ring": ring, "section": sec, "place": "",
                    "period": period,
                    "count": int(c), "max_count": NFILL,
                    "cause": infer_cause(row, mode),
                    "severity": sev,
                    "abort_time": row[COL_ABORT],
                    "beam_at_check": beam_val,
                    "series": build_series(
                        rec, data, normal_rows,
                        pres_col=(COL_TAIL_RAWPRE if mode == "Tail" else COL_MAXPRE)),
                })
                # リングマップ: より深刻な方を残す
                prev = sections[ring][sec]
                if prev != "danger":
                    sections[ring][sec] = sev

            # 直近フィルのビーム電流（ヘッダの current_mA 用）
            latest_row = max(sub, key=lambda r: r[COL_FILL][15:])
            try:
                ring_beam_recent[ring] = float(latest_row[COL_BEAM])
            except Exception:
                pass

    # ── ヘッダ（リング状態）: detector_status を最優先で実態に合わせる ──
    def ring_header(ring):
        st = (status or {}).get(ring)
        chk = ring_check[ring]
        last_check = chk.strftime("%Y-%m-%d %H:%M:%S") if chk else None
        if st is not None:
            beam_on = bool(st.get("beam_on"))
            cur = ring_beam_recent[ring] if beam_on else 0
            return {"beam_on": beam_on, "current_mA": cur or 0,
                    "last_check": st.get("check_time") or last_check}
        # status が無い場合: 直近データから推定（無ければビーム不明）
        cur = ring_beam_recent[ring]
        return {"beam_on": cur is not None and cur > 50,
                "current_mA": cur or 0, "last_check": last_check}

    # 「最後の異常」note（A案: 現在異常なしでも履歴を小さく添える）
    def last_anom_note():
        parts = []
        for ring in ("LER", "HER"):
            la = last_anomaly[ring]
            if la:
                parts.append("%s %s（%s）" % (ring, la["abort_time"], la["record"]))
        return " / ".join(parts) if parts else None

    # サマリー
    critical = sum(1 for a in anomalies if a["count"] >= 6)
    warning = sum(1 for a in anomalies if 3 <= a["count"] < 6)

    # 監視台数（CSV から実数を数える。リング別内訳つき）
    ler_n, her_n, total_n = count_monitored()

    state = {
        "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classifier": METHOD,
        "rings": {"LER": ring_header("LER"), "HER": ring_header("HER")},
        "summary": {"monitored": {"total": total_n, "LER": ler_n, "HER": her_n},
                    "critical": critical, "warning": warning},
        "anomalies": anomalies,
        "sections": sections,
        "recent_window_days": RECENT_DAYS,
        "last_anomaly": last_anom_note(),
        "_note": "原因=純numpy再現／トレンド=フィル要約／異常は検知時刻から%d日以内に限定" % RECENT_DAYS,
    }
    # 拡張（開発中）：イオンポンプ放電電流。ip_state が保存した要約があれば載せる。
    # 無ければキー自体を出さない（ダッシュボードは無ければ非表示）。
    try:
        import ip_state
        ip_block = ip_state.load_saved()
        if ip_block:
            state["ion_pumps"] = ip_block
    except Exception:
        pass
    return state


def main():
    state = build_state()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print("%s を書き出しました（異常 %d 件）" % (OUT, len(state["anomalies"])))


if __name__ == "__main__":
    main()
