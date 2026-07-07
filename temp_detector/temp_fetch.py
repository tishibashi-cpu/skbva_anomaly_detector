#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_fetch.py — ビームパイプ温度計の履歴を kblogrd で取得する。

CCG/イオンポンプと同じく kblogrd でログを引く。末次プログラムの .sh が
  /usr/local/bin/kblogrd -r PV1,PV2,... -t yyyymmddhhmmss-yyyymmddhhmmssd<秒> -f kaleida <ログ群> > out
で呼ぶのを踏襲し、Python から実行する（実機 kekb-co-user01/02/03 でのみ動作）。

ログ群（最後の引数）はサブシステム名:
    CCG=VA/CCG, イオンポンプ電流=VA/IPump, ビーム電流=BM/DCCT, 温度計=VA/VATemp（実機確認済み）

温度計ならではの設計差（ここが ip_fetch と違う点）:
  1) 欠測は NaN/None で表す。**温度は負値が正常な測定値になりうる**（短絡故障で -14℃ 等）ので、
     ip_fetch の NODATA=1e-10 のような「数値フラグで欠測を表す」方式は使わない。数値はすべて
     そのまま残し（負値も）、数値化できないトークンだけ None（→ judge 側で np.isfinite が落とす）。
  2) サンプリング間隔は既定 300 秒と長め。理由: 監視本数が多く（LER 1550 / HER 1260 ≒ 2810 本）、
     かつ温度は熱質量で緩慢に動くため細かく刻む必要がない。grad（接触不良の高速グリッチ）を
     狙うときだけ一時的に間隔を詰めればよい。
  3) 上下ペアは扱わない（まずは単独センサの故障予兆のみ）。本モジュールは取得だけを担当する。

PV 名リストは <RING>_TEMP_PV.csv（1列目 PV、先頭行はヘッダ "TEMP PV"）から読む。
ファイル名取り違え対策として、読んだ PV の接頭辞（VAL/VAH）から実リングを判定し、
要求リングと一致するものだけを使う（不一致は警告して除外）。
"""

import csv
import os
import re
import subprocess
import sys

import numpy as np

import temp_pv

HERE = os.path.dirname(os.path.abspath(__file__))
PV_INFO_DIR = os.path.join(os.path.dirname(HERE), "pv_info")

KBLOGRD = "/usr/local/bin/kblogrd"
LOG_GROUP = {"LER": "VA/VATemp", "HER": "VA/VATemp", "IR": "BM/BMOthers"}  # リング別ログ群（実機確認済み）
CHUNK = int(os.environ.get("TEMP_KBLOGRD_CHUNK", 26))
    # 1回の kblogrd に渡す PV 数。元は末次さんの CCG 用 .sh（legacy/HERD01CCG.sh 等）が
    # D01セクションのCCG27本を14本+13本に手分けして kblogrd -r していたのを踏襲し、
    # 当初 13 を既定にしていたが、kblogrd 自体に「1回13本まで」という制限は無いと分かり、
    # 実機で26本一括取得も問題なく動作したことを確認できたため既定値を26に引き上げた。
    # さらに増やせるか試す場合は環境変数 TEMP_KBLOGRD_CHUNK で上書きできる
    # （例: setenv TEMP_KBLOGRD_CHUNK 52 ; python temp_equipment.py learn ... 　※tcshの場合。
    # bashなら env TEMP_KBLOGRD_CHUNK=52 python ...）。段階的に増やして動作・所要時間を
    # 確認するのを推奨（kblogrd側の未知の上限に備え、いきなり大きくしない）。
DEFAULT_INTERVAL = 300       # サンプリング間隔[秒]。本数が多く温度は緩慢なので長め（5分）
TIMEOUT = int(os.environ.get("TEMP_KBLOGRD_TIMEOUT", 300))
                            # kblogrd 1回あたりのタイムアウト[秒]。既定300s（旧120sではlearnの
                            # 数週間規模の窓取得でTimeoutExpiredが実機で発生したため延長。
                            # 詳細はip_fetch.pyの同種の修正コメント参照）。
                            # 環境変数 TEMP_KBLOGRD_TIMEOUT でコード変更なしに上書き可

# kblogrd が「指定 record 名がアーカイブに無い」ときに stderr に出す目印。
# 過去窓では当時未設置/改名/撤去の PV が現行 CSV に含まれて出る。該当 PV を落として続行する。
_NOMATCH_MARK = "specified record name doesn't match"
_PV_TOKEN_RE = re.compile(r"[A-Za-z][\w]*:[\w]+(?::[\w]+)*")

# kblogrd が温度の欠測に専用の数値センチネルを使う場合はここに足す（実機で確認できたら）。
# 既定は空＝数値はすべて実測として残す（温度は負値も正常値になりうるため数値での欠測判定はしない）。
NODATA_VALUES = set()


# ───────────────────────── PV リスト ─────────────────────────

RING_OVERRIDE_FILE = os.path.join(PV_INFO_DIR, "TEMP_RING_OVERRIDE.csv")


def load_ring_overrides(path=None):
    """温度計センサのビームリング例外を読む（配線間違い、一時的な物理移設など、
    PV名から機械的に決まる所属と実際の設置リングが食い違う個体用）。全形式共通
    （VA{L,H}TMP のビームパイプ本体センサ、FB_MOVE の IR センサ、両方に効く）。

    形式: ヘッダ 'PV,ring' の2列 CSV。例:
        PV,ring
        FB_MOVE:D02:QC1H:BWS:TEMP,LER          ← 配線間違い（恒久的な例外）
        VAHTMP:D10_139:QD3E_11:BL,LER          ← 一時的にLERへ物理移設した温度計
    ファイルが無ければ空（上書き無し）として扱う。ここを直接編集すれば、コードを
    触らずに例外を追加・修正・削除できる（pv_info/TEMP_RING_OVERRIDE.csv、CCG/IP
    の PV リストと同じ置き場）。PV名（アーカイブ上の取得先）は変わらないので、
    取得は常に natural_ring 側の <RING>_TEMP_PV.csv から行われる（temp_fetch内部で
    自動処理。ここを気にする必要はない）。
    """
    path = path or RING_OVERRIDE_FILE
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or not row[0].strip():
                continue
            if i == 0 and row[0].strip().upper() in ("PV", "PV NAME"):
                continue                       # ヘッダ行を捨てる
            pv = row[0].strip()
            ring = (row[1].strip().upper() if len(row) > 1 else "")
            if ring not in ("LER", "HER"):
                sys.stderr.write("[TEMP_RING_OVERRIDE] 行 %d: ring は LER/HER のみ有効です（%r は無視）: %s\n"
                                 % (i + 1, ring, pv))
                continue
            out[pv] = ring
    return out


def csv_path(ring):
    """<RING>_TEMP_PV.csv の絶対パス（skbva_anomaly_detector/pv_info/、CCG/IP と共通の置き場）。"""
    return os.path.join(PV_INFO_DIR, "%s_TEMP_PV.csv" % ring.upper())


def load_pv_list(ring, path=None, strict_ring=True):
    """指定リングの温度計 PV を CSV から読み、temp_pv で解析した dict のリストを返す。

    先頭行（"TEMP PV"）はヘッダとして捨てる。BOM/CRLF を吸収する。
    strict_ring=True のとき、PV名から機械的に決まる所属（natural_ring。IR形式は
    tag のH/L）が要求と違うものは（ファイル名取り違え対策として）警告して除外する。
    ※ TEMP_RING_OVERRIDE.csv による上書きが適用されていても、この判定は natural_ring
    （上書き前）で行う。PV は常にその natural_ring 側のファイルに実在するはずで、
    ここでの判定は「取り違えていないか」のチェックであり、物理的な設置場所の話とは
    別物のため。返り値の d["ring"] には上書き後の実効リングが入る。
    ring="IR" のときは IR_TEMP_PV.csv（FB_MOVE 形式）を読み、各 PV のビームリングは
    tag(QC1H/QC1L)から自動推定する。このバケットは元々 LER/HER が混在する設計なので
    strict_ring は適用しない（IR自体が「所属ファイル」の概念）。
    """
    path = path or csv_path(ring)
    ring = ring.upper()
    is_ir = (ring == "IR")
    pvs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            cell = row[0].strip()
            if i == 0 and not cell.upper().startswith("VA"):
                continue                       # ヘッダ行 "TEMP PV" を捨てる
            if cell:
                pvs.append(cell)

    ring_override = load_ring_overrides()
    records, mismatch = [], 0
    for pv in pvs:
        d = temp_pv.parse_pv(pv, ring_override=ring_override)
        if not d:
            continue
        if strict_ring and not is_ir and d["natural_ring"] != ring:
            mismatch += 1
            continue
        records.append(d)
    if mismatch:
        sys.stderr.write(
            "[%s] 警告: 接頭辞が %s と一致しない PV を %d 本除外しました"
            "（ファイル %s の中身が別リングの可能性）。\n"
            % (ring, ring, mismatch, os.path.basename(path)))
    return records


# ───────────────────────── kblogrd ─────────────────────────

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_ttime(start, end, interval_sec=DEFAULT_INTERVAL):
    """kblogrd の -t 引数 'yyyymmddhhmmss-yyyymmddhhmmssd<秒>' を作る。"""
    return "%s-%sd%d" % (start, end, interval_sec)


def parse_kaleida(text, pvs):
    """kblogrd -f kaleida の出力を {pv: [(ts, value_or_None), ...]} に変換する。

    出力（スペース区切り）:
        time VALTMP:D10M001:QDWNP_4:BL        ← ヘッダ行（'/' を含まない）はスキップ
        06/10/2026 00:00:00 25.31 24.98 ...   ← 日付 時刻 値[ 値...]（-r の順）
    ・先頭2語が時刻 'MM/DD/YYYY HH:MM:SS'。
    ・数値化できないトークンは None（欠測）。**負値はそのまま残す**（短絡故障の証拠）。
    ・NODATA_VALUES に登録した数値だけは None 化（既定は空）。
    """
    series = {pv: [] for pv in pvs}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        tok = s.split()
        if len(tok) < 3 or "/" not in tok[0]:    # ヘッダ/不正行をスキップ
            continue
        ts = tok[0] + " " + tok[1]
        vals = tok[2:]
        for i, pv in enumerate(pvs):
            if i >= len(vals):
                break
            try:
                v = float(vals[i])
                if v in NODATA_VALUES:
                    v = None
            except ValueError:
                v = None
            series[pv].append((ts, v))
    return series


def _kblogrd_once(pvs, ttime, log_group, kblogrd):
    """kblogrd を1回実行し (rc, stdout, stderr) を返す。stdin は閉じ、TIMEOUT で打ち切る。
    タイムアウト時は rc=None を返す（raise しない。_fetch_chunk が二分探索で再試行できる
    ようにするため。ip_fetch.py と同じ方針）。"""
    cmd = [kblogrd, "-r", ",".join(pvs), "-t", ttime, "-f", "kaleida", log_group]
    try:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=TIMEOUT)
    except FileNotFoundError:
        raise RuntimeError(
            "kblogrd が見つかりません: %s\n"
            "温度の取得は kblogrd のある実機（kekb-co-user01/02/03）で実行してください。"
            "手元では `python temp_fetch.py selftest`（kblogrd 不要）で確認できます。" % kblogrd)
    except subprocess.TimeoutExpired:
        return None, "", ""
    return res.returncode, res.stdout, res.stderr


def _fetch_chunk(pvs, ttime, log_group, kblogrd, dropped_out=None, _depth=0):
    """1チャンクを頑健に取得し {pv: [(ts, value), ...]} を返す。

    kblogrd は1本でも存在しない record があるとチャンク全体を失敗させるため、
    stderr から犯人を特定して落とすか、特定できなければ二分探索で1本単位まで隔離して落とす
    （ip_fetch と同じ方針）。落とした PV は dropped_out に追記。不一致以外の失敗は raise。

    タイムアウト（rc=None）も同じ二分探索で対処する（本数を減らせば完走することがあるため）。
    1本まで割ってもなおタイムアウトする場合は、期間そのものが重いということなので分かりやすい
    エラーに変換する。
    """
    pvs = list(pvs)
    if not pvs:
        return {}
    rc, out, err = _kblogrd_once(pvs, ttime, log_group, kblogrd)
    if rc == 0:
        return parse_kaleida(out, pvs)

    if rc is None:                          # タイムアウト
        if len(pvs) == 1:
            raise RuntimeError(
                "kblogrd が %d 秒以内に応答しませんでした（PV 1本: %s, 期間 %s）。\n"
                "本数を1本まで減らしても解消しないため、期間そのものが重いと考えられます。"
                "対処法:\n"
                "  1) 環境変数 TEMP_KBLOGRD_TIMEOUT でタイムアウトを延ばして再実行する"
                "（例: env TEMP_KBLOGRD_TIMEOUT=900 python temp_equipment.py learn ...）\n"
                "  2) 期間を短く区切って複数回に分けて取得する\n"
                "  3) しばらく待って（アーカイバ/共用サーバーの負荷が下がってから）再実行する"
                % (TIMEOUT, pvs[0], ttime))
        mid = len(pvs) // 2
        d = {}
        d.update(_fetch_chunk(pvs[:mid], ttime, log_group, kblogrd, dropped_out, _depth + 1))
        d.update(_fetch_chunk(pvs[mid:], ttime, log_group, kblogrd, dropped_out, _depth + 1))
        return d

    if _NOMATCH_MARK not in (err or ""):
        raise RuntimeError("kblogrd 失敗 (rc=%d): %s" % (rc, (err or "").strip()))

    bad = {tok for tok in _PV_TOKEN_RE.findall(err or "") if tok in set(pvs)}
    if bad:
        if dropped_out is not None:
            dropped_out.extend(sorted(bad))
        return _fetch_chunk([p for p in pvs if p not in bad],
                            ttime, log_group, kblogrd, dropped_out, _depth + 1)

    if len(pvs) == 1:
        if dropped_out is not None:
            dropped_out.append(pvs[0])
        return {}
    mid = len(pvs) // 2
    d = {}
    d.update(_fetch_chunk(pvs[:mid], ttime, log_group, kblogrd, dropped_out, _depth + 1))
    d.update(_fetch_chunk(pvs[mid:], ttime, log_group, kblogrd, dropped_out, _depth + 1))
    return d


def fetch_history(ring, start, end, interval_sec=DEFAULT_INTERVAL,
                  log_group=None, kblogrd=None, csv_path_=None, progress=True, pvs=None):
    """指定リングの温度計履歴を取得する。

    返り値: {pv: {"ring","section","tag","suffix","sensor_id","series":[(ts,value_or_None),...]}}
    start/end は 'yyyymmddhhmmss' 文字列。
    pvs を渡すと CSV 全件ではなくその PV リストだけを取得する（特定センサの精査・再取得用）。
    """
    log_group = log_group or LOG_GROUP.get(ring.upper(), "VA/VATemp")
    kblogrd = kblogrd or KBLOGRD
    if pvs is not None:
        records = [d for d in (temp_pv.parse_pv(p) for p in pvs) if d]
    else:
        records = load_pv_list(ring, path=csv_path_)
    meta = {r["pv"]: r for r in records}
    pvs = [r["pv"] for r in records]
    ttime = make_ttime(start, end, interval_sec)

    chunks = list(_chunks(pvs, CHUNK))
    out, dropped = {}, []
    for ci, chunk in enumerate(chunks, 1):
        if progress:
            sys.stderr.write("\r[%s] 取得中 %d/%d チャンク (%d本)..."
                             % (ring, ci, len(chunks), len(pvs)))
            sys.stderr.flush()
        parsed = _fetch_chunk(chunk, ttime, log_group, kblogrd, dropped_out=dropped)
        for pv in chunk:
            m = meta[pv]
            out[pv] = {"ring": m["ring"], "section": m["section"], "tag": m["tag"],
                       "suffix": m["suffix"], "sensor_id": m["sensor_id"],
                       "series": parsed.get(pv, [])}
    if progress:
        sys.stderr.write("\r[%s] 取得完了 %d チャンク / %d 本        \n" % (ring, len(chunks), len(pvs)))
        if dropped:
            sys.stderr.write("[%s] この期間にアーカイブに無く除外した PV %d 本: %s\n"
                             % (ring, len(dropped),
                                ", ".join(dropped[:8]) + (" ..." if len(dropped) > 8 else "")))
        sys.stderr.flush()
    return out


def series_to_arrays(series):
    """[(ts, value_or_None), ...] → (timestamps:list[str], T:np.ndarray[float])。
    欠測（None）は NaN にする。temp_judge.judge_sensor / learn_sensor にそのまま渡せる。"""
    ts = [t for t, _ in series]
    T = np.array([np.nan if v is None else float(v) for _, v in series], dtype=float)
    return ts, T


# ───────────────────────── ビーム電流（温度との相関確認用）─────────────────────────

BEAM_LOG_GROUP = "BM/DCCT"
BEAM_PV = {"LER": "BMLDCCT:CURRENT", "HER": "BMHDCCT:CURRENT"}

# バンチ数 Nb（Suetsugu et al., PRAB 27, 063201 (2024) 式(5) の HOM 項 (I^2/Nb)^2 の再現に必要）。
# skbva_anomaly_detector/beam_fetch.py と同じ PV・ログ群（Misc/Base）。temp_detector は
# CCG/IP のコードに依存しない自己完結パッケージにしているため、ここで独立に実装する。
NB_LOG_GROUP = "Misc/Base"
NB_PV = {"LER": "CGLINJ:BKSEL:NOB_SET", "HER": "CGHINJ:BKSEL:NOB_SET"}


def fetch_beam(ring, start, end, interval_sec=DEFAULT_INTERVAL, kblogrd=None):
    """蓄積ビーム電流[mA]の履歴 [(ts, value_or_None), ...]（ログ群 BM/DCCT）。

    温度と同じ -t グリッド（同 start/end/interval）で引けるので、時刻文字列で突き合わせて
    温度とビームの相関や「無ビーム時の温度」を見られる。多くの温度はビーム発熱（SR/HOM）で
    上下するので、温度低下が①ビーム減（正常な冷却）か②センサ故障かの切り分けに使う。
    """
    pv = BEAM_PV.get(ring.upper())
    if pv is None:
        return []
    kblogrd = kblogrd or KBLOGRD
    ttime = make_ttime(start, end, interval_sec)
    d = _fetch_chunk([pv], ttime, BEAM_LOG_GROUP, kblogrd)
    return d.get(pv, [])


def fetch_nb(ring, start, end, interval_sec=DEFAULT_INTERVAL, kblogrd=None):
    """バンチ数 Nb の履歴 [(ts, value_or_None), ...]（ログ群 Misc/Base）。
    温度・ビームと同じ -t グリッドで引ける。HOM項 (I^2/Nb)^2 の計算に使う
    （temp_equipment.py の Suetsugu et al. 式(5)型フィット参照）。"""
    pv = NB_PV.get(ring.upper())
    if pv is None:
        return []
    kblogrd = kblogrd or KBLOGRD
    ttime = make_ttime(start, end, interval_sec)
    d = _fetch_chunk([pv], ttime, NB_LOG_GROUP, kblogrd)
    return d.get(pv, [])


# ───────────────────────── selftest（kblogrd 不要）─────────────────────────

def _selftest():
    print("=== temp_fetch selftest（kblogrd 不要）===")
    ok = True

    # 1) PV リスト読み込み＋リング判定
    try:
        ler = load_pv_list("LER")
        her = load_pv_list("HER")
        print("  load_pv_list: LER %d 本 / HER %d 本" % (len(ler), len(her)))
        ok &= (len(ler) > 1000 and len(her) > 1000)
        ok &= all(d["ring"] == "LER" for d in ler) and all(d["ring"] == "HER" for d in her)
    except FileNotFoundError as ex:
        print("  (CSV 無し: %s) — リスト検証はスキップ" % ex)
        ler = her = []

    # 1b) IR（FB_MOVE, BM/BMOthers）PV リスト読み込み
    try:
        ir = load_pv_list("IR")
        print("  load_pv_list: IR %d 本" % len(ir))
        ok &= (len(ir) == 12)
        ok &= all(d["ring"] in ("LER", "HER") for d in ir)   # H/L から自動推定
        ok &= (LOG_GROUP.get("IR") == "BM/BMOthers")
        # 配線間違いの例外（TEMP_RING_OVERRIDE.csv）が実際に反映されているか
        exc = next((d for d in ir if d["pv"] == "FB_MOVE:D02:QC1H:BWS:TEMP"), None)
        if exc is not None:
            print("  例外PV %s → ring=%s（TEMP_RING_OVERRIDE.csv 反映）" % (exc["pv"], exc["ring"]))
            ok &= (exc["ring"] == "LER")   # タグは QC1H だが上書きで LER になっているはず
    except FileNotFoundError as ex:
        print("  (IR CSV 無し: %s) — IR リスト検証はスキップ" % ex)

    # 1c) VA形式（LER/HER本体センサ）でも一時的な物理移設が上書きできること。
    #     実ファイルは汚さず、一時的な上書きファイルで動作確認する。
    if her:
        import tempfile
        moved_pv = her[0]["pv"]   # 実在する HER センサを1本借りて「LERへ一時移設」を模擬
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as tf:
            tf.write("PV,ring\n%s,LER\n" % moved_pv)
            tmp_override_path = tf.name
        try:
            her_moved = load_pv_list("HER", strict_ring=True)
            # 一時ファイルを差し替えて再読込（load_ring_overrides の path 引数を直接使う簡易確認）
            ov = load_ring_overrides(path=tmp_override_path)
            d_direct = temp_pv.parse_pv(moved_pv, ring_override=ov)
            print("  一時移設テスト(VA): %s → ring=%s (natural_ring=%s、HER_TEMP_PV.csvには残る)"
                  % (moved_pv, d_direct["ring"], d_direct["natural_ring"]))
            ok &= (d_direct["ring"] == "LER" and d_direct["natural_ring"] == "HER")
            # かつ、strict_ring のファイル所属チェックは natural_ring 基準なので、
            # 上書きしても HER ファイルの一覧から消えないこと（取得先を見失わない）
            ok &= any(d["pv"] == moved_pv for d in her_moved)
        finally:
            os.remove(tmp_override_path)

    # 2) -t 引数
    tt = make_ttime("20260617000000", "20260618000000", 300)
    print("  make_ttime:", tt)
    ok &= (tt == "20260617000000-20260618000000d300")

    # 3) kaleida パース: 正常値・負値（短絡）・欠測（非数値）を含む合成出力
    pvs = ["VAHTMP:D01M095:QLC3LE:BL", "VAHTMP:D01M094:BLC1LE:BL"]
    text = ("time %s %s\n" % (pvs[0], pvs[1]) +
            "06/17/2026 00:00:00 25.30 24.90\n"
            "06/17/2026 00:05:00 -14.20 24.95\n"      # 片方が短絡でマイナス
            "06/17/2026 00:10:00 nan 25.01\n"         # 片方が欠測（非数値）
            "06/17/2026 00:15:00 -15.00 25.00\n")
    series = parse_kaleida(text, pvs)
    ts, T = series_to_arrays(series[pvs[0]])
    print("  parse: %s 点 / 先頭値 %s / 負値保持 %s / 欠測NaN %s"
          % (len(T), T[0], np.nanmin(T), np.isnan(T[2])))
    ok &= (len(T) == 4)
    ok &= (T[0] == 25.30)                   # 正常値
    ok &= (np.nanmin(T) == -15.0)           # 負値はそのまま残る（数値フラグ化しない）
    ok &= bool(np.isnan(T[2]))              # 非数値は NaN

    # 4) judge へ素通しできること（pipeline スモークテスト。severity 期待値は judge 較正依存
    #    なのでここでは縛らない。実 D01M095 の較正は temp_judge 側で別途行う）。
    import temp_judge
    base = np.r_[25 + 0.3 * np.sin(np.arange(120) / 20.0),
                 np.full(120, -14.0) + np.random.RandomState(0).normal(0, 0.4, 120)]
    r = temp_judge.judge_sensor(base)
    print("  judge(模擬短絡列): sev=%d reason=%s t_min=%.1f n_glitch=%d  ※現閾値では参考値"
          % (r["severity"], r["reason"], r["layers"]["H0_range"]["t_min"],
             r["layers"]["G_glitch"]["n_glitch"]))
    ok &= (r["severity"] in (0, 1, 2, 3))  # 値が返り plumbing が通ること

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(0 if _selftest() else 1)

    if len(sys.argv) >= 4:
        # 実機での使い方: python temp_fetch.py LER 20260617000000 20260618000000 [interval_sec]
        ring, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
        interval = int(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_INTERVAL
        data = fetch_history(ring, start, end, interval_sec=interval)
        n_pts = sum(len(v["series"]) for v in data.values())
        n_fin = sum(1 for v in data.values() for _, val in v["series"] if val is not None)
        print("%s: %d PV / 総サンプル %d 点 / 有効(非欠測) %d 点（ログ群 %s, 間隔 %ds）"
              % (ring, len(data), n_pts, n_fin, LOG_GROUP.get(ring.upper(), "?"), interval))
        for pv, v in list(data.items())[:3]:
            print("  [%s %s %s] %s" % (v["section"], v["suffix"], pv, v["series"][:3]))
    else:
        print("usage: python temp_fetch.py <LER|HER|IR> <start yyyymmddhhmmss> <end> [interval_sec]")
        print("       python temp_fetch.py selftest        # kblogrd 不要の自己テスト")
        print("ログ群 LOG_GROUP = %s / 既定間隔 %ds。" % (LOG_GROUP, DEFAULT_INTERVAL))
