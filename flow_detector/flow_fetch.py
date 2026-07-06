#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flow_fetch.py — 冷却水流量計の履歴を kblogrd で取得する。

CCG/温度計と同じく kblogrd で履歴を引く。温度計(temp_fetch.py)と同様に自己完結パッケージ
（flow_detector/ 単独で動く。他パッケージに依存しない）。

流量計ならではの設計差:
  1) **ビーム電流と無関係の機器**なので、ビーム/バンチ数の取得は無い（fetch_beam/fetch_nb 相当は
     このパッケージには存在しない。flow_judge.py も指示値だけで判定する）。
  2) **リング(LER/HER)の概念を持たない**。PVリストは pv_info/FLOW_PV.csv の1本のみ
     （<RING>_TEMP_PV.csv のようなリング別ファイルではない）。
  3) 判定は直近1窓だけを見る設計（flow_judge.py 参照）なので、取得も直近窓1回で足りる
     （CCG式ローリング基準のような別窓の再取得は不要）。
  4) 単位は %（ある時点の基準流量に対する比）。負値になることは想定しないが、パースは
     temp_fetch と同じ方針で数値はそのまま残す（今後の解析の幅を狭めないため）。

ログ群（kblogrd 最後の引数）は VA/VAFlow（実機確認済み）。環境変数 FLOW_KBLOGRD_LOG_GROUP で
コード変更なしに上書きもできる（一時的な確認用途など）。
"""

import csv
import os
import re
import subprocess
import sys

import numpy as np

import flow_pv

HERE = os.path.dirname(os.path.abspath(__file__))
PV_INFO_DIR = os.path.join(os.path.dirname(HERE), "pv_info")
FLOW_PV_CSV = os.path.join(PV_INFO_DIR, "FLOW_PV.csv")

KBLOGRD = "/usr/local/bin/kblogrd"
# 流量計のログ群名（実機確認済み）。環境変数 FLOW_KBLOGRD_LOG_GROUP でコード変更なしに上書きも可能。
LOG_GROUP = os.environ.get("FLOW_KBLOGRD_LOG_GROUP", "VA/VAFlow")
CHUNK = int(os.environ.get("FLOW_KBLOGRD_CHUNK", 26))   # temp_detector と同じ既定値から開始
DEFAULT_INTERVAL = 30       # サンプリング間隔[秒]。実アーカイブは5s刻みだが、678本×24hの
                            # データ量を抑えるため既定は30sに間引く（flow_judge のCV閾値は
                            # 実測5s生データで較正したが、白色雑音的なノイズは間引いても
                            # 分散の程度は大きく変わらない想定。実機で気になれば5に戻せる）
TIMEOUT = 120

_NOMATCH_MARK = "specified record name doesn't match"
_PV_TOKEN_RE = re.compile(r"[A-Za-z][\w]*:[\w]+(?::[\w]+)*")


# ───────────────────────── PV リスト ─────────────────────────

def load_pv_list(path=None):
    """pv_info/FLOW_PV.csv から流量計 PV を読み、flow_pv で解析した dict のリストを返す。
    先頭行（"FLOW PV"）はヘッダとして捨てる。BOM/CRLF を吸収する。リング分けは無い（全セクション1本）。"""
    path = path or FLOW_PV_CSV
    pvs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            cell = row[0].strip()
            if i == 0 and not cell.upper().startswith("VA_FLS"):
                continue                       # ヘッダ行 "FLOW PV" を捨てる
            if cell:
                pvs.append(cell)
    return flow_pv.parse_all(pvs)


# ───────────────────────── kblogrd ─────────────────────────

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_ttime(start, end, interval_sec=DEFAULT_INTERVAL):
    """kblogrd の -t 引数 'yyyymmddhhmmss-yyyymmddhhmmssd<秒>' を作る。"""
    return "%s-%sd%d" % (start, end, interval_sec)


def parse_kaleida(text, pvs):
    """kblogrd -f kaleida の出力を {pv: [(ts, value_or_None), ...]} に変換する（temp_fetch と同方式）。"""
    series = {pv: [] for pv in pvs}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        tok = s.split()
        if len(tok) < 3 or "/" not in tok[0]:
            continue
        ts = tok[0] + " " + tok[1]
        vals = tok[2:]
        for i, pv in enumerate(pvs):
            if i >= len(vals):
                break
            try:
                v = float(vals[i])
            except ValueError:
                v = None
            series[pv].append((ts, v))
    return series


def _kblogrd_once(pvs, ttime, log_group, kblogrd):
    cmd = [kblogrd, "-r", ",".join(pvs), "-t", ttime, "-f", "kaleida", log_group]
    try:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=TIMEOUT)
    except FileNotFoundError:
        raise RuntimeError(
            "kblogrd が見つかりません: %s\n"
            "流量計の取得は kblogrd のある実機（kekb-co-user01/02/03）で実行してください。"
            "手元では `python flow_fetch.py selftest`（kblogrd 不要）で確認できます。" % kblogrd)
    return res.returncode, res.stdout, res.stderr


def _fetch_chunk(pvs, ttime, log_group, kblogrd, dropped_out=None, _depth=0):
    """1チャンクを頑健に取得する（temp_fetch/ip_fetch と同じ二分探索フォールバック方式）。"""
    pvs = list(pvs)
    if not pvs:
        return {}
    rc, out, err = _kblogrd_once(pvs, ttime, log_group, kblogrd)
    if rc == 0:
        return parse_kaleida(out, pvs)
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


def fetch_history(start, end, interval_sec=DEFAULT_INTERVAL, log_group=None, kblogrd=None,
                  pvs=None, progress=True):
    """流量計の履歴を取得する（リング指定は無い。全セクション共通の1リストから引く）。

    返り値: {pv: {"section","idx","tag","sensor_id","series":[(ts,value_or_None),...]}}
    start/end は 'yyyymmddhhmmss' 文字列。
    pvs を渡すと CSV 全件ではなくその PV リストだけを取得する（特定センサの精査・再取得用）。
    """
    log_group = log_group or LOG_GROUP
    kblogrd = kblogrd or KBLOGRD
    if pvs is not None:
        records = [d for d in (flow_pv.parse_pv(p) for p in pvs) if d]
    else:
        records = load_pv_list()
    meta = {r["pv"]: r for r in records}
    pv_names = [r["pv"] for r in records]
    ttime = make_ttime(start, end, interval_sec)

    chunks = list(_chunks(pv_names, CHUNK))
    out, dropped = {}, []
    for ci, chunk in enumerate(chunks, 1):
        if progress:
            sys.stderr.write("\r[FLOW] 取得中 %d/%d チャンク (%d本)..." % (ci, len(chunks), len(pv_names)))
            sys.stderr.flush()
        parsed = _fetch_chunk(chunk, ttime, log_group, kblogrd, dropped_out=dropped)
        for pv in chunk:
            m = meta[pv]
            out[pv] = {"section": m["section"], "idx": m["idx"], "tag": m["tag"],
                       "sensor_id": m["sensor_id"], "series": parsed.get(pv, [])}
    if progress:
        sys.stderr.write("\r[FLOW] 取得完了 %d チャンク / %d 本        \n" % (len(chunks), len(pv_names)))
        if dropped:
            sys.stderr.write("[FLOW] この期間にアーカイブに無く除外した PV %d 本: %s\n"
                             % (len(dropped), ", ".join(dropped[:8]) + (" ..." if len(dropped) > 8 else "")))
        sys.stderr.flush()
    return out


def series_to_arrays(series):
    """[(ts, value_or_None), ...] → (timestamps:list[str], V:np.ndarray[float])。欠測はNaN。"""
    ts = [t for t, _ in series]
    V = np.array([np.nan if v is None else float(v) for _, v in series], dtype=float)
    return ts, V


# ───────────────────────── selftest（kblogrd 不要）─────────────────────────

def _selftest():
    print("=== flow_fetch selftest（kblogrd 不要）===")
    ok = True

    # 1) PV リスト読み込み
    try:
        pvs = load_pv_list()
        print("  load_pv_list: %d 本" % len(pvs))
        ok &= (len(pvs) > 600)
        ok &= all("section" in d and "sensor_id" in d for d in pvs)
    except FileNotFoundError as ex:
        print("  (CSV 無し: %s) — リスト検証はスキップ" % ex)
        pvs = []

    # 2) -t 引数
    tt = make_ttime("20260617000000", "20260618000000", 30)
    print("  make_ttime:", tt)
    ok &= (tt == "20260617000000-20260618000000d30")

    # 3) kaleida パース: 正常値・欠測（非数値）を含む合成出力
    test_pvs = ["VA_FLS:D01_11_XXX:RATE", "VA_FLS:D02_26_XXX:RATE"]
    text = ("time %s %s\n" % (test_pvs[0], test_pvs[1]) +
            "06/17/2026 00:00:00 98.75 140.87\n"
            "06/17/2026 00:00:30 nan 141.02\n"          # 片方が欠測（非数値）
            "06/17/2026 00:01:00 99.12 140.65\n")
    series = parse_kaleida(text, test_pvs)
    ts, V = series_to_arrays(series[test_pvs[0]])
    print("  parse: %d 点 / 先頭値 %s / 欠測NaN %s" % (len(V), V[0], np.isnan(V[1])))
    ok &= (len(V) == 3 and V[0] == 98.75 and bool(np.isnan(V[1])))

    # 4) judge へ素通しできること（pipeline スモークテスト）
    import flow_judge
    good = np.r_[100 + np.random.RandomState(0).normal(0, 1.5, 200)]
    r = flow_judge.judge_series(good)
    print("  judge(正常模擬列): sev=%s reason=%s" % (r["severity"], r["reason"]))
    ok &= (r["severity"] == 0)

    print("=== selftest:", "PASS" if ok else "FAIL", "===")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(0 if _selftest() else 1)

    if len(sys.argv) >= 3:
        # 実機での使い方: python flow_fetch.py <start yyyymmddhhmmss> <end> [interval_sec]
        start, end = sys.argv[1], sys.argv[2]
        interval = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_INTERVAL
        data = fetch_history(start, end, interval_sec=interval)
        n_pts = sum(len(v["series"]) for v in data.values())
        n_fin = sum(1 for v in data.values() for _, val in v["series"] if val is not None)
        print("流量計: %d PV / 総サンプル %d 点 / 有効(非欠測) %d 点（ログ群 %s, 間隔 %ds）"
              % (len(data), n_pts, n_fin, LOG_GROUP, interval))
        for pv, v in list(data.items())[:3]:
            print("  [%s %s] %s" % (v["section"], pv, v["series"][:3]))
    else:
        print("usage: python flow_fetch.py <start yyyymmddhhmmss> <end> [interval_sec]")
        print("       python flow_fetch.py selftest        # kblogrd 不要の自己テスト")
        print("ログ群 LOG_GROUP = %s / 既定間隔 %ds。" % (LOG_GROUP, DEFAULT_INTERVAL))
