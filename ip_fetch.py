#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_fetch.py — イオンポンプ放電電流の履歴を kblogrd で取得する（拡張：開発中）

CCG と同じく kblogrd でログを引く。末次プログラムの .sh（例 LERD01CCG.sh）が
  /usr/local/bin/kblogrd -r PV1,PV2,... -t yyyymmddhhmmss-yyyymmddhhmmssd<秒> -f kaleida <ログ群> > out
という形で呼んでいるのを踏襲し、これを Python から（レガシー無改変のまま）実行する。

PV が多いと1回の kblogrd 呼び出しに載らないので、CCG の .sh と同様に分割して呼ぶ。

kblogrd の最後の引数「ログ群」は、データの所属サブシステム名。
  CCG は VA/CCG、ビーム電流(BMLDCCT:CURRENT)は BM/DCCT、
  イオンポンプ電流は VA/IPump（実機確認済み）。
"""

import os
import subprocess
import sys

import ip_pv

KBLOGRD = "/usr/local/bin/kblogrd"
LOG_GROUP = "VA/IPump"     # イオンポンプ電流のログ群（実機確認済み。CCG は VA/CCG）
CHUNK = 13                 # 1回の kblogrd に渡す PV 数（CCG .sh の分割に倣う）
DEFAULT_INTERVAL = 60      # サンプリング間隔[秒]
NODATA = 1e-10             # kblogrd が「データ無し」に使う下限フラグ値
TIMEOUT = 120              # kblogrd 1回あたりのタイムアウト[秒]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_ttime(start, end, interval_sec=DEFAULT_INTERVAL):
    """kblogrd の -t 引数 'yyyymmddhhmmss-yyyymmddhhmmssd<秒>' を作る。
    start/end は 'yyyymmddhhmmss' 文字列。"""
    return "%s-%sd%d" % (start, end, interval_sec)


def parse_kaleida(text, pvs):
    """kblogrd -f kaleida の出力を {pv: [(timestamp_str, value_or_None), ...]} に変換する。

    実際の出力形式（スペース区切り）:
        time VALIP:D01_IP_L01:CUR            ← 1行目はヘッダ
        06/10/2026 00:00:00 9.402521e-08     ← 日付 時刻 値[ 値...]
        ...
    ・先頭2語（MM/DD/YYYY HH:MM:SS）が時刻。以降が各 PV の値（-r の順）。
    ・ヘッダ行（先頭が日付でない＝'/'を含まない）はスキップ。
    ・値が数値でなければ None（欠測）。
    ・値 NODATA(=1e-10) は「データ無し」を表す下限フラグ。ここでは値はそのまま残し、
      表示・判定の層で必要に応じて除外する（解析は無損失にしておく）。
    """
    series = {pv: [] for pv in pvs}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        tok = s.split()
        if len(tok) < 3:
            continue
        if "/" not in tok[0]:          # ヘッダ行 'time ...' 等をスキップ
            continue
        ts = tok[0] + " " + tok[1]     # 'MM/DD/YYYY HH:MM:SS'
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


def _run_kblogrd(pvs, ttime, log_group, kblogrd):
    """1チャンク分の kblogrd を実行し、標準出力（kaleida テキスト）を返す。
    stdin は閉じる（対話待ちでハングしないように）。TIMEOUT 秒で打ち切る。"""
    cmd = [kblogrd, "-r", ",".join(pvs), "-t", ttime, "-f", "kaleida", log_group]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True, timeout=TIMEOUT)
    if res.returncode != 0:
        raise RuntimeError("kblogrd 失敗 (rc=%d): %s" % (res.returncode, res.stderr.strip()))
    return res.stdout


def fetch_history(ring, start, end, interval_sec=DEFAULT_INTERVAL,
                  log_group=None, kblogrd=None, csv_path=None, progress=True):
    """指定リングのイオンポンプ放電電流の履歴を取得する。

    返り値: {pv: {"section":..., "supply":..., "series":[(ts, value), ...]}, ...}
    start/end は 'yyyymmddhhmmss' 文字列。log_group を省略すると LOG_GROUP を使う。
    progress=True なら、チャンクごとの進捗を標準エラーに出す（止まって見えないように）。
    """
    log_group = log_group or LOG_GROUP
    kblogrd = kblogrd or KBLOGRD
    path = csv_path or ip_pv.csv_path(ring)
    records = ip_pv.load(path)
    meta = {r["pv"]: r for r in records}
    pvs = [r["pv"] for r in records]
    ttime = make_ttime(start, end, interval_sec)

    chunks = list(_chunks(pvs, CHUNK))
    out = {}
    for ci, chunk in enumerate(chunks, 1):
        if progress:
            sys.stderr.write("\r[%s] 取得中 %d/%d チャンク..." % (ring, ci, len(chunks)))
            sys.stderr.flush()
        text = _run_kblogrd(chunk, ttime, log_group, kblogrd)
        parsed = parse_kaleida(text, chunk)
        for pv in chunk:
            out[pv] = {
                "section": meta[pv]["section"],
                "supply": meta[pv]["supply"],
                "series": parsed.get(pv, []),
            }
    if progress:
        sys.stderr.write("\r[%s] 取得完了 %d チャンク        \n" % (ring, len(chunks)))
        sys.stderr.flush()
    return out


if __name__ == "__main__":
    # 実機での使い方の例（kblogrd が必要）:
    #   python ip_fetch.py LER 20260610000000 20260611000000
    import json
    if len(sys.argv) >= 4:
        ring, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
        interval = int(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_INTERVAL
        data = fetch_history(ring, start, end, interval_sec=interval)
        n_pts = sum(len(v["series"]) for v in data.values())
        n_valid = sum(1 for v in data.values() for _, val in v["series"]
                      if val is not None and val > NODATA)
        print("%s: %d PV / 総サンプル %d 点 / 有効値(>%g) %d 点（ログ群 %s）"
              % (ring, len(data), n_pts, NODATA, n_valid, LOG_GROUP))
        # KEK と Agilent_4U を1本ずつ、先頭3点を表示して値を確認
        shown = {"KEK": False, "Agilent_4U": False}
        for pv, v in data.items():
            sup = v["supply"]
            if not shown.get(sup):
                shown[sup] = True
                print("  [%s/%s %s] %s" % (sup, v["section"], pv, v["series"][:3]))
            if all(shown.values()):
                break
    else:
        print("usage: python ip_fetch.py <LER|HER> <start yyyymmddhhmmss> <end> [interval_sec]")
        print("ログ群 LOG_GROUP = '%s'（イオンポンプ電流）。" % LOG_GROUP)
