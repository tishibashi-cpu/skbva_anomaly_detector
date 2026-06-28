#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ccg_fetch.py — CCG 圧力の履歴を kblogrd で取得する（観察・拡張用）

イオンポンプ放電電流の判定（同一地点の CCG 圧力との整合性）を検討するには、
CCG 圧力の履歴も必要になる。レガシーの検知も内部で CCG 圧力を取っているが、
ここではレガシーに触らず、ip_fetch と同じ kblogrd の仕組みを使って独立に取得する
（CCG とイオンポンプを同型の取得部で並べられる）。

kblogrd のログ群は CCG = VA/CCG（イオンポンプは VA/IPump）。
低レベルの呼び出し・解析は ip_fetch のものを再利用する（一箇所に集約）。
"""

import os
import re
import sys

import ccg_pv
import ip_fetch   # parse_kaleida / _run_kblogrd / make_ttime / _chunks を再利用

LOG_GROUP = "VA/CCG"
CHUNK = ip_fetch.CHUNK
DEFAULT_INTERVAL = ip_fetch.DEFAULT_INTERVAL
NODATA = ip_fetch.NODATA

_SEC_RE = re.compile(r"^VA[LH]CCG:(D\d+)_")


def section_of(pv):
    """VALCCG:D01_L08:PRES → 'D01'。判別不可なら '?'。"""
    m = _SEC_RE.match(pv)
    return m.group(1) if m else "?"


def fetch_history(ring, start, end, interval_sec=DEFAULT_INTERVAL,
                  log_group=None, kblogrd=None, csv_path=None, progress=True):
    """指定リングの CCG 圧力の履歴を取得する。

    返り値: {pv: {"section":..., "series":[(ts, value), ...]}, ...}
    start/end は 'yyyymmddhhmmss' 文字列。値は生のまま（圧力単位）。
    """
    log_group = log_group or LOG_GROUP
    kblogrd = kblogrd or ip_fetch.KBLOGRD
    path = csv_path or ccg_pv.csv_path(ring)
    pvs = ccg_pv.load_flat(path)
    ttime = ip_fetch.make_ttime(start, end, interval_sec)

    chunks = list(ip_fetch._chunks(pvs, CHUNK))
    out = {}
    dropped = []
    for ci, chunk in enumerate(chunks, 1):
        if progress:
            sys.stderr.write("\r[%s CCG] 取得中 %d/%d チャンク..." % (ring, ci, len(chunks)))
            sys.stderr.flush()
        parsed = ip_fetch._fetch_chunk(chunk, ttime, log_group, kblogrd, dropped_out=dropped)
        for pv in chunk:
            out[pv] = {"section": section_of(pv), "series": parsed.get(pv, [])}
    if progress:
        sys.stderr.write("\r[%s CCG] 取得完了 %d チャンク        \n" % (ring, len(chunks)))
        if dropped:
            sys.stderr.write("[%s CCG] この期間にアーカイブに無く除外した PV %d 本: %s\n"
                             % (ring, len(dropped),
                                ", ".join(dropped[:8]) + (" ..." if len(dropped) > 8 else "")))
        sys.stderr.flush()
    return out


if __name__ == "__main__":
    # 例: python ccg_fetch.py LER 20260610000000 20260610010000
    if len(sys.argv) >= 4:
        ring, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
        interval = int(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_INTERVAL
        data = fetch_history(ring, start, end, interval_sec=interval)
        n_pts = sum(len(v["series"]) for v in data.values())
        print("%s: %d PV / 総サンプル %d 点（ログ群 %s）" % (ring, len(data), n_pts, LOG_GROUP))
        first = next(iter(data))
        print("例 %s [%s]: %s" % (first, data[first]["section"], data[first]["series"][:3]))
    else:
        print("usage: python ccg_fetch.py <LER|HER> <start yyyymmddhhmmss> <end> [interval_sec]")
        print("ログ群 LOG_GROUP = '%s'（CCG 圧力）。" % LOG_GROUP)
