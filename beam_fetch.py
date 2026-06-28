#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beam_fetch.py — 蓄積ビーム電流の履歴を kblogrd で取得する（イオンポンプ判定用）

イオンポンプ放電電流の異常判定では、放電電流が「ビーム電流（=ガス負荷の源）」に
素直に追従するかどうかが重要な手掛かりになる（フィードスルー破損ポンプはビーム電流と
相関しにくくなる）。そのため、CCG 圧力（ccg_fetch）と同様に、ビーム電流の履歴も
ip_fetch と同型の kblogrd 取得で引けるようにする。

リアルタイムの現在値は beamcurrent.py（epics caget）が担当する。こちらは履歴専用で、
判定の窓内でビーム相関や「無ビーム時の放電」を見るために使う。

kblogrd のログ群はビーム電流 = BM/DCCT（CCG は VA/CCG、イオンポンプは VA/IPump）。
低レベルの呼び出し・解析は ip_fetch のものを再利用する（一箇所に集約）。

PV（蓄積電流, 単位 mA）:
  LER : BMLDCCT:CURRENT
  HER : BMHDCCT:CURRENT
"""

import sys

import ip_fetch   # parse_kaleida / _run_kblogrd / make_ttime を再利用

LOG_GROUP = "BM/DCCT"
NODATA = ip_fetch.NODATA

PVS = {"LER": "BMLDCCT:CURRENT", "HER": "BMHDCCT:CURRENT"}

# バンチ数 Nb（HOM_2=(I*I/Nb)^2 の再現に必要）。legacy の .sh と同じく ログ群 Misc/Base。
NB_LOG_GROUP = "Misc/Base"
NB_PVS = {"LER": "CGLINJ:BKSEL:NOB_SET", "HER": "CGHINJ:BKSEL:NOB_SET"}


def pv_for(ring):
    return PVS.get(ring)


def nb_pv_for(ring):
    return NB_PVS.get(ring)


def fetch_nb(ring, start, end, interval_sec=ip_fetch.DEFAULT_INTERVAL, kblogrd=None):
    """バンチ数 Nb の履歴を取得 [(ts, value_or_None), ...]（ログ群 Misc/Base）。
    HOM_2=(I*I/Nb)^2 の再現に使う。CCG/ビームと同じ時刻グリッド。"""
    pv = nb_pv_for(ring)
    if pv is None:
        return []
    kblogrd = kblogrd or ip_fetch.KBLOGRD
    ttime = ip_fetch.make_ttime(start, end, interval_sec)
    d = ip_fetch._fetch_chunk([pv], ttime, NB_LOG_GROUP, kblogrd)
    return d.get(pv, [])


def fetch_series(ring, start, end, interval_sec=ip_fetch.DEFAULT_INTERVAL,
                 log_group=None, kblogrd=None):
    """指定リングの蓄積ビーム電流の履歴を取得する。

    返り値: [(timestamp_str, value_or_None), ...]（CCG/IP と同じ時刻グリッド）。
    値は mA。データ無し（NODATA 以下）は値そのまま残す（判定層で扱う）。
    """
    pv = pv_for(ring)
    if pv is None:
        return []
    log_group = log_group or LOG_GROUP
    kblogrd = kblogrd or ip_fetch.KBLOGRD
    ttime = ip_fetch.make_ttime(start, end, interval_sec)
    text = ip_fetch._run_kblogrd([pv], ttime, log_group, kblogrd)
    parsed = ip_fetch.parse_kaleida(text, [pv])
    return parsed.get(pv, [])


if __name__ == "__main__":
    # 例: python beam_fetch.py LER 20260610000000 20260610010000
    if len(sys.argv) >= 4:
        ring, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
        interval = int(sys.argv[4]) if len(sys.argv) >= 5 else ip_fetch.DEFAULT_INTERVAL
        s = fetch_series(ring, start, end, interval_sec=interval)
        nv = sum(1 for _, v in s if v is not None and v > NODATA)
        print("%s ビーム電流 %s: %d 点 / 有効 %d 点（ログ群 %s）"
              % (ring, pv_for(ring), len(s), nv, LOG_GROUP))
        print("先頭3点:", s[:3])
    else:
        print("usage: python beam_fetch.py <LER|HER> <start yyyymmddhhmmss> <end> [interval_sec]")
        print("ログ群 LOG_GROUP = '%s'（蓄積ビーム電流, mA）。" % LOG_GROUP)
