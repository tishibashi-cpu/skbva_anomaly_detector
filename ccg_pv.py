#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ccg_pv.py — 監視対象 CCG PV リストを CSV から読む

CSV 形式（1 行目はヘッダ。1 列目だけを使う）:
    CCG PV
    VALCCG:D01_L01A:PRES
    ...

PV 名の側室部（D01〜D12）で振り分け、
112p の Make_{LER,HER}_Record と同じ「側室別リスト・各先頭に 'none'」形式にする。
これを検知プログラムに渡せば、監視 CCG リストと CCG 数（CCG_n）が CSV 準拠になる。
（旧版は CCG PV, S [m], Order, Remarks の4列だったが、使うのは1列目のみなので
 1列だけの CSV でもそのまま動く。）
"""

import csv


def load_side_rooms(csv_path):
    """CSV を D01〜D12 の側室別 PV リストにして返す。

    戻り値: [room0, room1, ..., room11]  （room0 = D01）
            各 room は ['none', 'PV1', 'PV2', ...] （先頭 'none' は 112p 仕様）
    """
    rooms = [["none"] for _ in range(12)]
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # ヘッダ行を読み飛ばす
        for row in reader:
            if not row or not row[0].strip():
                continue
            pv = row[0].strip()
            # 'VALCCG:D06_L04:PRES' -> 'D06' -> index 5
            try:
                d = pv.split(":")[1].split("_")[0]   # 'D06'
                idx = int(d[1:]) - 1
            except (IndexError, ValueError):
                continue
            if 0 <= idx < 12:
                rooms[idx].append(pv)
    return rooms


def ccg_n(side_rooms):
    """側室別リストから CCG_n（各側室の長さ・'none' 込み）を作る。

    112p の LER_CCG_n[k] = len(self.LER_Record_Name_List_box[k]) と同義。
    """
    return [len(r) for r in side_rooms]


if __name__ == "__main__":
    # 動作確認: 各 CSV を読んで側室別の本数を表示する
    import sys
    for path in sys.argv[1:]:
        rooms = load_side_rooms(path)
        n = ccg_n(rooms)
        total = sum(len(r) - 1 for r in rooms)  # 'none' を除いた実 CCG 数
        print("%s: 実CCG %d 本" % (path, total))
        for i in range(12):
            print("  D%02d: %2d 本  CCG_n=%d" % (i + 1, len(rooms[i]) - 1, n[i]))
