#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_npy.py — レガシーが実行時に生成する Record_Data 等の .npy を調べる診断ツール

`could not convert string to float: '攀'` / `chr() arg not in range` の原因
（実行時生成 .npy の文字化け・破損）を特定するためのもの。検知本体には影響しない。

使い方（実機の legacy/ で実行するのが簡単）:
    cd ~/skbva_anomaly_detector/legacy
    python ../check_npy.py                 # カレントの *Record_Data*.npy を全部調べる
    python ../check_npy.py <ファイル名>     # 特定の .npy を調べる
    python ../check_npy.py --all           # カレントの全 .npy を調べる
"""

import glob
import os
import sys

import numpy as np


def check_file(path):
    print("=" * 70)
    print("ファイル: %s" % path)
    try:
        a = np.load(path, allow_pickle=True)
    except Exception as ex:
        print("  np.load 失敗: %r" % ex)
        return
    print("  shape=%s  dtype=%s" % (a.shape, a.dtype))
    if a.ndim != 2:
        print("  （2次元でないため、セル単位チェックはスキップ）")
        # それでも一部を表示
        flat = a.flatten()
        print("  先頭5要素:", [repr(x) for x in flat[:5]])
        return

    print("  1行目(ヘッダ?):", [repr(x) for x in a[0][:3]])
    if a.shape[0] > 1:
        print("  2行目:", [repr(x) for x in a[1][:3]])

    bad = []
    weird_codepoints = []
    for i in range(1, a.shape[0]):          # 1行目(PV名ヘッダ)は除く
        for j in range(a.shape[1]):
            s = a[i, j]
            try:
                float(s)
            except (ValueError, TypeError):
                bad.append((i, j, repr(s)))
                # 非ASCII文字（化けの疑い）を記録
                try:
                    for ch in str(s):
                        if ord(ch) > 0x7F:
                            weird_codepoints.append((i, j, ch, hex(ord(ch))))
                            break
                except Exception:
                    pass

    print("  数値変換できないセル数（ヘッダ除く）: %d / 全%d セル" %
          (len(bad), (a.shape[0] - 1) * a.shape[1]))
    if bad:
        print("  例（最大10件）:")
        for i, j, r in bad[:10]:
            print("    [%d,%d] = %s" % (i, j, r))
    if weird_codepoints:
        print("  非ASCII（化け疑い）の例（最大10件）:")
        for i, j, ch, hx in weird_codepoints[:10]:
            print("    [%d,%d] '%s' (U+%s)" % (i, j, ch, hx[2:].upper()))
    if not bad:
        print("  → このファイルは正常（全データセルが数値）")


def main():
    args = [a for a in sys.argv[1:]]
    if args and args[0] not in ("--all",):
        # 明示ファイル
        for p in args:
            check_file(p)
        return
    pattern = "*.npy" if "--all" in args else "*Record_Data*.npy"
    files = sorted(glob.glob(pattern))
    if not files:
        print("カレントディレクトリに %s が見つかりません。" % pattern)
        print("実機の legacy/ で実行してください:  cd ~/skbva_anomaly_detector/legacy ; python ../check_npy.py")
        return
    print("対象 %d ファイル（パターン %s）" % (len(files), pattern))
    nbad = 0
    for p in files:
        check_file(p)
    print("=" * 70)
    print("完了。『数値変換できないセル数』が 0 より大きいファイルが破損しています。")


if __name__ == "__main__":
    main()
