#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apply_labels.py — ダッシュボードで溜めた教師ラベルを legacy 形式の学習データに反映する。

二段構えの「後処理」側。ダッシュボードの Save Normal/Abnormal ボタンは
`label_queue.jsonl` に1行追記するだけ（材料計算はしない）。本スクリプトを【実機で人が】
実行すると、キューの各ラベルについて、その abort timing 用の解析キャッシュ（.npz）を
legacy のパイプラインで作り直し、legacy の Save_Manual_* を呼んで
`{Ring}_Manual_{Normal|Abnormal}_Class2_Result_{Strg|Tail}_{WB|NB}.npy/.txt` に
byte 互換の教師行を追記する。再学習は別途人手で。

設計上の注意：
- legacy（Anomaly_Detection_112p.py）は一行も書き換えず import して関数だけ呼ぶ
  （detector_headless.py の足場 det/CCG_LIST/CCG_N/CONFIG を再利用）。
- 教師行を作るには kblogrd で当時のデータを引き直す必要があるため、本スクリプトは実機
  （kekb-co-user01 等）で実行すること。手元（kblogrd 無し）では取得段で失敗する。
- 各エントリの処理後、キューの status を applied / error:… に更新して書き戻す
  （applied は監査用に残す。再実行しても applied はスキップ）。

使い方:
  python apply_labels.py            # queued を順に処理して反映
  python apply_labels.py --dry-run  # 何を処理するか一覧表示のみ（legacy は呼ばない）
  python apply_labels.py --queue path/to/label_queue.jsonl
"""

import os
import sys
import json
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUEUE = os.path.join(HERE, "label_queue.jsonl")


def _read_queue(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                rows.append({"_raw": ln, "status": "error: parse failed"})
    return rows


def _write_queue(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            if "_raw" in r and len(r) <= 2:
                f.write(r["_raw"] + "\n")
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _norm_period(p):
    """period 表記を Storage / Tail に正規化。"""
    if not p:
        return "Storage"
    p = str(p).lower()
    if p.startswith("tail"):
        return "Tail"
    return "Storage"


def _apply_one(dh, entry):
    """1エントリを legacy パイプラインで反映。成功で 'applied'、失敗で 'error: …' を返す。"""
    det = dh.det
    cfg = dh.CONFIG
    ring = entry.get("ring")
    record = entry.get("record")
    klass = entry.get("klass")           # 'Normal' / 'Abnormal' = Save_Class
    abort_timing = entry.get("abort_time")
    period = _norm_period(entry.get("period"))

    if ring not in ("LER", "HER"):
        return "error: bad ring"
    if klass not in ("Normal", "Abnormal"):
        return "error: bad klass"
    if not record or not abort_timing:
        return "error: record/abort_time 欠落"

    List_Para = dh.CCG_LIST[ring]
    CCG_n = dh.CCG_N[ring]
    Method = cfg["Method"]

    # (a) 日付範囲とビーム有無
    No_Beam, err = det.Define_Date_Range(
        abort_timing, List_Para, cfg["Run_Mode"], cfg["Auto_Mode"],
        cfg["Cns_Mode"], cfg["Arun"], cfg["Hadv"], cfg["Last_Refd"], cfg["Ref_Pd"])
    No_STD_Abort = 1 if err == "No_STD_Abort" else 0
    if err not in ("none", "No_STD_Abort"):
        return "error: Define_Date_Range=%s" % err

    if period == "Tail":
        if No_Beam != 0:
            return "error: Tail はビームなし時は非対応"
        if No_STD_Abort == 1:
            return "error: 基準にアボートが無く Tail 基準を作れない"
        # STD/CHK/DIF（Tail）を用意してから Save_Manual_Tail
        _, e = det.Get_Fit_STD_Strg(List_Para, Method, cfg["Ref_Pd"], CCG_n)
        if e != "none":
            return "error: STD_Strg=%s" % e
        det.Get_Fit_STD_Tail(List_Para, Method)
        _, e = det.Get_Fit_CHK_Strg(Method, cfg["Check_Record_Name"], "CHK_Strg",
                                    List_Para, abort_timing, CCG_n)
        if e != "none":
            return "error: CHK_Strg=%s" % e
        _, e = det.Get_Fit_CHK_Tail(Method, cfg["Check_Record_Name"], "CHK_Tail",
                                    List_Para, abort_timing)
        if e != "none":
            return "error: CHK_Tail=%s" % e
        _, e = det.Get_DIF_Strg(List_Para, CCG_n)
        if e != "none":
            return "error: DIF_Strg=%s" % e
        _, e = det.Get_DIF_Tail(List_Para)
        if e != "none":
            return "error: DIF_Tail=%s" % e
        det.Save_Manual_Tail(None, Method, "DIF_Tail", List_Para, record, klass, abort_timing)
        return "applied"

    # period == Storage
    if No_Beam == 0:
        _, e = det.Get_Fit_STD_Strg(List_Para, Method, cfg["Ref_Pd"], CCG_n)
        if e != "none":
            return "error: STD_Strg=%s" % e
        _, e = det.Get_Fit_CHK_Strg(Method, cfg["Check_Record_Name"], "CHK_Strg",
                                    List_Para, abort_timing, CCG_n)
        if e != "none":
            return "error: CHK_Strg=%s" % e
        _, e = det.Get_DIF_Strg(List_Para, CCG_n)
        if e != "none":
            return "error: DIF_Strg=%s" % e
        det.Save_Manual_Strg(None, Method, "DIF_Strg", List_Para, record, klass, abort_timing)
        return "applied"
    else:
        _, e = det.Get_Fit_STD_Strg_NB(List_Para, Method, cfg["Ref_Pd"], CCG_n)
        if e != "none":
            return "error: STD_Strg_NB=%s" % e
        _, e = det.Get_Fit_CHK_Strg_NB(Method, cfg["Check_Record_Name"], "CHK_Strg",
                                       List_Para, abort_timing, CCG_n)
        if e != "none":
            return "error: CHK_Strg_NB=%s" % e
        _, e = det.Get_DIF_Strg_NB(List_Para, CCG_n)
        if e != "none":
            return "error: DIF_Strg_NB=%s" % e
        det.Save_Manual_Strg_NB(None, Method, "DIF_Strg", List_Para, record, klass, abort_timing)
        return "applied"


def main(argv):
    queue = DEFAULT_QUEUE
    if "--queue" in argv:
        queue = argv[argv.index("--queue") + 1]
    dry = "--dry-run" in argv

    rows = _read_queue(queue)
    todo = [r for r in rows if r.get("status") == "queued"]
    print("キュー: %s（全 %d 行、未処理 %d 行）" % (queue, len(rows), len(todo)))
    if not todo:
        print("処理対象なし。")
        return 0

    if dry:
        for r in todo:
            print("  [dry] %s %s %s %s @%s"
                  % (r.get("klass"), r.get("ring"), r.get("record"),
                     _norm_period(r.get("period")), r.get("abort_time")))
        print("--dry-run のため legacy は呼んでいません。")
        return 0

    # legacy 足場（detector_headless）を import。実機以外では失敗しうる。
    try:
        import detector_headless as dh
    except Exception as ex:
        print("detector_headless を import できませんでした（実機で実行してください）: %s" % ex)
        return 2

    applied = failed = 0
    for r in rows:
        if r.get("status") != "queued":
            continue
        try:
            st = _apply_one(dh, r)
        except Exception as ex:
            st = "error: %s" % ex
        r["status"] = st
        r["applied_ts"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if st == "applied":
            applied += 1
            print("  ✓ %s %s %s @%s" % (r.get("klass"), r.get("ring"),
                                        r.get("record"), r.get("abort_time")))
        else:
            failed += 1
            print("  ✗ %s %s %s @%s → %s" % (r.get("klass"), r.get("ring"),
                                             r.get("record"), r.get("abort_time"), st))
        # 1件ごとに書き戻し（途中で落ちても進捗が残る）
        _write_queue(queue, rows)

    print("完了: 反映 %d 件 / 失敗 %d 件。学習データに追記しました（再学習は別途）。"
          % (applied, failed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
