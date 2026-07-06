#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""temp_equipment_plot.py — 温度 vs ビーム電流の散布図を基準期間/現在期間で重ね描きする。

temp_equipment.py の learn/judge は数値（傾き dT/dI・切片 a・比・差）で判定するが、
数値だけでは「本当に機器が劣化したのか」「季節等の環境要因で全体がシフトしただけか」の
最終判断が難しい（実際、IRセクションD01で切片差Δaの符号がPVごとにバラバラという実例が出た）。
本スクリプトは各センサについて

    横軸：ビーム電流 [mA]、縦軸：温度 [℃]

の散布図を、基準期間（学習期間）＝青、現在期間（判定期間）＝赤で重ねて PNG に出力する。
各期間の頑健フィット直線（temp_equipment.fit_t_vs_i と同じ Theil-Sen）も重ね描きし、
傾き・切片・相関係数を凡例に出す。ビーム閾値（既定50mA、判定に使った点の下限）も
縦の目安線で示す（無ビーム点も見えるよう、点自体は全部プロットする）。

使い方:
    # 学習済みモデル（temp_equipment_models.json）の基準期間を自動で使う（推奨・楽）
    python temp_equipment_plot.py IR 20260301000000 20260401000000

    # 基準・現在ともに明示指定
    python temp_equipment_plot.py IR --ref-start 20220401000000 --ref-end 20220501000000 \\
        --now-start 20260301000000 --now-end 20260401000000

    # 実データCSV（変化ログ形式）から読む場合
    python temp_equipment_plot.py IR --ref-csv 2022.csv --now-csv 2026.csv --pv "FB_MOVE:D01:QC1L:BWS:TEMP"

    # 1本だけ
    python temp_equipment_plot.py IR 20260301000000 20260401000000 --pv "FB_MOVE:D01:QC1L:BWS:TEMP"
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")          # ヘッドレス（画面に出さず PNG 保存）。ip_observe.py と同じ作法
import matplotlib.pyplot as plt

import temp_equipment as te
import temp_fetch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "equipment_plots")


def _safe_name(pv):
    return pv.replace(":", "_")


def plot_one(pv, T_ref, I_ref, T_now, I_now, ring=None, out_dir=DEFAULT_OUT_DIR,
            ref_label="Reference", now_label="Current", cfg=te.CONFIG):
    """1本分の T-I 散布図（基準/現在を重ね描き）を PNG に保存し、保存パスを返す。
    図中の文字列はすべて英語（実機に日本語フォントが無い環境があるため。ip_observe.py と同じ方針）。"""
    os.makedirs(out_dir, exist_ok=True)

    # フィットはビームあり点のみ（judge/learn と同じ条件）。点自体は無ビームも含め全部見せる。
    Tf_ref, If_ref = te.filter_beam_on(T_ref, I_ref, cfg)
    Tf_now, If_now = te.filter_beam_on(T_now, I_now, cfg)
    fit_ref = te.fit_t_vs_i(Tf_ref, If_ref, cfg)
    fit_now = te.fit_t_vs_i(Tf_now, If_now, cfg)

    fig, ax = plt.subplots(figsize=(7.5, 6), tight_layout=True)
    ax.scatter(I_ref, T_ref, s=10, alpha=0.35, color="#4a7fd6", label="%s (n=%d)" % (ref_label, len(T_ref)))
    ax.scatter(I_now, T_now, s=10, alpha=0.35, color="#e2574a", label="%s (n=%d)" % (now_label, len(T_now)))

    xmax = max(np.max(I_ref) if len(I_ref) else 0, np.max(I_now) if len(I_now) else 0, cfg["beam_on_ma"] * 2)
    # 各期間のフィット曲線は、その期間で実際に観測された電流範囲までしか描かない
    # （データの無い領域まで外挿すると、特に非線形モデルで誇張された線になるため）。
    xs_ref = np.linspace(0, np.max(If_ref) if len(If_ref) else cfg["beam_on_ma"] * 2, 100)
    xs_now = np.linspace(0, np.max(If_now) if len(If_now) else cfg["beam_on_ma"] * 2, 100)
    if fit_ref["trust"]:
        ax.plot(xs_ref, fit_ref["a"] + fit_ref["b"] * xs_ref, color="#1c4fa0", lw=2,
               label="%s fit: dT/dI=%.4f C/mA (%.2f C/A) r=%.2f" % (ref_label, fit_ref["b"], fit_ref["b"] * 1000, fit_ref["r"]))
    if fit_now["trust"]:
        ax.plot(xs_now, fit_now["a"] + fit_now["b"] * xs_now, color="#a83226", lw=2,
               label="%s fit: dT/dI=%.4f C/mA (%.2f C/A) r=%.2f" % (now_label, fit_now["b"], fit_now["b"] * 1000, fit_now["r"]))

    ax.axvline(cfg["beam_on_ma"], color="gray", ls="--", lw=1, alpha=0.6)
    ax.annotate("beam threshold %gmA\n(fit uses points above this only)" % cfg["beam_on_ma"],
               xy=(cfg["beam_on_ma"], ax.get_ylim()[1]), xytext=(5, -5),
               textcoords="offset points", fontsize=8, color="gray", va="top")

    title = pv if ring is None else "[%s] %s" % (ring, pv)
    if fit_ref["trust"] and fit_now["trust"] and fit_ref["b"] > 0:
        ratio = fit_now["b"] / fit_ref["b"]
        title += "\ndT/dI ratio (current/reference) = %.2fx" % ratio
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Beam current [mA]")
    ax.set_ylabel("Temperature [degC]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    path = os.path.join(out_dir, "%s.png" % _safe_name(pv))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_one_hom(pv, T_ref, I_ref, Nb_ref, T_now, I_now, Nb_now, ring=None, out_dir=DEFAULT_OUT_DIR,
                 ref_label="Reference", now_label="Current", cfg=te.CONFIG):
    """HOMモデル（T=w0+w1*I+w2*(I^2/Nb)^2）版の T-I 散布図。
    Nb は期間内で変動しうるため、フィット曲線は各期間の中央値Nbで評価した近似曲線として描く
    （実データの各点はそのPV固有のNbで決まるので、曲線からの散らばりにはNb変動の影響も含まれる）。
    """
    os.makedirs(out_dir, exist_ok=True)
    Tf_ref, If_ref, Nf_ref = te.filter_beam_on3(T_ref, I_ref, Nb_ref, cfg)
    Tf_now, If_now, Nf_now = te.filter_beam_on3(T_now, I_now, Nb_now, cfg)
    fit_ref = te.fit_t_vs_i_hom(Tf_ref, If_ref, Nf_ref, cfg)
    fit_now = te.fit_t_vs_i_hom(Tf_now, If_now, Nf_now, cfg)
    nb_med_ref = float(np.median(Nf_ref)) if len(Nf_ref) else float("nan")
    nb_med_now = float(np.median(Nf_now)) if len(Nf_now) else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 6), tight_layout=True)
    ax.scatter(I_ref, T_ref, s=10, alpha=0.35, color="#4a7fd6", label="%s (n=%d)" % (ref_label, len(T_ref)))
    ax.scatter(I_now, T_now, s=10, alpha=0.35, color="#e2574a", label="%s (n=%d)" % (now_label, len(T_now)))

    xmax = max(np.max(I_ref) if len(I_ref) else 0, np.max(I_now) if len(I_now) else 0, cfg["beam_on_ma"] * 2)
    # 各期間のフィット曲線は、その期間で実際に観測された電流範囲までしか描かない
    # （HOM項は I の4乗に近い増え方をするので、データの無い領域への外挿は特に誇張されやすい）。
    xs_ref = np.linspace(0, np.max(If_ref) if len(If_ref) else cfg["beam_on_ma"] * 2, 100)
    xs_now = np.linspace(0, np.max(If_now) if len(If_now) else cfg["beam_on_ma"] * 2, 100)
    if fit_ref["trust"]:
        w0, w1, w2 = fit_ref["w"]
        ax.plot(xs_ref, w0 + w1 * xs_ref + w2 * (xs_ref ** 2 / nb_med_ref) ** 2, color="#1c4fa0", lw=2,
               label="%s fit (Nb=%.0f): w1=%.4f w2=%.2e R2=%.2f" % (ref_label, nb_med_ref, w1, w2, fit_ref["r2"]))
    if fit_now["trust"]:
        w0, w1, w2 = fit_now["w"]
        ax.plot(xs_now, w0 + w1 * xs_now + w2 * (xs_now ** 2 / nb_med_now) ** 2, color="#a83226", lw=2,
               label="%s fit (Nb=%.0f): w1=%.4f w2=%.2e R2=%.2f" % (now_label, nb_med_now, w1, w2, fit_now["r2"]))

    ax.axvline(cfg["beam_on_ma"], color="gray", ls="--", lw=1, alpha=0.6)
    ax.annotate("beam threshold %gmA\n(fit uses points above this only)" % cfg["beam_on_ma"],
               xy=(cfg["beam_on_ma"], ax.get_ylim()[1]), xytext=(5, -5),
               textcoords="offset points", fontsize=8, color="gray", va="top")

    title = pv if ring is None else "[%s] %s" % (ring, pv)
    title += "\nHOM model: T = w0 + w1*I + w2*(I^2/Nb)^2  (Suetsugu et al. PRAB 27,063201(2024) Eq.5-form)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Beam current [mA]")
    ax.set_ylabel("Temperature [degC]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    path = os.path.join(out_dir, "%s_hom.png" % _safe_name(pv))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _load_period(ring, pv, start=None, end=None, csv_path=None, with_nb=False):
    if csv_path:
        if with_nb:
            T, I, Nb, _ = te.load_raw_csv(csv_path, pv=pv, with_nb=True)
        else:
            T, I, _ = te.load_raw_csv(csv_path, pv=pv)
        label = os.path.basename(csv_path)
    else:
        if with_nb:
            T, I, Nb = te.load_live(ring, pv, start, end, with_nb=True)
        else:
            T, I = te.load_live(ring, pv, start, end)
        label = "%s-%s" % (start, end)
    return (T, I, Nb, label) if with_nb else (T, I, label)


def cmd_plot(args):
    # 明示指定（--model / --ref-start,--ref-end / --ref-csv）があればそれを全PV共通で使う。
    # 無ければ PV ごとに保存済みモデル（learn の結果）から model種別・学習期間を自動判定する
    # （judge() が model="hom"/"linear" を保存済み値から自動で見るのと同じ考え方）。
    store = te._load_models(args.models)
    rd = store.get(args.ring, {})
    explicit_model = args.model                # None なら自動判定
    explicit_ref = bool(args.ref_csv or (args.ref_start and args.ref_end))

    recs = temp_fetch.load_pv_list(args.ring)
    if args.pv:
        recs = [r for r in recs if r["pv"] == args.pv]
        if not recs:
            sys.exit("エラー: PV %r が %s のリストに見つかりません。" % (args.pv, args.ring))
    elif args.match:
        recs = [r for r in recs if args.match in r["pv"]]

    print("対象 %d 本をプロット → %s/" % (len(recs), args.out_dir))
    paths = []
    n_skipped = 0
    for rec in recs:
        pv = rec["pv"]
        saved = rd.get(pv)

        # モデル種別: 明示指定 > 保存済みモデルの種別 > 既定 linear（保存済みが無い場合のみ警告）
        if explicit_model:
            model_type = explicit_model
        elif saved:
            model_type = saved.get("model", "linear")
        else:
            model_type = "linear"
            sys.stderr.write("[%s] 学習済みモデルが無いため model=linear で仮定します"
                             "（--model で明示するか、先に learn してください）。\n" % pv)

        # 基準期間: 明示指定 > 保存済みモデルの学習期間 > エラー
        if explicit_ref:
            ref_start, ref_end, ref_csv = args.ref_start, args.ref_end, args.ref_csv
        elif saved and "trained_start" in saved:
            ref_start, ref_end, ref_csv = saved["trained_start"], saved["trained_end"], None
        else:
            sys.stderr.write("[%s] 基準期間が不明（学習済みモデルも --ref-start/--ref-csv も無し）。"
                             "スキップします。\n" % pv)
            n_skipped += 1
            continue

        try:
            if model_type == "hom":
                T_ref, I_ref, Nb_ref, ref_label = _load_period(args.ring, pv, ref_start, ref_end,
                                                                ref_csv, with_nb=True)
                T_now, I_now, Nb_now, now_label = _load_period(args.ring, pv, args.now_start, args.now_end,
                                                                args.now_csv, with_nb=True)
            else:
                T_ref, I_ref, ref_label = _load_period(args.ring, pv, ref_start, ref_end, ref_csv)
                T_now, I_now, now_label = _load_period(args.ring, pv, args.now_start, args.now_end, args.now_csv)
        except Exception as ex:
            sys.stderr.write("[%s] 取得失敗（スキップ）: %s\n" % (pv, ex))
            continue
        if model_type == "hom":
            path = plot_one_hom(pv, T_ref, I_ref, Nb_ref, T_now, I_now, Nb_now, ring=args.ring,
                                out_dir=args.out_dir, ref_label="Ref(%s)" % ref_label,
                                now_label="Now(%s)" % now_label)
        else:
            path = plot_one(pv, T_ref, I_ref, T_now, I_now, ring=args.ring, out_dir=args.out_dir,
                            ref_label="Ref(%s)" % ref_label, now_label="Now(%s)" % now_label)
        print("  %-32s [model=%s ref=%s-%s] → %s" % (pv, model_type, ref_start, ref_end, path))
        paths.append(path)
    print("完了: %d 枚%s" % (len(paths), "（%d本スキップ）" % n_skipped if n_skipped else ""))


def main():
    ap = argparse.ArgumentParser(description="温度 vs ビーム電流の比較散布図を出力する")
    ap.add_argument("ring", choices=["LER", "HER", "IR"])
    ap.add_argument("now_start", nargs="?", default=None, help="現在期間の開始（--now-csv使用時は省略可）")
    ap.add_argument("now_end", nargs="?", default=None, help="現在期間の終了（--now-csv使用時は省略可）")
    ap.add_argument("--ref-start", default=None)
    ap.add_argument("--ref-end", default=None)
    ap.add_argument("--now-start", dest="now_start_kw", default=None)
    ap.add_argument("--now-end", dest="now_end_kw", default=None)
    ap.add_argument("--ref-csv", default=None, help="基準期間を変化ログCSVから読む")
    ap.add_argument("--now-csv", default=None, help="現在期間を変化ログCSVから読む")
    ap.add_argument("--models", default=te.MODELS_FILE, help="基準期間自動取得に使うモデルファイル")
    ap.add_argument("--pv", default=None, help="1本だけプロット（完全なPV名）")
    ap.add_argument("--match", default=None, help="PV名の部分一致でしぼる（例: BWS）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", choices=["linear", "hom"], default=None,
                    help="既定: PVごとに学習済みモデル(temp_equipment_models.json)の種別を自動判定。"
                         "明示指定すると全PVでその種別を強制する。"
                         "linear: T=a+b*I。hom: T=w0+w1*I+w2*(I^2/Nb)^2"
                         "（Suetsugu et al. PRAB 27,063201(2024) 式(5)型。Nb取得が必要）")
    args = ap.parse_args()

    # now_start/now_end は位置引数と --now-start/--now-end のどちらでも受ける
    if args.now_start_kw:
        args.now_start = args.now_start_kw
    if args.now_end_kw:
        args.now_end = args.now_end_kw
    if not args.now_csv and not (args.now_start and args.now_end):
        sys.exit("エラー: 現在期間を指定してください（位置引数 now_start now_end、"
                 "または --now-start/--now-end、または --now-csv）。")

    cmd_plot(args)


if __name__ == "__main__":
    main()
