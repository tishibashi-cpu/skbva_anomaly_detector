#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cause_infer.py — 推定原因（Possible Cause）をヘッドレスで再現する

末次プログラムの Find_Possible_Cause は、異常レコードの回帰パラメータを
  funclog1 → 標準化(sms) → 2層FNN(Dense tanh → Dense softmax) → argmax
に通して原因を当てる（model_pc_*.h5）。本モジュールはこの推論を
【keras を使わず純 numpy で】再現する。判定ロジックには触れず、結果ファイルに
既に保存されている列だけを入力にするので、検知本体は無改変のまま使える。

なぜ numpy 再現で十分か:
  モデルは Sequential の Dense(tanh)→Dense(softmax) のみ（乱数なし）。
  .h5 に保存された重み・構成をそのまま順伝播すれば keras と同一の argmax になる
  （softmax は単調なので argmax は logits と一致）。h5py だけで動く。

入力列は Find_Possible_Cause と完全一致:
  Storage(WB) 8入力 : W0_std,W1_std,W2_std,W0_dif,W1_dif,W2_dif,Max_Pre,Max_Cal
  Storage(NB) 9入力 : 上記6W + Max_Pre,Max_Cal,RMSE_dif
  Tail(WB)   10入力 : W0_std..W3_std,W0_dif..W3_dif,Max_Pre,Max_Cal
"""

import os
import json
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

# ── 原因ラベル（Find_Possible_Cause の Cause_List と同義。第0版/Table III 準拠）──
CAUSE_LABELS = {
    ("strg", "wb"): ["リーク or ポンプ故障", "異常加熱 or 放電", "軌道異常 or リーク"],
    ("strg", "nb"): ["リーク or ポンプ(CCG)故障", "排気後 or リーク"],
    ("tail", "wb"): ["リーク or ポンプ故障", "異常加熱 or 放電", "圧力バースト"],
}

# ── 各ケースで結果ファイルのどの列を入力にするか（ヘッダ準拠の 0-index）──
#   Strg ヘッダ: ...,Max_Pre(8),...,Max_Cal(11),...,W0_std(17),W1_std(18),W2_std(19),
#                W0_dif(20),W1_dif(21),W2_dif(22),RMSE_dif(23),...
#   Tail ヘッダ: ...,Max_Pre(8),...,Max_Cal(10),...,W0_std(16)..W3_std(19),
#                W0_dif(21)..W3_dif(24),...
INPUT_COLUMNS = {
    ("strg", "wb"): [17, 18, 19, 20, 21, 22, 8, 11],
    ("strg", "nb"): [17, 18, 19, 20, 21, 22, 8, 11, 23],
    ("tail", "wb"): [16, 17, 18, 19, 21, 22, 23, 24, 8, 10],
}


def funclog1(x):
    """末次プログラムの funclog1 と同一（符号付き log10、下限 1e-12）。"""
    if x >= 0.0:
        x1 = max(x, 1.0e-12)
        return np.log10(x1 * 1.0e12)
    else:
        x1 = max(abs(x), 1.0e-12)
        return -np.log10(x1 * 1.0e12)


def _load_sms(path):
    """sms_pc_*.txt を平坦な float リストにして返す（[mean0,std0,mean1,std1,...]）。"""
    vals = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            for tok in line.replace(",", " ").split():
                vals.append(float(tok))
    return vals


def _load_keras_sequential(h5_path):
    """keras Sequential(.h5) の Dense 層を順序どおりに [(kernel,bias,activation),...] で返す。"""
    if h5py is None:
        raise RuntimeError("h5py が必要です（pip install h5py）。")
    with h5py.File(h5_path, "r") as f:
        cfg = f.attrs.get("model_config")
        if isinstance(cfg, bytes):
            cfg = cfg.decode("utf-8")
        cfg = json.loads(cfg)
        layers_cfg = cfg["config"]["layers"] if "layers" in cfg["config"] else cfg["config"]
        wg = f["model_weights"] if "model_weights" in f else f

        layers = []
        for L in layers_cfg:
            if L["class_name"] != "Dense":
                continue
            name = L["config"]["name"]
            act = L["config"].get("activation", "linear")
            g = wg[name][name]
            kernel = np.array(g["kernel:0"])
            bias = np.array(g["bias:0"])
            layers.append((kernel, bias, act))
    return layers


def _forward(layers, x):
    """2層FNN（Dense tanh → Dense softmax）の順伝播。x: shape (n_in,)。"""
    h = x
    for kernel, bias, act in layers:
        h = h @ kernel + bias
        if act == "tanh":
            h = np.tanh(h)
        elif act == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-h))
        elif act == "softmax":
            e = np.exp(h - np.max(h))
            h = e / e.sum()
        # 'linear' は素通り
    return h


class CauseInferer:
    """legacy/ のモデル・標準化ファイルを読み、結果ファイル行 → 原因テキストを返す。"""

    def __init__(self, legacy_dir):
        self.legacy_dir = legacy_dir
        self._models = {}   # (period, beam) -> (layers, sms)

    def _get(self, period, beam):
        key = (period, beam)
        if key not in self._models:
            h5 = os.path.join(self.legacy_dir, "model_pc_%s_%s.h5" % (period, beam))
            sms = os.path.join(self.legacy_dir, "sms_pc_%s_%s.txt" % (period, beam))
            self._models[key] = (_load_keras_sequential(h5), _load_sms(sms))
        return self._models[key]

    def available(self, period, beam):
        h5 = os.path.join(self.legacy_dir, "model_pc_%s_%s.h5" % (period, beam))
        sms = os.path.join(self.legacy_dir, "sms_pc_%s_%s.txt" % (period, beam))
        return os.path.isfile(h5) and os.path.isfile(sms)

    def infer_row(self, row, period, beam):
        """結果ファイルの1行(list[str]) → (cause_text, class_index, probabilities)。

        row が短すぎる/数値化できない場合は (None, -1, None) を返す（呼び側で無視）。
        """
        key = (period, beam)
        cols = INPUT_COLUMNS[key]
        labels = CAUSE_LABELS[key]
        try:
            raw = [float(row[c]) for c in cols]
        except (IndexError, ValueError):
            return None, -1, None

        x = np.array([funclog1(v) for v in raw], dtype=float)

        layers, sms = self._get(period, beam)
        # 標準化（Training data の mean/std を使用。legacy と同式）
        for i in range(len(x)):
            me1 = sms[i * 2]
            si1 = sms[i * 2 + 1]
            x[i] = (x[i] - me1) / (si1 if si1 != 0.0 else 1.0)

        probs = _forward(layers, x)
        cmax = int(np.argmax(probs))
        return labels[cmax], cmax, probs


if __name__ == "__main__":
    # 動作確認: legacy/ の WB 異常結果ファイルから先頭数件の原因を推定して表示
    import csv
    import sys

    legacy = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "legacy")
    inf = CauseInferer(legacy)

    for ring in ("LER", "HER"):
        for period, mode in (("strg", "Strg"), ("tail", "Tail")):
            path = os.path.join(legacy, "%s_FNN_Abnormal_Class2_Result_%s_WB.txt" % (ring, mode))
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))[1:]
            print("\n%s %s（%d 行）先頭5件の推定原因:" % (ring, mode, len(rows)))
            for r in rows[:5]:
                cause, ci, probs = inf.infer_row(r, period, "wb")
                if cause is None:
                    continue
                p = ", ".join("%.2f" % v for v in probs)
                print("  %-22s -> %-16s (class %d, p=[%s])" % (r[3], cause, ci, p))
