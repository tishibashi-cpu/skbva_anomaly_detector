#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperKEKB 圧力異常検知 — 監視ダッシュボード（プロトタイプ）

依存ゼロ（Python 標準ライブラリのみ）。外部 CDN も使いません。
制御ネット上の kekb-co-user0X で:
    /cont/python/x86_64-AlmaLinux9/3.9.17/bin/python3 dashboard.py
で起動し、自席 PC からは SSH トンネル経由でブラウザ http://localhost:18050 を開きます。

データの渡し方:
  - 同じディレクトリに dashboard_state.json があればそれを表示します（本番モード）。
  - 無ければ内蔵のダミーデータを表示します（プロトタイプ確認用）。
  実運用では、既存の検知プログラムに dashboard_state.json を書き出すフックを
  足すだけで、このダッシュボードはそのまま使えます（判定ロジックには触りません）。
"""

import json
import os
import math
import random
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 共用サーバーの負荷表示（標準ライブラリのみ）。無くてもダッシュボードは動く。
try:
    import sysload
    _cpu_sampler = sysload.CpuSampler()
    _cpu_sampler.percent()   # 基準サンプルを1回取っておく
    _proc_sampler = sysload.ProcSampler()
    _proc_sampler.sample()   # 自プロセスも基準取り
except Exception:
    sysload = None
    _cpu_sampler = None
    _proc_sampler = None

# 蓄積電流（リアルタイム）取得。EPICS が無くてもダッシュボードは動く。
try:
    import beamcurrent
except Exception:
    beamcurrent = None

PORT = 18050
# dashboard_state.json はこのスクリプトと同じ場所（トップ）に置かれる。
# CWD に依存しないよう __file__ 基準で解決する。
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_state.json")

# ----------------------------------------------------------------------------
# ダミーデータ生成（実運用 JSON と同じスキーマ）
# ----------------------------------------------------------------------------
def _series_rising(n=60, base=6e-8, slope=2.2e-9, noise=3e-9):
    """加熱・放電型: 圧力が時間とともに上昇。"""
    t = list(range(n))
    pres = [base + slope * i + random.uniform(-noise, noise) for i in range(n)]
    beam = [800 + random.uniform(-15, 15) for _ in range(n)]
    return t, pres, beam

def _series_burst(n=60, base=8e-8, noise=4e-9):
    """Tail バースト型: ところどころ突発的なスパイク。"""
    t = list(range(n))
    pres = []
    for i in range(n):
        spike = random.choice([0, 0, 0, 0, 1.2e-7]) if i % 7 == 0 else 0
        pres.append(base + spike + random.uniform(-noise, noise))
    beam = [700 + random.uniform(-20, 20) for _ in range(n)]
    return t, pres, beam

def _series_gradual(n=60, base=5e-8, slope=9e-10, noise=2e-9):
    """軌道異常・リーク型: 緩やかな上昇。"""
    t = list(range(n))
    pres = [base + slope * i + random.uniform(-noise, noise) for i in range(n)]
    beam = [802 + random.uniform(-10, 10) for _ in range(n)]
    return t, pres, beam

def make_dummy_state():
    t1, p1, b1 = _series_rising()
    t2, p2, b2 = _series_burst()
    t3, p3, b3 = _series_gradual()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    anomalies = [
        {
            "id": "her-d01-h16",
            "record": "VAHCCG:D01_H16:PRES",
            "ring": "HER", "section": "D01", "place": "大穂",
            "period": "Storage", "count": 8, "max_count": 8,
            "cause": "異常加熱 or 放電", "severity": "danger",
            "series": {"t": t1, "pressure": p1, "beam": b1},
        },
        {
            "id": "ler-d04-l07",
            "record": "VALCCG:D04_L07:PRES",
            "ring": "LER", "section": "D04", "place": "富士",
            "period": "Tail", "count": 6, "max_count": 6,
            "cause": "圧力バースト or リーク", "severity": "danger",
            "series": {"t": t2, "pressure": p2, "beam": b2},
        },
        {
            "id": "ler-d10-l09",
            "record": "VALCCG:D10_L09:PRES",
            "ring": "LER", "section": "D10", "place": "日光 (Wiggler)",
            "period": "Storage", "count": 4, "max_count": 4,
            "cause": "軌道異常 or リーク", "severity": "warning",
            "series": {"t": t3, "pressure": p3, "beam": b3},
        },
    ]

    sections = {"LER": {}, "HER": {}}
    for ring in ("LER", "HER"):
        for i in range(1, 13):
            sections[ring]["D%02d" % i] = None
    for a in anomalies:
        sections[a["ring"]][a["section"]] = a["severity"]

    return {
        "updated": now,
        "classifier": "Keras",
        "rings": {
            "LER": {"beam_on": True, "current_mA": 802},
            "HER": {"beam_on": True, "current_mA": 712},
        },
        "summary": {"monitored": {"total": 605, "LER": 308, "HER": 297}, "critical": 2, "warning": 1},
        "anomalies": anomalies,
        "sections": sections,
    }

def load_state():
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": "state file read failed: %s" % e}
    return make_dummy_state()

# ----------------------------------------------------------------------------
# フロントエンド（HTML/CSS/JS を 1 つにまとめて配信）
# ----------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>圧力異常モニター — SuperKEKB</title>
<style>
  :root {
    --bg: #14161b; --surface: #1c1f27; --surface2: #232732;
    --line: #2c313d; --text: #e7e8ec; --muted: #969aa6; --faint: #6b7080;
    --danger: #e2574a; --danger-bg: #2a1a18; --danger-fg: #f0a097;
    --warning: #e6a02a; --warning-bg: #2a2316; --warning-fg: #f0c878;
    --ok: #4fc59a; --accent: #5dcaa5;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --sans: system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: var(--sans); font-size: 14px; line-height: 1.5; }
  a { color: inherit; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 18px 22px 60px; }

  .topbar { display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 12px; padding-bottom: 16px; }
  .brand { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  .brand h1 { font-size: 17px; font-weight: 600; margin: 0; letter-spacing: .2px; }
  .ring-stat { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
  .dot.on { background: var(--ok); }
  .meta { display: flex; align-items: center; gap: 14px; font-size: 12.5px; color: var(--muted); }
  .pill { padding: 3px 10px; border-radius: 6px; background: var(--surface2); color: var(--muted); }
  #sysload.load-ok   { color: var(--ok); }
  #sysload.load-warn { color: #e6a02a; }
  #sysload.load-high { color: var(--danger-fg); background: var(--danger-bg); }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 10px; margin: 4px 0 22px; }
  .stat { background: var(--surface); border: 1px solid var(--line);
          border-radius: 10px; padding: 13px 15px; }
  .stat .lbl { font-size: 12.5px; color: var(--muted); }
  .stat .num { font-size: 25px; font-weight: 600; margin-top: 2px; }
  .stat .sub { font-size: 11.5px; color: var(--muted); margin-top: 3px; }
  .num.danger { color: var(--danger-fg); }
  .num.warning { color: var(--warning-fg); }
  .num.ok { color: var(--ok); }

  .section-title { font-size: 13px; color: var(--muted); margin: 0 0 10px; font-weight: 500; }

  .alist { display: flex; flex-direction: column; gap: 9px; }
  .acard { background: var(--surface); border: 1px solid var(--line);
           border-left: 3px solid var(--faint); border-radius: 0 10px 10px 0;
           padding: 13px 16px; display: flex; align-items: center; gap: 16px;
           flex-wrap: wrap; cursor: pointer; transition: background .12s; }
  .acard:hover { background: var(--surface2); }
  .acard.danger { border-left-color: var(--danger); }
  .acard.warning { border-left-color: var(--warning); }
  .acard.sel { background: var(--surface2); border-color: var(--accent); }
  .count { min-width: 60px; text-align: center; }
  .count .n { font-size: 25px; font-weight: 600; line-height: 1; }
  .count.danger .n { color: var(--danger-fg); }
  .count.warning .n { color: var(--warning-fg); }
  .count .of { font-size: 11px; color: var(--faint); }
  .ainfo { flex: 1; min-width: 190px; }
  .rec { font-family: var(--mono); font-size: 14px; }
  .loc { font-size: 12px; color: var(--muted); margin-top: 1px; }
  .cause { display: inline-block; margin-top: 5px; font-size: 12px;
           padding: 2px 9px; border-radius: 6px; }
  .cause.danger { background: var(--danger-bg); color: var(--danger-fg); }
  .cause.warning { background: var(--warning-bg); color: var(--warning-fg); }
  .spark { flex-shrink: 0; }

  .detail { margin-top: 10px; background: var(--surface); border: 1px solid var(--accent);
            border-radius: 10px; padding: 16px 18px; }
  .detail h3 { margin: 0 0 2px; font-size: 15px; font-family: var(--mono); font-weight: 600; }
  .detail .sub { font-size: 12.5px; color: var(--muted); margin-bottom: 12px; }
  .plotrow { display: grid; grid-template-columns: 1fr; gap: 14px; }
  .plotbox { background: var(--surface2); border: 1px solid var(--line);
             border-radius: 8px; padding: 10px 12px; }
  .plotbox .cap { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .btns { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
  button { font-family: var(--sans); font-size: 13px; color: var(--text);
           background: var(--surface2); border: 1px solid var(--line);
           border-radius: 7px; padding: 7px 14px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.close { margin-left: auto; }

  .map { margin-top: 26px; }
  .maprow { display: flex; gap: 5px; align-items: center; margin-top: 6px; }
  .maprow .rl { width: 34px; font-size: 12px; color: var(--faint); }
  .cells { display: flex; gap: 5px; flex: 1; }
  .cell { flex: 1; min-width: 34px; text-align: center; font-size: 11px;
          padding: 7px 0; border-radius: 6px; background: var(--surface);
          border: 1px solid var(--line); color: var(--muted); cursor: default; }
  .cell.danger { background: var(--danger-bg); border-color: var(--danger);
                 color: var(--danger-fg); cursor: pointer; }
  .cell.warning { background: var(--warning-bg); border-color: var(--warning);
                  color: var(--warning-fg); cursor: pointer; }

  .legend { display: flex; gap: 16px; margin-top: 14px; font-size: 12px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .sw { width: 11px; height: 11px; border-radius: 3px; }
  .toast { position: relative; }
  .footnote { margin-top: 22px; font-size: 12px; color: var(--faint); }
  .empty { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
           padding: 34px 20px; text-align: center; }
  .empty .emoji { font-size: 26px; color: var(--ok); margin-bottom: 8px; }
  .empty .etxt { font-size: 15px; color: var(--text); }
  .empty .esub { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
  /* イオンポンプパネル */
  .devtag { font-size: 10.5px; color: #b06a1a; background: #f6e8d2; border-radius: 4px;
            padding: 1px 6px; margin-left: 8px; vertical-align: middle; }
  .ipmeta { font-size: 11.5px; color: var(--muted); margin: -4px 0 10px; }
  .iprow { display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start; }
  .iplabel { width: 38px; font-weight: 600; color: var(--muted); padding-top: 6px; flex-shrink: 0; }
  .ipgrid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 6px; flex: 1; }
  @media (max-width: 900px) { .ipgrid { grid-template-columns: repeat(6, 1fr); } }
  .ipcard { background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
            padding: 6px 7px; cursor: pointer; transition: border-color .12s; }
  .ipcard:hover { border-color: #4a90d9; }
  .ipcard.sel { border-color: #4a90d9; box-shadow: 0 0 0 1px #4a90d9 inset; }
  .ipsec { font-size: 12px; font-weight: 600; display: flex; justify-content: space-between;
           align-items: center; }
  .badge { font-size: 9px; border-radius: 3px; padding: 0 4px; font-weight: 600; }
  .bkek { background: #e3eef7; color: #2c6291; }
  .b4u { background: #f0e6f7; color: #7a3fa0; }
  .ipbar { height: 4px; background: var(--line); border-radius: 2px; margin: 5px 0 3px; overflow: hidden; }
  .ipbar span { display: block; height: 100%; background: linear-gradient(90deg,#5aa0e0,#e2574a); }
  .ipval { font-size: 11px; font-variant-numeric: tabular-nums; }
  .ipn { font-size: 10px; color: var(--muted); }
  .ippvs { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
           padding: 10px 12px; margin: 2px 0 12px 48px; }
  .ippvs-h { font-size: 12.5px; font-weight: 600; margin-bottom: 6px; }
  .ippv { display: flex; align-items: center; gap: 10px; padding: 3px 0;
          border-top: 1px solid var(--line); }
  .ippv:first-of-type { border-top: none; }
  .ippv-name { flex: 1; font-size: 11.5px; font-family: ui-monospace, monospace; color: var(--text); }
  .ippv-val { width: 92px; text-align: right; font-size: 11px; font-variant-numeric: tabular-nums;
              color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <h1>圧力異常モニター</h1>
      <span class="ring-stat" id="ler-stat"><span class="dot"></span> LER —</span>
      <span class="ring-stat" id="her-stat"><span class="dot"></span> HER —</span>
    </div>
    <div class="meta">
      <span class="pill" id="sysload" title="共用サーバー全体の負荷">負荷 —</span>
      <span class="pill" id="selfload" title="このダッシュボード（＋子プロセス）自身の使用量">本プログラム —</span>
      <span id="updated">— 更新待ち</span>
      <span class="pill" id="classifier">判定: —</span>
    </div>
  </div>

  <div class="cards" id="summary"></div>

  <div class="section-title">検知された異常 — カウントの多い順</div>
  <div class="alist" id="alist"></div>
  <div id="detail-host"></div>

  <div class="map">
    <div class="section-title">リング配置（異常箇所をハイライト）</div>
    <div id="map"></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--danger)"></span> 要注意（カウント ≥6）</span>
      <span><span class="sw" style="background:var(--warning)"></span> 注意（3–5）</span>
      <span><span class="sw" style="background:var(--surface)"></span> 正常</span>
    </div>
  </div>

  <div id="ionpumps"></div>

  <div class="footnote" id="footnote"></div>
</div>

<script>
let STATE = null;
let SELECTED = null;

function fmtPres(v) { return (v * 1e8).toFixed(2) + "e-8"; }

function spark(series, color, w, h) {
  const p = series.pressure;
  const min = Math.min.apply(null, p), max = Math.max.apply(null, p);
  const rng = (max - min) || 1;
  const pts = p.map((v, i) => {
    const x = (i / (p.length - 1)) * w;
    const y = h - ((v - min) / rng) * (h - 4) - 2;
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
  return '<svg class="spark" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">' +
         '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2"/></svg>';
}

function sevColor(sev) {
  return sev === "danger" ? "#e2574a" : sev === "warning" ? "#e6a02a" : "#6b7080";
}

function renderSummary(s) {
  const el = document.getElementById("summary");
  const m = s.summary.monitored;
  let mainNum, sub = "";
  if (m && typeof m === "object") {
    mainNum = m.total;
    sub = "LER " + m.LER + " / HER " + m.HER;
  } else {
    mainNum = "~" + m;   // 旧形式（数値）との後方互換
  }
  el.innerHTML =
    statCard("監視中の真空計", mainNum, "", sub) +
    statCard("要注意 (≥6)", s.summary.critical, "danger") +
    statCard("注意 (3–5)", s.summary.warning, "warning") +
    statCard("状態", s.summary.critical > 0 ? "ALERT" : "OK",
             s.summary.critical > 0 ? "danger" : "ok");
}
function statCard(lbl, num, cls, sub) {
  return '<div class="stat"><div class="lbl">' + lbl + '</div>' +
         '<div class="num ' + cls + '">' + num + '</div>' +
         (sub ? '<div class="sub">' + sub + '</div>' : '') + '</div>';
}

function renderTop(s) {
  document.getElementById("updated").textContent = "最終チェック " + (s.updated || "—");
  document.getElementById("classifier").textContent = "判定: " + (s.classifier || "—");
  renderRings();
}

let BEAM_RT = {LER: null, HER: null};   // リアルタイム蓄積電流（/api/beam）

function renderRings() {
  if (!STATE || !STATE.rings) return;
  setRing("ler-stat", "LER", STATE.rings.LER, BEAM_RT.LER);
  setRing("her-stat", "HER", STATE.rings.HER, BEAM_RT.HER);
}

function setRing(id, name, r, rt) {
  const el = document.getElementById(id);
  const hasRT = (rt !== null && rt !== undefined);
  const rtOn = hasRT && rt >= 1;          // 1 mA 以上を ON とみなす
  // ドットはリアルタイムがあればそれを、無ければ最終チェック由来を反映
  const dotOn = hasRT ? rtOn : (r && r.beam_on);
  let parts = [];
  if (hasRT) parts.push("現在 " + (rtOn ? (Math.round(rt) + " mA") : "OFF"));
  if (r) parts.push("最終チェック " + (r.beam_on ? ("ON " + Math.round(r.current_mA) + " mA") : "OFF"));
  if (!parts.length) parts.push("—");
  el.innerHTML = '<span class="dot ' + (dotOn ? "on" : "") + '"></span> ' +
                 name + " " + parts.join(" ⋅ ");
}

async function refreshBeam() {
  try {
    const r = await fetch("/api/beam", {cache: "no-store"});
    const d = await r.json();
    BEAM_RT.LER = d.LER; BEAM_RT.HER = d.HER;
    renderRings();
  } catch (e) { /* 取得できなくてもヘッダは最終チェック表示のまま */ }
}

function renderList(s) {
  const host = document.getElementById("alist");
  const items = s.anomalies.slice().sort((a, b) => b.count - a.count);
  if (items.length === 0) {
    const days = s.recent_window_days || 3;
    let msg = '<div class="empty"><div class="emoji">✓</div>' +
      '<div class="etxt">直近 ' + days + ' 日に検知された異常はありません</div>';
    if (s.last_anomaly)
      msg += '<div class="esub">最後に記録された異常: ' + s.last_anomaly + '</div>';
    host.innerHTML = msg + '</div>';
    return;
  }
  host.innerHTML = items.map(a => {
    const sev = a.severity;
    return '<div class="acard ' + sev + (SELECTED === a.id ? ' sel' : '') +
      '" onclick="select(\'' + a.id + '\')">' +
      '<div class="count ' + sev + '"><div class="n">' + a.count + '</div>' +
      '<div class="of">/ ' + a.max_count + ' checks</div></div>' +
      '<div class="ainfo"><div class="rec">' + a.record + '</div>' +
      '<div class="loc">' + a.ring + ' ' + a.section + (a.place ? ' · ' + a.place : '') + ' · ' + a.period + '</div>' +
      '<span class="cause ' + sev + '">推定: ' + a.cause + '</span></div>' +
      spark(a.series, sevColor(sev), 120, 38) + '</div>';
  }).join("");
}

function renderDetail() {
  const host = document.getElementById("detail-host");
  if (!SELECTED) { host.innerHTML = ""; return; }
  const a = STATE.anomalies.find(x => x.id === SELECTED);
  if (!a) { host.innerHTML = ""; return; }
  const place = a.place ? (a.place + " · ") : "";
  const hasSeries = a.series && a.series.pressure && a.series.pressure.length > 0;
  const plots = hasSeries
    ? ('<div class="plotrow">' +
        '<div class="plotbox"><div class="cap">圧力 vs フィル（直近）[Pa]</div>' +
          linePlot(a.series.t, a.series.pressure, "#e2574a", 640, 150, fmtPres) + '</div>' +
        '<div class="plotbox"><div class="cap">ビーム電流 vs フィル（直近）[mA]</div>' +
          linePlot(a.series.t, a.series.beam, "#5dcaa5", 640, 110, v => Math.round(v)) + '</div>' +
      '</div>')
    : ('<div class="plotbox" style="text-align:center;color:var(--muted);padding:22px">' +
        '圧力トレンドは次段階で接続予定</div>');
  host.innerHTML =
    '<div class="detail"><h3>' + a.record + '</h3>' +
    '<div class="sub">' + a.ring + ' ' + a.section + ' · ' + place +
      a.period + ' 部 · 推定原因: ' + a.cause + '</div>' +
    plots +
    '<div class="btns">' +
      '<button onclick="saveLabel(\'Normal\')">Normal として保存</button>' +
      '<button onclick="saveLabel(\'Abnormal\')">Abnormal として保存</button>' +
      '<button class="close" onclick="select(null)">閉じる</button>' +
    '</div></div>';
  host.scrollIntoView({behavior: "smooth", block: "nearest"});
}

function linePlot(t, y, color, w, h, fmt) {
  const pad = 38, padR = 10, padT = 10, padB = 22;
  const min = Math.min.apply(null, y), max = Math.max.apply(null, y);
  const rng = (max - min) || 1;
  const X = i => pad + (i / (t.length - 1)) * (w - pad - padR);
  const Y = v => padT + (1 - (v - min) / rng) * (h - padT - padB);
  const pts = y.map((v, i) => X(i).toFixed(1) + "," + Y(v).toFixed(1)).join(" ");
  let svg = '<svg width="100%" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" style="display:block">';
  for (let g = 0; g <= 2; g++) {
    const yy = padT + g * (h - padT - padB) / 2;
    const val = max - g * rng / 2;
    svg += '<line x1="' + pad + '" y1="' + yy + '" x2="' + (w - padR) + '" y2="' + yy +
           '" stroke="#2c313d" stroke-width="1"/>';
    svg += '<text x="4" y="' + (yy + 4) + '" fill="#6b7080" font-size="11">' + fmt(val) + '</text>';
  }
  svg += '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2"/>';
  svg += '</svg>';
  return svg;
}

function renderMap(s) {
  const host = document.getElementById("map");
  host.innerHTML = mapRow("LER", s.sections.LER) + mapRow("HER", s.sections.HER);
}
function mapRow(ring, secs) {
  let cells = "";
  for (let i = 1; i <= 12; i++) {
    const key = "D" + String(i).padStart(2, "0");
    const sev = secs[key];
    const onClick = sev ? ' onclick="selectSection(\'' + ring + '\',\'' + key + '\')"' : '';
    cells += '<div class="cell ' + (sev || '') + '"' + onClick + '>' + key + '</div>';
  }
  return '<div class="maprow"><div class="rl">' + ring + '</div><div class="cells">' + cells + '</div></div>';
}

function select(id) { SELECTED = id; renderList(STATE); renderDetail(); }
function selectSection(ring, sec) {
  const a = STATE.anomalies.find(x => x.ring === ring && x.section === sec);
  if (a) select(a.id);
}
function saveLabel(kind) {
  // プロトタイプではダミー。実運用では検知側の Save 機能に POST して
  // *_FNN_{Normal,Abnormal}_* に追記する想定。
  const a = STATE.anomalies.find(x => x.id === SELECTED);
  alert("[プロトタイプ] " + (a ? a.record : "") + " を " + kind + " として保存（実運用では学習データに追記）");
}

async function refreshLoad() {
  try {
    const r = await fetch("/api/sysload", {cache: "no-store"});
    const d = await r.json();
    const el = document.getElementById("sysload");
    el.className = "pill load-" + (d.level || "ok");
    if (!d.loadavg) { el.textContent = "負荷 取得不可"; el.title = ""; return; }
    const parts = ["負荷 " + d.loadavg[0]];
    if (d.cpu_percent !== null && d.cpu_percent !== undefined)
      parts.push("CPU " + d.cpu_percent + "%");
    if (d.mem_percent !== null && d.mem_percent !== undefined)
      parts.push("Mem " + d.mem_percent + "%");
    el.textContent = parts.join(" ⋅ ");
    el.title = "load avg " + d.loadavg.join(" / ") + "（1/5/15分）, " +
               d.ncpu + " コア, 比 " + d.load_ratio;
    // このプログラム自身（＋子プロセス）の使用量
    const se = document.getElementById("selfload");
    if (se && d.self_mem_mb !== undefined && d.self_mem_mb !== null) {
      const sc = (d.self_cpu_percent === null || d.self_cpu_percent === undefined)
                 ? "—" : (d.self_cpu_percent + "%");
      se.textContent = "本プログラム CPU " + sc + " ⋅ Mem " + d.self_mem_mb + " MB";
      se.title = "このダッシュボード（＋子プロセス " + d.self_nproc + " 個）自身の使用量。" +
                 "CPU% はサーバー全体に対する割合なので、左の全体CPU%と直接比べられる。";
    }
  } catch (e) {
    const el = document.getElementById("sysload");
    el.className = "pill"; el.textContent = "負荷 —";
  }
}

// ── イオンポンプ放電電流（開発中・判定なし）──────────────────────
let IP_SEL = {};   // {ring: section} 展開中のセクション

function fmtCur(v) {
  if (v === null || v === undefined) return "—";
  return v.toExponential(2) + " A";
}

// 放電電流は桁が広い(1e-10〜1e-6)ので対数スケール。null は線を切る（欠測）。
function sparkIP(trend, color, w, h) {
  const logs = trend.map(v => (v === null || v <= 0) ? null : Math.log10(v));
  const valid = logs.filter(x => x !== null);
  if (!valid.length) return '<svg width="' + w + '" height="' + h + '"></svg>';
  let lo = Math.min.apply(null, valid), hi = Math.max.apply(null, valid);
  const rng = (hi - lo) || 1;
  let segs = [], cur = [];
  logs.forEach((x, i) => {
    if (x === null) { if (cur.length) { segs.push(cur); cur = []; } return; }
    const px = (i / (logs.length - 1)) * w;
    const py = h - ((x - lo) / rng) * (h - 4) - 2;
    cur.push(px.toFixed(1) + "," + py.toFixed(1));
  });
  if (cur.length) segs.push(cur);
  const lines = segs.map(s =>
    '<polyline points="' + s.join(" ") + '" fill="none" stroke="' + color +
    '" stroke-width="1.5"/>').join("");
  return '<svg class="spark" width="' + w + '" height="' + h +
         '" viewBox="0 0 ' + w + ' ' + h + '">' + lines + '</svg>';
}

function renderIonPumps(s) {
  const host = document.getElementById("ionpumps");
  const ip = s.ion_pumps;
  if (!ip || !ip.rings) { host.innerHTML = ""; return; }   // データ無ければ非表示
  let html = '<div class="section-title">イオンポンプ 放電電流' +
             '<span class="devtag">開発中・判定なし</span></div>' +
             '<div class="ipmeta">取得: ' + (ip.updated || "—") +
             ' ／ 間隔 ' + (ip.interval_sec || "?") + ' 秒 ／ 値が大きい色ほど放電電流が大きい（相対）</div>';
  ["LER", "HER"].forEach(ring => {
    const r = ip.rings[ring];
    if (!r) return;
    html += '<div class="iprow"><div class="iplabel">' + ring + '</div><div class="ipgrid">';
    // 色スケール用: このリングの max_latest の最大
    const maxs = r.sections.map(x => x.max_latest || 0);
    const gmax = Math.max.apply(null, maxs.concat([1e-12]));
    r.sections.forEach(sec => {
      const frac = sec.max_latest ? Math.min(1, Math.log10(sec.max_latest / 1e-10) / 4) : 0;
      const sel = (IP_SEL[ring] === sec.section) ? " sel" : "";
      html += '<div class="ipcard' + sel + '" onclick="toggleIP(\'' + ring + '\',\'' +
              sec.section + '\')">' +
              '<div class="ipsec">' + sec.section +
              (sec.supply === "Agilent_4U" ? '<span class="badge b4u">4U</span>' : '') +
              '</div>' +
              '<div class="ipbar"><span style="width:' + (frac * 100).toFixed(0) + '%"></span></div>' +
              '<div class="ipval">' + fmtCur(sec.max_latest) + '</div>' +
              '<div class="ipn">' + sec.n_active + '/' + sec.n + ' 本</div>' +
              '</div>';
    });
    html += '</div></div>';
    // 展開: 選択セクションの PV 一覧＋トレンド
    if (IP_SEL[ring]) {
      const sec = r.sections.find(x => x.section === IP_SEL[ring]);
      if (sec) {
        html += '<div class="ippvs"><div class="ippvs-h">' + ring + ' ' + sec.section +
                '（' + (sec.supply === "Agilent_4U" ? "Agilent 4U" : "KEK") + '／' +
                sec.n + ' 本）</div>';
        sec.pvs.forEach(p => {
          html += '<div class="ippv"><span class="ippv-name">' + p.pv + '</span>' +
                  sparkIP(p.trend, "#4a90d9", 120, 26) +
                  '<span class="ippv-val">' + fmtCur(p.latest) + '</span></div>';
        });
        html += '</div>';
      }
    }
  });
  host.innerHTML = html;
}

function toggleIP(ring, section) {
  IP_SEL[ring] = (IP_SEL[ring] === section) ? null : section;
  renderIonPumps(STATE);
}

async function refresh() {
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    STATE = await r.json();
    if (STATE.error) { document.getElementById("footnote").textContent = STATE.error; return; }
    renderTop(STATE); renderSummary(STATE); renderList(STATE);
    renderMap(STATE); renderDetail(); renderIonPumps(STATE);
    document.getElementById("footnote").textContent =
      "5 秒ごとに自動更新。データ源: " + (STATE._source || "dashboard_state.json / ダミー");
  } catch (e) {
    document.getElementById("footnote").textContent = "更新失敗: " + e;
  }
}
refresh();
setInterval(refresh, 5000);
refreshLoad();
setInterval(refreshLoad, 5000);
refreshBeam();
setInterval(refreshBeam, 3000);
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------------
# HTTP サーバー
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/api/beam"):
            data = beamcurrent.read() if beamcurrent is not None else {"LER": None, "HER": None}
            self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/api/sysload"):
            if sysload is not None:
                body = json.dumps(sysload.snapshot(_cpu_sampler, _proc_sampler), ensure_ascii=False)
            else:
                body = json.dumps({"level": "ok", "loadavg": None})
            self._send(200, body.encode("utf-8"), "application/json; charset=utf-8")
        elif self.path.startswith("/api/state"):
            state = load_state()
            state["_source"] = "dashboard_state.json" if os.path.isfile(STATE_FILE) else "内蔵ダミーデータ"
            self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args):
        pass  # アクセスログを抑制

def main():
    # 二重起動防止（共用サーバーで複数立ち上げを防ぐ）
    try:
        import singleton
        lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dashboard.lock")
        if not singleton.guard(lock, "dashboard.py", "ダッシュボード"):
            return
    except Exception:
        pass  # singleton が無くても起動はできる（ポート占有でも二重起動は実質防がれる）

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as ex:
        print("ポート %d を開けませんでした（既に使用中かもしれません）: %s" % (PORT, ex))
        return
    src = "dashboard_state.json" if os.path.isfile(STATE_FILE) else "内蔵ダミーデータ"
    print("圧力異常モニター（プロトタイプ）起動")
    print("  データ源: %s" % src)
    print("  ローカル:  http://localhost:%d" % PORT)
    print("  停止: Ctrl-C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")
        srv.shutdown()

if __name__ == "__main__":
    main()
