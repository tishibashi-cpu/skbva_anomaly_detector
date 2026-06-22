#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sysload.py — 共用計算機サーバーの負荷状況を取得する（標準ライブラリのみ）

dashboard が定期的に呼び、ヘッダに「ロードアベレージ・CPU使用率・メモリ使用率」を
表示するためのモジュール。psutil 等の追加依存は使わず、os.getloadavg() と /proc を読む。
読み取り自体が共用機に負荷をかけないよう、軽い読み込みのみ。失敗しても None を返して
ダッシュボードを壊さない。

負荷指標の考え方:
  ロードアベレージ(1分) を CPU コア数で割った比 load_ratio が最も分かりやすい目安。
    load_ratio < 0.7 : 余裕         -> level "ok"
    0.7 <= ratio < 1.0: やや高い     -> level "warn"
    ratio >= 1.0      : コア数を超過 -> level "high"
  CPU% / メモリ% も併せて level を引き上げる。
"""

import os


def loadavg():
    """(1分, 5分, 15分) のロードアベレージ。取得不可なら None。"""
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return None


def ncpu():
    return os.cpu_count() or 1


def meminfo():
    """(total_mb, used_mb, percent) を返す。/proc/meminfo から。失敗なら None。"""
    try:
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.strip().split()[0])  # kB
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total is None or avail is None or total == 0:
            return None
        used = total - avail
        return (total / 1024.0, used / 1024.0, 100.0 * used / total)
    except (OSError, ValueError, IndexError):
        return None


class CpuSampler:
    """/proc/stat を前回値と比較してシステム全体のCPU使用率[%]を出す。

    初回呼び出しは基準が無いため None。以降は「前回呼び出しからの区間」のCPU%。
    dashboard が 5 秒間隔で呼べば、5 秒窓の実使用率になる。
    """

    def __init__(self):
        self._prev = None  # (total, idle)

    def _read(self):
        with open("/proc/stat", encoding="ascii") as f:
            parts = f.readline().split()
        # parts[0] == 'cpu', 以降 user nice system idle iowait irq softirq steal ...
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals)
        return total, idle

    def percent(self):
        try:
            total, idle = self._read()
        except (OSError, ValueError, IndexError):
            return None
        prev = self._prev
        self._prev = (total, idle)
        if prev is None:
            return None
        dt = total - prev[0]
        di = idle - prev[1]
        if dt <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (dt - di) / dt))


def _proc_ticks(pid):
    """/proc/<pid>/stat の utime+stime（clock ticks）。"""
    with open("/proc/%d/stat" % pid, encoding="ascii") as f:
        data = f.read()
    rp = data.rfind(")")              # comm に空白/括弧があるので最後の ')' 以降を見る
    fields = data[rp + 2:].split()
    return int(fields[11]) + int(fields[12])   # utime(14), stime(15)


def _proc_rss_kb(pid):
    with open("/proc/%d/status" % pid, encoding="ascii") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])    # kB
    return 0


def _descendants(pid):
    """pid の子孫プロセス PID 一覧（detector の kblogrd 等を含めるため）。"""
    kids = {}
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % e, encoding="ascii") as f:
                data = f.read()
            ppid = int(data[data.rfind(")") + 2:].split()[1])   # ppid(4)
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(e))
    out, stack = [], [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


class ProcSampler:
    """自プロセス（＋子プロセス）の CPU%・RSS を出す。CPU% はシステム全体に対する割合
    （/api/sysload の system CPU% と同じ分母）なので「全体のうち自分の寄与」を直接比較できる。"""

    def __init__(self, pid=None):
        self.pid = pid or os.getpid()
        self._prev = None   # (proc_ticks, total_ticks)

    def _total_ticks(self):
        with open("/proc/stat", encoding="ascii") as f:
            return sum(int(x) for x in f.readline().split()[1:])

    def sample(self):
        """(cpu_percent, rss_mb, nproc) を返す。初回 CPU% は None。"""
        pids = [self.pid] + _descendants(self.pid)
        proc = 0
        rss = 0
        for p in pids:
            try:
                proc += _proc_ticks(p)
            except (OSError, ValueError, IndexError):
                pass
            try:
                rss += _proc_rss_kb(p)
            except (OSError, ValueError):
                pass
        try:
            total = self._total_ticks()
        except (OSError, ValueError):
            return None, rss / 1024.0, len(pids)
        prev = self._prev
        self._prev = (proc, total)
        cpu = None
        if prev is not None and total - prev[1] > 0:
            cpu = max(0.0, min(100.0, 100.0 * (proc - prev[0]) / (total - prev[1])))
        return cpu, rss / 1024.0, len(pids)


def _level(load_ratio, cpu_pct, mem_pct):
    lvl = "ok"
    def bump(to):
        nonlocal lvl
        order = {"ok": 0, "warn": 1, "high": 2}
        if order[to] > order[lvl]:
            lvl = to
    if load_ratio is not None:
        if load_ratio >= 1.0: bump("high")
        elif load_ratio >= 0.7: bump("warn")
    if cpu_pct is not None:
        if cpu_pct >= 90: bump("high")
        elif cpu_pct >= 75: bump("warn")
    if mem_pct is not None:
        if mem_pct >= 90: bump("high")
        elif mem_pct >= 75: bump("warn")
    return lvl


def snapshot(sampler=None, proc_sampler=None):
    """負荷状況の辞書を返す。dashboard の /api/sysload がこれを JSON で返す。"""
    la = loadavg()
    n = ncpu()
    cpu_pct = sampler.percent() if sampler is not None else None
    mem = meminfo()
    mem_pct = mem[2] if mem else None
    load_ratio = (la[0] / n) if la else None
    out = {
        "loadavg": [round(x, 2) for x in la] if la else None,
        "ncpu": n,
        "load_ratio": round(load_ratio, 2) if load_ratio is not None else None,
        "cpu_percent": round(cpu_pct, 1) if cpu_pct is not None else None,
        "mem_total_mb": round(mem[0]) if mem else None,
        "mem_used_mb": round(mem[1]) if mem else None,
        "mem_percent": round(mem_pct, 1) if mem_pct is not None else None,
        "level": _level(load_ratio, cpu_pct, mem_pct),
    }
    # 自プロセス（＋子プロセス）の寄与
    if proc_sampler is not None:
        scpu, srss, nproc = proc_sampler.sample()
        mem_total = mem[0] if mem else None
        out["self_cpu_percent"] = round(scpu, 1) if scpu is not None else None
        out["self_mem_mb"] = round(srss)
        out["self_mem_percent"] = (round(100.0 * srss / mem_total, 1)
                                   if mem_total else None)
        out["self_nproc"] = nproc
    return out


def _snapshot(sampler=None):   # 後方互換
    return snapshot(sampler)


if __name__ == "__main__":
    import json
    import time
    s = CpuSampler()
    ps = ProcSampler()
    s.percent(); ps.sample()      # 基準取り
    time.sleep(1.0)
    print(json.dumps(snapshot(s, ps), ensure_ascii=False, indent=2))
