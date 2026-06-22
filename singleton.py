#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
singleton.py — 二重起動を防ぐ PID ロック（標準ライブラリのみ）

共用サーバーで同じプログラムが複数同時に立ち上がらないようにする。
起動時に acquire() を呼び、既に生きている同名プロセスがロックを持っていれば
AlreadyRunning を送出する。前回が異常終了してロックが残っていても（PIDが死んで
いる or 別プログラムに再利用されている場合）、古いロックとみなして掃除して起動する。

使い方:
    import singleton
    try:
        singleton.acquire(lockpath, tag="dashboard.py")
    except singleton.AlreadyRunning as e:
        print("既に PID %d で起動中です。" % e.pid)
        sys.exit(1)
"""

import os
import atexit


class AlreadyRunning(Exception):
    def __init__(self, pid):
        super().__init__("already running (pid=%s)" % pid)
        self.pid = pid


def _alive(pid):
    """pid のプロセスが存在するか（シグナル0を送るだけ）。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # 存在するが権限が無い＝生きている
    except OSError:
        return False
    return True


def _cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _holds_lock(pid, tag):
    """pid が生きていて、かつ我々のプログラム（tag）であるか。
    /proc が無い環境では cmdline 照合をスキップし、生存だけで判定する。"""
    if not _alive(pid):
        return False
    if tag and os.path.isdir("/proc"):
        cl = _cmdline(pid)
        if cl and tag not in cl:
            return False   # PID は生きているが別プログラム＝古いロックの残骸
    return True


def acquire(lockpath, tag=""):
    """ロックを取得する。既に同名プロセスが保持していれば AlreadyRunning。"""
    if os.path.exists(lockpath):
        old = None
        try:
            with open(lockpath, encoding="ascii") as f:
                old = int(f.read().strip().split()[0])
        except (OSError, ValueError, IndexError):
            old = None
        if old is not None and _holds_lock(old, tag):
            raise AlreadyRunning(old)
        # 残骸ロック → 掃除
        try:
            os.remove(lockpath)
        except OSError:
            pass

    with open(lockpath, "w", encoding="ascii") as f:
        f.write("%d %s\n" % (os.getpid(), tag))
    atexit.register(release, lockpath)
    return True


def release(lockpath):
    """自分が持っているロックだけを消す。"""
    try:
        with open(lockpath, encoding="ascii") as f:
            pid = int(f.read().strip().split()[0])
        if pid == os.getpid():
            os.remove(lockpath)
    except (OSError, ValueError, IndexError):
        pass


def guard(lockpath, tag, label):
    """acquire のラッパー。二重起動なら親切なメッセージを出して False を返す。"""
    try:
        acquire(lockpath, tag)
        return True
    except AlreadyRunning as e:
        print("%s は既に起動中です（PID %d）。二重起動はできません。" % (label, e.pid))
        print("  本当に起動したい場合は、先に動いている方を止めるか、")
        print("  残骸なら %s を削除してください。" % lockpath)
        return False
