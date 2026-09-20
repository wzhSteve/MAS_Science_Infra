#!/usr/bin/env python3
"""AutoDL 公网访问转发器: 0.0.0.0:6006 -> 127.0.0.1:8787

AutoDL 容器只代理 6006/6008 端口到公网 (https://<uuid>.<region>.seetacloud.com:8443)。
Science Control UI 跑在容器内 8787，不在代理列表 → 从外网打不开。

本脚本在 6006 上监听（AutoDL 代理转发目标），把流量转发给 127.0.0.1:8787。
纯标准库实现，SSE (Server-Sent Events) 长连接同样支持（单向透传，不缓存整个响应）。

用法:
    nohup python3 scripts/ui_public_forwarder.py >/tmp/ui_forwarder.log 2>&1 &
    # 停止: pkill -f ui_public_forwarder
"""
import socket
import select
import threading
import sys
import os

LISTEN_PORT = int(os.environ.get("FORWARD_LISTEN_PORT", "6006"))
TARGET_HOST = os.environ.get("FORWARD_TARGET_HOST", "127.0.0.1")
TARGET_PORT = int(os.environ.get("FORWARD_TARGET_PORT", "8787"))
BUF = 65536


def pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(BUF)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client: socket.socket) -> None:
    try:
        upstream = socket.create_connection((TARGET_HOST, TARGET_PORT), timeout=15)
    except OSError as e:
        client.sendall(
            b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain; charset=utf-8\r\n"
            b"Connection: close\r\nContent-Length: 60\r\n\r\n"
            b"Science Control (127.0.0.1:8787) unreachable: " + str(e).encode()[:30] + b"\n"
        )
        client.close()
        return
    # 双向透传; SSE 靠 socket 直通, 数据一到就转, 无整体缓冲
    t = threading.Thread(target=pipe, args=(client, upstream), daemon=True)
    t.start()
    pipe(upstream, client)
    t.join(timeout=5)
    try:
        client.close()
        upstream.close()
    except OSError:
        pass


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", LISTEN_PORT))
    except OSError as e:
        print(f"[forwarder] cannot bind :{LISTEN_PORT}: {e}", file=sys.stderr)
        sys.exit(1)
    srv.listen(64)
    print(f"[forwarder] 0.0.0.0:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}", flush=True)
    while True:
        client, addr = srv.accept()
        print(f"[forwarder] conn from {addr[0]}:{addr[1]}", flush=True)
        threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
