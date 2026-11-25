#!/usr/bin/env python3
"""从 ModelScope 下载模型快照到指定目录。

用法:
    python scripts/download_ms_model.py <repo_id> <local_dir> [--revision master] [--jobs N]

特性:
  - 通过 ModelScope open API 获取文件清单(含 sha256)
  - 使用 aria2c 多连接并发下载(每文件 16 连接), 断点续传
  - 已存在且 sha256 匹配的文件直接跳过
  - 下载完成后逐个校验 sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev}"
FILE_URL = (
    "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision={rev}&FilePath={path}"
)


def fetch_file_list(repo: str, revision: str) -> list[dict]:
    url = API.format(repo=repo, rev=revision)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    if not data.get("Success"):
        raise RuntimeError(f"获取文件列表失败: {data.get('Message')}")
    files = [f for f in data["Data"]["Files"] if f.get("Type") == "blob"]
    return files


def sha256_of(path: str, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_complete(path: str, size: int, sha256: str | None) -> bool:
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) != size:
        return False
    if not sha256:
        return True
    return sha256_of(path) == sha256


def aria2_download(url: str, out_dir: str, out_name: str, log_dir: str) -> None:
    cmd = [
        "aria2c",
        "-x", "16",
        "-s", "16",
        "-k", "4M",
        "--continue=true",
        "--allow-overwrite=false",
        "--auto-file-renaming=false",
        "--max-tries=10",
        "--retry-wait=5",
        "--timeout=60",
        "--connect-timeout=30",
        "--summary-interval=30",
        "--console-log-level=warn",
        "--log-level=warn",
        f"--log={os.path.join(log_dir, out_name + '.aria2.log')}",
        "-d", out_dir,
        "-o", out_name,
        url,
    ]
    # ModelScope 国内直连远快于走代理, 显式清空代理变量
    env = dict(os.environ)
    for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(k, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("local_dir")
    ap.add_argument("--revision", default="master")
    args = ap.parse_args()

    repo, local_dir = args.repo, os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    log_dir = os.path.join(local_dir, ".dl_logs")
    os.makedirs(log_dir, exist_ok=True)

    files = fetch_file_list(repo, args.revision)
    total = sum(f["Size"] for f in files)
    print(f"[{repo}] {len(files)} files, {total / 1e9:.2f} GB -> {local_dir}", flush=True)

    t0 = time.time()
    failed: list[str] = []
    for i, f in enumerate(files, 1):
        path, size, sha = f["Path"], f["Size"], f.get("Sha256")
        dst = os.path.join(local_dir, path)
        os.makedirs(os.path.dirname(dst) or local_dir, exist_ok=True)

        if is_complete(dst, size, sha):
            print(f"  [{i}/{len(files)}] SKIP {path}", flush=True)
            continue

        url = FILE_URL.format(repo=repo, rev=args.revision, path=urllib.parse.quote(path))
        print(f"  [{i}/{len(files)}] GET  {path} ({size / 1e9:.3f} GB)", flush=True)
        try:
            aria2_download(url, os.path.dirname(dst) or local_dir, os.path.basename(dst), log_dir)
        except subprocess.CalledProcessError as e:
            print(f"  !! aria2 failed on {path}: {e}", flush=True)
            failed.append(path)
            continue

        if not is_complete(dst, size, sha):
            print(f"  !! checksum/size mismatch after download: {path}", flush=True)
            failed.append(path)

    # 清理 aria2 控制文件
    for root, _, names in os.walk(local_dir):
        for n in names:
            if n.endswith(".aria2"):
                os.remove(os.path.join(root, n))

    dt = time.time() - t0
    print(f"[{repo}] DONE in {dt / 60:.1f} min, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
