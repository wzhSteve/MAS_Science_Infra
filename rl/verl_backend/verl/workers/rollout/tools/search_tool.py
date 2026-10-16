import os
import json
import time
import queue
import atexit
import fcntl
import hashlib
import pathlib
import random
import threading
import requests
from typing import Optional, Union, Dict, List, Any
from urllib.parse import urlencode

import langid

from verl.workers.rollout.tools.base_tool import BaseTool


# --------------------------------------------------------------------
# 账户级速率：serp_api1 控制台写明 100 req/min，验证前不会更高。
# 两个 Ray worker 是两个进程，进程内锁盖不住合计速率，所以用文件锁共享一个窗口。
# 默认 80/min，给重试留出余量。同一条 query 在飞时，其它线程/进程只等缓存。
# --------------------------------------------------------------------
class _KeyThrottle:
    """单 key 节流器：令牌间隔 + 限流冷却时间."""

    def __init__(self, min_interval: float, max_cooldown: float):
        self.min_interval = min_interval
        self.max_cooldown = max_cooldown
        self._lock = threading.Lock()
        self._next_slot = time.time()
        self._limit_until = 0.0

    def acquire(self, extra_cooldown: float = 0.0) -> None:
        """阻塞至本 key 允许发下一个请求。extra_cooldown>0 表示刚被限流，追加冷却
        （只在进入时施加一次，不参与后续循环的 ready 计算）。"""
        pending_cooldown = extra_cooldown
        while True:
            with self._lock:
                now = time.time()
                if pending_cooldown > 0:
                    # 一次性把冷却截止时刻定死，避免循环里反复后移（追及死循环）
                    self._limit_until = max(self._limit_until, now + pending_cooldown)
                    pending_cooldown = 0.0
                ready = max(self._next_slot, self._limit_until)
                if now >= ready:
                    self._next_slot = now + self.min_interval
                    return
                wait = ready - now
            time.sleep(min(wait, 5.0))

    def is_cooling(self) -> bool:
        with self._lock:
            return time.time() < self._limit_until


class _AccountRateGate:
    """跨进程滑动窗口。所有 worker 共用一个文件，保证账户合计不超过 max_per_minute。"""

    def __init__(self, path: str, max_per_minute: int, window_seconds: float = 60.0):
        self.path = path
        self.max_per_minute = int(max_per_minute)
        self.window = float(window_seconds)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    def acquire(self) -> None:
        if self.max_per_minute <= 0:
            return
        lock_path = self.path + ".lock"
        while True:
            wait = 0.05
            with open(lock_path, "a+") as lockf:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
                try:
                    now = time.time()
                    stamps = self._read(now)
                    if len(stamps) < self.max_per_minute:
                        stamps.append(now)
                        self._write(stamps)
                        return
                    wait = max(0.05, self.window - (now - stamps[0]))
                finally:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            if wait >= 0.5:
                print(f"[SearchTool] account rate gate wait {wait:.1f}s "
                      f"({self.max_per_minute}/min shared)")
            time.sleep(min(wait, 1.0))

    def _read(self, now: float) -> list:
        file_path = pathlib.Path(self.path)
        if not file_path.exists():
            return []
        try:
            data = json.loads(file_path.read_text() or "[]")
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return []
        kept = []
        for item in data:
            try:
                stamp = float(item)
            except (TypeError, ValueError):
                continue
            if now - stamp < self.window:
                kept.append(stamp)
        kept.sort()
        return kept

    def _write(self, stamps: list) -> None:
        tmp = self.path + ".tmp"
        pathlib.Path(tmp).write_text(json.dumps(stamps))
        os.replace(tmp, self.path)


class BingSearchTool(BaseTool):
    """
    Bing search tool that provides web search capability with caching.
    
    This tool interfaces with the Brightdata API to perform Bing searches.
    It includes robust caching to minimize redundant API calls and supports
    both synchronous and asynchronous cache writing modes.
    
    Thread-safety is ensured via memory locks, and process-safety via file locks
    for the cache file.

    在此基础上：
    1. 账户级共享窗口（默认 80 req/min，控制台上限是 100），两个 GPU 进程一起计数；
    2. 同一条 query 只允许一个 in-flight 请求，其它调用等缓存；
    3. 限流后退避 8/16/32s，并加最多 25% 的随机抖动，避免两个进程同时醒来。
    """

    # 进程级共享：key -> _KeyThrottle（Ray worker 内多实例/多线程共用）
    _THROTTLES: Dict[str, "_KeyThrottle"] = {}
    _THROTTLES_LOCK = threading.Lock()

    def __init__(
        self,
        api_key: str,
        zone: str = "web_unlocker1",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        cache_file: Optional[str] = None,
        async_cache_write: bool = True,
        cache_refresh_interval: float = 15.0,
        param_base_url: str = "https://api.brightdata.com",
        api_keys: str = "",
        min_request_interval: float = 3.0,
        backoff_base: float = 8.0,
        backoff_max: float = 120.0,
        max_api_retries: int = 4,
        rl_status_codes: str = "429,502,503",
        max_requests_per_minute: int = 80,
        rate_window_seconds: float = 60.0,
    ):
        """
        Initialize the Bing search tool.
        
        Args:
            api_key: Brightdata API key
            zone: Brightdata zone name (e.g. serp_api1)
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
            location: Country code for search localization
            cache_file: Path to cache file (if None, uses ~/.verl_cache/bing_search_cache.json)
            async_cache_write: Whether to write cache updates asynchronously
            cache_refresh_interval: Minimum seconds between cache file checks
            param_base_url: Brightdata API base URL（直连 https://api.brightdata.com；
                             若需经公司代理网关，则填网关地址并通过 X-Dst-* 头转发）
            api_keys: 逗号分隔的额外 key 列表（与 api_key 合并去重后轮换使用；
                      每个独立账号的 key 配额独立，吞吐随 key 数线性扩展）
            min_request_interval: 同一进程、同一 key 的额外最小间隔（秒）。
                      账户总速率由 max_requests_per_minute 限制；双卡脚本把它设为 0。
            max_requests_per_minute: 所有进程合计的请求上限。serp_api1 当前是 100/min，
                      默认 80 给重试留余量。<=0 表示关闭这道跨进程门。
            rate_window_seconds: 滑动窗口长度，默认 60 秒。
            backoff_base: 限流退避基数（秒），第 n 次连续限流冷却 backoff_base*2^(n-1)
            backoff_max: 单次限流冷却的上限（秒）
            max_api_retries: execute() 内部限流感知重试次数（退避 8/16/32/64/120s）；
                      这与外层 tool_executor 的"立即重试"不同——每次重试之间会冷却
            rl_status_codes: 上游 body 里视为"限流"的 status_code 集合（逗号分隔）
        """
        # API configuration —— 多 key 轮换
        keys = [k.strip() for k in (api_keys or "").split(",") if k.strip()]
        if api_key:
            keys.append(api_key.strip())
        seen = set()
        self._api_keys = [k for k in keys if not (k in seen or seen.add(k))]
        if not self._api_keys:
            self._api_keys = [""]  # 占位，请求会 401 由上层报错
        self._key_idx = -1
        self._key_lock = threading.Lock()
        self._current_key = self._api_keys[0]  # execute() 里每次请求前会轮换更新

        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        # Brightdata API 基地址：默认直连官方端点；也可指向内网代理网关
        self._api_base_url = param_base_url.rstrip("/")
        self._use_proxy_gateway = self._api_base_url != "https://api.brightdata.com"

        # 节流/退避参数
        self._min_request_interval = float(min_request_interval)
        self._backoff_base = float(backoff_base)
        self._backoff_max = float(backoff_max)
        self._max_api_retries = int(max_api_retries)
        self._rl_status_codes = set(
            int(s) for s in str(rl_status_codes).split(",") if s.strip()
        )
        self._max_requests_per_minute = int(max_requests_per_minute)
        self._rate_window_seconds = float(rate_window_seconds)
        # 进程级注册每个 key 的节流器（多实例/多线程共享）
        with BingSearchTool._THROTTLES_LOCK:
            for k in self._api_keys:
                if k not in BingSearchTool._THROTTLES:
                    BingSearchTool._THROTTLES[k] = _KeyThrottle(
                        self._min_request_interval, self._backoff_max
                    )
        
        # Cache and synchronization
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._async_cache_write = async_cache_write
        self._write_queue = queue.Queue() if async_cache_write else None
        self._cache_refresh_interval = cache_refresh_interval
        self._last_cache_check = 0.0
        self._cache_mod_time = 0.0
        
        # Setup cache file paths
        self._setup_cache_paths(cache_file)
        self._rate_gate = _AccountRateGate(
            str(self._cache_file) + ".rpm",
            self._max_requests_per_minute,
            self._rate_window_seconds,
        )
        self._inflight_dir = pathlib.Path(str(self._cache_file) + ".inflight")
        
        # Load existing cache
        self._load_cache()
        
        # Initialize async cache writer if enabled
        if self._async_cache_write:
            self._init_async_writer()
    
    def _setup_cache_paths(self, cache_file: Optional[str]) -> None:
        """
        Set up cache file and lock file paths.
        
        Args:
            cache_file: Path to cache file or None for default
        """
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / "bing_search_cache.json"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create lock file path
        self._lock_file = str(self._cache_file) + ".lock"
    
    def _init_async_writer(self) -> None:
        """Initialize the asynchronous cache writer thread."""
        self._stop_writer = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._cache_writer_thread, 
            daemon=True,
            name="BingSearchCacheWriter"
        )
        self._writer_thread.start()
        atexit.register(self._cleanup)
    
    def _cleanup(self) -> None:
        """Ensure cache is saved when program exits."""
        if self._async_cache_write and hasattr(self, '_stop_writer'):
            self._stop_writer.set()
            if self._writer_thread.is_alive():
                self._writer_thread.join(timeout=5.0)
            # Save any remaining cache updates
            self._save_cache_sync()
    
    def _cache_writer_thread(self) -> None:
        """Background thread for asynchronous cache writing."""
        while not self._stop_writer.is_set():
            try:
                # Wait for write requests, timeout after 1 second
                try:
                    _ = self._write_queue.get(timeout=1.0)
                    self._save_cache_sync()
                    self._write_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                print(f"Cache writer thread error: {str(e)}")
    
    def _acquire_file_lock(self, timeout: int = 10) -> Optional[Any]:
        """
        Acquire a file lock to prevent concurrent cache file access.
        
        Args:
            timeout: Maximum seconds to wait for lock acquisition
            
        Returns:
            File descriptor if lock acquired, None if failed
        """
        start_time = time.time()
        lock_fd = None
        
        try:
            # Create or open lock file
            lock_fd = open(self._lock_file, 'w+')
            
            while True:
                try:
                    # Try to acquire exclusive lock
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock_fd  # Successfully acquired lock
                except IOError:
                    # Could not immediately acquire lock
                    if time.time() - start_time > timeout:
                        lock_fd.close()
                        raise TimeoutError(f"Failed to acquire file lock within {timeout} seconds")
                    # Retry after a short delay
                    time.sleep(0.1)
        except Exception as e:
            if lock_fd:
                lock_fd.close()
            print(f"Failed to acquire file lock: {str(e)}")
            return None
    
    def _release_file_lock(self, lock_fd: Any) -> bool:
        """
        Release file lock.
        
        Args:
            lock_fd: File descriptor to release
            
        Returns:
            True if successfully released, False otherwise
        """
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                return True
            except Exception as e:
                print(f"Failed to release file lock: {str(e)}")
                return False
        return False
    
    def _load_cache(self) -> None:
        """Load the cache from disk with file locking."""
        if not self._cache_file.exists():
            return
            
        lock_fd = None
        try:
            # Acquire file lock for reading
            lock_fd = self._acquire_file_lock()
            if not lock_fd:
                print("Unable to acquire file lock, using empty cache")
                return
            
            # Record file modification time
            self._cache_mod_time = os.path.getmtime(self._cache_file)
            
            with open(self._cache_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)
            
            # Update in-memory cache
            with self._cache_lock:
                self._cache = file_data
            
            self._last_cache_check = time.time()
            print(f"Loaded {len(self._cache)} cache entries from {self._cache_file}")
        except json.JSONDecodeError:
            print(f"Cache file {self._cache_file} contains invalid JSON, using empty cache")
            self._cache = {}
        except Exception as e:
            print(f"Failed to load cache file: {str(e)}")
            self._cache = {}
        finally:
            if lock_fd:
                self._release_file_lock(lock_fd)
    
    def _check_cache_update(self, force: bool = False) -> bool:
        """
        Check if cache file has been updated by another process.
        
        Returns:
            True if cache was reloaded, False otherwise
        """
        now = time.time()
        # Limit check frequency to avoid excessive I/O. 发请求前 force=True，避免 15s 空窗里重复打 API。
        if not force and now - self._last_cache_check < self._cache_refresh_interval:
            return False
        
        self._last_cache_check = now
        
        if not self._cache_file.exists():
            return False
        
        try:
            current_mod_time = os.path.getmtime(self._cache_file)
            if current_mod_time > self._cache_mod_time:
                print(f"Cache file update detected, reloading")
                self._load_cache()
                return True
        except Exception as e:
            print(f"Failed to check cache file updates: {str(e)}")
        
        return False

    def _save_cache(self) -> None:
        """Save cache to disk, either synchronously or asynchronously."""
        if self._async_cache_write:
            # Queue write request for background thread
            try:
                self._write_queue.put(True, block=False)
            except queue.Full:
                print("Cache write queue full, skipping this update")
        else:
            # Direct synchronous write
            self._save_cache_sync()
    
    def _save_cache_sync(self) -> None:
        """Save cache to disk synchronously with file locking."""
        lock_fd = None
        try:
            # Acquire exclusive file lock
            lock_fd = self._acquire_file_lock()
            if not lock_fd:
                print("Unable to acquire file lock, skipping cache save")
                return
            
            # Create temporary file for atomic write
            temp_file = self._cache_file.with_suffix('.tmp')
            
            # Copy cache data to minimize lock time
            with self._cache_lock:
                cache_copy = dict(self._cache)
            
            # If cache file exists, read and merge with current cache
            merged_cache = self._merge_with_existing_cache(cache_copy)
            
            # Write to temp file and replace original (atomic operation)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(merged_cache, f, ensure_ascii=False, indent=2)
            
            temp_file.replace(self._cache_file)
            
            # Update modification time record
            self._cache_mod_time = os.path.getmtime(self._cache_file)
            print(f"Cache saved to {self._cache_file} with {len(merged_cache)} entries")
            
            # Update in-memory cache with merged data
            with self._cache_lock:
                self._cache = merged_cache
                
        except Exception as e:
            print(f"Failed to save cache file: {str(e)}")
        finally:
            if lock_fd:
                self._release_file_lock(lock_fd)
    
    def _merge_with_existing_cache(self, cache_copy: Dict) -> Dict:
        """
        Merge in-memory cache with existing cache file content.
        
        Args:
            cache_copy: Copy of the current in-memory cache
            
        Returns:
            Merged cache dictionary
        """
        merged_cache = cache_copy
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    existing_cache = json.load(f)
                # Update existing cache with new entries
                existing_cache.update(cache_copy)
                merged_cache = existing_cache
                print(f"Merged with existing cache, total entries: {len(merged_cache)}")
            except Exception as e:
                print(f"Failed to read existing cache file, using new cache: {str(e)}")
        return merged_cache

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        """Tag used to trigger this tool."""
        return "search"

    def _make_request(self, query: str, timeout: int) -> requests.Response:
        """
        Send request to Brightdata SERP API.

        默认直连 https://api.brightdata.com/request（zone=serp_api1），
        目标站为 Bing 搜索，返回 parsed_bing 格式的 JSON（含 organic 字段）。
        若 _api_base_url 指向公司内网代理网关，则附加 X-Dst-Host / X-Dst-Scheme 头转发。

        使用 self._current_key（由 execute() 的 _pick_key 轮换选出）做鉴权。

        Args:
            query: Search query
            timeout: API request timeout in seconds

        Returns:
            API response object
        """
        # Determine language settings based on query language
        lang_code, lang_confidence = langid.classify(query)

        # 目标站：Bing SERP（serp_api1 zone 对 bing 稳定；
        # google.com 在该 zone 上偶发 selector 超时，故不使用）
        from urllib.parse import urlencode
        if lang_code == 'zh':
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        encoded_query = urlencode({"q": query, "mkt": mkt, "setLang": setLang})
        target_url = f"https://www.bing.com/search?{encoded_query}"
        # 准备 headers：鉴权（轮换 key）+（可选）代理网关转发头
        headers = {
            "Authorization": f"Bearer {self._current_key}",
            "Content-Type": "application/json",
        }
        if self._use_proxy_gateway:
            headers.update({
                "X-Dst-Host": "api.brightdata.com",
                "X-Dst-Scheme": "https",
            })

        # parsed_bing 格式：返回结构化 JSON，字段与 Bing organic 兼容（title/link/description）
        payload = {
            "zone": self._zone,
            "url": target_url,
            "format": "json",
            "data_format": "parsed_bing",
        }

        request_url = f"{self._api_base_url}/request"
        print(f"[SearchTool] Sending request to: {request_url}, zone={self._zone}, query={query!r}")
        response = requests.post(
            request_url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        print(f"[SearchTool] Response status={response.status_code}, body_len={len(response.text)}, body_preview={repr(response.text[:200])}")
        return response

    # ------------------------------------------------------------ key 轮换
    def _pick_key(self) -> tuple:
        """轮换选一个 key：优先未冷却的；全部冷却则选冷却最快结束的（由 acquire 阻塞等待）."""
        with self._key_lock:
            self._key_idx = (self._key_idx + 1) % len(self._api_keys)
            order = [self._api_keys[(self._key_idx + i) % len(self._api_keys)]
                     for i in range(len(self._api_keys))]
        for k in order:
            th = BingSearchTool._THROTTLES.get(k)
            if th is not None and not th.is_cooling():
                return k, th
        # 全部冷却中：取第一个（acquire 会阻塞到它解冻）
        k = order[0]
        return k, BingSearchTool._THROTTLES[k]

    @staticmethod
    def _is_rate_limited(data: Any, rl_codes: set) -> bool:
        """识别 BrightData 限流信封：{status_code: 429/502/...}或 body 内 x-brd-error."""
        if not isinstance(data, dict):
            return False
        if data.get("status_code") in rl_codes:
            return True
        body = data.get("body")
        if isinstance(body, str) and ("exceeded the allowed rate limits" in body
                                     or "http_request_denied" in body):
            return True
        if isinstance(body, dict) and isinstance(body.get("headers"), dict):
            return any("brd-error" in str(k).lower() for k in body["headers"])
        return False

    def _cache_get(self, query: str):
        self._check_cache_update(force=True)
        with self._cache_lock:
            if query in self._cache:
                return self._cache[query]
        return None

    def _inflight_path(self, query: str) -> pathlib.Path:
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
        return self._inflight_dir / digest

    def _try_claim_inflight(self, query: str) -> bool:
        self._inflight_dir.mkdir(parents=True, exist_ok=True)
        path = self._inflight_path(query)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                return False
            if age > 180.0:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return False

    def _release_inflight(self, query: str) -> None:
        try:
            self._inflight_path(query).unlink()
        except FileNotFoundError:
            pass

    def execute(self, query: str, timeout: int = 60) -> str:
        """
        Execute Bing search query.

        先查共享缓存。未命中时用文件独占认领这条 query，其它线程和另一个 GPU
        进程只等待缓存。真正发出的请求先过账户级 80/min 窗口，再按 key 冷却。

        Args:
            query: Search query string
            timeout: API request timeout in seconds

        Returns:
            Formatted search results as string
        """
        query = query.replace('"', '')
        cached = self._cache_get(query)
        if cached is not None:
            print(f"Cache hit for query: {query}")
            return cached

        deadline = time.time() + 90.0
        while time.time() < deadline:
            cached = self._cache_get(query)
            if cached is not None:
                print(f"Cache hit for query: {query}")
                return cached
            if not self._try_claim_inflight(query):
                time.sleep(0.4)
                continue
            # 占位文件可能刚被上一个请求释放，缓存已经写好了。
            cached = self._cache_get(query)
            if cached is not None:
                self._release_inflight(query)
                print(f"Cache hit for query: {query}")
                return cached
            try:
                return self._fetch_remote(query, timeout)
            finally:
                self._release_inflight(query)
        cached = self._cache_get(query)
        return cached if cached is not None else ""

    def _fetch_remote(self, query: str, timeout: int) -> str:
        api_attempt = 0
        while True:
            # 1) 选 key 并阻塞至该 key 满足节流间隔（限流冷却在 acquire 里生效）
            key, throttle = self._pick_key()
            throttle.acquire()
            self._rate_gate.acquire()
            headers_key = key
            try:
                # 直接复用 _make_request，但把 key 传进去（避免大改签名）
                self._current_key = headers_key
                response = self._make_request(query, timeout)
            except requests.RequestException as e:
                # 网络层失败（超时等）：不算限流，短退避后重试
                api_attempt += 1
                if api_attempt > self._max_api_retries:
                    print(f"Bing search failed: {str(e)}")
                    return ""
                time.sleep(min(self._backoff_base * (2 ** (api_attempt - 1)), self._backoff_max))
                continue

            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(error_msg)
                api_attempt += 1
                if api_attempt > self._max_api_retries:
                    return ""
                time.sleep(min(self._backoff_base * (2 ** (api_attempt - 1)), self._backoff_max))
                continue

            # 解析响应 JSON
            try:
                data = json.loads(response.text)
            except json.JSONDecodeError:
                data = None

            # 2) 限流信封检测：命中则冷却该 key，指数退避后换 key 重试
            if self._is_rate_limited(data, self._rl_status_codes):
                api_attempt += 1
                cooldown = min(self._backoff_base * (2 ** (api_attempt - 1)), self._backoff_max)
                cooldown *= 1.0 + random.uniform(0.0, 0.25)
                print(f"[SearchTool] rate-limited on key ...{key[-6:]}, "
                      f"attempt {api_attempt}/{self._max_api_retries}, cooling {cooldown:.1f}s")
                throttle.acquire(extra_cooldown=cooldown)
                if api_attempt > self._max_api_retries:
                    print("Bing search failed: rate limited after all retries")
                    return ""
                continue
            if data is None:
                # 非限流的解析失败：也退避重试（可能上游临时抖动）
                api_attempt += 1
                if api_attempt > self._max_api_retries:
                    print("Bing search failed: Expecting value (non-JSON response) after retries")
                    return ""
                time.sleep(min(self._backoff_base * (2 ** (api_attempt - 1)), self._backoff_max))
                continue

            # Brightdata SERP API (parsed_bing) 返回 {status_code, headers, body}，
            # body 为字符串，需要二次 json.loads 才能拿到 organic 结果
            if isinstance(data, dict) and "organic" not in data and "body" in data:
                inner_body = data.get("body")
                if isinstance(inner_body, str):
                    try:
                        inner_body = json.loads(inner_body)
                    except json.JSONDecodeError:
                        inner_body = None
                if isinstance(inner_body, dict) and "organic" in inner_body:
                    data = inner_body
                else:
                    print(f"[SearchTool] upstream status_code={data.get('status_code')}, no organic in nested body")
            elif isinstance(data, dict):
                upstream_status = data.get("status_code")
                if upstream_status is not None and upstream_status != 200:
                    # 非 200 且非限流码（如 404 页面类）：重试意义小，直接尝试提取
                    print(f"[SearchTool] upstream status_code={upstream_status}, treating as no results")

            # 3) 提取结果；即便没有 organic 也算"完成"，写缓存防重复请求
            result = self._extract_and_format_results(data)

            # 先同步落盘再返回。另一个进程在占位文件消失后会立刻读这份缓存。
            with self._cache_lock:
                self._cache[query] = result
            self._save_cache_sync()

            return result
    
    def _extract_and_format_results(self, data: Dict) -> str:
        """
        Extract and format search results from API response.
        
        Args:
            data: API response data
            
        Returns:
            Formatted search results as string
        """
        # If no organic results, return empty response
        if 'organic' not in data:
            data['chunk_content'] = []
            return self._format_results(data)

        # Extract unique snippets
        chunk_content_list = []
        seen_snippets = set()
        for result in data['organic']:
            snippet = result.get('description', '').strip()
            if len(snippet) > 0 and snippet not in seen_snippets:
                chunk_content_list.append(snippet)
                seen_snippets.add(snippet)

        data['chunk_content'] = chunk_content_list
        return self._format_results(data)

    def _format_results(self, results: Dict) -> str:
        """
        Format search results into readable text.
        
        Args:
            results: Dictionary containing search results
            
        Returns:
            Formatted string of search results
        """
        if not results.get("chunk_content"):
            return "No search results found."

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][:self._max_results], 1):
            snippet = snippet[:self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        
        return "\n".join(formatted)


if __name__ == "__main__":
    import sys
    
    # Add parent directory to path for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Test configuration: read from env vars if available
    cache_file = os.environ.get("SEARCH_CACHE_FILE", "/tmp/bing_search_cache_test.json")
    api_key = os.environ.get("BRIGHTDATA_API_KEY", "YOUR_BING_SEARCH_API_KEY")
    zone = os.environ.get("BRIGHTDATA_ZONE", "serp_api1")
    
    # Create search tool instance
    search_tool = BingSearchTool(
        api_key=api_key, 
        zone=zone, 
        cache_file=cache_file
    )

    # Test query
    query = "甄嬛传导演"
    print(f"Searching for: {query}")

    # Execute search
    result = search_tool.execute(query)
    print("\nSearch results:\n")
    print(result)
    
    # Test cache hit
    print("\nExecuting same query again (should hit cache):")
    result2 = search_tool.execute(query)
    print("\nSearch results:\n")
    print(result2)

