"""
Local SERP scraper (patched for this environment).

Root causes fixed (why the original version hung / failed):
1. No system Chrome on AutoDL -> uc.Chrome() auto-detection set binary_location
   to a non-string and raised "Binary Location Must be a String".
   Fix: always pass CHROME_BINARY (Playwright Chromium) explicitly.
2. Playwright build chromium-1243 is actually Chrome 153, so a chromedriver 124
   mismatch caused "This version of ChromeDriver only supports Chrome version 124".
   Fix: detect the browser's real major version and download/point to a
   matching chromedriver (see _prepare_patched_chromedriver).
3. Exporting http_proxy/https_proxy made selenium route its loopback DevTools
   traffic through the proxy -> "Bad Gateway".
   Fix: keep the process env clean and give Chrome only --proxy-server, so the
   chromedriver <-> chrome loopback stays direct.
4. Fresh ephemeral profile + hard-coded Chrome/120 UA got CAPTCHA'd instantly
   (/sorry/). Fix: persistent profile, UA major aligned with the real binary,
   stealth CDP patches, and a Chrome-free DuckDuckGo HTTP fallback that still
   returns results when Google blocks the session.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # ensure GOOGLE_SEARCH_* / CHROME_* from .env apply at import

from bs4 import BeautifulSoup

DRIVER_INIT_RETRIES = int(os.getenv("GOOGLE_DRIVER_INIT_RETRIES", "5"))
# Default ON: run Chrome in background without a visible window.
# Set GOOGLE_SEARCH_HEADLESS=0 only when debugging CAPTCHA /sorry/ issues.
HEADLESS = os.getenv("GOOGLE_SEARCH_HEADLESS", "1").strip().lower() not in ("0", "false", "no")
# Default OFF here: reuse GOOGLE_CHROME_PROFILE for cookie/trust accumulation
# (a brand-new profile on every call is what triggers /sorry/ instantly).
INCOGNITO = os.getenv("GOOGLE_SEARCH_INCOGNITO", "0").strip().lower() not in ("0", "false", "no")
# Persist profile across runs when INCOGNITO=0.
PERSISTENT_PROFILE = os.path.abspath(
    os.getenv("GOOGLE_CHROME_PROFILE", "/root/autodl-tmp/MAS_Science_Infra/.chrome_profile")
)
SEARCH_LANG = os.getenv("GOOGLE_SEARCH_HL", "en")
SEARCH_COUNTRY = os.getenv("GOOGLE_SEARCH_GL", "us")
PAGE_LOAD_TIMEOUT = int(os.getenv("GOOGLE_PAGE_LOAD_TIMEOUT", "25"))
RESULT_WAIT_SECONDS = int(os.getenv("GOOGLE_RESULT_WAIT_SECONDS", "15"))
# udm=14 = Google "Web" tab; skips AI Overview / sparse layouts that break parsers.
USE_WEB_TAB = os.getenv("GOOGLE_SEARCH_UDM14", "1").strip().lower() not in ("0", "false", "no")
MAX_QUERY_LEN = int(os.getenv("GOOGLE_SEARCH_MAX_QUERY_LEN", "160"))
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "").strip()
CHROME_BINARY = os.getenv(
    "CHROME_BINARY",
    "/root/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome",
).strip()
# Tool-level Chrome proxy only (does NOT set process-wide http_proxy).
# Used for SSH RemoteForward tunnels / local Clash on AutoDL.
CHROME_PROXY = (
    os.getenv("GOOGLE_CHROME_PROXY", "").strip()
    or os.getenv("CHROME_PROXY", "").strip()
    or os.getenv("WEB_SEARCH_PROXY", "").strip()
)

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with", "from",
    "by", "at", "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "that", "this", "these", "those", "who", "whom", "whose", "how", "when", "where",
    "why", "if", "as", "into", "about", "over", "after", "before", "between",
    "identify", "exact", "needed", "correct", "following", "output", "code",
    "character", "text", "added", "add", "needs", "need", "please", "find",
    "retrieve", "search", "query", "answer", "using", "use", "does", "do",
}

# Unlambda / esolang blobs: long backtick runs, dotted char streams, apply chains.
_CODE_BLOB_RE = re.compile(
    r"(?:`{2,}[^\s]{0,80})|(?:`+\.?[A-Za-z0-9_. ]{0,40}`+)|"
    r"(?:(?:`|\.){3,}[A-Za-z0-9_.` ]{4,})",
)
_IMPERATIVE_PREFIX_RE = re.compile(
    r"^(?:identify|find|determine|retrieve|search\s+for|look\s+up|what|which|"
    r"how|tell\s+me|give\s+me)\b[\s:,-]*",
    re.I,
)


_chrome_deps_ready = False

# Circuit-breaker state: backend name -> last failure timestamp.
_backend_failures: dict[str, float] = {}


def _ensure_chrome_deps():
    """Lazy-import Selenium / undetected_chromedriver only when Chrome backend is used."""
    global uc, Patcher, By, EC, WebDriverWait, _chrome_deps_ready
    if _chrome_deps_ready:
        return uc, Patcher, By, EC, WebDriverWait

    import undetected_chromedriver as uc_mod
    from selenium.webdriver.common.by import By as By_mod
    from selenium.webdriver.support import expected_conditions as EC_mod
    from selenium.webdriver.support.ui import WebDriverWait as WebDriverWait_mod
    from undetected_chromedriver.patcher import Patcher as Patcher_mod

    uc = uc_mod
    Patcher = Patcher_mod
    By = By_mod
    EC = EC_mod
    WebDriverWait = WebDriverWait_mod
    _chrome_deps_ready = True
    _patch_uc_arm64_platform()
    return uc, Patcher, By, EC, WebDriverWait


def _patch_uc_arm64_platform() -> None:
    """undetected_chromedriver hardcodes darwin -> mac-x64 even on Apple Silicon."""
    if getattr(Patcher, "_epc_aw_arm64_patched", False):
        return
    original = Patcher._set_platform_name

    def _set_platform_name(self):
        original(self)
        if sys.platform == "darwin" and platform.machine() == "arm64" and not self.is_old_chromedriver:
            self.platform_name = "mac-arm64"

    Patcher._set_platform_name = _set_platform_name
    Patcher._epc_aw_arm64_patched = True


def _uc_cache_path() -> str:
    if sys.platform.endswith("win32"):
        return os.path.expanduser("~/appdata/roaming/undetected_chromedriver")
    if sys.platform.startswith(("linux", "linux2")):
        return os.path.expanduser("~/.local/share/undetected_chromedriver")
    if sys.platform.endswith("darwin"):
        return os.path.expanduser("~/Library/Application Support/undetected_chromedriver")
    return os.path.expanduser("~/.undetected_chromedriver")


def _clear_uc_cache() -> None:
    """Remove potentially corrupted chromedriver downloads from uc cache."""
    try:
        cache_path = os.path.abspath(_uc_cache_path())
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
    except Exception:
        pass


def _normalize_chrome_proxy(proxy: str) -> str:
    """Normalize proxy URL for Chrome --proxy-server."""
    proxy = (proxy or "").strip()
    if not proxy:
        return ""
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def _requests_proxies() -> dict[str, str] | None:
    """Proxy dict for `requests` (DuckDuckGo HTTP fallback). Same source as Chrome."""
    proxy = _normalize_chrome_proxy(CHROME_PROXY)
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _detect_chrome_version_main() -> int:
    """Detect installed Chrome major version; 0 lets uc auto-detect.

    NOTE: playwright build dirs like chromium-1243 do NOT mean Chrome 124 —
    the only truth is `chrome --version` (here: Chrome 153).
    """
    candidates = []
    if CHROME_BINARY:
        candidates.append(CHROME_BINARY)
    if sys.platform == "darwin":
        candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    candidates.extend(["google-chrome", "chromium", "chrome", "/usr/bin/google-chrome"])

    for cmd in candidates:
        try:
            if cmd.startswith("/") and not os.path.exists(cmd):
                continue
            out = subprocess.check_output([cmd, "--version"], text=True, timeout=10)
            match = re.search(r"(\d+)\.", out)
            if match:
                return int(match.group(1))
        except Exception:
            continue

    env_version = os.getenv("CHROME_VERSION_MAIN", "").strip()
    if env_version.isdigit():
        return int(env_version)
    return 0


def _is_retriable_driver_error(error: Exception) -> bool:
    msg = f"{type(error).__name__} {error}".lower()
    retriable_markers = (
        "retrieval incomplete",
        "urlopen error",
        "content too short",
        "incomplete read",
        "incompleteread",
        "connection reset",
        "connection aborted",
        "timed out",
        "temporarily unavailable",
        "chrome version",
        "session not created",
        "this version of chromedriver",
        "status code was: -9",
        "unexpectedly exited",
        "binary location",
    )
    return any(marker in msg for marker in retriable_markers)


def _resolve_driver_executable() -> str | None:
    """Only use an explicitly provided driver path."""
    if CHROMEDRIVER_PATH and os.path.isfile(CHROMEDRIVER_PATH) and os.access(CHROMEDRIVER_PATH, os.X_OK):
        return CHROMEDRIVER_PATH
    return None


def _codesign_chromedriver(executable_path: str) -> None:
    """Re-sign patched chromedriver on macOS."""
    if sys.platform != "darwin" or not executable_path or not os.path.isfile(executable_path):
        return
    try:
        subprocess.run(["xattr", "-cr", executable_path], check=False, capture_output=True)
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", executable_path],
            check=False,
            capture_output=True,
        )
    except Exception as e:
        print(f"Warning: codesign chromedriver failed: {e}")


def _prepare_patched_chromedriver(version_main: int) -> str:
    """Download/patch chromedriver via uc.Patcher, then codesign on darwin.

    uc.Patcher downloads via urllib, which ignores Chrome's --proxy-server, so we
    temporarily export the proxy env vars for that download only.
    """
    _, Patcher, _, _, _ = _ensure_chrome_deps()
    last_error = None
    proxy = _normalize_chrome_proxy(CHROME_PROXY)
    saved_proxy_env = {
        k: os.environ.get(k) for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
    }
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    try:
        for attempt in range(DRIVER_INIT_RETRIES):
            try:
                patcher = Patcher(version_main=version_main or 0)
                need_force = not os.path.isfile(patcher.executable_path)
                patcher.auto(force=need_force, version_main=version_main or None)
                _codesign_chromedriver(patcher.executable_path)

                probe = subprocess.run(
                    [patcher.executable_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if probe.returncode != 0:
                    raise RuntimeError(
                        f"chromedriver smoke-check failed rc={probe.returncode}: "
                        f"{probe.stderr or probe.stdout}"
                    )
                return patcher.executable_path
            except Exception as e:
                last_error = e
                err_l = f"{type(e).__name__} {e}".lower()
                if any(
                    k in err_l
                    for k in (
                        "retrieval incomplete",
                        "urlopen error",
                        "content too short",
                        "incomplete read",
                        "incompleteread",
                        "status code was: -9",
                        "smoke-check failed",
                    )
                ):
                    _clear_uc_cache()
                print(
                    f"Prepare chromedriver failed ({attempt + 1}/{DRIVER_INIT_RETRIES}): {e}. "
                    "Retrying..."
                )
                time.sleep(min(2 ** attempt, 8))
    finally:
        # IMPORTANT: restore env so selenium's loopback traffic never goes
        # through the proxy (that caused "Bad Gateway" during NEW_SESSION).
        for k, v in saved_proxy_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    raise RuntimeError(f"Failed to prepare chromedriver: {last_error}")


def _build_chrome_options(version_main: int, *, headless: bool | None = None) -> Any:
    """Build Chrome flags that minimize easy automation fingerprints."""
    uc, _, _, _, _ = _ensure_chrome_deps()
    if headless is None:
        headless = HEADLESS
    options = uc.ChromeOptions()

    # Stability
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--lang=en-US")

    if INCOGNITO:
        options.add_argument("--incognito")

    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--mute-audio")

    # Route Chrome network through the proxy WITHOUT exporting http_proxy for
    # the whole process: keeps chromedriver <-> chrome loopback direct.
    proxy = _normalize_chrome_proxy(CHROME_PROXY)
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
        options.add_argument("--proxy-bypass-list=<-loopback>")
        print(f"[google_search] Chrome proxy-server={proxy}")

    # Keep UA major version aligned with the installed Chrome binary (153 here);
    # the original hard-coded Chrome/120 UA was a mismatch fingerprint signal.
    major = version_main or 150
    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.0.0.0 Safari/537.36"
    )

    return options


def _clear_profile_locks(profile_path: str) -> None:
    for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock_path = os.path.join(profile_path, lock_name)
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except Exception:
                pass


def _ephemeral_profile_dir(attempt: int) -> str:
    """Unique profile when the persistent dir is locked by another Chrome."""
    base = os.path.dirname(PERSISTENT_PROFILE) or "./chrome_profile"
    path = os.path.abspath(
        os.path.join(base, f"ephemeral_{os.getpid()}_{attempt}_{int(time.time())}")
    )
    os.makedirs(path, exist_ok=True)
    return path


def get_stealth_driver(version_main: int | None = None):
    uc, _, _, _, _ = _ensure_chrome_deps()
    if version_main is None:
        version_main = _detect_chrome_version_main()

    os.makedirs(PERSISTENT_PROFILE, exist_ok=True)
    last_error = None
    ephemeral_paths: list[str] = []

    driver_path = _resolve_driver_executable() or _prepare_patched_chromedriver(version_main)

    for attempt in range(DRIVER_INIT_RETRIES):
        use_ephemeral = INCOGNITO or attempt >= 2 or (
            last_error is not None
            and any(
                k in f"{last_error}".lower()
                for k in (
                    "chrome not reachable",
                    "session not created",
                    "user data directory is already in use",
                    "devtoolsactiveport",
                )
            )
        )
        profile_path = _ephemeral_profile_dir(attempt) if use_ephemeral else PERSISTENT_PROFILE
        if use_ephemeral:
            ephemeral_paths.append(profile_path)

        use_headless = HEADLESS
        if attempt >= 3 and HEADLESS:
            use_headless = False

        try:
            _clear_profile_locks(profile_path)
            options = _build_chrome_options(version_main, headless=use_headless)

            chrome_kwargs = {
                "options": options,
                "user_data_dir": profile_path,
                "use_subprocess": True,
                "headless": use_headless,
                "driver_executable_path": driver_path,
            }
            if version_main:
                chrome_kwargs["version_main"] = version_main
            if CHROME_BINARY and os.path.isfile(CHROME_BINARY):
                chrome_kwargs["browser_executable_path"] = CHROME_BINARY

            driver = uc.Chrome(**chrome_kwargs)
            driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

            # Extra stealth patches on top of undetected_chromedriver.
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                    window.chrome = window.chrome || { runtime: {} };
                    """
                },
            )
            return driver, profile_path
        except Exception as e:
            last_error = e
            err_l = f"{type(e).__name__} {e}".lower()
            if any(
                k in err_l
                for k in (
                    "retrieval incomplete",
                    "urlopen error",
                    "content too short",
                    "incomplete read",
                    "incompleteread",
                    "status code was: -9",
                )
            ):
                _clear_uc_cache()
                driver_path = _prepare_patched_chromedriver(version_main)
            if "chrome version" in err_l or "this version of chromedriver" in err_l:
                version_main = _detect_chrome_version_main()
                _clear_uc_cache()
                driver_path = _prepare_patched_chromedriver(version_main)

            print(
                f"ChromeDriver init failed ({attempt + 1}/{DRIVER_INIT_RETRIES}) "
                f"[profile={'ephemeral' if use_ephemeral else 'persistent'}, "
                f"incognito={INCOGNITO}, headless={use_headless}]: {e}. Retrying..."
            )
            if not _is_retriable_driver_error(e) and attempt >= 1:
                if not any(
                    k in err_l
                    for k in (
                        "chrome not reachable",
                        "session not created",
                        "already in use",
                        "devtoolsactiveport",
                    )
                ):
                    break
            time.sleep(min(1.5 * (attempt + 1), 6))

    for path in ephemeral_paths:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    raise RuntimeError(
        f"Failed to initialize ChromeDriver after {DRIVER_INIT_RETRIES} retries: {last_error}"
    )


def _is_blocked_page(url: str, html: str) -> bool:
    u = (url or "").lower()
    h = (html or "").lower()
    return (
        "/sorry/" in u
        or "unusual traffic" in h
        or "g-recaptcha" in h
        or "id=\"recaptcha\"" in h
        or "detected unusual traffic" in h
        or "our systems have detected unusual traffic" in h
        or "/httpservice/retry/enablejs" in u
    )


def _decode_goto_url(raw: str) -> str:
    """Decode Google /goto?url= base64 / base64url payloads to http(s) URLs."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    padded = raw + "=" * (-len(raw) % 4)
    for decoder in (
        lambda s: base64.urlsafe_b64decode(s),
        lambda s: base64.b64decode(s),
    ):
        try:
            decoded = decoder(padded).decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if decoded.startswith("http://") or decoded.startswith("https://"):
            return decoded
        if "http://" in decoded or "https://" in decoded:
            m = re.search(r"https?://[^\s\"'<>]+", decoded)
            if m:
                return m.group(0)
    return ""


def _url_from_cite(result_el) -> str:
    if result_el is None:
        return ""
    cite = result_el.select_one("cite")
    if not cite:
        return ""
    cite_text = cite.get_text(" ", strip=True)
    if not cite_text or "reactions" in cite_text.lower() or "answers" in cite_text.lower():
        return ""
    parts = [p.strip() for p in re.split(r"\s*[›»]\s*", cite_text) if p.strip()]
    clean_parts = []
    for part in parts:
        if part in (".", "..") or part.startswith("...") or part.endswith("..."):
            break
        if "…" in part or "..." in part:
            break
        clean_parts.append(part.replace(" ", ""))
    if not clean_parts:
        return ""
    joined = "/".join(clean_parts)
    joined = re.sub(r"(?<!:)/{2,}", "/", joined)
    m = re.search(r"(https?://[^\s]+)", joined)
    if m:
        return m.group(1).rstrip("/")
    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}", joined, re.I):
        return "https://" + joined.rstrip("/")
    return ""


def _normalize_result_url(href: str, result_el=None) -> str:
    """Resolve Google obfuscated hrefs (/goto?url=..., /url?q=...) to http(s) URLs."""
    href = (href or "").strip()
    if href.startswith("http://") or href.startswith("https://"):
        if "google." in urllib.parse.urlparse(href).netloc and (
            "/url?" in href or "/goto?" in href
        ):
            pass
        else:
            return href

    full = href if href.startswith("http") else ("https://www.google.com" + href if href.startswith("/") else href)

    if href.startswith("/url?") or ("/url?" in full and "google." in full):
        parsed = urllib.parse.urlparse(full if full.startswith("http") else "https://www.google.com" + href)
        q = urllib.parse.parse_qs(parsed.query)
        for key in ("q", "url"):
            vals = q.get(key) or []
            if vals and str(vals[0]).startswith("http"):
                return str(vals[0])

    if href.startswith("/goto?") or "/goto?" in full:
        parsed = urllib.parse.urlparse(full if full.startswith("http") else "https://www.google.com" + href)
        q = urllib.parse.parse_qs(parsed.query)
        for key in ("url", "q", "u"):
            for val in q.get(key) or []:
                decoded = _decode_goto_url(str(val))
                if decoded:
                    return decoded

    cite_url = _url_from_cite(result_el)
    if cite_url:
        return cite_url

    return ""


def sanitize_google_query(query: str) -> str:
    """Strip instruction fluff + source-code blobs that break Google SERPs."""
    q = str(query or "").strip()
    if not q:
        return q

    q = _CODE_BLOB_RE.sub(" ", q)
    q = re.sub(r"`+", " ", q)
    q = re.sub(r"(?:\.[A-Za-z0-9]){3,}\.?", " ", q)
    q = q.replace("\\'", "'").replace('\\"', '"')
    q = re.sub(r"\s+", " ", q).strip(" \t\r\n-;,:.")

    phrases = re.findall(r'"([^"]{2,60})"', q)
    q_no_quotes = re.sub(r'"[^"]{2,60}"', " ", q)
    q_no_quotes = _IMPERATIVE_PREFIX_RE.sub("", q_no_quotes)
    q_no_quotes = re.sub(
        r"\b(?:to|that|which|needed|correct|output|character|text)\b",
        " ",
        q_no_quotes,
        flags=re.I,
    )

    tokens = []
    # Allow digit-leading tokens ("2024", "5g") — the old letter-start regex
    # silently dropped years/numbers from queries like "Who won the euro 2024?".
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+#.-]{1,40}", q_no_quotes):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if tok not in tokens:
            tokens.append(tok)

    parts = tokens[:12]
    for phrase in phrases[:2]:
        quoted = f'"{phrase.strip()}"'
        if quoted not in parts and phrase.strip().lower() not in _STOPWORDS:
            parts.append(quoted)

    cleaned = " ".join(parts).strip()
    if not cleaned:
        cleaned = re.sub(r"\s+", " ", _CODE_BLOB_RE.sub(" ", str(query))).strip()
    if len(cleaned) > MAX_QUERY_LEN:
        cleaned = cleaned[:MAX_QUERY_LEN].rsplit(" ", 1)[0].strip()
    return cleaned or str(query or "").strip()[:MAX_QUERY_LEN]


def _usable_organic(organic: list, *, min_keep: int = 1) -> list:
    """Drop social / benchmark-leak hits so variant search can continue."""
    junk_markers = (
        "instagram.com", "tiktok.com", "facebook.com", "pinterest.com",
        "huggingface.co/datasets", "gaia", "bamboogle",
    )
    kept = []
    for item in organic or []:
        link = str(item.get("link") or "").lower()
        title = str(item.get("title") or "").lower()
        blob = f"{link} {title}"
        if any(m in blob for m in junk_markers):
            continue
        kept.append(item)
    return kept if len(kept) >= min_keep else []


def build_query_variants(query: str) -> list[str]:
    """Ordered unique queries: sanitized first, then progressively shorter."""
    original = str(query or "").strip()
    sanitized = sanitize_google_query(original)
    variants: list[str] = []

    def _add(q: str) -> None:
        q = re.sub(r"\s+", " ", (q or "").strip())
        if not q:
            return
        if "`" in q or re.search(r"(?:\.[A-Za-z0-9]){3,}", q):
            q = sanitize_google_query(q)
        if not q or q in variants:
            return
        variants.append(q)

    _add(sanitized)

    tokens = [t for t in sanitized.replace('"', " ").split() if t]
    if len(tokens) > 4:
        _add(" ".join(tokens[:6]))
        _add(" ".join(tokens[:4]))

    for phrase in re.findall(r'"([^"]{2,60})"', original):
        phrase = phrase.strip()
        if not phrase or "`" in phrase:
            continue
        _add(f'"{phrase}"')

    return variants or [sanitized or original or ""]


def _parse_organic_results(soup: BeautifulSoup) -> list:
    """Parse Google SERP organic results with resilient selectors."""
    organic = []
    seen_links = set()

    def _push(title: str, href: str, snippet: str) -> None:
        if not title or not href or href in seen_links:
            return
        if href.startswith("https://www.google.") and "/search" in href:
            return
        seen_links.add(href)
        organic.append({
            "title": title,
            "link": href,
            "snippet": snippet or "",
        })

    selectors = (
        "div.MjjYud",
        "div.g",
        "div[data-sokoban-container]",
        "div.tF2Cxc",
        "div.N54PNb",
    )
    for result in soup.select(", ".join(selectors)):
        title_el = result.select_one("h3")
        if not title_el:
            continue
        link_el = title_el.find_parent("a") or result.select_one("a[href]")
        if not link_el:
            continue
        snippet_el = result.select_one(
            "div.VwiC3b, span.aCOpRe, div[data-sncf], div.IsZvec, span.st"
        )
        href = _normalize_result_url(link_el.get("href") or "", result)
        _push(
            title_el.get_text(strip=True),
            href,
            snippet_el.get_text(" ", strip=True) if snippet_el else "",
        )

    if not organic:
        for title_el in soup.select("div#search a h3, #rso a h3, #search a h3, a h3"):
            link_el = title_el.find_parent("a")
            if not link_el:
                continue
            parent = link_el
            card = None
            for _ in range(10):
                parent = parent.parent if parent else None
                if parent is None:
                    break
                classes = parent.get("class") or []
                if parent.name == "div" and (
                    "MjjYud" in classes
                    or "g" in classes
                    or "tF2Cxc" in classes
                    or "N54PNb" in classes
                    or parent.has_attr("data-sokoban-container")
                ):
                    card = parent
                    break
            href = _normalize_result_url(link_el.get("href") or "", card or link_el.parent)
            snippet = ""
            walk = link_el
            for _ in range(8):
                walk = walk.parent if walk else None
                if walk is None:
                    break
                snippet_el = walk.select_one(
                    "div.VwiC3b, span.aCOpRe, div[data-sncf], div.IsZvec, span.st"
                )
                if snippet_el:
                    snippet = snippet_el.get_text(" ", strip=True)
                    break
            _push(title_el.get_text(strip=True), href, snippet)

    return organic


def _search_url(query: str) -> str:
    params = {
        "q": query,
        "hl": SEARCH_LANG,
        "gl": SEARCH_COUNTRY,
        "pws": "0",
    }
    if USE_WEB_TAB:
        params["udm"] = "14"
    return "https://www.google.com/search?" + urllib.parse.urlencode(params)


def _dismiss_consent(driver) -> None:
    _, _, By, _, _ = _ensure_chrome_deps()
    for sel in (
        "button#L2AGLb",
        'button[aria-label="Accept all"]',
        'button[aria-label="同意全部"]',
        "#L2AGLb",
    ):
        try:
            buttons = driver.find_elements(By.CSS_SELECTOR, sel)
            if buttons:
                buttons[0].click()
                time.sleep(0.8)
                return
        except Exception:
            pass


def _fetch_serp_organic(driver, query: str) -> list:
    """Navigate once and parse organic results for a single query."""
    _, _, By, EC, WebDriverWait = _ensure_chrome_deps()
    url = _search_url(query)
    driver.get(url)
    time.sleep(random.uniform(1.0, 1.8))
    _dismiss_consent(driver)

    html = driver.page_source or ""
    current_url = driver.current_url or ""
    if _is_blocked_page(current_url, html):
        raise RuntimeError(
            "Google blocked the local scraper with a CAPTCHA/sorry page "
            f"(url={current_url[:180]}). "
            "Try GOOGLE_SEARCH_HEADLESS=0 and reuse GOOGLE_CHROME_PROFILE; "
            "or wait before retrying."
        )

    try:
        WebDriverWait(driver, RESULT_WAIT_SECONDS).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div#search a h3, #rso a h3, a h3, div.tF2Cxc h3")
            )
        )
    except Exception:
        pass

    html = driver.page_source or ""
    current_url = driver.current_url or ""
    if _is_blocked_page(current_url, html):
        raise RuntimeError(
            "Google blocked the local scraper with a CAPTCHA/sorry page "
            f"(url={current_url[:180]})."
        )

    return _parse_organic_results(BeautifulSoup(html, "html.parser"))


def _parse_ddg_html(html: str, max_results: int = 8) -> list[dict]:
    soup = BeautifulSoup(html or "", "html.parser")
    organic: list[dict] = []
    seen: set[str] = set()

    def _unwrap(href: str) -> str:
        href = (href or "").strip()
        if "uddg=" in href:
            parsed = urllib.parse.urlparse(
                href if href.startswith("http") else "https://duckduckgo.com" + href
            )
            qs = urllib.parse.parse_qs(parsed.query)
            vals = qs.get("uddg") or qs.get("u") or []
            if vals:
                return urllib.parse.unquote(vals[0])
        return href

    for result in soup.select("div.result, div.web-result, article"):
        link_el = result.select_one("a.result__a, a.result-link, a[href]")
        if not link_el:
            continue
        href = _unwrap(link_el.get("href") or "")
        if not href.startswith("http"):
            continue
        if any(b in href for b in ("duckduckgo.com", "youtube.com/watch")):
            continue
        title = link_el.get_text(" ", strip=True)
        snip_el = result.select_one(
            "a.result__snippet, td.result-snippet, div.result__snippet"
        )
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        if not title or href in seen:
            continue
        seen.add(href)
        organic.append({"title": title, "link": href, "snippet": snippet})
        if len(organic) >= max_results:
            return organic

    if not organic:
        for link_el in soup.select("a.result__a, a[href]"):
            href = _unwrap(link_el.get("href") or "")
            if not href.startswith("http") or "duckduckgo.com" in href:
                continue
            title = link_el.get_text(" ", strip=True)
            if len(title) < 8 or href in seen:
                continue
            seen.add(href)
            organic.append({"title": title, "link": href, "snippet": ""})
            if len(organic) >= max_results:
                break
    return organic


_BING_CK_URL_RE = re.compile(r"[?&]u=a1([A-Za-z0-9_-]+)")


def _decode_bing_ck(href: str) -> str:
    """Decode Bing /ck/a tracking links (?u=a1<base64url>) to the real URL."""
    m = _BING_CK_URL_RE.search(href or "")
    if not m:
        return href or ""
    try:
        decoded = base64.urlsafe_b64decode(m.group(1) + "=" * (-len(m.group(1)) % 4))
        url = decoded.decode("utf-8", errors="ignore")
        return url if url.startswith("http") else (href or "")
    except Exception:
        return href or ""


def chrome_bing_search(query: str, max_results: int = 8) -> dict:
    """Bing SERP via the stealth Chrome session (real browser TLS fingerprint).

    Plain-requests Bing SERPs from this egress IP come back polluted (results
    unrelated to the query), and top b_algo cards are often injected ads.
    Guards: decode /ck/a trackers to real URLs, then drop results sharing no
    token with the query; if fewer than 2 survive, treat as junk and raise so
    the backend chain moves on.
    """
    from urllib.parse import quote

    driver, profile_path = get_stealth_driver()
    try:
        url = "https://www.bing.com/search?q=" + quote(query) + "&setlang=en&cc=US"
        driver.get(url)
        # Bing sometimes lands on an intermediate redirect (…&rdr=1) before
        # rendering results; poll instead of a fixed sleep.
        soup = BeautifulSoup("", "html.parser")
        deadline = time.time() + 15
        while time.time() < deadline:
            soup = BeautifulSoup(driver.page_source or "", "html.parser")
            if soup.select("li.b_algo h2 a[href]"):
                break
            time.sleep(1.0)

        # Query tokens used for a cheap junk/ads relevance filter.
        q_tokens = {
            t.lower()
            for t in re.findall(r"[A-Za-z0-9]{3,}", str(query or ""))
            if t.lower() not in _STOPWORDS
        }

        raw: list[dict] = []
        for item in soup.select("li.b_algo")[:24]:
            a = item.select_one("h2 a")
            if not a or not a.get("href"):
                continue
            snippet_el = item.select_one(".b_caption p, .b_caption, p")
            raw.append({
                "title": a.get_text(" ", strip=True),
                "link": _decode_bing_ck(str(a.get("href"))),
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            })

        # Relevance filter: keep results sharing at least one query token
        # (drops injected-ads cards like "Summer Place Apartment Homes" for a
        # "capital of France" query). Ignore bing.com internal links.
        organic = []
        for o in raw:
            if "bing.com" in o["link"]:
                continue
            blob = f"{o['title']} {o['snippet']}".lower()
            if q_tokens and not any(t in blob for t in q_tokens):
                continue
            organic.append(o)
        if len(organic) < 2:
            raise RuntimeError(
                f"Chrome Bing search returned junk/no organic results for query={query!r} "
                f"(raw={len(raw)}, relevant={len(organic)})"
            )
        return {
            "query": query,
            "organic": organic[:max_results],
            "original_query": str(query or ""),
            "backend": "chrome_bing",
        }
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def _decode_baidu_link(href: str) -> str:
    """Resolve Baidu /link?url=... redirect wrappers via a HEAD request."""
    href = (href or "").strip()
    if not href.startswith("http"):
        return href
    if "baidu.com/link?" not in href:
        return href
    try:
        import requests as _requests

        r = _requests.head(
            href,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=8,
            allow_redirects=False,
            proxies=_requests_proxies(),
        )
        loc = r.headers.get("Location", "")
        return loc if loc.startswith("http") else href
    except Exception:
        return href


def baidu_chrome_search(query: str, max_results: int = 8) -> dict:
    """Baidu SERP via the stealth Chrome session (final fallback).

    Baidu is far more permissive than Google/Bing toward this egress IP: plain
    requests work until rate-limited ("百度安全验证"), and the stealth Chrome
    session (cookies + real TLS fingerprint) is stable. Works for both Chinese
    and English queries.
    """
    from urllib.parse import quote

    driver, profile_path = get_stealth_driver()
    try:
        url = "https://www.baidu.com/s?wd=" + quote(query) + "&rn=10"
        driver.get(url)
        # Poll for results (Baidu sometimes needs an extra beat to render).
        soup = BeautifulSoup("", "html.parser")
        deadline = time.time() + 15
        while time.time() < deadline:
            soup = BeautifulSoup(driver.page_source or "", "html.parser")
            if soup.select("div#content_left div.c-container h3 a, div.result h3 a"):
                break
            time.sleep(1.0)

        raw: list[dict] = []
        for item in soup.select("div#content_left div.c-container, div.result")[:24]:
            a = item.select_one("h3 a")
            if not a or not a.get("href") or not a.get_text(strip=True):
                continue
            snippet_el = item.select_one(
                ".c-abstract, [class*='content-right'], [class*='abstract'], "
                ".c-span-last, span[class*='text']"
            )
            raw.append({
                "title": a.get_text(" ", strip=True),
                "link": _decode_baidu_link(str(a.get("href"))),
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            })
        # Relevance ordering: Baidu mixes in promo/edu cards unrelated to the
        # query; rank by query-token overlap (count + title hits first).
        q_tokens = [
            t.lower()
            for t in re.findall(r"[A-Za-z0-9]{3,}", str(query or ""))
            if t.lower() not in _STOPWORDS
        ]
        def _score(o: dict) -> int:
            title = o["title"].lower()
            snip = o["snippet"].lower()
            return sum((2 if t in title else 0) + (1 if t in snip else 0) for t in q_tokens)
        raw.sort(key=_score, reverse=True)
        if len(raw) < 1:
            page_title = soup.title.get_text(strip=True) if soup.title else ""
            raise RuntimeError(
                f"Baidu search returned no organic results for query={query!r} "
                f"(page={page_title[:60]!r})"
            )
        return {
            "query": query,
            "organic": raw[:max_results],
            "original_query": str(query or ""),
            "backend": "baidu_chrome",
        }
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def http_duckduckgo_search(query: str, max_results: int = 8) -> dict:
    """Chrome-free fallback via DuckDuckGo HTML (no API key).

    Used when ChromeDriver cannot start or Google blocks the session.
    """
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
    }
    variants = build_query_variants(query)
    tried: list[str] = []
    best: list[dict] = []
    best_q = variants[0] if variants else str(query or "")
    proxies = _requests_proxies()
    if proxies:
        print(f"[google_search] DuckDuckGo HTTP proxies={proxies}")

    for q in variants:
        tried.append(q)
        html = ""
        for method, url, data in (
            ("POST", "https://html.duckduckgo.com/html/", {"q": q}),
            ("GET", "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q), None),
        ):
            try:
                if method == "POST":
                    resp = requests.post(
                        url, data=data, headers=headers, timeout=20, proxies=proxies
                    )
                else:
                    resp = requests.get(
                        url, headers=headers, timeout=20, proxies=proxies
                    )
                if resp.status_code >= 400:
                    continue
                html = resp.text or ""
                if "result__a" in html or "web-result" in html:
                    break
            except Exception:
                continue
        if not html:
            continue
        organic = _usable_organic(_parse_ddg_html(html, max_results=max_results * 2))
        organic = organic[:max_results]
        if len(organic) > len(best):
            best, best_q = organic, q
        if len(organic) >= 5:
            break

    if not best:
        raise RuntimeError(
            f"DuckDuckGo HTTP fallback returned no organic results "
            f"(tried={tried!r})"
        )

    return {
        "query": best_q,
        "organic": best,
        "original_query": str(query or ""),
        "tried_queries": tried,
        "backend": "duckduckgo_http",
    }


def local_google_search(query: str):
    """Search Google locally; sanitize + variant-retry on one Chrome session."""
    driver = None
    profile_path = None
    variants = build_query_variants(query)
    tried: list[str] = []
    last_block_error: Exception | None = None

    try:
        driver, profile_path = get_stealth_driver()

        for q in variants:
            tried.append(q)
            try:
                organic = _fetch_serp_organic(driver, q)
            except RuntimeError as e:
                last_block_error = e
                break
            usable = _usable_organic(organic)
            if usable:
                return {
                    "query": q,
                    "organic": usable,
                    "original_query": str(query or ""),
                    "tried_queries": tried,
                    "backend": "chrome_google",
                }
            time.sleep(random.uniform(0.6, 1.2))

        if last_block_error is not None:
            raise last_block_error

        raise RuntimeError(
            "local Google search returned no organic results after query variants "
            f"tried={tried!r}. This is a scraper/coverage failure, not a verified absence."
        )
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        if profile_path and os.path.abspath(profile_path) != PERSISTENT_PROFILE:
            try:
                shutil.rmtree(profile_path, ignore_errors=True)
            except Exception:
                pass


def multi_backend_search(query: str) -> dict:
    """Search engines only (no Wikipedia), with graceful backend fallback:

    1. chrome_google  — stealth Chrome on Google (best relevance, may CAPTCHA)
    2. chrome_bing    — same Chrome session tech on Bing (real-browser TLS)
    3. duckduckgo_http— plain-requests DDG HTML (no Chrome; often challenged)
    4. baidu_chrome   — stealth Chrome on Baidu (very permissive; CN/EN queries)

    GOOGLE_SEARCH_BACKEND env var selects the chain:
      "auto" (default)      — full chain above
      "chrome_first"        — chrome_google, then everything else
      "http" | "ddg"        — DuckDuckGo HTTP only
      "baidu"               — Baidu only
      "chrome_bing"         — Bing only

    Returns the first backend that yields usable organic results; the result
    dict carries a `backend` key saying which one won.
    """
    errors: list[str] = []
    all_attempts = (
        ("chrome_google", local_google_search),
        ("chrome_bing", chrome_bing_search),
        ("duckduckgo_http", http_duckduckgo_search),
        ("baidu_chrome", baidu_chrome_search),
    )
    chain = os.getenv("GOOGLE_SEARCH_BACKEND", "auto").strip().lower()
    if chain in ("http", "ddg", "duckduckgo"):
        attempts = [a for a in all_attempts if a[0] == "duckduckgo_http"]
    elif chain == "baidu":
        attempts = [a for a in all_attempts if a[0] == "baidu_chrome"]
    elif chain == "chrome_bing":
        attempts = [a for a in all_attempts if a[0] == "chrome_bing"]
    else:
        # auto / chrome_first / anything else: full chain
        attempts = list(all_attempts)
    for name, fn in attempts:
        # Circuit breaker: skip Chrome backends that recently failed hard
        # (CAPTCHA / junk SERP) so a multi-question run doesn't burn ~100s per
        # query retrying a blocked IP. Set SEARCH_BREAKER_SECONDS=0 to disable.
        breaker_s = int(os.getenv("SEARCH_BREAKER_SECONDS", "600"))
        if breaker_s > 0:
            failed_at = _backend_failures.get(name)
            if failed_at is not None and (time.time() - failed_at) < breaker_s:
                errors.append(f"{name}: skipped (circuit open, failed {int(time.time() - failed_at)}s ago)")
                continue
        try:
            result = fn(query)
            if result.get("organic"):
                _backend_failures.pop(name, None)
                return result
            errors.append(f"{name}: no organic results")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            _backend_failures[name] = time.time()
            print(f"[google_search] backend {name} failed: {e}")
    raise RuntimeError(
        f"All search backends failed for query={query!r}:\n  " + "\n  ".join(errors)
    )


# 测试
if __name__ == "__main__":
    result = multi_backend_search("apple inc")
    print(json.dumps(result, indent=2, ensure_ascii=False))