"""
Local Google SERP scraper via undetected_chromedriver.

Root-cause notes (why Google returns /sorry/ CAPTCHA):
1. --headless=new is heavily fingerprinted; Google often serves sorry/recaptcha.
2. A brand-new user-data-dir on every call (then deleted) looks like a fresh bot.
3. --incognito + disposable profile prevents any cookie/trust accumulation.
4. Hard-coded Chrome/140 UA while the real browser is Chrome/150 is a mismatch signal.
5. Typing into the box from automation is more suspicious than opening /search?q=...
"""

from __future__ import annotations

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

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from undetected_chromedriver.patcher import Patcher

DRIVER_INIT_RETRIES = int(os.getenv("GOOGLE_DRIVER_INIT_RETRIES", "5"))
# Default ON: run Chrome in background without a visible window.
# Set GOOGLE_SEARCH_HEADLESS=0 only when debugging CAPTCHA /sorry/ issues.
HEADLESS = os.getenv("GOOGLE_SEARCH_HEADLESS", "1").strip().lower() not in ("0", "false", "no")
# Persist profile across runs so cookies/local state can accumulate.
PERSISTENT_PROFILE = os.path.abspath(
    os.getenv("GOOGLE_CHROME_PROFILE", "./chrome_profile/persistent")
)
SEARCH_LANG = os.getenv("GOOGLE_SEARCH_HL", "en")
SEARCH_COUNTRY = os.getenv("GOOGLE_SEARCH_GL", "us")
PAGE_LOAD_TIMEOUT = int(os.getenv("GOOGLE_PAGE_LOAD_TIMEOUT", "25"))
RESULT_WAIT_SECONDS = int(os.getenv("GOOGLE_RESULT_WAIT_SECONDS", "15"))
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "").strip()
CHROME_BINARY = os.getenv(
    "CHROME_BINARY",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if sys.platform == "darwin"
    else "",
).strip()


def _patch_uc_arm64_platform() -> None:
    """undetected_chromedriver hardcodes darwin -> mac-x64 even on Apple Silicon.

    On arm64 Macs that downloads the wrong chromedriver zip and amplifies
    flaky IncompleteRead / broken-driver failures. Force mac-arm64 instead.
    """
    if getattr(Patcher, "_epc_aw_arm64_patched", False):
        return
    original = Patcher._set_platform_name

    def _set_platform_name(self):
        original(self)
        if sys.platform == "darwin" and platform.machine() == "arm64" and not self.is_old_chromedriver:
            self.platform_name = "mac-arm64"

    Patcher._set_platform_name = _set_platform_name
    Patcher._epc_aw_arm64_patched = True


_patch_uc_arm64_platform()


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


def _detect_chrome_version_main() -> int:
    """Detect installed Chrome major version; 0 lets uc auto-detect."""
    candidates = []
    if sys.platform == "darwin":
        candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    candidates.extend(["google-chrome", "chromium", "chrome"])

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
    )
    return any(marker in msg for marker in retriable_markers)


def _resolve_driver_executable() -> str | None:
    """Only use an explicitly provided driver path."""
    if CHROMEDRIVER_PATH and os.path.isfile(CHROMEDRIVER_PATH) and os.access(CHROMEDRIVER_PATH, os.X_OK):
        return CHROMEDRIVER_PATH
    return None


def _codesign_chromedriver(executable_path: str) -> None:
    """Re-sign patched chromedriver on macOS.

    undetected_chromedriver patches the binary; after that, Apple Silicon macOS
    often SIGKILLs it (exit 137 / status -9) until an ad-hoc codesign is applied.
    """
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
    """Download/patch chromedriver via uc.Patcher, then codesign on darwin."""
    last_error = None
    for attempt in range(DRIVER_INIT_RETRIES):
        try:
            patcher = Patcher(version_main=version_main or 0)
            # Force refresh when binary is missing/broken.
            need_force = not os.path.isfile(patcher.executable_path)
            patcher.auto(force=need_force, version_main=version_main or None)
            _codesign_chromedriver(patcher.executable_path)

            # Smoke-check: binary must start and print a version.
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
    raise RuntimeError(f"Failed to prepare chromedriver: {last_error}")


def _build_chrome_options(version_main: int) -> uc.ChromeOptions:
    """Build Chrome flags that minimize easy automation fingerprints."""
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

    # Do NOT use --incognito with a persistent profile: it wipes the trust signal.
    # Do NOT recreate a random user-data-dir here; pass it to uc.Chrome(...).
    if HEADLESS:
        # Background mode: no popup window. Also pass headless=True to uc.Chrome().
        options.add_argument("--headless=new")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--mute-audio")

    # Keep UA major version aligned with the installed Chrome binary.
    major = version_main or 150
    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.0.0.0 Safari/537.36"
    )
    # Let undetected_chromedriver own automation-flag patching.
    # Do NOT set excludeSwitches here: recent chromedriver rejects it as
    # "unrecognized chrome option: excludeSwitches".

    return options


def get_stealth_driver(version_main: int | None = None):
    if version_main is None:
        version_main = _detect_chrome_version_main()

    os.makedirs(PERSISTENT_PROFILE, exist_ok=True)
    last_error = None

    # Pre-patch + codesign BEFORE uc.Chrome starts the service. Otherwise macOS
    # kills the freshly patched binary with status -9 during Service.start().
    driver_path = _resolve_driver_executable() or _prepare_patched_chromedriver(version_main)

    for attempt in range(DRIVER_INIT_RETRIES):
        try:
            # Clear stale profile locks from previous crashed sessions.
            for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
                lock_path = os.path.join(PERSISTENT_PROFILE, lock_name)
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass

            options = _build_chrome_options(version_main)
            chrome_kwargs = {
                "options": options,
                # Persistent profile: critical vs always-new temp dirs.
                "user_data_dir": PERSISTENT_PROFILE,
                "use_subprocess": True,
                "headless": HEADLESS,
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
            return driver, PERSISTENT_PROFILE
        except Exception as e:
            last_error = e
            if not _is_retriable_driver_error(e) or attempt >= DRIVER_INIT_RETRIES - 1:
                break

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
                f"ChromeDriver init failed ({attempt + 1}/{DRIVER_INIT_RETRIES}): {e}. "
                "Retrying..."
            )
            time.sleep(min(2 ** attempt, 8))

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
    )


def _normalize_result_url(href: str, result_el=None) -> str:
    """Resolve Google obfuscated hrefs (/goto?url=..., /url?q=...) to http(s) URLs."""
    href = (href or "").strip()
    if href.startswith("http://") or href.startswith("https://"):
        return href

    # Classic redirect: /url?q=https://example.com&sa=...
    if href.startswith("/url?") or href.startswith("https://www.google.") and "/url?" in href:
        parsed = urllib.parse.urlparse(href if href.startswith("http") else "https://www.google.com" + href)
        q = urllib.parse.parse_qs(parsed.query)
        for key in ("q", "url"):
            vals = q.get(key) or []
            if vals and str(vals[0]).startswith("http"):
                return str(vals[0])

    # Newer obfuscated cards: href=/goto?url=BASE64... but real URL is in <cite>
    if result_el is not None:
        cite = result_el.select_one("cite")
        if cite:
            cite_text = cite.get_text(" ", strip=True)
            if not cite_text or "reactions" in cite_text.lower() or "answers" in cite_text.lower():
                return ""
            # e.g. "https://en.wikipedia.org › wiki › Moon"
            # Truncated crumbs like "h..." / "... › Moon" are dropped.
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


def _parse_organic_results(soup: BeautifulSoup) -> list:
    """Parse Google SERP organic results with resilient selectors."""
    organic = []
    seen_links = set()

    for result in soup.select("div.MjjYud, div.g"):
        link_el = result.select_one("a[href]")
        title_el = result.select_one("h3")
        snippet_el = result.select_one("div.VwiC3b, span.aCOpRe, div[data-sncf]")
        if not (title_el and link_el):
            continue
        href = _normalize_result_url(link_el.get("href") or "", result)
        if not href or href in seen_links:
            continue
        seen_links.add(href)
        organic.append({
            "title": title_el.get_text(strip=True),
            "link": href,
            "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
        })

    if not organic:
        for title_el in soup.select("div#search a h3, #rso a h3, a h3"):
            link_el = title_el.find_parent("a")
            if not link_el:
                continue
            # climb to a card-like parent for cite lookup
            parent = link_el
            card = None
            for _ in range(8):
                parent = parent.parent if parent else None
                if parent is None:
                    break
                if parent.name == "div" and (
                    "MjjYud" in (parent.get("class") or []) or "g" in (parent.get("class") or [])
                ):
                    card = parent
                    break
            href = _normalize_result_url(link_el.get("href") or "", card or link_el.parent)
            if not href or href in seen_links:
                continue
            snippet = ""
            walk = link_el
            for _ in range(6):
                walk = walk.parent if walk else None
                if walk is None:
                    break
                snippet_el = walk.select_one("div.VwiC3b, span.aCOpRe, div[data-sncf]")
                if snippet_el:
                    snippet = snippet_el.get_text(" ", strip=True)
                    break
            seen_links.add(href)
            organic.append({
                "title": title_el.get_text(strip=True),
                "link": href,
                "snippet": snippet,
            })

    return organic


def _search_url(query: str) -> str:
    params = {
        "q": query,
        "hl": SEARCH_LANG,
        "gl": SEARCH_COUNTRY,
        "pws": "0",
    }
    return "https://www.google.com/search?" + urllib.parse.urlencode(params)


def local_google_search(query: str):
    driver = None
    profile_path = None

    try:
        driver, profile_path = get_stealth_driver()

        # Prefer direct SERP navigation over homepage typing.
        # Typing from automation is a stronger bot signal and often trips /sorry/.
        url = _search_url(query)
        driver.get(url)

        # Small human-like settle delay.
        time.sleep(random.uniform(1.2, 2.2))

        # Optional consent banner.
        for sel in (
            "button#L2AGLb",
            'button[aria-label="Accept all"]',
            'button[aria-label="同意全部"]',
        ):
            try:
                buttons = driver.find_elements(By.CSS_SELECTOR, sel)
                if buttons:
                    buttons[0].click()
                    time.sleep(0.8)
                    break
            except Exception:
                pass

        html = driver.page_source or ""
        current_url = driver.current_url or ""
        if _is_blocked_page(current_url, html):
            raise RuntimeError(
                "Google blocked the local scraper with a CAPTCHA/sorry page "
                f"(url={current_url[:180]}). "
                "Try GOOGLE_SEARCH_HEADLESS=0 and reuse GOOGLE_CHROME_PROFILE; "
                "or wait before retrying."
            )

        # Wait for organic titles when the page is not blocked.
        try:
            WebDriverWait(driver, RESULT_WAIT_SECONDS).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#search a h3, #rso a h3, a h3"))
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

        organic = _parse_organic_results(BeautifulSoup(html, "html.parser"))
        return {
            "query": query,
            "organic": organic,
        }
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        # Keep PERSISTENT_PROFILE on disk. Only remove accidental temp dirs.
        if profile_path and os.path.abspath(profile_path) != PERSISTENT_PROFILE:
            try:
                shutil.rmtree(profile_path, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    result = local_google_search("apple inc")
    print(json.dumps(result, indent=2, ensure_ascii=False))
