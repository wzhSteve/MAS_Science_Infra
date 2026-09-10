import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import random
from bs4 import BeautifulSoup
import json
import time
import os
from selenium.common.exceptions import WebDriverException

def human_like_type(element, text, base_delay=0.13, jitter=0.1):
    for ch in text:
        element.send_keys(ch)
        delay = abs(random.gauss(base_delay, jitter))
        time.sleep(delay)

def get_stealth_driver():
    # 用 undetected_chromedriver 启动一个更难被识别的 Chrome
    options = uc.ChromeOptions()
    # 你可以加之前提到的一些 prefs 禁用定位、通知
    # ====== 1. 基础稳定性 ======
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")  # 无头模式下更稳
    options.add_argument("--window-size=783,768")
    options.add_argument("--incognito")

    # ====== 2. 无头模式（新版） ======
    options.add_argument("--headless=new")

    # ====== 3. 反自动化检测 ======
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-geolocation")

    # ====== 4. 用户行为一致性（非常关键） ======
    options.add_argument("--disable-sync")
    options.add_argument("--disable-extensions")

    # ====== 5. User-Agent（与窗口尺寸匹配） ======
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )

    # ====== 6. 独立 profile（强烈推荐保留） ======
    profile_path = os.path.abspath(f"./chrome_profile/{int(time.time())}")
    os.makedirs(profile_path, exist_ok=True)
    options.add_argument(f"--user-data-dir={profile_path}")

    # ====== 7. 修复：明确指定 Chrome 版本为 149（与系统 Chrome 匹配） ======
    # version_main=149 与系统 Chrome 149.0.7827.156 匹配
    driver = uc.Chrome(options=options, version_main=149)

    # 注入 navigator.webdriver = undefined
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {
              get: () => undefined
            })
            """
        }
    )

    return driver, profile_path

def local_google_search(query: str):
    driver = None
    profile_path = None
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            driver, profile_path = get_stealth_driver()
            break  # Successfully created driver
        except WebDriverException as e:
            error_msg = str(e).lower()
            if "chromedriver" in error_msg and "chrome version" in error_msg:
                # ChromeDriver version mismatch - retry with automatic download
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed to initialize ChromeDriver after {max_retries} retries. Please ensure Chrome and ChromeDriver versions match.")
                print(f"ChromeDriver version mismatch detected. Retrying ({retry_count}/{max_retries})...")
                time.sleep(1)
            elif "no such window" in error_msg and driver:
                driver.quit()
                driver, profile_path = get_stealth_driver()
                break
            else:
                raise

    try:
        driver.get("https://www.google.com")
        time.sleep(1)

        box = driver.find_element(By.NAME, "q")
        human_like_type(box, query)
        box.send_keys(Keys.RETURN)

        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        organic = []

        # Google 新版自然结果结构（2023–2025）
        results = soup.select("div.MjjYud")

        for r in results:
            link_el = r.select_one("a")
            title_el = r.select_one("h3")
            snippet_el = r.select_one("div.VwiC3b")

            if title_el and link_el:
                organic.append({
                    "title": title_el.get_text(strip=True),
                    "link": link_el.get('href'),
                    "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else ""
                })

        return {
            "query": query,
            "organic": organic
        }
    finally:
        if driver:
            driver.quit()
        # 清理用户数据目录
        if profile_path:
            try:
                import shutil
                shutil.rmtree(profile_path)
            except Exception as e:
                pass

# 测试
if __name__ == "__main__":
    result = local_google_search("apple inc")
    print(json.dumps(result, indent=2, ensure_ascii=False))