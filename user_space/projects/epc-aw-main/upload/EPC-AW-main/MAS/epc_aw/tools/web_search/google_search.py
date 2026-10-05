import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import random
from bs4 import BeautifulSoup
import json

def human_like_type(element, text, base_delay=0.12, jitter=0.1):
    for ch in text:
        element.send_keys(ch)
        delay = abs(random.gauss(base_delay, jitter))
        time.sleep(delay)

def get_stealth_driver():
    # 用 undetected_chromedriver 启动一个更难被识别的 Chrome
    options = uc.ChromeOptions()
    # 你可以加之前提到的一些 prefs 禁用定位、通知
    options.add_argument("--disable-geolocation")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--incognito")
    # 禁用自动同步之类
    options.add_argument("--disable-sync")
    options.add_argument("--disable-extensions")

    driver = uc.Chrome(options=options)
    
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

    return driver

def google_search(query: str):
    driver = get_stealth_driver()

    driver.get("https://www.google.com")
    time.sleep(1)

    box = driver.find_element(By.NAME, "q")
    human_like_type(box, query)
    box.send_keys(Keys.RETURN)

    time.sleep(2)

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

    driver.quit()

    return {
        "query": query,
        "organic": organic
    }

# 测试
if __name__ == "__main__":
    result = google_search("apple inc")
    print(json.dumps(result, indent=2, ensure_ascii=False))