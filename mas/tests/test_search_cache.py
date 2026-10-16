"""Search-result cache unit tests: hit/miss/gap tri-state, incremental
write-back with file locking, MAS_SEARCH_CACHE gating, and
and the agentflow project passing raw queries to the kernel key.

Network is mocked everywhere: these tests lock in the cache contract,
not the availability of any upstream SERP.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.search as search_mod  # noqa: E402
import tools.wikipedia as wiki_mod  # noqa: E402
from tools.search import (  # noqa: E402
    _cache_query_key,
    bing_search,
    fetch_page,
)
from tools.search_cache import (  # noqa: E402
    GAP_SENTINEL,
    cache_lookup,
    cache_store,
    load_search_cache,
    reload_search_cache,
)
from tools.wikipedia import wikipedia_search  # noqa: E402


def _env_cache(path: Path) -> mock._patch_dict:  # type: ignore[name-defined]
    """patch.dict turning the cache layer ON and pointing at `path`."""
    return mock.patch.dict(os.environ, {"MAS_SEARCH_CACHE": str(path)})


class _CacheFileCase(unittest.TestCase):
    """Base: fresh tmp cache file per test, env-driven cache activation."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "search_cache.json"
        saved = os.environ.pop("MAS_SEARCH_CACHE", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "MAS_SEARCH_CACHE", saved)

    def write_cache(self, data: dict) -> None:
        self.cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def read_cache(self) -> dict:
        return json.loads(self.cache_path.read_text(encoding="utf-8"))


class TestCacheLookupTriState(_CacheFileCase):
    """cache_lookup: hit / miss / gap 三态. Module cache lazy-loads once."""

    def test_hit_miss_gap(self) -> None:
        self.write_cache({
            "query::ada lovelace":
                "wikipedia results for: ada lovelace\n"
                "1. Ada Lovelace\n   URL: https://en.wikipedia.org/wiki/Ada_Lovelace",
            "query::royal flash": GAP_SENTINEL,
            "url::https://example.com/ada": "Ada Lovelace wrote notes.",
        })
        with _env_cache(self.cache_path):
            reload_search_cache()
            # hit: 正常文本原样返回
            text, hit_type = cache_lookup("query", "ada lovelace")
            self.assertEqual(hit_type, "hit")
            self.assertIn("Ada_Lovelace", text)
            # url 类目同样 hit
            body, hit_type2 = cache_lookup("url", "https://example.com/ada")
            self.assertEqual(hit_type2, "hit")
            self.assertEqual(body, "Ada Lovelace wrote notes.")
            # gap: 显式缺失标记（区别于未命中）
            text_gap, hit_gap = cache_lookup("query", "royal flash")
            self.assertEqual(hit_gap, "gap")
            self.assertIsNone(text_gap)
            # miss: 无此 key
            text_miss, hit_miss = cache_lookup("query", "no such query")
            self.assertEqual(hit_miss, "miss")
            self.assertIsNone(text_miss)

    def test_missing_file_returns_empty_dict_no_raise(self) -> None:
        with _env_cache(self.cache_path):  # file does not exist yet
            cache = reload_search_cache()
            self.assertEqual(cache, {})
            text, hit_type = cache_lookup("query", "anything")
            self.assertEqual(hit_type, "miss")

    def test_invalid_category_raises(self) -> None:
        with self.assertRaises(ValueError):
            cache_lookup("bogus", "x")

    def test_explicit_path_arg_overrides_env(self) -> None:
        other = Path(self._tmp.name) / "other.json"
        other.write_text(json.dumps({"query::k": "v"}), encoding="utf-8")
        cache = reload_search_cache(str(other))
        self.assertEqual(cache.get("query::k"), "v")


class TestCacheStore(_CacheFileCase):
    """cache_store: 增量回写 + fcntl 侧车锁基本路径."""

    def test_store_creates_and_merges_incrementally(self) -> None:
        with _env_cache(self.cache_path):
            reload_search_cache()
            self.assertTrue(cache_store("query", "ada lovelace", "result text 1"))
            data = self.read_cache()
            self.assertEqual(data["query::ada lovelace"], "result text 1")
            # 增量：第二条不冲掉第一条
            self.assertTrue(cache_store("url", "https://example.com/x", "body x"))
            data = self.read_cache()
            self.assertEqual(len(data), 2)
            self.assertEqual(data["url::https://example.com/x"], "body x")
            # 回写后的条目立刻可在内存 cache 中 hit（无需重读盘）
            text, hit_type = cache_lookup("query", "ada lovelace")
            self.assertEqual(hit_type, "hit")
            self.assertEqual(text, "result text 1")

    def test_store_invalid_category_raises(self) -> None:
        with _env_cache(self.cache_path):
            with self.assertRaises(ValueError):
                cache_store("bogus", "k", "v")

    def test_store_write_failure_only_logs_no_raise(self) -> None:
        # 目录不存在 -> 写失败必须吞掉、只 log（返回 False）
        with mock.patch.dict(os.environ, {"MAS_SEARCH_CACHE": "/nonexistent_dir/x/cache.json"}):
            reload_search_cache()
            self.assertFalse(cache_store("query", "q", "v"))
            # 不抛错即通过；再次确认 lookup 仍可用
            _text, hit_type = cache_lookup("query", "q")
            self.assertEqual(hit_type, "miss")

    def test_two_threads_store_isolated_by_file_lock(self) -> None:
        with _env_cache(self.cache_path):
            reload_search_cache()
            results: list = []
            errs: list = []

            def worker(i: int) -> None:
                try:
                    results.append(cache_store("query", f"q{i}", f"v{i}"))
                except Exception as exc:  # pragma: no cover
                    errs.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errs, [])
            self.assertEqual(len([r for r in results if r]), 8)
            data = self.read_cache()
            # 8 个写入在锁下互不覆盖：全都在盘上
            for i in range(8):
                self.assertEqual(data[f"query::q{i}"], f"v{i}")


class TestToolWiringSearch(_CacheFileCase):
    """bing/google/fetch_page 三分支接线的 hit/miss/store 回填."""

    def setUp(self) -> None:
        super().setUp()
        # A .env loaded earlier in the process enables Bright Data, which would
        # bypass the mocked _request and hit the real SERP API.
        patcher = mock.patch.object(search_mod, "BRIGHT_DATA_API_KEY", "")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _mock_serp_html(self, title: str, url: str) -> str:
        return (
            f'<html><body><ul><li class="b_algo"><h2>'
            f'<a href="{url}">{title}</a></h2></li></ul></body></html>'
        )

    def test_query_hit_returns_cached_no_network(self) -> None:
        # _cache_query_key("Ada Lovelace") 短路分支保留大小写 → key 为原样
        self.write_cache({"query::Ada Lovelace": "CACHED SERP TEXT"})
        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0"}
        ), mock.patch.object(
            search_mod, "_request", side_effect=AssertionError("network must not run on hit")
        ):
            reload_search_cache()
            self.assertEqual(bing_search("Ada Lovelace"), "CACHED SERP TEXT")

    def test_query_miss_network_success_stores_back(self) -> None:
        html = self._mock_serp_html("Ada Lovelace", "https://example.com/ada")

        def fake_request(method, url, **_kw):
            return html, ""

        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0"}
        ), mock.patch.object(search_mod, "_request", side_effect=fake_request):
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertIn("bing results for: Ada Lovelace", out)
        self.assertIn("https://example.com/ada", out)
        # 回填落盘，key 是归一化后的 query key
        data = self.read_cache()
        self.assertEqual(data["query::Ada Lovelace"], out)

    def test_query_miss_network_failure_keeps_gap_response_clean(self) -> None:
        # miss → 现网失败 → 明确的 search_unavailable 前缀（非垃圾错误串）
        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0"}
        ), mock.patch.object(search_mod, "_request", return_value=("", "GET 503")):
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertTrue(
            out.startswith("Bing unreachable:") or out.startswith("search_unavailable"),
            out,
        )

    def test_gap_marker_maps_to_explicit_unavailable(self) -> None:
        self.write_cache({"query::Ada Lovelace": GAP_SENTINEL})
        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0"}
        ), mock.patch.object(
            search_mod, "_request", side_effect=AssertionError("gap must short-circuit")
        ):
            reload_search_cache()
            # gap 不直接返回 sentinel 裸值；网络也不该被重试 —— 但当前接线
            # 对 gap 走现有网络逻辑，此处网络被 mock 为“必须短路”以锁死
            # 行为契约；若实现选择重试网络则该测试失败并提醒接线者。
            out = bing_search("Ada Lovelace")
            self.assertIn("search_unavailable", out)

    def test_fetch_page_url_hit(self) -> None:
        self.write_cache({"url::https://example.com/ada": "cached body text"})
        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0"}
        ), mock.patch.object(
            search_mod, "_request", side_effect=AssertionError("hit must not fetch")
        ):
            reload_search_cache()
            out = fetch_page("https://example.com/ada")
        self.assertEqual(out, "url: https://example.com/ada\ncached body text")

    def test_fetch_page_miss_success_stores_back(self) -> None:
        html = "<html><body><p>Hello page body.</p><script>x()</script></body></html>"

        def fake_request(_m, _u, **_kw):
            return html, ""

        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0"}
        ), mock.patch.object(search_mod, "_request", side_effect=fake_request):
            reload_search_cache()
            out = fetch_page("https://example.com/ada")
        self.assertTrue(out.startswith("url: https://example.com/ada\n"))
        self.assertIn("Hello page body.", out)
        data = self.read_cache()
        self.assertIn("Hello page body.", data["url::https://example.com/ada"])

    def test_offline_flag_beats_cache_hit(self) -> None:
        self.write_cache({"query::ada lovelace": "CACHED"})
        with _env_cache(self.cache_path), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "1"}
        ):
            reload_search_cache()
            self.assertEqual(
                bing_search("Ada Lovelace"),
                "search_unavailable: TIR_OFFLINE_SEARCH=1",
            )

    def test_env_off_tools_use_original_network_path(self) -> None:
        # MAS_SEARCH_CACHE 未设（含空串）：不消费缓存，即使盘上有缓存文件
        self.write_cache({"query::Ada Lovelace": "CACHED BUT INACTIVE"})
        html = self._mock_serp_html("fresh", "https://example.com/fresh")

        def fake_request(method, url, **_kw):
            return html, ""

        env = {k: v for k, v in os.environ.items() if k != "MAS_SEARCH_CACHE"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(
            os.environ, {"TIR_OFFLINE_SEARCH": "0", "MAS_SEARCH_CACHE": ""}
        ), mock.patch.object(search_mod, "_request", side_effect=fake_request):
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertIn("fresh", out)
        self.assertNotIn("CACHED BUT INACTIVE", out)
        # 未设 env 时不回填：盘上内容保持测试写入的原样（未被覆盖）
        data = self.read_cache()
        self.assertEqual(data, {"query::Ada Lovelace": "CACHED BUT INACTIVE"})


class TestToolWiringWikipedia(_CacheFileCase):
    """wikipedia_search 三分支: hit / miss→网络成功回填 / miss→失败."""

    def _serp_payload(self, titles):
        return {"query": {"search": [{"title": t, "snippet": "s"} for t in titles]}}

    def test_wikipedia_hit(self) -> None:
        # key 归一化保留大小写
        self.write_cache({"query::Ada Lovelace": "CACHED WIKI TEXT"})
        with _env_cache(self.cache_path), mock.patch.object(
            wiki_mod.requests, "get", side_effect=AssertionError("hit must not http")
        ):
            reload_search_cache()
            self.assertEqual(wikipedia_search("Ada Lovelace"), "CACHED WIKI TEXT")

    def test_wikipedia_miss_success_stores_back(self) -> None:
        resp = mock.Mock(status_code=200)
        resp.json.return_value = self._serp_payload(["Ada Lovelace"])
        with _env_cache(self.cache_path), mock.patch.object(
            wiki_mod.requests, "get", return_value=resp
        ):
            reload_search_cache()
            out = wikipedia_search("Ada Lovelace")
        self.assertIn("wikipedia results for: Ada Lovelace", out)
        data = self.read_cache()
        self.assertEqual(data["query::Ada Lovelace"], out)

    def test_wikipedia_miss_failure_no_store_and_clean_message(self) -> None:
        resp = mock.Mock(status_code=403)

        def fake_get(_url, **_kw):
            return resp

        with _env_cache(self.cache_path), mock.patch.object(
            wiki_mod.requests, "get", side_effect=fake_get
        ):
            reload_search_cache()
            out = wikipedia_search("Ada Lovelace")
        self.assertTrue(out.startswith("search_unavailable"), out)
        # 失败不回填垃圾：文件不存在或不含该 key
        if self.cache_path.exists():
            data = self.read_cache()
            self.assertNotIn("query::Ada Lovelace", data)


class TestAgentflowUsesKernelKey(unittest.TestCase):
    """AgentFlow 不再自带查询归一化：原始 query 直接交给内核，缓存键只由
    _cache_query_key 计算，避免两份规则漂移。"""

    def test_local_bing_passes_raw_query_to_kernel(self) -> None:
        adapted = REPO / "user_space" / "projects" / "agentflow" / "adapted"
        spec = importlib.util.spec_from_file_location("af_tools_probe", adapted / "tools_local.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = "Search the web for: Oliver Reed Royal Flash?When ready, output the final answer"
        seen = []
        import types

        fake = types.SimpleNamespace(bing_search=lambda q: seen.append(q) or "search_unavailable: x")
        with mock.patch.dict(sys.modules, {"mas.tools.search": fake}):
            out = mod.local_bing_search(raw)
        self.assertEqual(seen, [raw])
        self.assertTrue(out.startswith("search_unavailable"))
        self.assertFalse(hasattr(mod, "_condense_query"))


if __name__ == "__main__":
    unittest.main()
