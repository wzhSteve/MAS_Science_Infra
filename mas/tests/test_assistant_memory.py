"""Assistant memory: fact store scoping/eviction and history compaction."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _store():
    from ai_assistant.memory.store import MemoryStore

    tmp = tempfile.TemporaryDirectory()
    store = MemoryStore(root=Path(tmp.name) / "assistant_memory")
    store._test_tmp = tmp  # keep alive for the test duration
    return store


class TestMemoryStore(unittest.TestCase):
    def test_write_and_read_fact(self) -> None:
        store = _store()
        store.write_fact("experiment:a", "这台机器有两张 A800")
        facts = store.facts("experiment:a")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["text"], "这台机器有两张 A800")
        self.assertFalse(facts[0]["pinned"])

    def test_scopes_are_isolated(self) -> None:
        store = _store()
        store.write_fact("experiment:a", "A 实验的记忆")
        store.write_fact("experiment:b", "B 实验的记忆")
        self.assertEqual([f["text"] for f in store.facts("experiment:a")], ["A 实验的记忆"])
        self.assertEqual([f["text"] for f in store.facts("experiment:b")], ["B 实验的记忆"])

    def test_max_facts_eviction_keeps_pinned(self) -> None:
        from ai_assistant.memory.store import MAX_FACTS

        self.assertEqual(MAX_FACTS, 64)
        store = _store()
        store.write_fact("experiment:a", "置顶事实", pinned=True)
        for i in range(64):
            store.write_fact("experiment:a", f"临时事实 {i}")
            time.sleep(0.001)
        facts = store.facts("experiment:a")
        self.assertEqual(len(facts), MAX_FACTS)
        texts = [f["text"] for f in facts]
        self.assertIn("置顶事实", texts)          # pinned survives eviction
        self.assertNotIn("临时事实 0", texts)      # oldest unpinned evicted first
        self.assertIn("临时事实 63", texts)        # newest kept

    def test_duplicate_text_updates_not_appends(self) -> None:
        store = _store()
        store.write_fact("experiment:a", "同一条记忆")
        first = store.facts("experiment:a")[0]
        time.sleep(0.01)
        store.write_fact("experiment:a", "同一条记忆")
        facts = store.facts("experiment:a")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["id"], first["id"])
        self.assertGreater(float(facts[0]["updated_at"]), float(first["updated_at"]))

    def test_session_summary_roundtrip(self) -> None:
        store = _store()
        self.assertEqual(store.latest_session_summary("experiment:a"), "")
        store.save_session_summary("experiment:a", "用户在调试 HIVE 封装。", experiment_id="demo")
        self.assertEqual(store.latest_session_summary("experiment:a"), "用户在调试 HIVE 封装。")
        time.sleep(0.001)
        store.save_session_summary("experiment:a", "第二轮摘要。", experiment_id="demo")
        self.assertEqual(store.latest_session_summary("experiment:a"), "第二轮摘要。")


class TestCompressHistory(unittest.TestCase):
    def test_short_history_passes_through(self) -> None:
        from ai_assistant.memory.compaction import compress_history

        history = [{"role": "user", "content": f"消息 {i}"} for i in range(23)]
        kept, summary = compress_history(history)
        self.assertEqual(kept, history)
        self.assertEqual(summary, "")

    def test_full_history_compresses_oldest_chunk(self) -> None:
        from ai_assistant.memory.compaction import COMPACT_CHUNK, compress_history

        history = [{"role": "user", "content": f"消息 {i}"} for i in range(24)]
        kept, summary = compress_history(history)
        self.assertEqual(len(kept), len(history) - COMPACT_CHUNK)
        self.assertEqual(kept[0]["content"], f"消息 {COMPACT_CHUNK}")
        self.assertTrue(summary.strip())
        self.assertIn("消息 0", summary)
        self.assertIn(f"消息 {COMPACT_CHUNK - 1}", summary)
        self.assertNotIn(f"消息 {COMPACT_CHUNK}", summary)

    def test_previous_summary_is_carried(self) -> None:
        from ai_assistant.memory.compaction import compress_history

        history = [{"role": "user", "content": f"消息 {i}"} for i in range(24)]
        _, summary = compress_history(history, previous_summary="旧摘要")
        self.assertIn("旧摘要", summary)

    def test_summary_system_message(self) -> None:
        from ai_assistant.memory.compaction import summary_system_message

        self.assertIsNone(summary_system_message(""))
        self.assertIsNone(summary_system_message("   "))
        msg = summary_system_message("此前聊过 HIVE 封装。")
        self.assertEqual(msg["role"], "system")
        self.assertIn("压缩摘要", msg["content"])
        self.assertIn("HIVE", msg["content"])


if __name__ == "__main__":
    unittest.main()
