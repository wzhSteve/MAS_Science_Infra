"""Streaming assistant loop, workspace snapshot, and tool jail."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [_Choice(_Delta(content, tool_calls))]


class _Fn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, index=0, id="", name="", arguments=""):
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _FakeCompletions:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs.get("stream") is True
        assert kwargs.get("max_tokens") == 4096
        if not self.rounds:
            return iter([_Chunk(content="done")])
        return iter(self.rounds.pop(0))


class _FakeOpenAI:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions(self.script))
        _FakeOpenAI.last = self

    script = []


class AssistantChatTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old = os.environ.get("SCIENCE_USER_SPACE_DIR")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("SCIENCE_USER_SPACE_DIR", None)
        else:
            os.environ["SCIENCE_USER_SPACE_DIR"] = self._old
        self._tmp.cleanup()


class TestWorkspaceSnapshot(unittest.TestCase):
    def test_snapshot_uses_summary_not_full_yaml(self) -> None:
        from science_infra.control.assistant import ChatBody, build_workspace_snapshot, summarize_workflow

        huge = {"agents": [{"id": "planner", "kind": "planner", "system_prompt": "x" * 5000}]}
        summary = summarize_workflow({
            "entry_agent": "planner",
            "agents": [{"id": "planner", "kind": "planner"}, {"id": "verifier", "kind": "verifier"}],
            "routers": [{"id": "route_exec"}],
            "sampling": {"sites": [{}, {}, {}]},
        })
        body = ChatBody(
            message="画布上有什么？",
            experiment_id="demo",
            project_id="hive",
            workflow_summary=summary,
            current_workflow=huge,
        )
        snap = build_workspace_snapshot(body)
        self.assertIn("planner(planner)", snap)
        self.assertIn("verifier(verifier)", snap)
        self.assertIn("sampling.sites: 3", snap)
        self.assertIn("route_exec", snap)
        self.assertNotIn("x" * 200, snap)


class TestHistoryFilter(unittest.TestCase):
    def test_welcome_stripped(self) -> None:
        from science_infra.control.assistant import _history_messages

        hist = _history_messages([
            {"role": "assistant", "content": "我只能改 user_space/ 里的用户代码，可以解答本仓库合同。"},
            {"role": "user", "content": "现在画布上有什么？"},
            {"role": "assistant", "content": "planner / verifier"},
        ])
        self.assertEqual([h["role"] for h in hist], ["user", "assistant"])
        self.assertFalse(any("我只能改 user_space/" in h["content"] for h in hist))


class TestStreamLoop(unittest.TestCase):
    def test_stream_tool_then_tokens_no_local_answer(self) -> None:
        from science_infra.control.assistant import ChatBody, iter_chat_events

        _FakeOpenAI.script = [
            [
                _Chunk(tool_calls=[_Call(0, "c1", "list_dir", '{"path":"user_space"}')]),
            ],
            [
                _Chunk(content="画布"),
                _Chunk(content="有 planner"),
            ],
        ]
        env = {
            "AI_ASSISTANT_API_KEY": "sk-unit-test-key",
            "AI_ASSISTANT_API_BASE": "https://example.test/v1",
            "AI_ASSISTANT_MODEL": "unit-model",
        }
        events = []
        with patch("science_infra.control.assistant.load_science_env", lambda **k: None):
            with patch.dict(os.environ, env, clear=False):
                with patch("openai.OpenAI", _FakeOpenAI):
                    body = ChatBody(
                        message="现在画布上有什么？采样点在哪？",
                        experiment_id="demo",
                        workflow_summary={
                            "agents": [{"id": "planner", "kind": "planner"}],
                            "routers": ["route_exec"],
                            "n_sites": 2,
                            "entry_agent": "planner",
                        },
                    )
                    events = list(iter_chat_events(body))
        kinds = [k for k, _ in events]
        self.assertIn("tool_start", kinds)
        self.assertIn("tool_end", kinds)
        self.assertIn("token", kinds)
        self.assertEqual(kinds[-1], "done")
        tokens = "".join(str(d.get("text") or "") for k, d in events if k == "token")
        self.assertIn("planner", tokens)
        self.assertNotIn("sk-unit-test-key", tokens)
        self.assertNotIn("当前没有配置 AI_ASSISTANT_", tokens)
        start_i = kinds.index("tool_start")
        token_i = next(i for i, k in enumerate(kinds) if k == "token" and events[i][1].get("text") == "画布")
        self.assertLess(start_i, token_i)

    def test_first_token_yielded_before_stream_exhausted(self) -> None:
        from science_infra.control.assistant import ChatBody, iter_chat_events

        state = {"pulled": 0, "exhausted": False, "token_when_first": None}

        def slow_chunks():
            state["pulled"] += 1
            yield _Chunk(content="先")
            state["pulled"] += 1
            yield _Chunk(content="后")
            state["exhausted"] = True

        class _LiveOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                self.assert_stream = kwargs.get("stream") is True
                return slow_chunks()

        env = {
            "AI_ASSISTANT_API_KEY": "sk-unit-test-key",
            "AI_ASSISTANT_API_BASE": "https://example.test/v1",
            "AI_ASSISTANT_MODEL": "unit-model",
        }
        with patch("science_infra.control.assistant.load_science_env", lambda **k: None):
            with patch.dict(os.environ, env, clear=False):
                with patch("openai.OpenAI", _LiveOpenAI):
                    body = ChatBody(message="解释一下采样站点")
                    for kind, data in iter_chat_events(body):
                        if kind == "token" and state["token_when_first"] is None:
                            state["token_when_first"] = {
                                "text": data.get("text"),
                                "exhausted": state["exhausted"],
                                "pulled": state["pulled"],
                            }
        self.assertIsNotNone(state["token_when_first"])
        self.assertEqual(state["token_when_first"]["text"], "先")
        self.assertFalse(state["token_when_first"]["exhausted"])
        self.assertLess(state["token_when_first"]["pulled"], 2)
        self.assertTrue(state["exhausted"])


class TestToolJail(AssistantChatTestBase):
    def test_grep_and_read_outside_rejected(self) -> None:
        from science_infra.control.assistant import assistant_tools

        tools = {t["name"]: t["fn"] for t in assistant_tools("hive")}
        with self.assertRaises((PermissionError, ValueError, FileNotFoundError)):
            tools["read_file"](path="/etc/passwd")
        with self.assertRaises((PermissionError, ValueError)):
            tools["grep"](query="root", path="/etc")

    def test_write_still_jailed(self) -> None:
        from workflow.user_gateway.registry import create_project
        from science_infra.control.assistant import assistant_tools

        create_project("hive", title="Hive", mode="native_mas")
        tools = {t["name"]: t["fn"] for t in assistant_tools("hive")}
        result = tools["write_file"](path="mas/tir_agent.py", content="nope")
        self.assertIn("adapted/tir_agent.py", result)
        self.assertFalse((TIR / "tir_agent.py").read_text(encoding="utf-8").startswith("nope"))


class TestOfflineOnlyLocal(unittest.TestCase):
    def test_no_llm_uses_local_answer(self) -> None:
        from science_infra.control.assistant import ChatBody, iter_chat_events

        env = {
            "AI_ASSISTANT_API_KEY": "",
            "AI_ASSISTANT_API_BASE": "",
            "AI_ASSISTANT_BASE_URL": "",
            "AI_ASSISTANT_MODEL": "",
        }
        with patch("science_infra.control.assistant.load_science_env", lambda **k: None):
            with patch.dict(os.environ, env, clear=False):
                events = list(iter_chat_events(ChatBody(message="你好")))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "token")
        self.assertTrue(events[-1][1].get("offline"))


if __name__ == "__main__":
    unittest.main()
