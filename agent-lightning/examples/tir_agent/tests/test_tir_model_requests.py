"""Exercise serialized model requests without contacting a model provider."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestTirModelRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import langchain
            import langchain_openai
            import langgraph
            import bs4
        except ImportError as error:
            raise unittest.SkipTest("Install the live extra to test model requests") from error

    def test_tool_fields_in_normal_and_finalize_requests(self):
        import tir_agent
        from langchain_core.messages import HumanMessage

        factory = tir_agent.init_chat_model
        for enabled_tools in ([], ["execute_python"], None):
            for finalize in (False, True):
                with self.subTest(enabled_tools=enabled_tools, finalize=finalize):
                    requests = []

                    def respond(request: httpx.Request) -> httpx.Response:
                        payload = json.loads(request.content)
                        requests.append(payload)
                        self.assertEqual(request.url.path, "/v1/chat/completions")
                        self.assertNotEqual(payload.get("tools"), [])
                        return httpx.Response(200, json={
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "test-model",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": "<answer>2</answer>"},
                                "finish_reason": "stop",
                            }],
                        })

                    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                        def create_model(*args, **kwargs):
                            return factory(*args, http_client=client, **kwargs)

                        with patch.object(tir_agent, "init_chat_model", side_effect=create_model):
                            agent = tir_agent.TirAgent(
                                endpoint="https://model.invalid/v1",
                                model_name="test-model",
                                api_key="test-key",
                                enabled_tools=enabled_tools,
                            )
                            result = agent.call_model({
                                "messages": [HumanMessage(content="What is 1 + 1?")],
                                "asked_finalize": finalize,
                            })

                    self.assertEqual(len(requests), 1)
                    payload = requests[0]
                    expected = list(tir_agent.TOOL_MAP) if enabled_tools is None else enabled_tools
                    if finalize or not expected:
                        self.assertNotIn("tools", payload)
                        self.assertNotIn("tool_choice", payload)
                    else:
                        self.assertEqual(
                            [tool["function"]["name"] for tool in payload["tools"]], expected,
                        )
                    self.assertEqual(result["messages"][-1].content, "<answer>2</answer>")


if __name__ == "__main__":
    unittest.main()
