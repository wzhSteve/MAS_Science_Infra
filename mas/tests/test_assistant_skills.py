"""Skill registry, skill queries and assistant tool surface."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestDiscoverSkills(unittest.TestCase):
    def test_discover_scans_custom_dir_and_parses_frontmatter(self) -> None:
        from ai_assistant.skills.registry import discover_skills

        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            (skills_dir / "my_skill").mkdir(parents=True)
            (skills_dir / "my_skill" / "SKILL.md").write_text(
                "---\n"
                "name: my_skill\n"
                "description: 测试技能说明\n"
                "triggers: [测试触发词, demo]\n"
                "---\n"
                "# 正文\n"
                "步骤一。\n",
                encoding="utf-8",
            )
            (skills_dir / "broken").mkdir()
            (skills_dir / "broken" / "not_a_skill.txt").write_text("ignored", encoding="utf-8")

            found = discover_skills(skills_dir)
            self.assertIn("my_skill", found)
            meta = found["my_skill"]
            self.assertEqual(meta.name, "my_skill")
            self.assertEqual(meta.description, "测试技能说明")
            self.assertEqual(meta.triggers, ("测试触发词", "demo"))
            self.assertTrue(meta.path.is_file())
            self.assertNotIn("broken", found)

    def test_discover_defaults_to_bundled_skills(self) -> None:
        from ai_assistant.skills.registry import discover_skills

        found = discover_skills()
        self.assertIn("training_resource", found)
        self.assertTrue(found["training_resource"].description)


class TestLoadSkill(unittest.TestCase):
    def test_load_training_resource_contains_ledger(self) -> None:
        from ai_assistant.skills.registry import load_skill

        body = load_skill("training_resource")
        self.assertIn("显存账本", body)
        self.assertIn("训练资源顾问", body)

    def test_load_unknown_skill_raises(self) -> None:
        from ai_assistant.skills.registry import load_skill

        with self.assertRaises(ValueError):
            load_skill("不存在的技能")


class TestQueryGpuEnv(unittest.TestCase):
    def test_query_gpu_env_uses_mocked_inventory(self) -> None:
        from ai_assistant.skills.queries import query_gpu_env

        fake = {
            "gpus": [
                {"index": 0, "name": "NVIDIA A800", "memory_total_mb": 81559, "memory_used_mb": 1024},
                {"index": 1, "name": "NVIDIA A800", "memory_total_mb": 81559, "memory_used_mb": 512},
            ],
            "count": 2,
        }
        with patch("science_infra.control.services.list_gpus", return_value=fake):
            payload = json.loads(query_gpu_env())
        self.assertEqual(payload["count"], 2)
        self.assertEqual(len(payload["gpus"]), 2)
        self.assertIn("recommendation_by_gpu_count", payload)
        rec = payload["recommendation_by_gpu_count"]
        self.assertIn("profile", rec)
        self.assertIn("trainer.n_gpus_per_node", rec)


class TestMatchSkills(unittest.TestCase):
    def test_training_message_matches_training_resource(self) -> None:
        from ai_assistant.skills.registry import match_skills

        matched = match_skills("训练 agentflow 要几张卡")
        names = [meta.name for meta in matched]
        self.assertIn("training_resource", names)

    def test_unrelated_message_matches_nothing(self) -> None:
        from ai_assistant.skills.registry import match_skills

        self.assertEqual(match_skills("今天天气如何"), [])


class TestAssistantToolSurface(unittest.TestCase):
    def test_tool_names_include_skills_and_memory(self) -> None:
        from ai_assistant import ToolSession, assistant_tools

        rows = assistant_tools("hive", session=ToolSession())
        names = {row["name"] for row in rows}
        for expected in (
            "list_skills",
            "load_skill",
            "query_gpu_env",
            "query_run",
            "query_experiment",
            "memory_read",
            "memory_write",
        ):
            self.assertIn(expected, names)

    def test_list_and_load_skill_tools_work(self) -> None:
        from ai_assistant import ToolSession, assistant_tools

        tools = {t["name"]: t["fn"] for t in assistant_tools("hive", session=ToolSession())}
        listing = tools["list_skills"]()
        self.assertIn("training_resource", listing)
        body = tools["load_skill"](name="training_resource")
        self.assertIn("显存账本", body)


if __name__ == "__main__":
    unittest.main()
