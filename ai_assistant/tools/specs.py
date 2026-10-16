"""Skill + memory tool descriptors (OpenAI function-calling schema)."""

from __future__ import annotations

SKILL_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List available skills (name + description). Cheap; call when unsure what a skill does.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load the full instructions of one skill. Call before using that skill's domain.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Skill name from list_skills"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_gpu_env",
            "description": "Read-only: real nvidia-smi inventory, per-GPU-count recommendation, and the experiment's derived training plan (profile/algorithm/gpu_ids/checks).",
            "parameters": {
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_run",
            "description": "Read-only: latest (or given) train/eval run status plus the stdout tail, for diagnosis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "tail_lines": {"type": "integer"},
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_experiment",
            "description": "Read-only: summarize an experiment's five YAML sections (llm/workflow/rl/harness).",
            "parameters": {
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
                "required": ["experiment_id"],
            },
        },
    },
]

MEMORY_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read long-term facts for a scope (defaults to the current experiment, else global).",
            "parameters": {
                "type": "object",
                "properties": {"scope": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Persist a durable fact for a scope (machine facts, user preferences, decisions). Max 500 chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "pinned": {"type": "boolean", "description": "Pinned facts are injected into every future request."},
                },
                "required": ["text"],
            },
        },
    },
]
