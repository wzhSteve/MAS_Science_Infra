"""Legacy research stacks physically isolated from the production ``mas/`` tree.

Phase 1 of the MAS refactor (docs/refactor/phase0_external_api.md):
  - ``legacy/structagent/`` — vendored StructAgent / EPC-AW engine
    (importable as ``legacy.structagent``; the dynamic ``sys.modules["MAS"]``
    registration performed by its ``mas_agent.bootstrap_mas_package`` is
    unchanged and now points at ``legacy/structagent/epc_aw``).

Nothing inside ``mas/`` may import from ``legacy.*`` at module import time;
the only sanctioned coupling is the lazy optional backend loader in
``mas/tools/tool_agents.py::_import_epc_aw_tool``.
"""
