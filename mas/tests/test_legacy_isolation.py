"""Phase 1 isolation tests for the legacy/structagent split (docs/refactor/phase0_*).

Covers the three mandatory checks from MAS_design.md §1.3:
  1. dual-stack same-process import: product stack (workflow/tools) first,
     then vendored bootstrap, then the reverse order — neither pollutes
     the other's module identities.
  2. module identity: ``sys.modules["MAS"].__path__`` points at the vendored
     engine's physical location under ``legacy/structagent/epc_aw``.
  3. missing optional dependency: removing ``legacy/structagent`` from
     sys.path must NOT break the product stack; the pro backend factory
     degrades to None explicitly (ImportError semantics, not a crash).

The tests need the repo root on sys.path (they add it themselves) and skip
when the legacy tree is not present in this checkout.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
MAS_ROOT = REPO_ROOT / "mas"
LEGACY_ROOT = REPO_ROOT / "legacy" / "structagent"
EPC_AW_PATH = LEGACY_ROOT / "epc_aw"

REQUIRES_LEGACY = unittest.skipUnless(
    LEGACY_ROOT.is_dir() and EPC_AW_PATH.is_dir(),
    "legacy/structagent vendored stack not present in this checkout",
)


def _ensure_repo_root() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _ensure_mas_root() -> None:
    if str(MAS_ROOT) not in sys.path:
        sys.path.insert(0, str(MAS_ROOT))


class TestDualStackSameProcess(unittest.TestCase):
    """Product main stack and vendored StructAgent must coexist in-process."""

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_repo_root()
        _ensure_mas_root()

    @REQUIRES_LEGACY
    def test_product_first_then_vendored(self) -> None:
        import workflow.runtime  # noqa: F401  (product main stack)
        import tools.tool_agents  # noqa: F401

        self._bootstrap_and_assert()

    @REQUIRES_LEGACY
    def test_vendored_first_then_product(self) -> None:
        self._bootstrap_and_assert()
        import workflow.runtime  # noqa: F401
        import tools.tool_agents  # noqa: F401

        # product modules must not be re-bound by the vendored bootstrap
        self.assertTrue(str(workflow.runtime.__file__ or "").startswith(str(MAS_ROOT)))
        self.assertTrue(str(tools.tool_agents.__file__ or "").startswith(str(MAS_ROOT)))

    def _bootstrap_and_assert(self) -> None:
        # The mas/ shim keeps the top-level ``mas_agent`` name importable
        # under PYTHONPATH=mas (training hot path contract, Phase 0 §1).
        _ensure_mas_root()
        mas_agent = importlib.import_module("mas_agent")
        root = mas_agent.bootstrap_mas_package()
        # bootstrap returns the structagent dir whose ``epc_aw/`` is the
        # vendored engine root (MAS.__path__ == <that dir>)
        self.assertEqual(root, LEGACY_ROOT.resolve(), "bootstrap must locate legacy/structagent")
        self.assertTrue((root / "epc_aw").is_dir(), "vendored engine root must contain epc_aw/")

        mas_mod = sys.modules.get("MAS")
        self.assertIsNotNone(mas_mod, "bootstrap must register sys.modules['MAS']")
        paths = [str(Path(p).resolve()) for p in getattr(mas_mod, "__path__", [])]
        # MAS.epc_aw resolves through __path__ = structagent dir
        self.assertIn(str(LEGACY_ROOT.resolve()), paths)
        epc = importlib.import_module("MAS.epc_aw")
        self.assertEqual(
            [str(Path(p).resolve()) for p in epc.__path__],
            [str(EPC_AW_PATH.resolve())],
        )


class TestModuleIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_repo_root()
        _ensure_mas_root()

    @REQUIRES_LEGACY
    def test_mas_registered_path_is_unique_and_vendored(self) -> None:
        mas_agent = importlib.import_module("mas_agent")
        mas_agent.bootstrap_mas_package()
        mas_mod = sys.modules["MAS"]
        # single physical root: the structagent dir (bootstrap invariants)
        paths = [str(Path(p).resolve()) for p in mas_mod.__path__]
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], str(LEGACY_ROOT.resolve()))
        # and the vendored engine internals resolve through the MAS name
        engine = importlib.import_module("MAS.epc_aw.engine")
        self.assertTrue(str(engine.__file__ or "").startswith(str(EPC_AW_PATH.resolve())))


class TestMissingOptionalDependency(unittest.TestCase):
    """Product stack must survive (lite fallback) without the vendored engine."""

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_mas_root()
        _ensure_repo_root()

    def setUp(self) -> None:
        self._saved_path = list(sys.path)
        # simulate "legacy/structagent not installed": drop every sys.path
        # entry that would let ``legacy.*`` resolve, plus any cached modules.
        sys.path = [p for p in sys.path if not str(p).startswith(str(REPO_ROOT))]
        for name in [n for n in list(sys.modules) if n == "legacy" or n.startswith("legacy.")]:
            del sys.modules[name]

    def tearDown(self) -> None:
        sys.path = self._saved_path
        sys.path.insert(0, str(REPO_ROOT))

    def test_product_imports_and_tools_kernel_still_work(self) -> None:
        import tools.tool_agents as ta  # product main stack import survives

        if not (MAS_ROOT / "tools" / "search.py").exists():  # pragma: no cover
            self.skipTest("search kernel unavailable")
        # kernel path (tier resolution) is independent of the vendored stack
        self.assertEqual(ta.effective_tier("pro"), "lite" if _kernel_forced() else "pro")
        self.assertEqual(ta.effective_tier(None), "lite")
        self.assertEqual(ta.effective_tier("lite"), "lite")

    def test_pro_backend_degrades_explicitly(self) -> None:
        import tools.tool_agents as ta

        # Degradation happens at *closure call time* (the vendored import is
        # lazy inside each backend fn), not at factory time — the factory
        # happily returns closures; invoking one without the vendored stack
        # must raise the explicit ImportError (missing-dependency error
        # semantics per MAS_design §1.3, not a silent bogus result).
        self.assertIsNotNone(ta._epc_aw_llm_backend("think"), "factory returns lazy closures")
        backend = ta._epc_aw_llm_backend("think")
        with mock.patch.object(ta, "_import_epc_aw_tool") as fake:
            fake.side_effect = ImportError("legacy.structagent.epc_aw missing")
            with self.assertRaises(ImportError):
                backend({"text": "hi"})

        # When the tree genuinely cannot be resolved (no repo root anywhere,
        # no fallback possible), the loader itself raises ImportError:
        sys.path = [p for p in sys.path if not str(p).startswith(str(REPO_ROOT))]
        for name in [n for n in list(sys.modules) if n == "legacy" or n.startswith("legacy.")]:
            del sys.modules[name]
        with self.assertRaises(ImportError):
            ta._import_epc_aw_tool("epc_aw.tools.base_generator.tool", "Base_Generator_Tool")
        sys.path[:] = self._saved_path  # hard restore before leaving this branch

        # The product stack itself never imports legacy at module import
        # time, so an unresolvable vendored stack never blocks the module:
        sys.path = [p for p in sys.path if not str(p).startswith(str(REPO_ROOT))]
        import importlib as _il
        _il.reload(ta)
        sys.path[:] = self._saved_path


def _kernel_forced() -> bool:
    import os

    return os.getenv("SCIENCE_INFRA_TOOL_KERNEL", "").strip().lower() in ("1", "true", "yes")


if __name__ == "__main__":
    unittest.main()
