"""Regression tests for Web_Search URL injection."""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAS.epc_aw.solver import Solver


@pytest.mark.parametrize(
    "command,url",
    [
        (
            'execution = tool.execute(query="Nature journal 2020 number of research articles")',
            "https://en.wikipedia.org/",
        ),
        (
            "execution = tool.execute(query='Pie Menus or Linear Menus Which Is Better? 2015 authors')",
            "https://en.wikipedia.org/",
        ),
    ],
)
def test_inject_url_produces_valid_python(command: str, url: str) -> None:
    fixed = Solver._inject_url_into_command(command, url)
    ast.parse(fixed)
    assert f'url="{url}"' in fixed
    assert fixed.count(")") == command.count(")")


def test_inject_url_replaces_existing() -> None:
    command = 'execution = tool.execute(query="x", url="https://old.example/")'
    fixed = Solver._inject_url_into_command(command, "https://new.example/")
    ast.parse(fixed)
    assert "https://new.example/" in fixed
    assert "https://old.example/" not in fixed
