"""science-infra: thin MAS research CLI (see science_infra.ui.cli)."""

from .env import load_science_env

__version__ = "0.1.0"

# Load Agent_Science_Infra/.env on import (process env wins).
load_science_env()
