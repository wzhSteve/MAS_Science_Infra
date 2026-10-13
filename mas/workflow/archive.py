"""Transition shim: the Archive now lives at ``workflow/memory/archive.py``.

Kept for tests/scripts that still import ``from workflow.archive import ...``
(27 references across mas/tests). Deletion condition
(docs/refactor/phase0_external_api.md §6): once all consumers switch to
``workflow.memory.archive``, expected by Phase 4—5 of the refactor.
"""

from .memory.archive import *  # noqa: F401,F403
from .memory.archive import (  # noqa: F401  (explicit for IDE)
    Archive,
    ArchiveRef,  # re-exported transitively for legacy importers
    branch_point_to_resume_task_fields,
    dump_resume_with_archive,
    get_archive,
    get_default_archive,
    load_resume_messages,
    register_archive,
)
