"""Make the service package importable when pytest runs from
inf-ver-orchestrator/. Tests are pure logic + a mocked ledger HTTP layer; the
kubernetes API is faked behind KubeClient's interface (never called)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
