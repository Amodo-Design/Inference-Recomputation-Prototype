"""Make the service package importable when pytest runs from prompt-runner/.

Tests stub the prompt suite and the per-prompt HTTP call: promptbench (and
its datasets) are never imported, and nothing talks to a real Open WebUI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
