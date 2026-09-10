"""Make the service package importable when pytest runs from inf-ver-runner/.

These tests deliberately avoid app.verifier (torch/transformers imports); they
cover the pure mapping, polling and ledger-client layers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
