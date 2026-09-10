"""Make the service package importable when pytest runs from inf-proxy/.

These tests cover the pure declaration logic (payload building, model-name
discovery parsing, retry behaviour) without a running ledger or model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
