"""Make the service package importable when pytest runs from frame-processor/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
