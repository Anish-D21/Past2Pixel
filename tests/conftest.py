"""
Pytest Configuration and Fixtures for Past2Pixel.
Configures sys.path to locate src package across all test runners.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Unless explicitly instructed to hit live Atlas (e.g. RUN_ATLAS_TESTS=1),
# isolate automated unit tests to local persistence for offline reliability.
import os
if not os.getenv("RUN_ATLAS_TESTS"):
    os.environ["MONGODB_URI"] = ""
