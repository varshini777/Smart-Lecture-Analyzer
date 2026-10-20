"""Script launcher for Lecture Engagement Pattern Analyzer desktop application."""

import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_analyzer.app import main

if __name__ == "__main__":
    sys.exit(main())
