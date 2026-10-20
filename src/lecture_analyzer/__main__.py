"""Package execution entry point for 'python -m lecture_analyzer'."""

import sys
from lecture_analyzer.app import main

if __name__ == "__main__":
    sys.exit(main())
