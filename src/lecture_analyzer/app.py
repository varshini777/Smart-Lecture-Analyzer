"""Application entry point and GUI launch orchestrator."""

import argparse
import logging
import sys

from PySide6.QtWidgets import QApplication

from lecture_analyzer.config import load_config
from lecture_analyzer.logging_config import setup_logging
from lecture_analyzer.ui.main_window import MainWindow

logger = logging.getLogger("lecture_analyzer.app")


def main() -> int:
    """Initializes configuration, logging, and PySide6 application."""
    parser = argparse.ArgumentParser(description="Lecture Engagement Pattern Analyzer")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # 1. Load validated configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[FATAL] Configuration validation error: {e}", file=sys.stderr)
        return 1

    # 2. Setup structured logging
    setup_logging(level=config.system.log_level, log_dir=config.system.log_dir)
    logger.info(f"Starting {config.system.app_name} v{config.system.version}...")

    # 3. Create Qt Application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName(config.system.app_name)
    app.setApplicationVersion(config.system.version)

    window = MainWindow(config)
    if hasattr(window, "tab_live") and hasattr(window.tab_live, "start_prewarm"):
        window.tab_live.start_prewarm()
    window.show()

    logger.info("Application main window launched.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
