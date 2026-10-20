"""CLI utility to export provenance-filtered datasets."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from lecture_analyzer.config import load_config
from lecture_analyzer.dataset.export_dataset import DatasetExporter
from lecture_analyzer.dataset.provenance import DataProvenance
from lecture_analyzer.storage.database import Database


def main():
    parser = argparse.ArgumentParser(description="Export datasets with strict provenance filtering.")
    parser.add_argument(
        "--provenance",
        choices=["REAL_HUMAN", "SIMULATED", "PIPELINE_VALIDATION", "UNKNOWN"],
        default="REAL_HUMAN",
        help="Target data provenance to export (default: REAL_HUMAN)"
    )
    parser.add_argument("--allow-unannotated", action="store_true", help="Include records without human annotations")
    parser.add_argument("--ignore-audit", action="store_true", help="Bypass audit pass requirement")
    args = parser.parse_args()

    cfg = load_config()
    db = Database(cfg.storage.database_path)
    exporter = DatasetExporter(db, cfg.storage.exports_dir)

    prov_enum = DataProvenance(args.provenance)
    print(f"[*] Exporting dataset with provenance: {prov_enum.value}...")

    csv_path, manifest = exporter.export_dataset(
        provenance=prov_enum,
        require_annotations=not args.allow_unannotated,
        require_passed_audit=not args.ignore_audit,
    )

    if manifest:
        print("\n=== Export Manifest ===")
        print(json.dumps(manifest.to_dict(), indent=2))

    if csv_path:
        print(f"\n[SUCCESS] Dataset exported to: {csv_path}")
        return 0
    else:
        print("\n[WARNING] No eligible records found matching criteria.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
