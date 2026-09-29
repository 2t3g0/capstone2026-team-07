"""Explicit-execution, no-vehicle Phase1 event rehearsal."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jolgwa_uav.field_event_bench import main

if __name__ == "__main__":
    raise SystemExit(main())
