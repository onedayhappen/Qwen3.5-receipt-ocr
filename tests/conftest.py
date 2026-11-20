"""Keep test temporary data inside the project; never auto-clean personal directories."""
from pathlib import Path
import uuid


def pytest_configure(config):
    if not config.option.basetemp:
        root = Path(__file__).resolve().parents[1] / ".pytest_runs"
        root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(root / ("run-" + uuid.uuid4().hex))
    config._inicache["tmp_path_retention_policy"] = "all"
    config._inicache["tmp_path_retention_count"] = 1000000
