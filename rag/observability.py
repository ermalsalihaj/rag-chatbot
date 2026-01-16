import csv
import os
from datetime import datetime
from typing import Any, Dict, List

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, int(len(text or "") / 4))

def append_log_csv(csv_path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            escapechar="\\",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
