from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize bot session logs")
    parser.add_argument("--log", required=True, help="Path to bot session log file")
    parser.add_argument("--exit-code", type=int, default=None, help="Optional process exit code")
    parser.add_argument("--max-errors", type=int, default=10, help="Maximum recent errors in summary")
    return parser.parse_args()


def summarize(log_path: Path, exit_code: int | None, max_errors: int) -> Dict[str, Any]:
    event_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    non_json_lines = 0
    json_event_lines = 0
    total_lines = 0
    first_event_utc = None
    last_event_utc = None
    errors: List[str] = []

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            total_lines += 1
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                non_json_lines += 1
                continue

            if not isinstance(payload, dict) or "event" not in payload:
                non_json_lines += 1
                continue

            json_event_lines += 1
            event = str(payload.get("event", "unknown"))
            event_counts[event] += 1

            if "action" in payload:
                action_counts[str(payload["action"])] += 1

            ts = payload.get("timestamp_utc")
            if isinstance(ts, str):
                if first_event_utc is None:
                    first_event_utc = ts
                last_event_utc = ts

            if event == "error":
                message = payload.get("message")
                errors.append(str(message) if message is not None else "<no message>")

    summary: Dict[str, Any] = {
        "log_file": str(log_path),
        "total_lines": total_lines,
        "json_event_lines": json_event_lines,
        "non_json_lines": non_json_lines,
        "first_event_utc": first_event_utc,
        "last_event_utc": last_event_utc,
        "event_counts": dict(sorted(event_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "error_count": len(errors),
        "recent_errors": errors[-max_errors:],
    }

    if exit_code is not None:
        summary["exit_code"] = exit_code
        summary["timed_out"] = exit_code == 124
        summary["successful"] = exit_code in (0, 124)

    return summary


def main() -> None:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    summary = summarize(log_path, args.exit_code, args.max_errors)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
