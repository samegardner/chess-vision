"""Structured per-game event log.

Writes one JSON object per line to a file. Each record has at least
`t` (seconds since session start) and `type` (event kind). Designed so
that after a game, the JSONL file plus the PGN are enough to fully
reconstruct what the detector saw and decided on every interesting
frame.

Event types currently emitted:
- snapshot: periodic dump of detector + main-loop state
- fire: a move was confirmed (greedy or two-move path)
- blocked: top candidate had a timer but couldn't fire (with reasons)
- undo: auto-undo retracted a move
- recalibrate: corner re-detection ran (whether it accepted or not)
- session_end: game ended or user quit
"""

import json
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        self.start = time.time()

    def log(self, event_type: str, **fields: Any) -> None:
        record = {
            "t": round(time.time() - self.start, 3),
            "type": event_type,
            **fields,
        }
        self.fh.write(json.dumps(record, default=str) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if not self.fh.closed:
            self.fh.close()


class NullEventLog:
    """No-op event log. Lets call sites stay unconditional."""
    def log(self, event_type: str, **fields: Any) -> None:
        pass
    def close(self) -> None:
        pass
