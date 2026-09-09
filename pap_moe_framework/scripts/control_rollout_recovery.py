#!/usr/bin/env python3
"""Publish expert chunks, request takeover, or mark a recovery outcome."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from pap_moe_framework.rollout_recovery.protocol import (
    PROTOCOL_VERSION,
    SKILL_PROGRESS_PHASES,
    publish_expert_chunk,
    request_takeover,
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish-expert")
    publish.add_argument("--session-dir", type=Path, required=True)
    publish.add_argument("--chunk", type=Path, required=True)
    publish.add_argument("--sequence", type=int, required=True)
    publish.add_argument("--publisher", required=True)
    publish.add_argument("--skill-progress-phase", choices=SKILL_PROGRESS_PHASES, required=True)
    publish.add_argument("--skill-progress", type=float, required=True)
    publish.add_argument("--transition-readiness", type=float, required=True)
    publish.add_argument("--label-confidence", type=float, default=1.0)

    takeover = subparsers.add_parser("takeover")
    takeover.add_argument("--session-dir", type=Path, required=True)
    takeover.add_argument("--trigger", required=True)
    takeover.add_argument("--requester", required=True)

    outcome = subparsers.add_parser("outcome")
    outcome.add_argument("--outcome-file", type=Path, required=True)
    outcome.add_argument("--value", choices=("success", "failure"), required=True)
    outcome.add_argument("--operator", required=True)
    outcome.add_argument("--note", default="")

    args = parser.parse_args()
    if args.command == "publish-expert":
        loaded = np.load(args.chunk, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                chunk = np.asarray(loaded["action_chunk"]).copy()
            finally:
                loaded.close()
        else:
            chunk = loaded
        path = publish_expert_chunk(
            args.session_dir,
            chunk,
            sequence=args.sequence,
            publisher=args.publisher,
            skill_progress_phase=SKILL_PROGRESS_PHASES.index(args.skill_progress_phase),
            skill_progress=args.skill_progress,
            transition_readiness=args.transition_readiness,
            label_confidence=args.label_confidence,
        )
        print(path)
    elif args.command == "takeover":
        request_takeover(
            args.session_dir,
            trigger=args.trigger,
            requester=args.requester,
        )
        print(args.session_dir / "input" / "takeover.json")
    else:
        _atomic_json(
            args.outcome_file,
            {
                "protocol_version": PROTOCOL_VERSION,
                "timestamp": time.time(),
                "outcome": args.value,
                "operator": args.operator,
                "note": args.note,
            },
        )
        print(args.outcome_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
