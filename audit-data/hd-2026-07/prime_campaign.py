"""Curated fixed-lattice campaigns for the prime/Radon coloring search.

Example:

    python prime_campaign.py E7*-ABPR \
        '[[7,7,7,3], [7,7,5,4], [7,7,5,5]]' \
        --restarts 40 --output prime_campaign_e7.json

The forbidden set is built once and reused across all quotient structures.
Results are checkpointed after every structure so long campaigns are resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

from prime_radon import PrimarySearch, load_forbidden


def parse_structures(text: str) -> list[list[int]]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("structures must be a non-empty JSON list")
    result: list[list[int]] = []
    for item in raw:
        if not isinstance(item, list) or not item:
            raise argparse.ArgumentTypeError("each structure must be a non-empty list")
        result.append([int(value) for value in item])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice")
    parser.add_argument("structures", type=parse_structures)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    _, forbidden, diameter = load_forbidden(args.lattice)
    print(
        f"{args.lattice}: n={forbidden.shape[1]} |F|={len(forbidden)} "
        f"diam={diameter:.12g}; {len(args.structures)} structures",
        flush=True,
    )
    payload: dict = {
        "lattice": args.lattice,
        "n": int(forbidden.shape[1]),
        "n_forbidden": int(len(forbidden)),
        "diameter": diameter,
        "budget": {
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "seed": args.seed,
        },
        "results": [],
    }
    campaign_start = time.perf_counter()
    for number, moduli in enumerate(args.structures, start=1):
        product = math.prod(moduli)
        print(
            f"\n[{number}/{len(args.structures)}] moduli={moduli} "
            f"product={product}",
            flush=True,
        )
        search = PrimarySearch(forbidden, moduli, seed=args.seed + 1009 * number)
        result = search.run(
            restarts=args.restarts,
            max_sweeps=args.sweeps,
            top=args.top,
            progress_every=max(1, args.restarts // 4),
        )
        record = {
            "target_product": product,
            **result.as_json(),
        }
        payload["results"].append(record)
        if args.output:
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"[{number}/{len(args.structures)}] done: killed={result.killed} "
            f"image={result.image_index} time={result.seconds:.2f}s",
            flush=True,
        )
        if result.found and result.image_index < 1372:
            print(
                f"*** NEW FIXED-LATTICE CANDIDATE: index={result.image_index} ***",
                flush=True,
            )
    payload["campaign_seconds"] = round(time.perf_counter() - campaign_start, 6)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved {args.output}", flush=True)
    print("\n=== SUMMARY ===")
    for record in sorted(payload["results"], key=lambda item: item["killed"]):
        print(
            f"  {record['moduli']}: product={record['target_product']} "
            f"killed={record['killed']} image={record['image_index']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
