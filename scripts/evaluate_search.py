from __future__ import annotations

import argparse
import json

from visual_memory.config import load_settings
from visual_memory.service import VisualMemoryService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSON file containing query and expected_event_id records")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    service = VisualMemoryService(load_settings(args.data_dir))
    with open(args.dataset, encoding="utf-8") as handle:
        queries = json.load(handle)["queries"]
    top1 = top5 = 0
    for item in queries:
        results = service.search.search(item["query"], limit=5)
        ids = [result["id"] for result in results]
        top1 += bool(ids and ids[0] == item["expected_event_id"])
        top5 += item["expected_event_id"] in ids
        print(f"{item['query']}: {ids}")
    count = len(queries)
    print(f"top-1: {top1}/{count}; top-5: {top5}/{count}")
    raise SystemExit(0 if top5 >= min(13, count) else 1)


if __name__ == "__main__":
    main()
