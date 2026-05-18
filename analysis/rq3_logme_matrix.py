from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.multi_scenario_eval import redirect_output_root
from src.train.utils import load_graph_bundle, save_dataframe
from src.transfer.transferability_analysis import compute_pair_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 8 E2 LogME transferability matrix wrapper.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    if args.smoke_test:
        redirect_output_root(bundle, "week8_smoke")
    pair_scores = compute_pair_scores(bundle, smoke_test=args.smoke_test)
    save_dataframe(pair_scores, bundle.output_paths["metrics"] / "E2_pair_scores.csv")


if __name__ == "__main__":
    main()
