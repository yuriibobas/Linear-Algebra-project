import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics_suite import plot_analytics, save_top_results


def _load_trials(json_path: Path) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    trials = payload.get("all_trials", [])
    if not trials:
        raise ValueError("Input JSON does not contain non-empty 'all_trials'.")

    rows = []
    for row in trials:
        item = dict(row)
        det_metrics = item.get("det_metrics") or {}

        if isinstance(item.get("win_stride"), list) and len(item["win_stride"]) == 2:
            item["win_stride"] = f"{item['win_stride'][0]}x{item['win_stride'][1]}"
        if isinstance(item.get("padding"), list) and len(item["padding"]) == 2:
            item["padding"] = f"{item['padding'][0]}x{item['padding'][1]}"

        item["continuity_score"] = float(item.get("continuity", 0.0))
        item["texture_score"] = float(item.get("texture", 0.0))
        item["num_boxes"] = int(det_metrics.get("tp", 0) + det_metrics.get("fp", 0))
        item["f1"] = float(det_metrics.get("f1", 0.0))
        item["precision"] = float(det_metrics.get("precision", 0.0))
        item["recall"] = float(det_metrics.get("recall", 0.0))
        item["mean_iou_matched"] = float(det_metrics.get("mean_iou_matched", 0.0))
        item["preset"] = item.get("preset", "smart_tune")

        item["total_score"] = float(item.get("heuristic_total", item.get("total_score", 0.0)))

        rows.append(item)

    df = pd.DataFrame(rows)
    if "trial" in df.columns:
        df = df.sort_values("trial").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Reuse analytics_suite plotting for smart_tune_results.json."
    )
    parser.add_argument("--input", default="smart_tune_results.json", help="Path to smart tune JSON.")
    parser.add_argument(
        "--out-dir",
        default="analytics_output_smart_tune",
        help="Directory for generated plots and tables.",
    )
    args = parser.parse_args()

    json_path = Path(args.input)
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find input JSON: {json_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_trials(json_path)
    df.to_csv(out_dir / "smart_tune_all_results.csv", index=False)
    save_top_results(df, out_dir=out_dir, top_k=20)
    plot_analytics(df, out_dir=out_dir)
    print(f"Saved smart_tune analytics to: {out_dir}")


if __name__ == "__main__":
    main()
