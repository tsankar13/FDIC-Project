from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline
from .synopsis_report import generate_synopsis_report
from .visualizations import generate_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-first FDIC ML pipeline runner.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--input-dir", help="Directory containing FDIC CSV files.")
    parser.add_argument("--output-dir", default="backend/artifacts", help="Directory for run artifacts.")
    parser.add_argument(
        "--mode",
        choices=["clustering", "supervised", "synopsis", "both"],
        default="clustering",
        help="Pipeline mode to execute.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--label-column", default="FAILED")
    parser.add_argument(
        "--generate-synopsis-report",
        action="store_true",
        help="Generate an automated markdown synopsis report from synopsis artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        help="Optional specific run directory to source synopsis artifacts from.",
    )
    parser.add_argument(
        "--reports-dir",
        default="backend/artifacts/reports/primary",
        help="Directory where generated reports are written.",
    )
    parser.add_argument(
        "--generate-visualizations",
        action="store_true",
        help="Generate matplotlib visualizations and a markdown report from run artifacts.",
    )
    parser.add_argument(
        "--secondary-reports-dir",
        default="backend/artifacts/reports/secondary",
        help="Directory where visualization markdown reports are written.",
    )
    parser.add_argument(
        "--figures-dir",
        default="backend/artifacts/reports/secondary/figures",
        help="Directory where generated matplotlib figures are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_synopsis_report and args.generate_visualizations:
        raise ValueError("Use either --generate-synopsis-report or --generate-visualizations, not both.")

    if args.generate_synopsis_report:
        report_path = generate_synopsis_report(
            output_dir=Path(args.output_dir),
            reports_dir=Path(args.reports_dir),
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(json.dumps({"generated_report": str(report_path)}, indent=2, sort_keys=True))
        return

    if args.generate_visualizations:
        output = generate_visualizations(
            output_dir=Path(args.output_dir),
            report_dir=Path(args.secondary_reports_dir),
            figures_dir=Path(args.figures_dir),
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(output.run_dir),
                    "report_path": str(output.report_path),
                    "figure_paths": [str(path) for path in output.figure_paths],
                    "skipped_charts": output.skipped_charts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.config:
        file_config = json.loads(Path(args.config).read_text())
        config = PipelineConfig.from_dict(file_config)
    else:
        if not args.input_dir:
            raise ValueError("--input-dir is required when --config is not provided.")
        config = PipelineConfig(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            mode=args.mode,
            random_seed=args.random_seed,
            max_clusters=args.max_clusters,
            label_column=args.label_column,
        )
    result = run_pipeline(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
