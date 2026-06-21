"""Report formatting helpers."""

from __future__ import annotations

from pathlib import Path


def build_report(
    instruction: str, plan: dict[str, object], result: dict[str, object]
):
    lines = [
        "Report",
        "",
        "Instruction:",
        instruction,
        "",
        "Resolved plan:",
    ]

    for key, value in plan.items():
        if key == "warnings":
            continue
        lines.append(f"- {key}: {value}")

    warnings = [
        *[str(warning) for warning in plan.get("warnings", [])],
        *[str(warning) for warning in result.get("warnings", [])],
    ]

    lines.extend(
        [
            "",
            f"Status: {result['status']}",
            f"Data path: {result['data_path']}",
            f"Rows: {result.get('rows', 'n/a')}",
            f"Train rows: {result.get('train_rows', 'n/a')}",
            f"Test rows: {result.get('test_rows', 'n/a')}",
            f"Model used: {result.get('model', 'n/a')}",
            f"Features used: {result.get('features', [])}",
            "",
            "Metrics:",
            _format_confusion_matrix(result.get("metrics", {}).get("confusion_matrix")),
            f"Accuracy: {_format_metric(result.get('metrics', {}).get('accuracy'))}",
            f"Precision: {_format_metric(result.get('metrics', {}).get('precision'))}",
            f"Recall: {_format_metric(result.get('metrics', {}).get('recall'))}",
            f"F1: {_format_metric(result.get('metrics', {}).get('f1'))}",
            "",
            "Warnings:",
        ]
    )

    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    return "\n".join(lines)


def write_report(path: Path, report: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")


def _format_confusion_matrix(value: object):
    if not value:
        return "Confusion matrix: n/a"
    return f"Confusion matrix: {value}"


def _format_metric(value: object):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
