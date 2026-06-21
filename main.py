"""Command-line entry point for the Titanic AI agent scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_parser import PlanResolutionError, parse_instruction
from ml_pipeline import run_pipeline
from reporting import build_report, write_report

# CLI argument parser definise sta korisnik u cli komandi moze proslijediti. 
def build_parser():
    parser = argparse.ArgumentParser(
        description="Parse a Titanic ML instruction and produce a scaffold report."
    )
    parser.add_argument(
        "--instruction",
        help="Natural-language instruction to parse. Use this or --infile.",
    )
    parser.add_argument(
        "--infile",
        type=Path,
        help="Path to a text file containing the natural-language instruction.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        help="Optional path where the report should be written.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/titanic.csv"),
        help="Path to the Titanic CSV file. Default: data/titanic.csv.",
    )
    return parser

# Citanje instrukcija iz odgovarajuceg izvora
def read_instruction(args: argparse.Namespace):
    if args.instruction and args.infile:
        raise SystemExit("Use either --instruction or --infile, not both.")

    if args.instruction:
        return args.instruction.strip()

    if args.infile:
        try:
            return args.infile.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise SystemExit(f"Instruction file not found: {args.infile}") from exc

    if sys.stdin.isatty():
        return input("Instruction: ").strip()

    return sys.stdin.read().strip()


def main():
    parser = build_parser()
    args = parser.parse_args()
    instruction = read_instruction(args)

    if not instruction:
        parser.error("provide an instruction with --instruction, --infile, or stdin")

    try:
        plan = parse_instruction(instruction)
    except PlanResolutionError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        result = run_pipeline(plan=plan, data_path=args.data)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = build_report(instruction=instruction, plan=plan, result=result)

    print(report)

    if args.outfile:
        write_report(args.outfile, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
