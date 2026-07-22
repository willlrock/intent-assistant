"""Evaluate the complete Rasa NLU pipeline, including fallback decisions.

Unlike ``rasa test nlu``, this script sends every held-out example through a
loaded ``Agent``.  The resulting report therefore reflects the configured
``FallbackClassifier`` and stores aggregates only, never example texts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


FALLBACK_INTENT = "nlu_fallback"
TOP_CONFUSIONS_LIMIT = 15


class InputValidationError(ValueError):
    """Raised when the evaluation inputs do not have the expected shape."""


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a JSON-friendly ratio, including for an empty denominator."""

    return numerator / denominator if denominator else 0.0


def _examples_from_block(value: Any, context: str) -> list[str]:
    """Extract annotated example strings from a Rasa YAML examples block."""

    if isinstance(value, str):
        examples: list[str] = []
        for line_number, line in enumerate(value.splitlines(), start=1):
            if not line.strip():
                continue
            stripped = line.strip()
            if not stripped.startswith("-"):
                raise InputValidationError(
                    f"{context}, line {line_number}: each multiline example must "
                    "start with '-'."
                )
            example = stripped[1:].strip()
            if not example:
                raise InputValidationError(
                    f"{context}, line {line_number}: example text is empty."
                )
            examples.append(example)
        return examples

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        examples = []
        for example_index, item in enumerate(value, start=1):
            if isinstance(item, str):
                example = item.strip()
            elif isinstance(item, Mapping):
                text = item.get("text")
                if not isinstance(text, str):
                    raise InputValidationError(
                        f"{context}, example {example_index}: list entries must "
                        "contain a string 'text' field."
                    )
                example = text.strip()
            else:
                raise InputValidationError(
                    f"{context}, example {example_index}: expected a string or "
                    "a mapping with a 'text' field."
                )

            if not example:
                raise InputValidationError(
                    f"{context}, example {example_index}: example text is empty."
                )
            examples.append(example)
        return examples

    raise InputValidationError(
        f"{context}: 'examples' must be a multiline string or a list."
    )


def load_labeled_examples(nlu_path: Path) -> list[tuple[str, str]]:
    """Load ``(expected_intent, plain_text)`` pairs from Rasa NLU YAML."""

    from rasa.shared.nlu.training_data.entities_parser import replace_entities

    try:
        document = yaml.safe_load(nlu_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"NLU file is not valid UTF-8: {nlu_path}") from exc
    except yaml.YAMLError as exc:
        raise InputValidationError(f"Invalid YAML in {nlu_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise InputValidationError("NLU YAML root must be a mapping.")

    version = document.get("version")
    if not isinstance(version, str) or not version.strip():
        raise InputValidationError("NLU YAML must contain a non-empty string 'version'.")

    nlu_blocks = document.get("nlu")
    if not isinstance(nlu_blocks, list):
        raise InputValidationError("NLU YAML field 'nlu' must be a list.")
    if not nlu_blocks:
        raise InputValidationError("NLU YAML field 'nlu' must not be empty.")

    labeled_examples: list[tuple[str, str]] = []
    for block_index, block in enumerate(nlu_blocks, start=1):
        context = f"NLU block {block_index}"
        if not isinstance(block, Mapping):
            raise InputValidationError(f"{context} must be a mapping.")

        intent = block.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            raise InputValidationError(
                f"{context} must contain a non-empty string 'intent'."
            )
        intent = intent.strip()
        if intent == FALLBACK_INTENT:
            raise InputValidationError(
                f"{context} uses reserved runtime intent '{FALLBACK_INTENT}' as "
                "a gold label. Label the message with its expected domain intent "
                "instead."
            )

        if "examples" not in block:
            raise InputValidationError(f"{context} is missing 'examples'.")
        annotated_examples = _examples_from_block(block["examples"], context)
        if not annotated_examples:
            raise InputValidationError(f"{context} contains no examples.")

        for annotated_text in annotated_examples:
            plain_text = replace_entities(annotated_text).strip()
            if not plain_text:
                raise InputValidationError(
                    f"{context} contains an example that is empty after removing "
                    "entity markup."
                )
            labeled_examples.append((intent, plain_text))

    if not labeled_examples:
        raise InputValidationError("NLU YAML contains no labeled examples.")
    return labeled_examples


def _raw_candidate_name(parsed: Mapping[str, Any], predicted: str) -> str:
    """Return the pre-fallback top intent for one runtime prediction."""

    if predicted != FALLBACK_INTENT:
        return predicted

    ranking = parsed.get("intent_ranking")
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
        raise RuntimeError(
            "Rasa fallback prediction did not contain an intent_ranking sequence."
        )

    for candidate in ranking:
        if not isinstance(candidate, Mapping):
            continue
        name = candidate.get("name")
        if isinstance(name, str) and name and name != FALLBACK_INTENT:
            return name

    raise RuntimeError(
        "Rasa fallback prediction did not contain a raw non-fallback candidate."
    )


async def evaluate(
    model_path: Path, labeled_examples: list[tuple[str, str]]
) -> dict[str, Any]:
    """Run all examples through one loaded agent and aggregate predictions."""

    from rasa.core.agent import Agent

    agent = Agent.load(str(model_path))
    if not agent.is_ready():
        raise RuntimeError(f"Rasa Agent could not load a ready model from {model_path}")

    correct = 0
    raw_correct = 0
    fallback_count = 0
    fallback_raw_correct_count = 0
    fallback_raw_incorrect_count = 0
    accepted_correct = 0
    fallback_expected: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()

    for expected, text in labeled_examples:
        parsed = await agent.parse_message(text)
        intent_data = parsed.get("intent")
        if not isinstance(intent_data, Mapping):
            raise RuntimeError("Rasa parser returned no intent mapping.")
        predicted = intent_data.get("name")
        if not isinstance(predicted, str) or not predicted:
            raise RuntimeError("Rasa parser returned an invalid intent name.")

        raw_predicted = _raw_candidate_name(parsed, predicted)
        is_correct = predicted == expected
        is_raw_correct = raw_predicted == expected
        if is_correct:
            correct += 1
        if is_raw_correct:
            raw_correct += 1

        if predicted == FALLBACK_INTENT:
            fallback_count += 1
            fallback_expected[expected] += 1
            if is_raw_correct:
                fallback_raw_correct_count += 1
            else:
                fallback_raw_incorrect_count += 1
        else:
            if is_correct:
                accepted_correct += 1

        if not is_correct:
            confusions[(expected, predicted)] += 1

    total = len(labeled_examples)
    accepted_count = total - fallback_count
    ordered_confusions = sorted(
        confusions.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )[:TOP_CONFUSIONS_LIMIT]
    ordered_fallback_distribution = sorted(
        fallback_expected.items(), key=lambda item: (-item[1], item[0])
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_ratio(correct, total),
        "raw_correct": raw_correct,
        "raw_accuracy": _safe_ratio(raw_correct, total),
        "fallback_count": fallback_count,
        "fallback_raw_correct_count": fallback_raw_correct_count,
        "fallback_raw_incorrect_count": fallback_raw_incorrect_count,
        "fallback_rate": _safe_ratio(fallback_count, total),
        "accepted_count": accepted_count,
        "coverage": _safe_ratio(accepted_count, total),
        "accepted_correct": accepted_correct,
        "selective_accuracy": _safe_ratio(accepted_correct, accepted_count),
        "fallback_expected_distribution": {
            intent: count for intent, count in ordered_fallback_distribution
        },
        "top_confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in ordered_confusions
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a full Rasa NLU pipeline, including FallbackClassifier, "
            "without storing source example texts."
        )
    )
    parser.add_argument("MODEL", type=Path, help="Path to a trained Rasa model")
    parser.add_argument(
        "--nlu",
        type=Path,
        default=Path("tests/nlu_test.yml"),
        help="Held-out Rasa NLU YAML (default: tests/nlu_test.yml)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Rasa configuration file to fingerprint (default: config.yml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/runtime_report.json"),
        help=(
            "Aggregate JSON report path "
            "(default: results/final/runtime_report.json)"
        ),
    )
    return parser


def _configure_rasa_logging() -> None:
    """Keep Rasa from logging evaluated message texts at debug level."""

    from rasa.utils.common import configure_logging_and_warnings
    from rasa.utils.log_utils import configure_structlog

    configure_logging_and_warnings(log_level=logging.ERROR)
    configure_structlog(log_level=logging.ERROR)


def _portable_path(path: Path) -> str:
    """Return a repository-relative path without leaking a local home path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    model_path: Path = args.MODEL.expanduser()
    nlu_path: Path = args.nlu.expanduser()
    config_path: Path = args.config.expanduser()
    output_path: Path = args.output.expanduser()

    if not model_path.is_file():
        parser.error(f"model file does not exist: {model_path}")
    if not nlu_path.is_file():
        parser.error(f"NLU file does not exist: {nlu_path}")
    if not config_path.is_file():
        parser.error(f"config file does not exist: {config_path}")

    try:
        labeled_examples = load_labeled_examples(nlu_path)
    except (OSError, InputValidationError) as exc:
        parser.error(str(exc))

    _configure_rasa_logging()
    report = asyncio.run(evaluate(model_path, labeled_examples))
    report["metadata"] = {
        "model_file": _portable_path(model_path),
        "model_sha256": _sha256(model_path),
        "nlu_file": _portable_path(nlu_path),
        "nlu_sha256": _sha256(nlu_path),
        "config_file": _portable_path(config_path),
        "config_sha256": _sha256(config_path),
        "python_version": platform.python_version(),
        "rasa_version": importlib.metadata.version("rasa"),
        "rasa_sdk_version": importlib.metadata.version("rasa-sdk"),
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote aggregate runtime NLU report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
