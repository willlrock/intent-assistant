"""Create a compact Markdown summary from Rasa NLU evaluation artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def count_nlu_examples(path: Path) -> int:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    blocks = document.get("nlu", []) if isinstance(document, dict) else []
    count = 0
    for block in blocks:
        if not isinstance(block, dict) or "intent" not in block:
            continue
        examples = block.get("examples", "")
        if isinstance(examples, str):
            count += sum(
                1 for line in examples.splitlines() if line.strip().startswith("-")
            )
        elif isinstance(examples, list):
            count += len(examples)
    return count


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "не установлена"


def build_summary(
    results_dir: Path,
    train_data: Path,
    test_data: Path,
) -> str:
    intent_report = load_json(results_dir / "intent_report.json")
    intent_errors = load_json(results_dir / "intent_errors.json")
    entity_report = load_json(results_dir / "DIETClassifier_report.json")
    entity_errors = load_json(results_dir / "DIETClassifier_errors.json")

    macro = intent_report["macro avg"]
    weighted = intent_report["weighted avg"]
    accuracy = intent_report["accuracy"]
    support = int(macro["support"])

    class_rows = []
    ignored = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    for name, metrics in intent_report.items():
        if name in ignored or not isinstance(metrics, dict):
            continue
        class_rows.append(
            (
                name,
                float(metrics["precision"]),
                float(metrics["recall"]),
                float(metrics["f1-score"]),
                int(metrics["support"]),
            )
        )
    class_rows.sort(key=lambda row: (row[3], row[0]))

    confusions = Counter(
        (error["intent"], error["intent_prediction"]["name"])
        for error in intent_errors
    )
    person_name_metrics = entity_report.get("person_name")
    if not isinstance(person_name_metrics, dict):
        # Backward-compatible fallback for older single-entity Rasa reports.
        person_name_metrics = entity_report["macro avg"]
    runtime_report_path = results_dir / "runtime_report.json"
    runtime_report = (
        load_json(runtime_report_path) if runtime_report_path.is_file() else None
    )
    core_report_path = results_dir / "story_report.json"
    core_report = load_json(core_report_path) if core_report_path.is_file() else None
    train_count = count_nlu_examples(train_data)
    test_count = count_nlu_examples(test_data)

    lines = [
        "# Итоги NLU-оценки",
        "",
        f"- Дата запуска (UTC): **{datetime.now(timezone.utc):%Y-%m-%d}**",
        f"- Окружение: **Python {platform.python_version()}, "
        f"Rasa {package_version('rasa')}, Rasa SDK {package_version('rasa-sdk')}**",
        f"- Данные: **{train_count} train / {test_count} test сообщений**",
        "- Конфигурация: **DIETClassifier/TEDPolicy, 100 эпох, seed 42**",
        f"- Оценено сообщений: **{support}**",
        f"- Accuracy: **{percent(accuracy)}**",
        f"- Macro-F1: **{percent(macro['f1-score'])}**",
        f"- Weighted-F1: **{percent(weighted['f1-score'])}**",
        f"- Ошибок intent: **{len(intent_errors)}**",
        f"- F1 `person_name` на уровне токенов/тегов: "
        f"**{percent(person_name_metrics['f1-score'])}**",
        f"- Сообщений с ошибкой entity: **{len(entity_errors)}**",
        "",
        "> `rasa test nlu` запускает NLU-конвейер, но перед подсчётом "
        "intent-метрик восстанавливает исходный top-intent из ranking, если "
        "сработал FallbackClassifier. Поэтому итоговое fallback-решение отдельно "
        "измеряется полным runtime-конвейером.",
        "",
    ]

    if isinstance(runtime_report, dict):
        raw_accuracy = float(runtime_report.get("raw_accuracy", accuracy))
        fallback_count = int(runtime_report.get("fallback_count", 0))
        fallback_raw_correct = runtime_report.get("fallback_raw_correct_count")
        fallback_raw_incorrect = runtime_report.get("fallback_raw_incorrect_count")
        lines.extend(
            [
                "## Поведение полного pipeline",
                "",
                f"- Raw accuracy до fallback: **{percent(raw_accuracy)}**",
                f"- Строгая intent accuracy после fallback: "
                f"**{percent(float(runtime_report['accuracy']))}**",
                f"- Coverage — доля сообщений с принятым intent "
                f"(не `nlu_fallback`): "
                f"**{percent(float(runtime_report['coverage']))}**",
                f"- Selective accuracy среди принятых intent: "
                f"**{percent(float(runtime_report['selective_accuracy']))}**",
                f"- Доля fallback: "
                f"**{percent(float(runtime_report['fallback_rate']))}**",
            ]
        )
        if fallback_raw_correct is not None and fallback_raw_incorrect is not None:
            lines.append(
                f"- Fallback отклонил **{fallback_count}** сообщений: "
                f"**{int(fallback_raw_correct)}** с верным raw intent и "
                f"**{int(fallback_raw_incorrect)}** с ошибочным raw intent."
            )
        lines.extend(
            [
                "",
                "> Это компромисс abstention: fallback может повысить точность "
                "среди принятых intent, одновременно снижая coverage и, если он "
                "отклоняет верный raw intent, строгую общую accuracy.",
                "",
            ]
        )

    if isinstance(core_report, dict):
        conversation = core_report.get("conversation_accuracy", {})
        total_actions = int(core_report["macro avg"]["support"])
        correct_actions = round(float(core_report["accuracy"]) * total_actions)
        action_listen = core_report.get("action_listen")
        action_listen_support = (
            int(action_listen.get("support", 0))
            if isinstance(action_listen, dict)
            else None
        )
        lines.extend(
            [
                "## Rasa Core",
                "",
                f"- Диалоговые истории: **{conversation.get('correct', 0)}/"
                f"{conversation.get('total', 0)}**",
                f"- Действия: **{correct_actions}/{total_actions}**",
                f"- Action accuracy: **{percent(float(core_report['accuracy']))}**",
            ]
        )
        core_caveat = (
            "> Это Core-only тесты с заранее заданными gold intents; NLU здесь "
            "не проверяется."
        )
        if action_listen_support is not None:
            core_caveat += (
                f" Шаги `action_listen` составляют "
                f"**{action_listen_support}/{total_actions}** всех проверенных "
                "действий."
            )
        lines.extend(["", core_caveat, ""])

    lines.extend(
        [
        "## Метрики по intent",
        "",
        "| Intent | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {precision:.3f} | {recall:.3f} | {f1:.3f} | {count} |"
        for name, precision, recall, f1, count in class_rows
    )

    lines.extend(
        [
            "",
            "## Частые смешения",
            "",
            "| Истинный intent | Предсказанный intent | Число |",
            "|---|---|---:|",
        ]
    )
    lines.extend(
        f"| `{expected}` | `{predicted}` | {count} |"
        for (expected, predicted), count in confusions.most_common(15)
    )

    lines.extend(
        [
            "",
            "> Финальный test-набор подготовлен отдельно и не использовался "
            "при обучении. Он остаётся вручную/синтетически составленным, "
            "поэтому результат не заменяет проверку на сообщениях реальных "
            "пользователей.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results/final")
    parser.add_argument("--output", help="Optional Markdown output path")
    parser.add_argument("--train-data", default="data/nlu.yml")
    parser.add_argument("--test-data", default="tests/nlu_test.yml")
    args = parser.parse_args()

    summary = build_summary(
        Path(args.results_dir),
        Path(args.train_data),
        Path(args.test_data),
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(summary)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
