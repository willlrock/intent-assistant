"""Static consistency checks for the Russian Rasa assistant.

The validator checks the project schema, the independent train/test NLU
corpora, dialogue references, custom actions, and pinned container versions.
It intentionally does not replace ``rasa data validate`` or trained-model
evaluation.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ModuleNotFoundError:
    print("STATIC VALIDATION: FAILED")
    print("  - PyYAML is not installed; install the dependencies from requirements.txt")
    raise SystemExit(2)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_INTENTS = {
    "greet",
    "goodbye",
    "thanks",
    "help",
    "ask_name",
    "ask_creator",
    "ask_project",
    "inform_name",
    "ask_restart",
    "ask_rasa",
    "ask_nlu",
    "ask_training",
    "ask_run",
    "ask_project_structure",
    "ask_docker",
    "feedback_bad",
    "ask_time",
    "ask_date",
    "smalltalk_howareyou",
    "smalltalk_joke",
    "out_of_scope",
}

SYSTEM_INTENTS = {"nlu_fallback"}

REQUIRED_FILES = (
    "requirements.txt",
    "config.yml",
    "domain.yml",
    "endpoints.yml",
    "docker-compose.yml",
    "actions/__init__.py",
    "actions/actions.py",
    "data/nlu.yml",
    "data/rules.yml",
    "data/stories.yml",
    "tests/nlu_dev.yml",
    "tests/nlu_test.yml",
    "tests/test_stories.yml",
    "README.md",
    ".gitignore",
)

SCHEMA_31_FILES = (
    "domain.yml",
    "data/nlu.yml",
    "data/rules.yml",
    "data/stories.yml",
    "tests/nlu_dev.yml",
    "tests/nlu_test.yml",
    "tests/test_stories.yml",
)

BUILT_IN_ACTIONS = {
    "action_back",
    "action_deactivate_loop",
    "action_default_ask_affirmation",
    "action_default_ask_rephrase",
    "action_default_fallback",
    "action_extract_slots",
    "action_listen",
    "action_restart",
    "action_revert_fallback_events",
    "action_session_start",
    "action_two_stage_fallback",
    "action_unlikely_intent",
}

INLINE_ENTITY_PATTERN = re.compile(
    r"\[(?P<value>[^\]]+)\]\((?P<entity>[A-Za-z_][A-Za-z0-9_.-]*)\)"
)
JSON_ENTITY_PATTERN = re.compile(r"\[(?P<value>[^\]]+)\](?P<meta>\{[^{}]+\})")


class ValidationContext:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.documents: dict[str, Any] = {}

    def error(self, message: str) -> None:
        self.errors.append(message)

    def load_yaml(self, relative_path: str) -> Any:
        path = ROOT / relative_path
        if relative_path in self.documents:
            return self.documents[relative_path]
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as source:
                document = yaml.safe_load(source)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            self.error(f"{relative_path}: invalid YAML: {exc}")
            document = None
        self.documents[relative_path] = document
        return document


def normalize_text(text: str) -> str:
    """Normalize examples for duplicate and leakage checks."""

    text = INLINE_ENTITY_PATTERN.sub(lambda match: match.group("value"), text)
    text = JSON_ENTITY_PATTERN.sub(lambda match: match.group("value"), text)
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def normalize_name(text: str) -> str:
    return normalize_text(text)


def normalize_entity_template(text: str) -> str:
    """Normalize wording while masking entity values for leakage checks."""

    text = INLINE_ENTITY_PATTERN.sub("__entity__", text)
    text = JSON_ENTITY_PATTERN.sub("__entity__", text)
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def string_set(values: Any, field: str, context: ValidationContext) -> set[str]:
    if values is None:
        return set()
    if not isinstance(values, list):
        context.error(f"{field}: expected a list")
        return set()

    result: set[str] = set()
    for value in values:
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, Mapping) and len(value) == 1:
            key = next(iter(value))
            if isinstance(key, str):
                result.add(key)
            else:
                context.error(f"{field}: mapping key must be a string: {value!r}")
        else:
            context.error(f"{field}: unsupported list item: {value!r}")
    return result


def parse_example_block(
    raw_examples: Any,
    location: str,
    context: ValidationContext,
) -> list[str]:
    if isinstance(raw_examples, str):
        result: list[str] = []
        for line_number, line in enumerate(raw_examples.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("-"):
                context.error(
                    f"{location}: example line {line_number} must start with '-': {line!r}"
                )
                continue
            example = stripped[1:].strip()
            if not example:
                context.error(f"{location}: empty example on line {line_number}")
                continue
            result.append(example)
        return result

    if isinstance(raw_examples, list):
        result = []
        for index, value in enumerate(raw_examples, start=1):
            if not isinstance(value, str) or not value.strip():
                context.error(f"{location}: invalid example #{index}: {value!r}")
                continue
            result.append(value.strip())
        return result

    context.error(f"{location}: examples must be a block string or a list")
    return []


def nlu_examples(
    document: Any,
    relative_path: str,
    context: ValidationContext,
) -> dict[str, list[str]]:
    if not isinstance(document, Mapping):
        context.error(f"{relative_path}: top-level YAML value must be a mapping")
        return {}
    items = document.get("nlu")
    if not isinstance(items, list):
        context.error(f"{relative_path}: top-level 'nlu' must be a list")
        return {}

    result: dict[str, list[str]] = {}
    block_counts: Counter[str] = Counter()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            context.error(f"{relative_path}: nlu item #{index} must be a mapping")
            continue
        intent = item.get("intent")
        if intent is None:
            # Rasa also allows regex, lookup and synonym entries in this list.
            continue
        if not isinstance(intent, str) or not intent:
            context.error(f"{relative_path}: nlu item #{index} has an invalid intent")
            continue
        block_counts[intent] += 1
        result.setdefault(intent, []).extend(
            parse_example_block(
                item.get("examples"),
                f"{relative_path}:{intent}",
                context,
            )
        )

    repeated = sorted(intent for intent, count in block_counts.items() if count > 1)
    if repeated:
        context.error(
            f"{relative_path}: intents are split across repeated blocks: "
            + ", ".join(repeated)
        )
    return result


def duplicate_examples(
    examples_by_intent: Mapping[str, Sequence[str]],
) -> dict[str, list[tuple[str, str]]]:
    grouped: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for intent, examples in examples_by_intent.items():
        for example in examples:
            grouped[normalize_text(example)].append((intent, example))
    return {
        normalized: occurrences
        for normalized, occurrences in grouped.items()
        if normalized and len(occurrences) > 1
    }


def entity_annotations(text: str, location: str, context: ValidationContext) -> list[tuple[str, str]]:
    annotations = [
        (match.group("value"), match.group("entity"))
        for match in INLINE_ENTITY_PATTERN.finditer(text)
    ]
    for match in JSON_ENTITY_PATTERN.finditer(text):
        try:
            metadata = json.loads(match.group("meta"))
        except json.JSONDecodeError as exc:
            context.error(f"{location}: invalid JSON entity annotation: {exc}")
            continue
        entity = metadata.get("entity") if isinstance(metadata, Mapping) else None
        if not isinstance(entity, str):
            context.error(f"{location}: JSON entity annotation has no string 'entity'")
            continue
        annotations.append((match.group("value"), entity))
    return annotations


def validate_entities(
    datasets: Mapping[str, Mapping[str, Sequence[str]]],
    declared_entities: set[str],
    context: ValidationContext,
) -> tuple[dict[str, int], dict[str, int]]:
    names_by_split: dict[str, set[str]] = {
        split: set() for split in datasets
    }
    counts: dict[str, int] = {split: 0 for split in datasets}

    for split, dataset in datasets.items():
        name_values = names_by_split[split]
        for intent, examples in dataset.items():
            for index, example in enumerate(examples, start=1):
                location = f"{split}:{intent} example #{index}"
                annotations = entity_annotations(example, location, context)
                for value, entity in annotations:
                    if entity not in declared_entities:
                        context.error(f"{location}: undeclared entity '{entity}'")
                    if entity == "person_name":
                        counts[split] += 1
                        name_values.add(normalize_name(value))
                        if intent != "inform_name":
                            context.error(
                                f"{location}: person_name may only be annotated in inform_name"
                            )

                person_annotations = [
                    value for value, entity in annotations if entity == "person_name"
                ]
                if intent == "inform_name" and len(person_annotations) != 1:
                    context.error(
                        f"{location}: expected exactly one person_name annotation, "
                        f"found {len(person_annotations)}"
                    )

    for left, right in combinations(names_by_split, 2):
        leaked_names = sorted(names_by_split[left] & names_by_split[right])
        if leaked_names:
            context.error(
                f"person_name values overlap between {left} and {right}: "
                + ", ".join(leaked_names)
            )

    train_names = names_by_split.get("train", set())
    unseen_from_train = {
        split: len(names - train_names)
        for split, names in names_by_split.items()
        if split != "train"
    }
    return counts, unseen_from_train


def maximum_same_intent_similarity(
    train: Mapping[str, Sequence[str]],
    test: Mapping[str, Sequence[str]],
) -> tuple[float, str | None, int | None]:
    """Return the largest train/test similarity after masking entity values."""

    maximum = 0.0
    maximum_intent: str | None = None
    maximum_index: int | None = None
    for intent, test_examples in test.items():
        train_templates = [
            normalize_entity_template(example)
            for example in train.get(intent, ())
        ]
        for index, example in enumerate(test_examples, start=1):
            template = normalize_entity_template(example)
            score = max(
                (
                    SequenceMatcher(None, template, candidate).ratio()
                    for candidate in train_templates
                ),
                default=0.0,
            )
            if score > maximum:
                maximum = score
                maximum_intent = intent
                maximum_index = index
    return maximum, maximum_intent, maximum_index


def walk_key_values(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for candidate_key, candidate_value in value.items():
            if candidate_key == key:
                yield candidate_value
            yield from walk_key_values(candidate_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from walk_key_values(item, key)


def collect_string_references(value: Any, key: str) -> set[str]:
    return {
        reference
        for reference in walk_key_values(value, key)
        if isinstance(reference, str)
    }


def action_implementation_details(source: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(source, filename="actions/actions.py")
    implemented: set[str] = set()
    response_references: set[str] = set()

    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        inherits_action = any(
            isinstance(base, ast.Name) and base.id == "Action"
            or isinstance(base, ast.Attribute) and base.attr == "Action"
            for base in class_node.bases
        )
        if not inherits_action:
            continue
        for function_node in (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "name"
        ):
            for return_node in ast.walk(function_node):
                if (
                    isinstance(return_node, ast.Return)
                    and isinstance(return_node.value, ast.Constant)
                    and isinstance(return_node.value.value, str)
                ):
                    implemented.add(return_node.value.value)

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        for keyword in call.keywords:
            if keyword.arg not in {"response", "template"}:
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, str
            ):
                response_references.add(keyword.value.value)

    return implemented, response_references


def schema_version(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    value = document.get("version")
    return str(value) if value is not None else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    context = ValidationContext()
    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            context.error(f"missing required file: {relative_path}")

    for relative_path in SCHEMA_31_FILES:
        document = context.load_yaml(relative_path)
        if document is not None and schema_version(document) != "3.1":
            context.error(f"{relative_path}: expected schema version 3.1")

    domain = context.load_yaml("domain.yml")
    train_document = context.load_yaml("data/nlu.yml")
    dev_document = context.load_yaml("tests/nlu_dev.yml")
    test_document = context.load_yaml("tests/nlu_test.yml")
    rules = context.load_yaml("data/rules.yml")
    stories = context.load_yaml("data/stories.yml")
    test_stories = context.load_yaml("tests/test_stories.yml")
    compose = context.load_yaml("docker-compose.yml")

    train = nlu_examples(train_document, "data/nlu.yml", context)
    dev = nlu_examples(dev_document, "tests/nlu_dev.yml", context)
    test = nlu_examples(test_document, "tests/nlu_test.yml", context)
    train_intents = set(train)
    dev_intents = set(dev)
    test_intents = set(test)

    for split, split_intents in (("dev", dev_intents), ("test", test_intents)):
        if train_intents != split_intents:
            context.error(
                f"train/{split} intent sets differ; only in train: "
                + ", ".join(sorted(train_intents - split_intents))
                + f"; only in {split}: "
                + ", ".join(sorted(split_intents - train_intents))
            )
    if train_intents != EXPECTED_INTENTS:
        context.error(
            "train intent taxonomy differs from the expected 21 intents; missing: "
            + ", ".join(sorted(EXPECTED_INTENTS - train_intents))
            + "; unexpected: "
            + ", ".join(sorted(train_intents - EXPECTED_INTENTS))
        )

    ordinary_intents = EXPECTED_INTENTS - {"inform_name", "out_of_scope"}
    expected_train_counts = {
        intent: 35 for intent in ordinary_intents
    } | {"inform_name": 60, "out_of_scope": 70}
    expected_eval_counts = {
        intent: 10 for intent in ordinary_intents
    } | {"inform_name": 15, "out_of_scope": 30}

    for intent, expected in sorted(expected_train_counts.items()):
        actual = len(train.get(intent, ()))
        if actual != expected:
            context.error(
                f"data/nlu.yml:{intent}: expected exactly {expected} train examples, "
                f"found {actual}"
            )
    for split, dataset in (("dev", dev), ("test", test)):
        for intent, expected in sorted(expected_eval_counts.items()):
            actual = len(dataset.get(intent, ()))
            if actual != expected:
                context.error(
                    f"tests/nlu_{split}.yml:{intent}: expected exactly {expected} "
                    f"examples, found {actual}"
                )

    train_duplicates = duplicate_examples(train)
    dev_duplicates = duplicate_examples(dev)
    test_duplicates = duplicate_examples(test)
    if train_duplicates:
        context.error(
            f"data/nlu.yml: found {len(train_duplicates)} normalized duplicate group(s)"
        )
    if dev_duplicates:
        context.error(
            f"tests/nlu_dev.yml: found {len(dev_duplicates)} normalized duplicate group(s)"
        )
    if test_duplicates:
        context.error(
            f"tests/nlu_test.yml: found {len(test_duplicates)} normalized duplicate group(s)"
        )

    train_texts = {
        normalize_text(example)
        for examples in train.values()
        for example in examples
    }
    dev_texts = {
        normalize_text(example)
        for examples in dev.values()
        for example in examples
    }
    test_texts = {
        normalize_text(example)
        for examples in test.values()
        for example in examples
    }
    leakage_by_pair = {
        "train/dev": sorted((train_texts & dev_texts) - {""}),
        "train/test": sorted((train_texts & test_texts) - {""}),
        "dev/test": sorted((dev_texts & test_texts) - {""}),
    }
    for pair, leakage in leakage_by_pair.items():
        if leakage:
            context.error(
                f"{pair} exact leakage: found {len(leakage)} normalized example(s)"
            )

    train_templates = {
        normalize_entity_template(example)
        for examples in train.values()
        for example in examples
    }
    test_templates = [
        normalize_entity_template(example)
        for examples in test.values()
        for example in examples
    ]
    repeated_test_templates = len(test_templates) - len(set(test_templates))
    if repeated_test_templates:
        context.error(
            "tests/nlu_test.yml: entity-masked templates contain "
            f"{repeated_test_templates} duplicate(s)"
        )
    template_leakage = (train_templates & set(test_templates)) - {""}
    if template_leakage:
        context.error(
            "train/test entity-template leakage: found "
            f"{len(template_leakage)} normalized template(s)"
        )

    maximum_similarity, similar_intent, similar_index = (
        maximum_same_intent_similarity(train, test)
    )
    if maximum_similarity >= 0.85:
        context.error(
            "train/test near-copy threshold exceeded: "
            f"{similar_intent} example #{similar_index} has similarity "
            f"{maximum_similarity:.3f} (limit < 0.850)"
        )

    dev_test_maximum_similarity, dev_test_similar_intent, dev_test_similar_index = (
        maximum_same_intent_similarity(dev, test)
    )
    if dev_test_maximum_similarity >= 0.85:
        context.error(
            "dev/test near-copy threshold exceeded: "
            f"{dev_test_similar_intent} example #{dev_test_similar_index} has "
            f"similarity {dev_test_maximum_similarity:.3f} (limit < 0.850)"
        )

    domain_mapping = domain if isinstance(domain, Mapping) else {}
    domain_intents = string_set(domain_mapping.get("intents"), "domain.yml:intents", context)
    expected_domain_intents = train_intents | SYSTEM_INTENTS
    if domain_intents != expected_domain_intents:
        context.error(
            "domain intents must equal train intents plus nlu_fallback; missing: "
            + ", ".join(sorted(expected_domain_intents - domain_intents))
            + "; unexpected: "
            + ", ".join(sorted(domain_intents - expected_domain_intents))
        )

    declared_entities = string_set(
        domain_mapping.get("entities"), "domain.yml:entities", context
    )
    if "person_name" not in declared_entities:
        context.error("domain.yml: entity person_name is not declared")

    slots = domain_mapping.get("slots")
    user_name = slots.get("user_name") if isinstance(slots, Mapping) else None
    mappings = user_name.get("mappings") if isinstance(user_name, Mapping) else None
    if not (
        isinstance(mappings, list)
        and any(
            isinstance(mapping, Mapping)
            and mapping.get("type") == "from_entity"
            and mapping.get("entity") == "person_name"
            for mapping in mappings
        )
    ):
        context.error("domain.yml: slot user_name is not mapped from person_name")

    entity_counts, unseen_names = validate_entities(
        {"train": train, "dev": dev, "test": test},
        declared_entities,
        context,
    )

    responses_value = domain_mapping.get("responses")
    responses = set(responses_value) if isinstance(responses_value, Mapping) else set()
    if not isinstance(responses_value, Mapping):
        context.error("domain.yml: responses must be a mapping")
    else:
        for response_name, variants in responses_value.items():
            if not isinstance(variants, list) or not variants:
                context.error(f"domain.yml:{response_name}: response variants are empty")

    declared_actions = string_set(
        domain_mapping.get("actions"), "domain.yml:actions", context
    )

    dialogue_documents = {
        "data/rules.yml": rules,
        "data/stories.yml": stories,
        "tests/test_stories.yml": test_stories,
    }
    intent_references: set[str] = set()
    action_references: set[str] = set()
    response_references: set[str] = set()
    for relative_path, document in dialogue_documents.items():
        if not isinstance(document, Mapping):
            context.error(f"{relative_path}: top-level YAML value must be a mapping")
            continue
        file_intents = collect_string_references(document, "intent")
        file_actions = collect_string_references(document, "action")
        file_responses = collect_string_references(document, "response")
        intent_references.update(file_intents)
        action_references.update(file_actions)
        response_references.update(file_responses)

        unknown_intents = file_intents - domain_intents
        if unknown_intents:
            context.error(
                f"{relative_path}: unknown intent reference(s): "
                + ", ".join(sorted(unknown_intents))
            )

    response_references.update(
        action for action in action_references if action.startswith("utter_")
    )
    missing_responses = response_references - responses
    if missing_responses:
        context.error(
            "dialogue/action response reference(s) missing from domain: "
            + ", ".join(sorted(missing_responses))
        )

    custom_action_references = {
        action
        for action in action_references
        if not action.startswith("utter_") and action not in BUILT_IN_ACTIONS
    }
    undeclared_action_references = custom_action_references - declared_actions
    if undeclared_action_references:
        context.error(
            "dialogue custom action reference(s) missing from domain: "
            + ", ".join(sorted(undeclared_action_references))
        )

    action_source_path = ROOT / "actions/actions.py"
    implemented_actions: set[str] = set()
    source_response_references: set[str] = set()
    if action_source_path.is_file():
        try:
            source = action_source_path.read_text(encoding="utf-8")
            implemented_actions, source_response_references = action_implementation_details(
                source
            )
        except (OSError, UnicodeError, SyntaxError) as exc:
            context.error(f"actions/actions.py: cannot inspect custom actions: {exc}")

    missing_implementations = declared_actions - implemented_actions
    undeclared_implementations = implemented_actions - declared_actions
    if missing_implementations:
        context.error(
            "declared custom action(s) without implementation: "
            + ", ".join(sorted(missing_implementations))
        )
    if undeclared_implementations:
        context.error(
            "implemented custom action(s) missing from domain: "
            + ", ".join(sorted(undeclared_implementations))
        )
    unused_actions = declared_actions - custom_action_references
    if unused_actions:
        context.error(
            "declared custom action(s) are not referenced by dialogue data: "
            + ", ".join(sorted(unused_actions))
        )
    missing_source_responses = source_response_references - responses
    if missing_source_responses:
        context.error(
            "actions/actions.py references unknown response(s): "
            + ", ".join(sorted(missing_source_responses))
        )

    services = compose.get("services") if isinstance(compose, Mapping) else None
    if not isinstance(services, Mapping):
        context.error("docker-compose.yml: top-level services mapping is missing")
    else:
        rasa_service = services.get("rasa")
        action_service = services.get("action_server")
        rasa_image = rasa_service.get("image") if isinstance(rasa_service, Mapping) else None
        action_image = (
            action_service.get("image") if isinstance(action_service, Mapping) else None
        )
        if rasa_image != "rasa/rasa:3.6.21":
            context.error(
                "docker-compose.yml: rasa image must be pinned to rasa/rasa:3.6.21"
            )
        if action_image != "rasa/rasa-sdk:3.6.2":
            context.error(
                "docker-compose.yml: action server image must be pinned to "
                "rasa/rasa-sdk:3.6.2"
            )

    metrics = {
        "train_intents": len(train_intents),
        "dev_intents": len(dev_intents),
        "test_intents": len(test_intents),
        "train_examples": sum(len(examples) for examples in train.values()),
        "dev_examples": sum(len(examples) for examples in dev.values()),
        "test_examples": sum(len(examples) for examples in test.values()),
        "train_person_name_annotations": entity_counts.get("train", 0),
        "dev_person_name_annotations": entity_counts.get("dev", 0),
        "test_person_name_annotations": entity_counts.get("test", 0),
        "unseen_dev_person_names": unseen_names.get("dev", 0),
        "unseen_test_person_names": unseen_names.get("test", 0),
        "normalized_train_duplicates": len(train_duplicates),
        "normalized_dev_duplicates": len(dev_duplicates),
        "normalized_test_duplicates": len(test_duplicates),
        "train_dev_exact_overlap": len(leakage_by_pair["train/dev"]),
        "train_test_exact_overlap": len(leakage_by_pair["train/test"]),
        "dev_test_exact_overlap": len(leakage_by_pair["dev/test"]),
        "train_test_template_overlap": len(template_leakage),
        "train_test_max_same_intent_similarity": f"{maximum_similarity:.3f}",
        "dev_test_max_same_intent_similarity": (
            f"{dev_test_maximum_similarity:.3f}"
        ),
        "domain_responses": len(responses),
        "declared_custom_actions": len(declared_actions),
        "implemented_custom_actions": len(implemented_actions),
        "rules": len(rules.get("rules", [])) if isinstance(rules, Mapping) else 0,
        "stories": len(stories.get("stories", [])) if isinstance(stories, Mapping) else 0,
        "test_stories": (
            len(test_stories.get("stories", []))
            if isinstance(test_stories, Mapping)
            else 0
        ),
    }

    print("Static project metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("Per-intent train/dev/test examples:")
    for intent in sorted(train_intents | dev_intents | test_intents):
        print(
            f"  {intent}: {len(train.get(intent, ()))} / "
            f"{len(dev.get(intent, ()))} / {len(test.get(intent, ()))}"
        )

    if context.errors:
        print("\nSTATIC VALIDATION: FAILED")
        for error in context.errors:
            print(f"  - {error}")
        return 1

    print("\nSTATIC VALIDATION: OK")
    print(
        "Static consistency passed. Run Rasa validation and trained-model tests "
        "for runtime and quality metrics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
