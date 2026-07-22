from datetime import datetime, timedelta
from typing import Any, Optional

import pytest
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions import actions as actions_module
from actions.actions import (
    ActionRememberName,
    ActionShowCapabilities,
    ActionShowDate,
    ActionShowTime,
)


def make_tracker(user_name: Optional[str] = None) -> Tracker:
    """Build the smallest real Rasa tracker needed by the custom actions."""
    return Tracker(
        sender_id="unit-test-user",
        slots={"user_name": user_name},
        latest_message={},
        events=[],
        paused=False,
        followup_action=None,
        active_loop={},
        latest_action_name=None,
    )


def run_action(action: Any, user_name: Optional[str] = None) -> tuple[list, str]:
    dispatcher = CollectingDispatcher()

    events = action.run(dispatcher, make_tracker(user_name), {})

    assert len(dispatcher.messages) == 1
    text = dispatcher.messages[0].get("text")
    assert isinstance(text, str) and text
    assert any("а" <= character.lower() <= "я" or character.lower() == "ё" for character in text)
    assert "Ð" not in text and "Ñ" not in text and "�" not in text
    return events, text


@pytest.mark.parametrize(
    ("action", "expected_name"),
    [
        (ActionRememberName(), "action_remember_name"),
        (ActionShowCapabilities(), "action_show_capabilities"),
        (ActionShowTime(), "action_show_time"),
        (ActionShowDate(), "action_show_date"),
    ],
)
def test_action_names(action: Any, expected_name: str) -> None:
    assert action.name() == expected_name


def test_remember_name_addresses_user_and_returns_no_events() -> None:
    events, text = run_action(ActionRememberName(), "Анна")

    assert events == []
    assert text == (
        "Приятно познакомиться, Анна! "
        "Я запомню ваше имя в рамках текущего диалога."
    )


@pytest.mark.parametrize("missing_name", [None, ""])
def test_remember_name_asks_to_repeat_when_slot_is_empty(
    missing_name: Optional[str],
) -> None:
    events, text = run_action(ActionRememberName(), missing_name)

    assert events == []
    assert text == (
        "Я не смог выделить имя. "
        "Напишите, например: «Меня зовут Анна»."
    )


@pytest.mark.parametrize(
    ("user_name", "expected_prefix"),
    [(None, "Я могу"), ("Илья", "Илья, я могу")],
)
def test_show_capabilities_lists_supported_topics(
    user_name: Optional[str], expected_prefix: str
) -> None:
    events, text = run_action(ActionShowCapabilities(), user_name)

    assert events == []
    assert text.startswith(expected_prefix)
    for topic in (
        "Rasa",
        "NLU",
        "Docker",
        "обучение",
        "запуск",
        "структуру файлов",
        "дату и время",
    ):
        assert topic in text


@pytest.fixture
def fixed_datetime(monkeypatch: pytest.MonkeyPatch) -> list:
    requested_timezones = []

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            requested_timezones.append(tz)
            return cls(2026, 7, 22, 9, 5, 0, tzinfo=tz)

    monkeypatch.setattr(actions_module, "datetime", FixedDateTime)
    return requested_timezones


def assert_tashkent_timezone_was_requested(requested_timezones: list) -> None:
    assert len(requested_timezones) == 1
    assert requested_timezones[0] is not None
    assert requested_timezones[0].utcoffset(None) == timedelta(hours=5)


def test_show_time_uses_tashkent_timezone_and_zero_padded_format(
    fixed_datetime: list,
) -> None:
    events, text = run_action(ActionShowTime())

    assert events == []
    assert text == "Сейчас 09:05 по времени Ташкента (UTC+5)."
    assert_tashkent_timezone_was_requested(fixed_datetime)


def test_show_date_uses_russian_month_and_tashkent_timezone(
    fixed_datetime: list,
) -> None:
    events, text = run_action(ActionShowDate())

    assert events == []
    assert text == "Сегодня 22 июля 2026 года по времени Ташкента."
    assert_tashkent_timezone_was_requested(fixed_datetime)
