from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionRememberName(Action):
    def name(self) -> Text:
        return "action_remember_name"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_name = tracker.get_slot("user_name")

        if user_name:
            dispatcher.utter_message(
                text=f"Приятно познакомиться, {user_name}! Я запомню ваше имя в рамках текущего диалога."
            )
        else:
            dispatcher.utter_message(
                text="Я не смог выделить имя. Напишите, например: «Меня зовут Анна»."
            )

        return []


class ActionShowCapabilities(Action):
    def name(self) -> Text:
        return "action_show_capabilities"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_name = tracker.get_slot("user_name")
        subject = f"{user_name}, я" if user_name else "Я"
        dispatcher.utter_message(
            text=(
                f"{subject} могу рассказать о себе, авторе и назначении проекта, "
                "объяснить Rasa, NLU, Docker, обучение, запуск и структуру файлов, "
                "запомнить имя, назвать дату и время, поддержать короткий разговор "
                "и пошутить."
            )
        )
        return []


class ActionShowTime(Action):
    def name(self) -> Text:
        return "action_show_time"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        tashkent_time = datetime.now(timezone(timedelta(hours=5)))
        dispatcher.utter_message(
            text=f"Сейчас {tashkent_time:%H:%M} по времени Ташкента (UTC+5)."
        )
        return []


class ActionShowDate(Action):
    def name(self) -> Text:
        return "action_show_date"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        tashkent_date = datetime.now(timezone(timedelta(hours=5)))
        month_names = (
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        )
        dispatcher.utter_message(
            text=(
                f"Сегодня {tashkent_date.day} "
                f"{month_names[tashkent_date.month - 1]} {tashkent_date.year} года "
                "по времени Ташкента."
            )
        )
        return []
