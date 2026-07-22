# Итоги NLU-оценки

- Дата запуска (UTC): **2026-07-22**
- Окружение: **Python 3.10.11, Rasa 3.6.21, Rasa SDK 3.6.2**
- Данные: **795 train / 235 test сообщений**
- Конфигурация: **DIETClassifier/TEDPolicy, 100 эпох, seed 42**
- Оценено сообщений: **235**
- Accuracy: **89.36%**
- Macro-F1: **90.23%**
- Weighted-F1: **89.26%**
- Ошибок intent: **25**
- F1 `person_name` на уровне токенов/тегов: **87.80%**
- Сообщений с ошибкой entity: **4**

> `rasa test nlu` запускает NLU-конвейер, но перед подсчётом intent-метрик восстанавливает исходный top-intent из ranking, если сработал FallbackClassifier. Поэтому итоговое fallback-решение отдельно измеряется полным runtime-конвейером.

## Поведение полного pipeline

- Raw accuracy до fallback: **89.36%**
- Строгая intent accuracy после fallback: **88.94%**
- Coverage — доля сообщений с принятым intent (не `nlu_fallback`): **97.02%**
- Selective accuracy среди принятых intent: **91.67%**
- Доля fallback: **2.98%**
- Fallback отклонил **7** сообщений: **1** с верным raw intent и **6** с ошибочным raw intent.

> Это компромисс abstention: fallback может повысить точность среди принятых intent, одновременно снижая coverage и, если он отклоняет верный raw intent, строгую общую accuracy.

## Rasa Core

- Диалоговые истории: **10/10**
- Действия: **47/47**
- Action accuracy: **100.00%**

> Это Core-only тесты с заранее заданными gold intents; NLU здесь не проверяется. Шаги `action_listen` составляют **23/47** всех проверенных действий.

## Метрики по intent

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `smalltalk_howareyou` | 0.875 | 0.700 | 0.778 | 10 |
| `help` | 0.667 | 1.000 | 0.800 | 10 |
| `out_of_scope` | 1.000 | 0.667 | 0.800 | 30 |
| `greet` | 0.750 | 0.900 | 0.818 | 10 |
| `inform_name` | 0.750 | 1.000 | 0.857 | 15 |
| `ask_project_structure` | 0.818 | 0.900 | 0.857 | 10 |
| `ask_run` | 0.818 | 0.900 | 0.857 | 10 |
| `thanks` | 0.818 | 0.900 | 0.857 | 10 |
| `ask_time` | 1.000 | 0.800 | 0.889 | 10 |
| `feedback_bad` | 1.000 | 0.800 | 0.889 | 10 |
| `smalltalk_joke` | 1.000 | 0.800 | 0.889 | 10 |
| `ask_name` | 0.900 | 0.900 | 0.900 | 10 |
| `ask_creator` | 1.000 | 0.900 | 0.947 | 10 |
| `ask_date` | 0.909 | 1.000 | 0.952 | 10 |
| `ask_nlu` | 0.909 | 1.000 | 0.952 | 10 |
| `ask_rasa` | 0.909 | 1.000 | 0.952 | 10 |
| `ask_training` | 0.909 | 1.000 | 0.952 | 10 |
| `ask_docker` | 1.000 | 1.000 | 1.000 | 10 |
| `ask_project` | 1.000 | 1.000 | 1.000 | 10 |
| `ask_restart` | 1.000 | 1.000 | 1.000 | 10 |
| `goodbye` | 1.000 | 1.000 | 1.000 | 10 |

## Частые смешения

| Истинный intent | Предсказанный intent | Число |
|---|---|---:|
| `out_of_scope` | `inform_name` | 3 |
| `ask_time` | `greet` | 2 |
| `out_of_scope` | `ask_run` | 2 |
| `out_of_scope` | `help` | 2 |
| `greet` | `help` | 1 |
| `thanks` | `help` | 1 |
| `ask_name` | `inform_name` | 1 |
| `ask_creator` | `thanks` | 1 |
| `ask_run` | `ask_training` | 1 |
| `ask_project_structure` | `ask_rasa` | 1 |
| `feedback_bad` | `thanks` | 1 |
| `feedback_bad` | `ask_date` | 1 |
| `smalltalk_howareyou` | `help` | 1 |
| `smalltalk_howareyou` | `ask_name` | 1 |
| `smalltalk_howareyou` | `ask_project_structure` | 1 |

> Финальный test-набор подготовлен отдельно и не использовался при обучении. Он остаётся вручную/синтетически составленным, поэтому результат не заменяет проверку на сообщениях реальных пользователей.
