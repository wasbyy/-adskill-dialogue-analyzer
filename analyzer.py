"""
AdSkill Dialogue Analyzer - AI/ML Test Task
Автоматический анализ диалогов менеджеров с клиентами
"""

import json
import os
from typing import List, Dict, Any
from openai import OpenAI


class DialogueAnalyzer:
    """
    Анализатор диалогов с использованием OpenAI GPT-4o.

    Выполняет два типа анализа:
    1. Инфо-якоря - важные темы, обсужденные в диалоге
    2. Отработка возражений - как менеджер реагировал на возражения клиента
    """

    # Ключевые темы для анализа
    INFO_ANCHORS = [
        "Цели и KPI/метрики успеха",
        "Текущие источники трафика",
        "Бюджет",
        "URL/артефакты",
        "Ожидания от партнёра"
    ]

    # Типы возражений
    OBJECTION_TYPES = {
        "Финансовые ограничения": ["дорого", "нет бюджета", "высокая комиссия", "дорогой", "дорогая"],
        "Невыгодные условия сотрудничества": ["не устраивают условия", "высокий минимальный депозит", "хотим другую модель"],
        "Потеря в пользу конкурента": ["нашли другое агентство", "у конкурента выгоднее", "более выгодные условия"]
    }

    def __init__(self, api_key: str = None):
        """
        Инициализация анализатора.

        Args:
            api_key: OpenAI API ключ (если не указан, берется из переменной окружения)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key не найден. Установите переменную OPENAI_API_KEY")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"

    def load_dialogues(self, file_path: str) -> List[Dict]:
        """Загрузка диалогов из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_dialogue(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Основной метод анализа диалога.

        Args:
            messages: Список сообщений диалога

        Returns:
            Словарь с результатами анализа инфо-якорей и возражений
        """
        dialogue_text = self._format_dialogue(messages)

        info_anchors_result = self._analyze_info_anchors(dialogue_text)
        objections_result = self._analyze_objections(dialogue_text)

        return {
            "info_anchors_found": info_anchors_result["found"],
            "info_anchors_missing": info_anchors_result["missing"],
            "objections_found": objections_result
        }

    def _format_dialogue(self, messages: List[Dict[str, str]]) -> str:
        """Форматирование диалога в текстовый формат"""
        lines = []
        for msg in messages:
            lines.append(f"{msg['sender']}: {msg['text']}")
        return "\n".join(lines)

    def _analyze_info_anchors(self, dialogue_text: str) -> Dict[str, List[str]]:
        """Анализ инфо-якорей с использованием structured output"""
        system_prompt = f"""Ты - эксперт по анализу продающих диалогов.
Твоя задача - определить, какие важные темы менеджер обсудил с клиентом из следующего списка:

1. "Цели и KPI/метрики успеха" - целевые KPI (CPA/ROMI/лиды), критерии успеха
2. "Текущие источники трафика" - где рекламируются сейчас, с кем работали
3. "Бюджет" - планируемый бюджет, модель оплаты, минимальные депозиты
4. "URL/артефакты" - ссылки на лендинги, креативы, материалы
5. "Ожидания от партнёра" - что важно клиенту, какие условия критичны

Анализируй диалог тщательно и укажи только те темы, которые ДЕЙСТВИТЕЛЬНО обсуждались.
"""

        user_prompt = f"""Проанализируй следующий диалог и определи, какие темы были обсуждены:

{dialogue_text}

Верни JSON со списками найденных и отсутствующих тем."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "info_anchors_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "found": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Список тем, которые были обсуждены"
                                },
                                "missing": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Список тем, которые НЕ были обсуждены"
                                }
                            },
                            "required": ["found", "missing"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Ошибка при анализе инфо-якорей: {e}")
            return {"found": [], "missing": self.INFO_ANCHORS}

    def _analyze_objections(self, dialogue_text: str) -> List[Dict[str, Any]]:
        """Анализ возражений клиента и их отработки менеджером"""
        system_prompt = """Ты - эксперт по анализу техник работы с возражениями в B2B продажах.

Типы возражений:
1. "Финансовые ограничения" - дорого, нет бюджета, высокая комиссия
2. "Невыгодные условия сотрудничества" - не устраивают условия, высокий депозит
3. "Потеря в пользу конкурента" - нашли другое агентство, у конкурента выгоднее

Для каждого возражения определи:
- Тип возражения
- Краткую цитату клиента
- Отработал ли менеджер (true/false)
- Конкретные действия менеджера (2-3 пункта)

Хорошая отработка включает:
- Уточнение деталей
- Объяснение ценности
- Предложение альтернатив/компромиссов
- Установка следующего шага (next step)
"""

        user_prompt = f"""Проанализируй диалог и найди все возражения клиента:

{dialogue_text}

Для каждого возражения укажи тип, цитату клиента, факт отработки и действия менеджера."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "objections_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "objections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "objection_type": {
                                                "type": "string",
                                                "description": "Тип возражения"
                                            },
                                            "client_quote": {
                                                "type": "string",
                                                "description": "Краткая цитата клиента"
                                            },
                                            "manager_handled": {
                                                "type": "boolean",
                                                "description": "Отработал ли менеджер возражение"
                                            },
                                            "manager_actions": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Список действий менеджера"
                                            }
                                        },
                                        "required": ["objection_type", "client_quote", "manager_handled", "manager_actions"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["objections"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("objections", [])

        except Exception as e:
            print(f"Ошибка при анализе возражений: {e}")
            return []

    def analyze_all_dialogues(self, file_path: str) -> List[Dict[str, Any]]:
        """Анализ всех диалогов из файла"""
        dialogues = self.load_dialogues(file_path)
        results = []

        for dialogue in dialogues:
            dialogue_id = dialogue["dialogue_id"]
            messages = dialogue["messages"]

            print(f"Анализирую диалог {dialogue_id}...")

            analysis = self.analyze_dialogue(messages)

            results.append({
                "dialogue_id": dialogue_id,
                "analysis": analysis
            })

        return results

    def save_results(self, results: List[Dict], output_path: str):
        """Сохранение результатов в JSON файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в {output_path}")


def main():
    """Главная функция для запуска анализа"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ОШИБКА: Установите переменную окружения OPENAI_API_KEY")
        print("Пример: export OPENAI_API_KEY='your-api-key-here'")
        return

    analyzer = DialogueAnalyzer(api_key=api_key)

    input_file = "dialogues_sample.json"
    output_file = "analysis_results.json"

    print(f"Загружаю диалоги из {input_file}...")
    results = analyzer.analyze_all_dialogues(input_file)

    analyzer.save_results(results, output_file)

    print("\n=== КРАТКАЯ СТАТИСТИКА ===")
    for result in results:
        print(f"\nДиалог {result['dialogue_id']}:")
        analysis = result['analysis']
        print(f"  Найдено инфо-якорей: {len(analysis['info_anchors_found'])}")
        print(f"  Пропущено инфо-якорей: {len(analysis['info_anchors_missing'])}")
        print(f"  Найдено возражений: {len(analysis['objections_found'])}")


if __name__ == "__main__":
    main()
