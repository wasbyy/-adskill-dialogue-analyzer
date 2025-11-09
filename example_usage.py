"""
Примеры использования DialogueAnalyzer
"""

from analyzer import DialogueAnalyzer
import os

def example_single():
    """Пример анализа одного диалога"""
    analyzer = DialogueAnalyzer()

    messages = [
        {"sender": "Менеджер", "text": "Какой у вас бюджет?"},
        {"sender": "Клиент", "text": "Ограниченный бюджет, максимум 50 тысяч."}
    ]

    result = analyzer.analyze_dialogue(messages)
    print(result)

def example_batch():
    """Пример массового анализа"""
    analyzer = DialogueAnalyzer()
    results = analyzer.analyze_all_dialogues("dialogues_sample.json")
    analyzer.save_results(results, "output.json")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Установите OPENAI_API_KEY")
    else:
        print("Примеры использования готовы!")
