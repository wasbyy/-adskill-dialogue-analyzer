# Deployment Guide

## Быстрый старт (5 минут)

### Установка

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
python analyzer.py
```

### С вашими диалогами

```python
from analyzer import DialogueAnalyzer

analyzer = DialogueAnalyzer()
results = analyzer.analyze_all_dialogues("your_dialogues.json")
analyzer.save_results(results, "output.json")
```

## Git setup

```bash
git init
git add .
git commit -m "AdSkill test task"
git remote add origin https://github.com/USERNAME/adskill-dialogue-analyzer.git
git push -u origin main
```

## Troubleshooting

**ImportError: openai**
```bash
pip install --upgrade openai
```

**API key not found**
```bash
export OPENAI_API_KEY='your-key'
```

---

**Готово к продакшену!**
