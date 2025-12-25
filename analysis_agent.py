import json
import logging
from typing import List, Dict, Any, Optional
from gigachat_client import GigaChatClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class AnalysisAgent:
    """Агент глубокого анализа контента"""

    def __init__(self, verify_ssl: bool = False):
        self.client = GigaChatClient(verify_ssl=verify_ssl)
        logger.info("AnalysisAgent инициализирован")
        # Системный промпт для анализа контента
        self.analysis_prompt = """Ты - экспертный аналитик контента. Твоя задача - глубоко анализировать предоставленный контент по нескольким критериям.

ТРЕБОВАНИЯ К АНАЛИЗУ:

1. РЕЛЕВАНТНОСТЬ ТЕМЕ:
   - Оценить насколько контент соответствует заданной теме (0-10 баллов)
   - Выделить ключевые моменты, подтверждающие релевантность
   - Отметить отклонения от темы, если они есть

2. УРОВЕНЬ СЛОЖНОСТИ:
   - Определить уровень сложности (начальный, средний, продвинутый, экспертный)
   - Указать факторы, влияющие на сложность
   - Оценить необходимость предварительных знаний

3. СООТВЕТСТВИЕ ЦЕЛЕВОЙ АУДИТОРИИ:
   - Определить целевую аудиторию контента
   - Оценить доступность изложения
   - Проверить соответствие стиля и тона аудитории

4. КЛЮЧЕВЫЕ ПОНЯТИЯ:
   - Выделить 5-10 основных концепций и терминов
   - Дать краткое объяснение каждого понятия
   - Оценить важность для понимания контента

5. ПОТЕНЦИАЛЬНЫЕ ОГРАНИЧЕНИЯ:
   - Выявить недостатки контента
   - Отметить устаревшую информацию
   - Указать на пробелы в освещении темы

6. КРАТКОЕ РЕЗЮМЕ:
   - Создать лаконичное резюме (3-5 предложений)
   - Выделить основную мысль
   - Отметить практическую ценность

ФОРМАТ ОТВЕТА:
Верни результат в формате JSON со следующей структурой:
{
  "relevance_analysis": {
    "score": 8,
    "explanation": "Текст объяснения",
    "key_points": ["пункт1", "пункт2"],
    "deviations": ["отклонение1", "отклонение2"]
  },
  "complexity_analysis": {
    "level": "средний",
    "factors": ["фактор1", "фактор2"],
    "prerequisites": ["знание1", "знание2"]
  },
  "audience_analysis": {
    "target_audience": "аудитория",
    "accessibility": "оценка доступности",
    "style_match": "соответствие стиля"
  },
  "key_concepts": [
    {"concept": "понятие1", "explanation": "объяснение", "importance": "высокая"},
    {"concept": "понятие2", "explanation": "объяснение", "importance": "средняя"}
  ],
  "limitations": [
    {"limitation": "ограничение1", "impact": "влияние", "suggestion": "рекомендация"},
    {"limitation": "ограничение2", "impact": "влияние", "suggestion": "рекомендация"}
  ],
  "summary": "Краткое резюме контента"
}"""

    def analyze_single_content(self, content: str, topic: str = None, target_audience: str = None) -> Dict[str, Any]:
        """Анализирует один контент по всем критериям"""

        print("🔍 Начинаю глубокий анализ контента...")

        # Формируем промпт для анализа
        analysis_prompt = f"""
КОНТЕНТ ДЛЯ АНАЛИЗА:
{content}

ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ АНАЛИЗА:
- Тема для оценки релевантности: {topic if topic else "не указана"}
- Целевая аудитория: {target_audience if target_audience else "не указана"}

Проведи полный анализ по всем критериям.
"""

        try:
            result = self.client.chat_json(
                prompt=analysis_prompt,
                system_prompt=self.analysis_prompt,
                temperature=0.2,  # Низкая температура для более консервативных ответов
                max_tokens=3000
            )

            print("✅ Анализ завершен успешно")
            return result

        except Exception as e:
            print(f"❌ Ошибка при анализе: {e}")
            return {"error": str(e)}

    def process(self, query: str = None, content: str = None, contents: List[Dict] = None,
                topic: str = None, target_audience: str = None, context: Dict = None) -> Dict[str, Any]:
        """
        Универсальный метод обработки для агента анализа.
        В зависимости от переданных параметров, вызывает либо анализ одного контента, либо нескольких.
        """
        logger.info(f"AnalysisAgent: обработка запроса - {query[:100] if query else 'без запроса'}")

        try:
            # Определяем, что анализировать
            if content is not None:
                logger.info("Выполняем анализ одного контента")
                return self.analyze_single_content(content, topic, target_audience)

            elif contents is not None and len(contents) > 0:
                logger.info(f"Выполняем анализ {len(contents)} контентов")
                return self.analyze_multiple_contents(contents, topic)

            elif context:
                # Извлекаем из контекста
                if 'content' in context:
                    logger.info("Найден контент в контексте")
                    return self.analyze_single_content(
                        content=context.get('content'),
                        topic=context.get('topic', topic),
                        target_audience=context.get('target_audience', target_audience)
                    )
                elif 'contents' in context:
                    logger.info(f"Найдены {len(context.get('contents', []))} контентов в контексте")
                    return self.analyze_multiple_contents(
                        context.get('contents', []),
                        context.get('topic', topic)
                    )
                else:
                    return {
                        "error": "Нет данных для анализа в контексте",
                        "context_keys": list(context.keys())
                    }
            else:
                # Возвращаем ошибку или дефолтный анализ
                return {
                    "error": "Для анализа необходим контент. Укажите content или contents в параметрах.",
                    "suggestion": "Используйте analyze_single_content() или analyze_multiple_contents() напрямую"
                }

        except Exception as e:
            logger.error(f"Ошибка в AnalysisAgent.process: {e}")
            return {"error": str(e)}

    def analyze_multiple_contents(self, contents: List[Dict[str, str]], topic: str = None) -> Dict[str, Any]:
        """Анализирует и сравнивает несколько контентов"""

        print(f"🔍 Начинаю сравнительный анализ {len(contents)} контентов...")

        # Анализируем каждый контент отдельно
        individual_analyses = []
        for i, content_item in enumerate(contents):
            print(f"📄 Анализ контента {i + 1}/{len(contents)}...")

            content_id = content_item.get("id", f"content_{i + 1}")
            content_text = content_item.get("text", "")
            title = content_item.get("title", f"Контент {i + 1}")

            analysis = self.analyze_single_content(content_text, topic)

            individual_analyses.append({
                "id": content_id,
                "title": title,
                "analysis": analysis
            })

        # Сравниваем все контенты
        comparison = self._compare_contents(individual_analyses, topic)

        return {
            "individual_analyses": individual_analyses,
            "comparative_analysis": comparison,
            "recommendations": self._generate_recommendations(individual_analyses)
        }

    def _compare_contents(self, analyses: List[Dict], topic: str = None) -> Dict[str, Any]:
        """Сравнивает несколько проанализированных контентов"""

        print("⚖️  Провожу сравнительный анализ...")

        comparison_prompt = f"""
АНАЛИЗЫ КОНТЕНТОВ:
{json.dumps(analyses, ensure_ascii=False, indent=2)}

ТЕМА ДЛЯ СРАВНЕНИЯ: {topic if topic else "общая оценка"}

Сравни контенты по следующим критериям:
1. Релевантность теме
2. Уровень сложности
3. Соответствие целевой аудитории
4. Качество изложения
5. Практическая ценность

Верни результат в формате JSON:
{{
  "comparison_matrix": [
    {{
      "content_id": "id1",
      "relevance_rank": 1,
      "complexity_rank": 2,
      "overall_score": 8.5,
      "best_for": ["ситуация1", "ситуация2"],
      "worst_for": ["ситуация3", "ситуация4"]
    }}
  ],
  "best_overall": "id_лучшего_контента",
  "best_for_beginners": "id_для_начинающих",
  "best_for_experts": "id_для_экспертов",
  "summary": "Сводка сравнения"
}}
"""

        try:
            result = self.client.chat_json(
                prompt=comparison_prompt,
                system_prompt="Ты - эксперт по сравнительному анализу контента. Сравни несколько контентов и выдели их сильные и слабые стороны.",
                temperature=0.3,
                max_tokens=2500
            )
            return result
        except Exception as e:
            return {"error": f"Ошибка сравнения: {str(e)}"}

    def _generate_recommendations(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Генерирует рекомендации на основе анализа"""

        recommendation_prompt = f"""
АНАЛИЗЫ КОНТЕНТОВ:
{json.dumps(analyses, ensure_ascii=False, indent=2)}

Сгенерируй рекомендации:
1. Какой контент использовать для разных целей
2. В какой последовательности изучать контенты
3. Какие дополнительные материалы могут понадобиться

Верни результат в формате JSON:
{{
  "usage_recommendations": [
    {{
      "purpose": "цель",
      "recommended_content": "id_контента",
      "reason": "обоснование",
      "study_time": "оценочное время"
    }}
  ],
  "learning_path": [
    {{
      "step": 1,
      "content_id": "id1",
      "action": "действие",
      "expected_outcome": "результат"
    }}
  ],
  "supplementary_materials": [
    {{
      "type": "тип материала",
      "description": "описание",
      "purpose": "для чего нужен"
    }}
  ]
}}
"""

        try:
            result = self.client.chat_json(
                prompt=recommendation_prompt,
                system_prompt="Ты - образовательный консультант. Создай персонализированные рекомендации на основе анализа контентов.",
                temperature=0.4,
                max_tokens=2000
            )
            return result
        except Exception as e:
            return {"error": f"Ошибка генерации рекомендаций: {str(e)}"}

    def analyze_file(self, file_path: str, topic: str = None, target_audience: str = None) -> Dict[str, Any]:
        """Анализирует контент из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Добавляем информацию о файле в анализ
            file_info = {
                "file_name": file_path.split('/')[-1],
                "file_size_chars": len(content),
                "file_size_words": len(content.split())
            }

            analysis = self.analyze_single_content(content, topic, target_audience)
            analysis["file_info"] = file_info

            return analysis

        except FileNotFoundError:
            return {"error": f"Файл не найден: {file_path}"}
        except Exception as e:
            return {"error": f"Ошибка при чтении файла: {str(e)}"}

    def save_analysis_report(self, analysis: Dict[str, Any], output_file: str = "analysis_report.json"):
        """Сохраняет отчет об анализе в JSON файл"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
            print(f"✅ Отчет сохранен в файл: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Ошибка при сохранении отчета: {e}")
            return False