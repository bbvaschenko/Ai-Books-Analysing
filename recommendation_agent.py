import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationAgent:
    """
    Агент рекомендаций - формирует финальный ответ пользователю.
    Адаптирует вывод под уровень пользователя и формат.
    """
    
    def __init__(self, gigachat_client):
        self.client = gigachat_client

    def process(self, query: str = None, analysis_results: Dict = None,
                user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Формирование финальных рекомендаций на основе анализа.
        """
        logger.info("RecommendationAgent: формирование рекомендаций")

        # Если параметры переданы по-другому (для обратной совместимости)
        actual_query = query
        actual_results = analysis_results

        # Проверяем context на наличие query и results
        if user_context:
            if actual_query is None and 'original_query' in user_context:
                actual_query = user_context['original_query']
            if actual_results is None and 'analysis_results' in user_context:
                actual_results = user_context['analysis_results']

        if actual_query is None:
            actual_query = "Неизвестный запрос"

        if actual_results is None:
            actual_results = {}

        try:
            # Агрегация всех результатов
            aggregated_data = self._aggregate_results(actual_results, user_context)

            # Генерация рекомендаций с учетом пользователя
            recommendations = self._generate_recommendations(aggregated_data, actual_query)

            # Адаптация под уровень пользователя
            adapted_response = self._adapt_to_user_level(recommendations, user_context)

            # Формирование финального ответа
            final_response = self._create_final_response(adapted_response, actual_query)

            # Добавление объяснений и альтернатив
            enriched_response = self._enrich_with_explanations(final_response, aggregated_data)

            return enriched_response

        except Exception as e:
            logger.error(f"Ошибка в RecommendationAgent: {e}")
            return self._create_error_response(actual_query, str(e))
    
    def _aggregate_results(self, results: Dict, user_context: Optional[Dict]) -> Dict[str, Any]:
        """
        Агрегация результатов от разных агентов.
        """
        aggregated = {
            "search_results": results.get("search_results", {}),
            "analysis_results": results.get("analysis_results", {}),
            "critique_results": results.get("critique_results", {}),
            "user_context": user_context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Извлечение ключевых данных
        key_data = {
            "top_content": self._extract_top_content(results),
            "key_concepts": self._extract_key_concepts(results),
            "comparisons": self._extract_comparisons(results),
            "limitations": self._extract_limitations(results),
            "user_preferences": user_context.get("preferences", {}) if user_context else {}
        }
        
        aggregated.update(key_data)
        return aggregated
    
    def _generate_recommendations(self, aggregated_data: Dict, query: str) -> Dict[str, Any]:
        """
        Генерация рекомендаций с помощью GigaChat.
        """
        system_prompt = """
        Ты - эксперт по рекомендациям и образовательный консультант.
        
        Твои задачи:
        1. ПРОАНАЛИЗИРОВАТЬ все полученные данные
        2. ВЫДЕЛИТЬ наиболее подходящие материалы
        3. СФОРМИРОВАТЬ персонализированные рекомендации
        4. ОБЪЯСНИТЬ выбор
        5. ПРЕДЛОЖИТЬ альтернативы
        
        Учитывай:
        - Уровень сложности материала
        - Целевую аудиторию
        - Практическую ценность
        - Доступность изложения
        """
        
        user_prompt = f"""
        ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"
        
        АГРЕГИРОВАННЫЕ ДАННЫЕ:
        {json.dumps(aggregated_data, ensure_ascii=False, indent=2)}
        
        Создай персонализированные рекомендации.
        
        Формат JSON:
        {{
            "primary_recommendation": {{
                "content": "основная рекомендация",
                "reason": "почему рекомендовано",
                "best_for": ["ситуация1", "ситуация2"],
                "expected_benefit": "какая польза",
                "time_requirement": "оценка времени"
            }},
            "alternative_recommendations": [
                {{
                    "content": "альтернатива1",
                    "reason": "почему альтернатива",
                    "when_to_choose": "когда выбирать эту альтернативу",
                    "pros_cons": ["плюс1", "минус1"]
                }}
            ],
            "learning_path": [
                {{
                    "step": 1,
                    "action": "что сделать",
                    "material": "какой материал",
                    "duration": "сколько времени",
                    "outcome": "чего достигнем"
                }}
            ],
            "avoid_recommendations": [
                {{
                    "what": "чего избегать",
                    "why": "почему",
                    "alternative": "чем заменить"
                }}
            ],
            "resource_list": [
                {{
                    "type": "тип ресурса",
                    "name": "название",
                    "why_useful": "чем полезен",
                    "access_info": "как получить"
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=2500
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка генерации рекомендаций: {e}")
            raise
    
    def _adapt_to_user_level(self, recommendations: Dict, user_context: Optional[Dict]) -> Dict[str, Any]:
        """
        Адаптация рекомендаций под уровень пользователя.
        """
        user_level = user_context.get("user_level", "intermediate") if user_context else "intermediate"
        
        system_prompt = f"""
        Ты адаптируешь рекомендации под уровень пользователя: {user_level}.
        
        Для начинающих:
        - Используй простой язык
        - Давай больше объяснений
        - Рекомендуй базовые материалы
        - Предлагай больше примеров
        
        Для продвинутых:
        - Используй профессиональную терминологию
        - Сосредоточься на деталях
        - Рекомендуй углубленные материалы
        - Предлагай практические задачи
        """
        
        user_prompt = f"""
        ИСХОДНЫЕ РЕКОМЕНДАЦИИ:
        {json.dumps(recommendations, ensure_ascii=False, indent=2)}
        
        Адаптируй эти рекомендации для пользователя уровня: {user_level}.
        
        Основные изменения:
        1. Язык и терминология
        2. Уровень детализации
        3. Сложность материалов
        4. Объем объяснений
        
        Верни адаптированные рекомендации в том же формате.
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.warning(f"Ошибка адаптации: {e}. Использую исходные рекомендации.")
            return recommendations
    
    def _create_final_response(self, adapted_recommendations: Dict, query: str) -> Dict[str, Any]:
        """
        Создание финального ответа пользователю.
        """
        system_prompt = """
        Ты формируешь финальный ответ пользователю на основе рекомендаций.
        Ответ должен быть:
        - Понятным и полезным
        - Структурированным
        - Содержать объяснения
        - Указывать ограничения
        
        Используй маркдаун для форматирования.
        """
        
        user_prompt = f"""
        ЗАПРОС ПОЛЬЗОВАТЕЛЬЯ: "{query}"
        
        АДАПТИРОВАННЫЕ РЕКОМЕНДАЦИИ:
        {json.dumps(adapted_recommendations, ensure_ascii=False, indent=2)}
        
        Создай финальный ответ, который включает:
        1. Краткое резюме ответа на запрос
        2. Основную рекомендацию с обоснованием
        3. Альтернативные варианты
        4. Пошаговый план изучения
        5. Полезные ресурсы
        6. Ограничения и предупреждения
        
        Формат ответа:
        {{
            "query": "оригинальный запрос",
            "executive_summary": "краткий ответ",
            "detailed_recommendation": {{
                "main": "главная рекомендация",
                "reasoning": "объяснение выбора",
                "evidence": "на чем основано"
            }},
            "alternatives": [
                {{
                    "option": "вариант",
                    "when_better": "когда лучше",
                    "tradeoffs": "компромиссы"
                }}
            ],
            "actionable_steps": [
                {{
                    "step": "шаг",
                    "action": "действие",
                    "resource": "ресурс",
                    "expected_outcome": "результат"
                }}
            ],
            "additional_resources": [
                {{
                    "type": "тип",
                    "name": "название",
                    "purpose": "для чего",
                    "access": "как получить"
                }}
            ],
            "limitations_and_disclaimers": [
                "ограничение1",
                "ограничение2"
            ],
            "formatted_response": "текст ответа в markdown"
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=3000
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка создания финального ответа: {e}")
            raise
    
    def _enrich_with_explanations(self, response: Dict, aggregated_data: Dict) -> Dict[str, Any]:
        """
        Обогащение ответа объяснениями и метаинформацией.
        """
        enriched = response.copy()
        
        # Добавление метаинформации
        enriched["meta"] = {
            "generation_timestamp": datetime.now().isoformat(),
            "data_sources_used": list(aggregated_data.keys()),
            "recommendation_confidence": self._calculate_confidence(aggregated_data),
            "user_context_applied": bool(aggregated_data.get("user_context")),
            "version": "1.0"
        }
        
        # Добавление рекомендаций по дальнейшим действиям
        enriched["next_actions"] = [
            "Начать с первого шага плана",
            "Сохранить список ресурсов",
            "Обратиться за уточнениями если нужно"
        ]
        
        return enriched
    
    def _calculate_confidence(self, aggregated_data: Dict) -> float:
        """Расчет уверенности в рекомендациях"""
        # Простая эвристика на основе полноты данных
        confidence = 0.5  # Базовая уверенность
        
        if aggregated_data.get("top_content"):
            confidence += 0.2
        if aggregated_data.get("key_concepts"):
            confidence += 0.1
        if aggregated_data.get("user_context"):
            confidence += 0.1
        if aggregated_data.get("comparisons"):
            confidence += 0.1
        
        return min(confidence, 0.95)  # Максимум 95%
    
    def _extract_top_content(self, results: Dict) -> List[Dict]:
        """Извлечение лучшего контента"""
        top_content = []
        
        # Из анализа
        if "analysis_results" in results:
            analysis = results["analysis_results"]
            if "best_overall" in analysis.get("comparative_analysis", {}):
                top_content.append({
                    "source": "analysis",
                    "content": analysis["comparative_analysis"]["best_overall"],
                    "reason": "Лучший по комплексной оценке"
                })
        
        return top_content
    
    def _extract_key_concepts(self, results: Dict) -> List[str]:
        """Извлечение ключевых концепций"""
        concepts = []
        
        if "analysis_results" in results:
            analysis = results["analysis_results"]
            if "key_concepts" in analysis:
                for concept in analysis["key_concepts"][:5]:
                    concepts.append(concept.get("concept", ""))
        
        return concepts
    
    def _extract_comparisons(self, results: Dict) -> List[Dict]:
        """Извлечение сравнений"""
        comparisons = []
        
        if "analysis_results" in results:
            analysis = results["analysis_results"]
            if "comparative_analysis" in analysis:
                comp = analysis["comparative_analysis"]
                if "comparison_matrix" in comp:
                    comparisons = comp["comparison_matrix"]
        
        return comparisons
    
    def _extract_limitations(self, results: Dict) -> List[str]:
        """Извлечение ограничений"""
        limitations = []
        
        # Из анализа
        if "analysis_results" in results:
            analysis = results["analysis_results"]
            if "limitations" in analysis:
                for limit in analysis["limitations"]:
                    limitations.append(limit.get("limitation", ""))
        
        # Из критики
        if "critique_results" in results:
            critique = results["critique_results"]
            if "missing_aspects" in critique:
                limitations.extend(critique["missing_aspects"])
        
        return limitations
    
    def _create_error_response(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Создание ответа об ошибке"""
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "agent": "RecommendationAgent",
            "original_query": query,
            "error": error_msg,
            "formatted_response": f"# Ошибка формирования рекомендаций\n\nПроизошла ошибка: {error_msg}\n\nПожалуйста, попробуйте переформулировать запрос или обратитесь к администратору системы.",
            "meta": {
                "confidence": 0.0,
                "data_sources_used": [],
                "generation_timestamp": datetime.now().isoformat()
            }
        }


# Пример использования
if __name__ == "__main__":
    from gigachat_client import GigaChatClient
    
    # Инициализация
    client = GigaChatClient(verify_ssl=False)
    agent = RecommendationAgent(client)
    
    # Тестовые данные
    test_results = {
        "search_results": {
            "best_match": "Учебник по Python",
            "alternatives": ["Курс по программированию", "Книга для начинающих"]
        },
        "analysis_results": {
            "key_concepts": [
                {"concept": "Переменные", "importance": "высокая"},
                {"concept": "Циклы", "importance": "средняя"}
            ],
            "comparative_analysis": {
                "best_overall": "Учебник по Python для начинающих",
                "best_for_beginners": "Интерактивный курс"
            }
        }
    }
    
    # Формирование рекомендаций
    recommendations = agent.process(
        analysis_results=test_results,
        user_query="как начать программировать на Python",
        user_context={"user_level": "beginner"}
    )
    
    print("=" * 80)
    print("РЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ:")
    print("=" * 80)
    print(recommendations.get("formatted_response", "Нет ответа"))