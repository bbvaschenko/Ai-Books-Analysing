"""
RecommendationAgent - агент формирования финального пользовательского ответа
"""
import json
from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentMessage


class RecommendationAgent(BaseAgent):
    """Агент формирования финальных рекомендаций"""

    def __init__(self):
        super().__init__(
            name="RecommendationAgent",
            role="Агент формирования финального пользовательского ответа",
            description="""Я агрегирую результаты анализа, адаптирую вывод под уровень пользователя
и формат. Формирую финальную рекомендацию с пояснением выбора, возможными альтернативами,
ограничениями и допущениями."""
        )

        self.user_profiles = {
            "beginner": {
                "language": "простой",
                "detail_level": "basic",
                "focus": "практическое применение"
            },
            "intermediate": {
                "language": "умеренно технический",
                "detail_level": "balanced",
                "focus": "теория и практика"
            },
            "advanced": {
                "language": "технический",
                "detail_level": "detailed",
                "focus": "глубокий анализ"
            }
        }

    async def process(self, message: AgentMessage, context: Dict = None) -> AgentMessage:
        """Формирование финальных рекомендаций"""
        self.add_to_history(message)

        analysis_results = message.content
        original_query = analysis_results.get("original_query", "")
        user_context = analysis_results.get("context", {})

        user_profile = self._determine_user_profile(user_context)

        print(f"  RecommendationAgent формирует рекомендации для профиля: {user_profile}")

        aggregated_results = await self._aggregate_results_with_llm(analysis_results)

        adapted_recommendations = await self._adapt_to_user_with_llm(
            aggregated_results,
            user_profile,
            original_query
        )

        explainable_result = await self._create_explainable_result_with_llm(
            adapted_recommendations,
            original_query
        )

        final_response = self._format_final_response(
            explainable_result,
            user_profile,
            original_query
        )

        response_content = {
            "original_query": original_query,
            "user_profile": user_profile,
            "recommendations": final_response,
            "reasoning": explainable_result.get("reasoning", ""),
            "alternatives": explainable_result.get("alternatives", []),
            "limitations": explainable_result.get("limitations", []),
            "assumptions": explainable_result.get("assumptions", []),
            "response_format": "structured_with_explanation"
        }

        print(f"✅ RecommendationAgent сформировал {len(final_response.get('top_recommendations', []))} рекомендаций")

        return AgentMessage(
            sender=self.name,
            recipient="User",
            content=response_content,
            conversation_id=message.conversation_id
        )

    def _determine_user_profile(self, context: Dict) -> str:
        """Определение профиля пользователя"""
        if context.get("user_level") == "beginner":
            return "beginner"
        elif context.get("user_level") == "expert":
            return "advanced"
        else:
            return "intermediate"

    async def _aggregate_results_with_llm(self, analysis_results: Dict) -> Dict[str, Any]:
        """Агрегация результатов анализа с помощью GigaChat"""
        system_prompt = """Ты - RecommendationAgent, агрегирующий результаты анализа.
Твоя задача: объединить результаты анализа нескольких кандидатов в единую структурированную форму.

Извлеки:
1. Лучшие кандидаты с их характеристиками
2. Общую оценку качества анализа
3. Ключевые выводы из сравнения
4. Рекомендации для дальнейшего использования

Верни ответ в формате JSON."""

        structured_report = analysis_results.get("structured_report", {})
        comparative_analysis = analysis_results.get("comparative_analysis", {})

        prompt = f"""Результаты анализа:
Структурированный отчет: {json.dumps(structured_report, ensure_ascii=False, indent=2)[:1000]}...
Сравнительный анализ: {json.dumps(comparative_analysis, ensure_ascii=False, indent=2)[:500]}...

Агрегируй эти результаты."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ RecommendationAgent: Ошибка GigaChat агрегации: {e}")

        return self._emulate_aggregation(analysis_results)

    def _emulate_aggregation(self, analysis_results: Dict) -> Dict[str, Any]:
        """Эмуляция агрегации результатов"""
        structured_report = analysis_results.get("structured_report", {})
        comparative_analysis = analysis_results.get("comparative_analysis", {})

        best_candidates = []

        if structured_report:
            candidates = structured_report.get("candidate_analyses", [])
            for candidate in candidates[:3]:
                cand_data = candidate.get("candidate", {})
                analysis = candidate.get("analysis", {})

                best_candidates.append({
                    "id": cand_data.get("book_id", ""),
                    "name": cand_data.get("filename", ""),
                    "area": cand_data.get("area", ""),
                    "relevance": analysis.get("relevance_score", 0),
                    "difficulty": analysis.get("difficulty_level", ""),
                    "audience": analysis.get("target_audience", []),
                    "key_concepts": analysis.get("key_concepts", []),
                    "summary": analysis.get("summary", "")
                })

        best_overall = comparative_analysis.get("best_overall", {})
        best_for_beginners = comparative_analysis.get("best_for_beginners", {})

        return {
            "best_candidates": best_candidates,
            "best_overall": best_overall,
            "best_for_beginners": best_for_beginners,
            "total_analyzed": len(best_candidates),
            "analysis_quality": analysis_results.get("analysis_methodology", "LLM-based"),
            "comparison_summary": comparative_analysis.get("comparison_summary", "")
        }

    async def _adapt_to_user_with_llm(self, aggregated: Dict, profile: str, query: str) -> Dict[str, Any]:
        """Адаптация рекомендаций под пользователя с помощью GigaChat"""
        system_prompt = f"""Ты - RecommendationAgent, адаптирующий рекомендации под пользователя.
Профиль пользователя: {profile}
Язык: {self.user_profiles[profile]['language']}
Уровень детализации: {self.user_profiles[profile]['detail_level']}
Фокус: {self.user_profiles[profile]['focus']}

Адаптируй рекомендации под этот профиль, сохраняя суть, но меняя форму изложения.
Верни ответ в формате JSON."""

        prompt = f"""Запрос пользователя: {query}
Агрегированные результаты: {json.dumps(aggregated, ensure_ascii=False, indent=2)[:1000]}...

Адаптируй эти рекомендации для профиля {profile}."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ RecommendationAgent: Ошибка GigaChat адаптации: {e}")

        return self._emulate_adaptation(aggregated, profile, query)

    def _emulate_adaptation(self, aggregated: Dict, profile: str, query: str) -> Dict[str, Any]:
        """Эмуляция адаптации под пользователя"""
        profile_settings = self.user_profiles.get(profile, self.user_profiles["intermediate"])

        recommendations = []

        for candidate in aggregated.get("best_candidates", []):
            adapted = self._adapt_candidate_recommendation(candidate, profile_settings, query)
            recommendations.append(adapted)

        recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return {
            "profile": profile,
            "language_style": profile_settings["language"],
            "detail_level": profile_settings["detail_level"],
            "top_recommendations": recommendations[:3],
            "alternative_options": recommendations[3:6] if len(recommendations) > 3 else [],
            "focus_area": profile_settings["focus"]
        }

    def _adapt_candidate_recommendation(self, candidate: Dict, profile: Dict, query: str) -> Dict[str, Any]:
        """Адаптация рекомендации по кандидату"""
        if profile["language"] == "простой":
            explanation = f"Рекомендую '{candidate.get('name', '')}' потому что он хорошо подходит для вашего запроса '{query}'. Уровень сложности: {candidate.get('difficulty', 'средний')}."
        elif profile["language"] == "технический":
            explanation = f"На основе анализа релевантности ({candidate.get('relevance', 0):.1%}) и соответствия целевой аудитории ({', '.join(candidate.get('audience', []))}), рекомендую материал: {candidate.get('name', '')}."
        else:
            explanation = f"Материал '{candidate.get('name', '')}' релевантен запросу '{query}' (оценка: {candidate.get('relevance', 0):.1%}). Уровень сложности: {candidate.get('difficulty', '')}."

        if profile["detail_level"] == "basic":
            details = {
                "key_points": candidate.get("key_concepts", [])[:2],
                "summary": candidate.get("summary", "")[:100] + "..." if len(
                    candidate.get("summary", "")) > 100 else candidate.get("summary", "")
            }
        elif profile["detail_level"] == "detailed":
            details = {
                "key_points": candidate.get("key_concepts", []),
                "summary": candidate.get("summary", ""),
                "strengths": ["Хорошая структура", "Соответствие теме"],
                "considerations": ["Проверьте актуальность", "Учтите свой уровень подготовки"]
            }
        else:
            details = {
                "key_points": candidate.get("key_concepts", [])[:3],
                "summary": candidate.get("summary", "")[:200] + "..." if len(
                    candidate.get("summary", "")) > 200 else candidate.get("summary", ""),
                "best_for": candidate.get("audience", [])
            }

        return {
            "id": candidate.get("id", ""),
            "name": candidate.get("name", ""),
            "area": candidate.get("area", ""),
            "relevance_score": candidate.get("relevance", 0),
            "difficulty_level": candidate.get("difficulty", ""),
            "explanation": explanation,
            "details": details,
            "why_better": f"Релевантность выше среднего" if candidate.get("relevance",
                                                                          0) > 0.5 else f"Соответствует запросу"
        }

    async def _create_explainable_result_with_llm(self, recommendations: Dict, query: str) -> Dict[str, Any]:
        """Создание объяснимого результата с помощью GigaChat"""
        system_prompt = """Ты - RecommendationAgent, создающий объяснимые рекомендации.
Твоя задача: создать понятное объяснение выбора, указать альтернативы, ограничения и допущения.
Сделай рекомендации максимально прозрачными и полезными.
Верни ответ в формате JSON."""

        prompt = f"""Запрос пользователя: {query}
Рекомендации: {json.dumps(recommendations, ensure_ascii=False, indent=2)[:1000]}...

Создай объяснимый результат с обоснованием выбора."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ RecommendationAgent: Ошибка GigaChat создания объяснения: {e}")

        return self._emulate_explainable_result(recommendations, query)

    def _emulate_explainable_result(self, recommendations: Dict, query: str) -> Dict[str, Any]:
        """Эмуляция создания объяснимого результата"""
        top_recommendations = recommendations.get("top_recommendations", [])

        if not top_recommendations:
            return {
                "reasoning": "По вашему запросу не найдено подходящих материалов.",
                "alternatives": [],
                "limitations": ["Ограниченная база данных", "Запрос слишком специфичен"],
                "assumptions": ["Предполагается базовое понимание темы"]
            }

        best_recommendation = top_recommendations[0]

        reasoning = f"Для запроса '{query}' рекомендую '{best_recommendation['name']}' потому что:\n"
        reasoning += f"1. Высокая релевантность теме ({best_recommendation['relevance_score']:.1%})\n"
        reasoning += f"2. Подходящий уровень сложности: {best_recommendation['difficulty_level']}\n"
        reasoning += f"3. Охватывает ключевые понятия: {', '.join(best_recommendation['details'].get('key_points', [])[:3])}"

        alternatives = []
        for i, rec in enumerate(top_recommendations[1:], 1):
            alternatives.append({
                "name": rec["name"],
                "reason": f"Альтернатива {i}: {rec['why_better']}",
                "when_to_choose": f"Если нужно больше акцента на {rec['area']}"
            })

        limitations = [
            "Анализ основан на метаданных, а не полном содержании",
            "Рекомендации могут не учитывать личные предпочтения"
        ]

        assumptions = [
            "Пользователь ищет учебный материал",
            "Точность запроса соответствует потребности"
        ]

        return {
            "reasoning": reasoning,
            "alternatives": alternatives,
            "limitations": limitations,
            "assumptions": assumptions,
            "best_choice": best_recommendation["name"],
            "confidence": "высокая" if best_recommendation["relevance_score"] > 0.7 else "средняя"
        }

    def _format_final_response(self, explainable: Dict, profile: str, query: str) -> Dict[str, Any]:
        """Форматирование финального ответа"""
        response = {
            "query": query,
            "profile": profile,
            "recommendation_summary": explainable.get("reasoning", ""),
            "top_recommendations": [],
            "alternative_options": explainable.get("alternatives", []),
            "important_notes": {
                "limitations": explainable.get("limitations", []),
                "assumptions": explainable.get("assumptions", []),
                "confidence": explainable.get("confidence", "средняя")
            },
            "next_steps": [
                "Изучите рекомендованный материал",
                "Рассмотрите альтернативные варианты",
                "При необходимости уточните запрос"
            ]
        }

        return response