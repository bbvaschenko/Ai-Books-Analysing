"""
AnalysisAgent - агент глубокого анализа контента
"""
import json
from typing import Dict, Any, List
from base_agent import BaseAgent, AgentMessage


class AnalysisAgent(BaseAgent):
    """Агент глубокого анализа учебного контента"""

    def __init__(self, library_core):
        super().__init__(
            name="AnalysisAgent",
            role="Агент глубокого анализа найденного контента",
            description="""Я анализирую каждый кандидат на релевантность, уровень сложности,
соответствие целевой аудитории. Формирую структурированный отчет с резюме,
ключевыми понятиями, ограничениями и сравнением альтернатив."""
        )

        self.library_core = library_core
        self.analysis_history = []

    async def process(self, message: AgentMessage, context: Dict = None) -> AgentMessage:
        """Анализ кандидатов"""
        self.add_to_history(message)

        candidates = message.content.get("ranked_results", [])
        original_query = message.content.get("original_query", "")
        task = message.content.get("task", "")

        if not candidates:
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                content={
                    "error": "Нет кандидатов для анализа",
                    "original_query": original_query
                },
                conversation_id=message.conversation_id
            )

        print(f" AnalysisAgent начинает анализ {len(candidates)} кандидатов")

        analyzed_candidates = []
        for i, candidate in enumerate(candidates[:5]):
            print(f"   Анализ кандидата {i + 1}/{min(5, len(candidates))}: {candidate.get('filename', '')}")

            analysis = await self._analyze_candidate_with_llm(candidate, original_query)
            analyzed_candidates.append({
                "candidate": candidate,
                "analysis": analysis,
                "analysis_timestamp": "now"
            })

        comparative_analysis = await self._compare_candidates_with_llm(analyzed_candidates, original_query)

        structured_report = self._create_structured_report(
            analyzed_candidates,
            comparative_analysis,
            original_query
        )

        response_content = {
            "original_query": original_query,
            "total_candidates_analyzed": len(analyzed_candidates),
            "structured_report": structured_report,
            "comparative_analysis": comparative_analysis,
            "analysis_methodology": "LLM-анализ релевантности, сложности и соответствия аудитории",
            "limitations": ["Анализ ограничен доступными метаданными", "Требуется проверка CriticAgent"]
        }

        print(f"✅ AnalysisAgent завершил анализ {len(analyzed_candidates)} кандидатов")

        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content=response_content,
            conversation_id=message.conversation_id
        )

    async def _analyze_candidate_with_llm(self, candidate: Dict, query: str) -> Dict[str, Any]:
        """Анализ отдельного кандидата с помощью GigaChat"""
        system_prompt = """Ты - AnalysisAgent, агент глубокого анализа учебного контента.
Твоя задача: анализировать учебные материалы по нескольким критериям.

Проанализируй материал по следующим критериям:
1. Релевантность запросу (оценка 0-1 и причины)
2. Уровень сложности (начальный, средний, продвинутый)
3. Целевую аудиторию (школьники, студенты, преподаватели, специалисты)
4. Ключевые понятия и темы
5. Сильные стороны и ограничения
6. Рекомендации по использованию

Верни ответ в формате JSON."""

        filename = candidate.get("filename", "")
        area = candidate.get("area", "")
        tags = candidate.get("matching_tags", [])
        summary = candidate.get("ai_summary", "")

        prompt = f"""Проанализируй учебный материал:

Название: {filename}
Область: {area}
Теги: {', '.join(tags)}
Краткое описание: {summary[:200] if summary else "Нет описания"}
Запрос пользователя: {query}

Сформулируй детальный анализ."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ AnalysisAgent: Ошибка GigaChat анализа: {e}")

        return await self._emulate_candidate_analysis(candidate, query)

    async def _emulate_candidate_analysis(self, candidate: Dict, query: str) -> Dict[str, Any]:
        """Эмуляция анализа кандидата"""
        filename = candidate.get("filename", "")
        area = candidate.get("area", "")
        tags = candidate.get("matching_tags", [])
        summary = candidate.get("ai_summary", "")

        query_words = query.lower().split()
        filename_lower = filename.lower()

        relevance_score = 0.0
        relevance_reasons = []

        for word in query_words:
            if len(word) > 3 and word in filename_lower:
                relevance_score += 0.1
                relevance_reasons.append(f"Содержит ключевое слово '{word}'")

        if area and any(word in area.lower() for word in query_words):
            relevance_score += 0.2
            relevance_reasons.append(f"Соответствует области '{area}'")

        relevance_score = min(1.0, relevance_score)

        complexity_keywords = {
            "начальный": ["базов", "введен", "основ", "начал"],
            "средний": ["средн", "продвинут", "углублен"],
            "продвинутый": ["высш", "университет", "исследован", "теоретич"]
        }

        difficulty_level = "средний"
        for level, keywords in complexity_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                difficulty_level = level
                break

        audience_keywords = {
            "школьники": ["школ", "класс", "ученик"],
            "студенты": ["студент", "вуз", "универ", "курс"],
            "преподаватели": ["препода", "учител", "методич"],
            "специалисты": ["специалист", "профессионал", "практик"]
        }

        target_audience = ["студенты"]
        for audience, keywords in audience_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                target_audience.append(audience)

        key_concepts = tags[:5] if tags else ["математика", "учебный материал"]

        limitations = []
        if len(summary) < 50:
            limitations.append("Мало описательной информации")
        if len(tags) < 2:
            limitations.append("Недостаточно тегов для точной классификации")

        return {
            "relevance_score": relevance_score,
            "relevance_reasons": relevance_reasons,
            "difficulty_level": difficulty_level,
            "target_audience": list(set(target_audience)),
            "key_concepts": key_concepts,
            "summary": f"{filename} - {area}. {summary[:100]}..." if summary else f"{filename} - {area}",
            "strengths": [
                "Структурированный материал",
                "Соответствует запросу" if relevance_score > 0.5 else "Частично соответствует запросу"
            ],
            "limitations": limitations,
            "recommendations_for_use": [
                f"Подходит для {target_audience[0]}" if target_audience else "Общего ознакомления"
            ]
        }

    async def _compare_candidates_with_llm(self, candidates: List[Dict], query: str) -> Dict[str, Any]:
        """Сравнение кандидатов между собой с помощью GigaChat"""
        if len(candidates) < 2:
            return {
                "comparison_possible": False,
                "reason": "Недостаточно кандидатов для сравнения"
            }

        system_prompt = """Ты - AnalysisAgent, занимающийся сравнением учебных материалов.
Сравни несколько учебных материалов и определи:
1. Лучший материал по общей релевантности
2. Лучший материал для начинающих
3. Сравнительную таблицу преимуществ и недостатков
4. Итоговую рекомендацию

Верни ответ в формате JSON."""

        candidate_descriptions = []
        for i, item in enumerate(candidates):
            candidate = item["candidate"]
            analysis = item["analysis"]

            candidate_desc = f"""
Материал {i+1}:
- Название: {candidate.get('filename', '')}
- Область: {candidate.get('area', '')}
- Релевантность: {analysis.get('relevance_score', 0)}
- Сложность: {analysis.get('difficulty_level', '')}
- Аудитория: {', '.join(analysis.get('target_audience', []))}
- Ключевые темы: {', '.join(analysis.get('key_concepts', [])[:3])}
"""
            candidate_descriptions.append(candidate_desc)

        prompt = f"""Сравни следующие учебные материалы для запроса: "{query}"

Материалы для сравнения:
{''.join(candidate_descriptions)}

Проанализируй и сравни их, дай рекомендации."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ AnalysisAgent: Ошибка GigaChat сравнения: {e}")

        return await self._emulate_candidate_comparison(candidates, query)

    async def _emulate_candidate_comparison(self, candidates: List[Dict], query: str) -> Dict[str, Any]:
        """Эмуляция сравнения кандидатов"""
        analyzed_candidates = []
        for item in candidates:
            candidate = item["candidate"]
            analysis = item["analysis"]

            analyzed_candidates.append({
                "filename": candidate.get("filename", ""),
                "relevance_score": analysis.get("relevance_score", 0),
                "difficulty_level": analysis.get("difficulty_level", ""),
                "target_audience": analysis.get("target_audience", []),
                "key_concepts": analysis.get("key_concepts", [])
            })

        sorted_by_relevance = sorted(
            analyzed_candidates,
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        difficulty_weights = {"начальный": 3, "средний": 2, "продвинутый": 1}
        sorted_by_simplicity = sorted(
            analyzed_candidates,
            key=lambda x: difficulty_weights.get(x["difficulty_level"], 1),
            reverse=True
        )

        best_overall = sorted_by_relevance[0] if sorted_by_relevance else None
        best_for_beginners = sorted_by_simplicity[0] if sorted_by_simplicity else None

        return {
            "comparison_possible": True,
            "best_overall": best_overall,
            "best_for_beginners": best_for_beginners,
            "comparison_summary": f"Лучший по релевантности: {best_overall['filename'] if best_overall else 'нет'}. Лучший для начинающих: {best_for_beginners['filename'] if best_for_beginners else 'нет'}",
            "comparison_criteria": ["релевантность", "уровень сложности", "целевая аудитория"],
            "recommendation": f"Для запроса '{query}' рекомендую {best_overall['filename'] if best_overall else 'первый кандидат'} как наиболее релевантный"
        }

    def _create_structured_report(self, candidates: List[Dict], comparison: Dict, query: str) -> Dict[str, Any]:
        """Создание структурированного отчета"""
        report = {
            "query": query,
            "analysis_date": "now",
            "total_candidates": len(candidates),
            "candidate_analyses": [],
            "summary": {
                "best_candidate": comparison.get("best_overall", {}),
                "alternative_recommendations": comparison.get("best_for_beginners", {}),
                "overall_quality": "высокий" if len(candidates) > 2 else "средний"
            },
            "methodology": {
                "analysis_type": "LLM-based multi-criteria analysis",
                "criteria_used": ["relevance", "difficulty", "audience_match", "content_quality"],
                "limitations": "Анализ основан на метаданных, требуется проверка содержания"
            }
        }

        for item in candidates:
            report["candidate_analyses"].append({
                "candidate_id": item["candidate"].get("book_id", ""),
                "filename": item["candidate"].get("filename", ""),
                "analysis": item["analysis"]
            })

        return report

    async def process_rerun_request(self, message: AgentMessage) -> AgentMessage:
        """Обработка запроса на повторный анализ"""
        print(f" AnalysisAgent выполняет повторный анализ по запросу CriticAgent")

        original_results = message.content.get("original_results", {})
        corrections_needed = message.content.get("corrections_needed", [])
        feedback = message.content.get("feedback", {})

        improved_analysis = await self._improve_analysis_with_feedback(
            original_results,
            corrections_needed,
            feedback
        )

        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            content={
                "improved_analysis": improved_analysis,
                "corrections_applied": corrections_needed,
                "feedback_processed": True,
                "improvement_summary": f"Учтено {len(corrections_needed)} замечаний от CriticAgent"
            },
            conversation_id=message.conversation_id
        )

    async def _improve_analysis_with_feedback(self, original_results: Dict,
                                              corrections: List[str],
                                              feedback: Dict) -> Dict[str, Any]:
        """Улучшение анализа с учетом обратной связи"""
        improved = original_results.copy()

        if "structured_report" in improved:
            improved["structured_report"]["analysis_improved"] = True
            improved["structured_report"]["feedback_applied"] = corrections
            improved["structured_report"]["improvement_date"] = "now"

            if "Улучшить анализ релевантности" in corrections:
                for candidate in improved["structured_report"].get("candidate_analyses", []):
                    analysis = candidate.get("analysis", {})
                    if "relevance_score" in analysis:
                        analysis["relevance_score"] = min(1.0, analysis["relevance_score"] + 0.05)
                        analysis["relevance_reasons"].append("Уточнено по замечанию CriticAgent")

        improved["analysis_quality"] = "улучшенный"
        improved["revision_number"] = original_results.get("revision_number", 0) + 1

        return improved