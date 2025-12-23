"""
CriticAgent - агент контроля качества и корректности решений
"""
import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentMessage


class CriticAgent(BaseAgent):
    """Агент-критик для проверки качества результатов"""

    def __init__(self):
        super().__init__(
            name="CriticAgent",
            role="Агент контроля качества и корректности решений",
            description="""Я проверяю релевантность результата запросу, достаточность ответа,
логические ошибки и пробелы. Принимаю решение: принять, отклонить или запросить повторный анализ.
Явно формулирую замечания, причины отклонения и рекомендации по исправлению."""
        )

        self.feedback_history = []

    async def process(self, message: AgentMessage, context: Dict = None) -> AgentMessage:
        """Критическая проверка результатов"""
        self.add_to_history(message)

        sender_agent = message.sender
        results = message.content
        original_query = results.get("original_query", "")

        print(f"  CriticAgent проверяет результаты от {sender_agent}")

        quality_analysis = await self._analyze_quality_with_llm(results, original_query, sender_agent)

        decision = await self._make_decision_with_llm(quality_analysis)

        detailed_feedback = self._create_detailed_feedback(quality_analysis, decision)

        response_content = {
            "results_reviewed": {
                "from_agent": sender_agent,
                "results_type": results.get("structured_report", {}).get("analysis_type", "unknown")
            },
            "quality_analysis": quality_analysis,
            "decision": decision,
            "detailed_feedback": detailed_feedback,
            "original_query": original_query,
            "needs_correction": decision.get("verdict") in ["reject", "needs_improvement"],
            "issues": quality_analysis.get("issues_found", []),
            "corrections_needed": detailed_feedback.get("specific_corrections", [])
        }

        self.feedback_history.append({
            "timestamp": "now",
            "target_agent": sender_agent,
            "decision": decision.get("verdict"),
            "issues_found": len(quality_analysis.get("issues_found", [])),
            "feedback_given": True
        })

        print(f"  CriticAgent вынес вердикт: {decision.get('verdict')}")
        if decision.get("verdict") != "approve":
            print(f"   Замечания: {len(quality_analysis.get('issues_found', []))}")

        return AgentMessage(
            sender=self.name,
            recipient="CoordinatorAgent",
            content=response_content,
            conversation_id=message.conversation_id
        )

    async def _analyze_quality_with_llm(self, results: Dict, query: str, sender_agent: str) -> Dict[str, Any]:
        """Анализ качества результатов с помощью GigaChat"""
        system_prompt = """Ты - CriticAgent, агент контроля качества.
Твоя задача: проверять результаты других агентов на качество.

Проверь следующее:
1. Релевантность результата исходному запросу
2. Полноту и достаточность информации
3. Логическую целостность и отсутствие противоречий
4. Качество структурирования и представления данных
5. Соответствие ожидаемому формату вывода

Верни ответ в формате JSON с полями: issues_found, strengths_found, total_issues, total_strengths, completeness_score, logic_score."""

        prompt = f"""Проверь качество результатов от агента {sender_agent}:

Исходный запрос: {query}
Результаты для проверки: {json.dumps(results, ensure_ascii=False, indent=2)[:1500]}...

Проанализируй качество и найди проблемы."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ CriticAgent: Ошибка GigaChat анализа: {e}")

        return await self._emulate_quality_analysis(results, query, sender_agent)

    async def _emulate_quality_analysis(self, results: Dict, query: str, sender_agent: str) -> Dict[str, Any]:
        """Эмуляция анализа качества"""
        issues_found = []
        strengths_found = []

        if sender_agent == "AnalysisAgent":
            structured_report = results.get("structured_report", {})

            if not structured_report:
                issues_found.append("Отсутствует структурированный отчет")

            candidates = structured_report.get("candidate_analyses", [])
            if len(candidates) == 0:
                issues_found.append("Нет проанализированных кандидатов")
            elif len(candidates) < 2:
                issues_found.append("Слишком мало кандидатов для сравнения")
            else:
                strengths_found.append(f"Проанализировано {len(candidates)} кандидатов")

            for i, candidate in enumerate(candidates[:3]):
                analysis = candidate.get("analysis", {})

                if not analysis.get("relevance_score"):
                    issues_found.append(f"Кандидат {i+1}: отсутствует оценка релевантности")

                if not analysis.get("difficulty_level"):
                    issues_found.append(f"Кандидат {i+1}: отсутствует оценка сложности")

                if analysis.get("relevance_score", 0) < 0.3:
                    issues_found.append(f"Кандидат {i+1}: низкая релевантность ({analysis.get('relevance_score')})")
                else:
                    strengths_found.append(f"Кандидат {i+1}: хорошая релевантность")

        elif sender_agent == "SearchAgent":
            found_candidates = results.get("found_candidates", 0)

            if found_candidates == 0:
                issues_found.append("Поиск не дал результатов")
            elif found_candidates < 3:
                issues_found.append(f"Мало результатов поиска: {found_candidates}")
            else:
                strengths_found.append(f"Найдено {found_candidates} кандидатов")

            if not results.get("search_strategy_explanation"):
                issues_found.append("Отсутствует объяснение стратегии поиска")

        original_query_lower = query.lower()
        results_summary = str(results).lower()

        query_match_score = 0
        for word in original_query_lower.split():
            if len(word) > 3 and word in results_summary:
                query_match_score += 1

        if query_match_score < 2 and len(original_query_lower.split()) > 3:
            issues_found.append(f"Низкое соответствие запросу: только {query_match_score} ключевых слов")

        logic_issues = self._check_logical_consistency(results)
        issues_found.extend(logic_issues)

        return {
            "issues_found": issues_found,
            "strengths_found": strengths_found,
            "query_match_score": query_match_score,
            "total_issues": len(issues_found),
            "total_strengths": len(strengths_found),
            "completeness_score": self._calculate_completeness_score(results),
            "logic_score": 10 - len(logic_issues) * 2
        }

    def _check_logical_consistency(self, results: Dict) -> List[str]:
        """Проверка логической целостности результатов"""
        issues = []

        if "structured_report" in results:
            report = results["structured_report"]
            candidates = report.get("candidate_analyses", [])

            relevance_scores = []
            for candidate in candidates:
                analysis = candidate.get("analysis", {})
                score = analysis.get("relevance_score", 0)
                if score > 0:
                    relevance_scores.append(score)

            if relevance_scores:
                avg_score = sum(relevance_scores) / len(relevance_scores)
                for i, score in enumerate(relevance_scores):
                    if score < avg_score * 0.3 and avg_score > 0.5:
                        issues.append(f"Кандидат {i+1}: несоответствие оценки релевантности ({score}) среднему ({avg_score:.2f})")

        return issues

    def _calculate_completeness_score(self, results: Dict) -> int:
        """Расчет оценки полноты результатов"""
        score = 10

        required_fields = ["original_query"]

        for field in required_fields:
            if field not in results:
                score -= 2

        if "structured_report" not in results and "ranked_results" not in results:
            score -= 3

        return max(0, score)

    async def _make_decision_with_llm(self, quality_analysis: Dict) -> Dict[str, Any]:
        """Принятие решения о качестве с помощью GigaChat"""
        system_prompt = """Ты - CriticAgent, принимающий решение о качестве анализа.
На основе анализа качества прими решение:
1. Принять (approve) - если качество высокое
2. Принять с замечаниями (approve_with_notes) - если есть мелкие проблемы
3. Требует улучшения (needs_improvement) - если есть существенные проблемы
4. Отклонить (reject) - если качество неприемлемо

Укажи причину решения и рекомендуемые действия.
Верни ответ в формате JSON с полями: verdict, reason, action, quality_score."""

        prompt = f"""Анализ качества:
- Всего проблем: {quality_analysis.get('total_issues', 0)}
- Оценка полноты: {quality_analysis.get('completeness_score', 0)}/10
- Оценка логики: {quality_analysis.get('logic_score', 0)}/10
- Проблемы: {quality_analysis.get('issues_found', [])[:3]}
- Достоинства: {quality_analysis.get('strengths_found', [])[:3]}

Прими решение о качестве."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ CriticAgent: Ошибка GigaChat принятия решения: {e}")

        return await self._emulate_decision_making(quality_analysis)

    async def _emulate_decision_making(self, quality_analysis: Dict) -> Dict[str, Any]:
        """Эмуляция принятия решения"""
        total_issues = quality_analysis.get("total_issues", 0)
        completeness_score = quality_analysis.get("completeness_score", 0)
        logic_score = quality_analysis.get("logic_score", 0)

        if total_issues == 0 and completeness_score >= 8 and logic_score >= 8:
            verdict = "approve"
            reason = "Высокое качество, все критерии выполнены"
            action = "Передать RecommendationAgent для формирования ответа"
        elif total_issues <= 2 and completeness_score >= 6:
            verdict = "approve_with_notes"
            reason = "Приемлемое качество, есть незначительные замечания"
            action = "Передать RecommendationAgent с пометкой о замечаниях"
        elif total_issues <= 5:
            verdict = "needs_improvement"
            reason = f"Требуется улучшение: {total_issues} замечаний"
            action = "Запросить повторный анализ у AnalysisAgent"
        else:
            verdict = "reject"
            reason = f"Низкое качество: {total_issues} существенных замечаний"
            action = "Запросить полный переанализ"

        return {
            "verdict": verdict,
            "reason": reason,
            "action": action,
            "quality_score": (completeness_score + logic_score) / 2,
            "thresholds_used": {
                "approve_threshold": 8,
                "improvement_threshold": 6,
                "reject_threshold": 4
            }
        }

    def _create_detailed_feedback(self, quality_analysis: Dict, decision: Dict) -> Dict[str, Any]:
        """Создание детализированной обратной связи"""
        issues = quality_analysis.get("issues_found", [])
        strengths = quality_analysis.get("strengths_found", [])

        specific_corrections = []

        for issue in issues:
            if "релевантность" in issue.lower():
                specific_corrections.append("Улучшить анализ релевантности")
            elif "сложност" in issue.lower():
                specific_corrections.append("Уточнить оценку уровня сложности")
            elif "кандидат" in issue.lower() and "мало" in issue.lower():
                specific_corrections.append("Найти больше кандидатов для анализа")
            elif "отчет" in issue.lower():
                specific_corrections.append("Улучшить структурирование отчета")
            else:
                specific_corrections.append("Исправить общие проблемы качества")

        specific_corrections = list(set(specific_corrections))

        return {
            "decision_summary": decision.get("reason", ""),
            "issues_detailed": issues,
            "strengths_acknowledged": strengths,
            "specific_corrections": specific_corrections,
            "improvement_suggestions": [
                "Использовать более точные метрики оценки",
                "Добавить сравнение с альтернативами",
                "Указать ограничения анализа"
            ],
            "feedback_timestamp": "now"
        }

    async def evaluate_final_recommendation(self, message: AgentMessage) -> AgentMessage:
        """Оценка финальных рекомендаций перед отправкой пользователю"""
        print(f"  CriticAgent оценивает финальные рекомендации")

        recommendations = message.content
        original_query = recommendations.get("original_query", "")

        final_check = await self._check_final_recommendations_with_llm(recommendations, original_query)

        approval_decision = await self._final_approval_decision_with_llm(final_check)

        return AgentMessage(
            sender=self.name,
            recipient="CoordinatorAgent",
            content={
                "final_recommendations_reviewed": True,
                "recommendations_quality": final_check,
                "approval_decision": approval_decision,
                "ready_for_user": approval_decision.get("approved", False),
                "final_feedback": "Рекомендации проверены и готовы к отправке" if approval_decision.get("approved") else "Требуются исправления"
            },
            conversation_id=message.conversation_id
        )

    async def _check_final_recommendations_with_llm(self, recommendations: Dict, query: str) -> Dict[str, Any]:
        """Проверка финальных рекомендаций с помощью GigaChat"""
        system_prompt = """Ты - CriticAgent, проверяющий финальные рекомендации.
Проверь рекомендации на:
1. Наличие объяснения выбора
2. Соответствие исходному запросу
3. Полноту информации
4. Наличие альтернативных вариантов
5. Четкость и понятность изложения

Верни ответ в формате JSON."""

        prompt = f"""Проверь финальные рекомендации:

Запрос: {query}
Рекомендации: {json.dumps(recommendations, ensure_ascii=False, indent=2)[:1000]}...

Проверь качество рекомендаций."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ CriticAgent: Ошибка GigaChat проверки: {e}")

        return self._check_final_recommendations_simple(recommendations, query)

    def _check_final_recommendations_simple(self, recommendations: Dict, query: str) -> Dict[str, Any]:
        """Простая проверка финальных рекомендаций"""
        has_explanation = "explanation" in recommendations or "reasoning" in recommendations
        has_alternatives = "alternatives" in recommendations or "comparison" in recommendations
        matches_query = query.lower() in str(recommendations).lower()

        return {
            "has_explanation": has_explanation,
            "has_alternatives": has_alternatives,
            "matches_query": matches_query,
            "completeness": "high" if has_explanation and has_alternatives else "medium",
            "clarity": "clear" if len(str(recommendations)) < 1000 else "too_detailed"
        }

    async def _final_approval_decision_with_llm(self, final_check: Dict) -> Dict[str, Any]:
        """Финальное решение об одобрении с помощью GigaChat"""
        system_prompt = """Ты - CriticAgent, принимающий финальное решение об отправке рекомендаций пользователю.
Прими решение: одобрить или не одобрить рекомендации.
Укажи причину и уверенность в решении.
Верни ответ в формате JSON."""

        prompt = f"""Проверка рекомендаций:
- Есть объяснение: {final_check.get('has_explanation', False)}
- Есть альтернативы: {final_check.get('has_alternatives', False)}
- Соответствует запросу: {final_check.get('matches_query', False)}
- Полнота: {final_check.get('completeness', 'medium')}
- Четкость: {final_check.get('clarity', 'clear')}

Прими решение об одобрении."""

        try:
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                return_json=True
            )

            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ CriticAgent: Ошибка GigaChat принятия решения: {e}")

        return self._final_approval_decision_simple(final_check)

    def _final_approval_decision_simple(self, final_check: Dict) -> Dict[str, Any]:
        """Простое финальное решение"""
        if final_check.get("has_explanation") and final_check.get("matches_query"):
            return {
                "approved": True,
                "reason": "Рекомендации имеют объяснение и соответствуют запросу",
                "confidence": "high"
            }
        else:
            return {
                "approved": False,
                "reason": "Недостаточно объяснений или несоответствие запросу",
                "confidence": "medium",
                "required_improvements": ["Добавить объяснение выбора", "Уточнить соответствие запросу"]
            }