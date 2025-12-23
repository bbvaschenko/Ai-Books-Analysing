"""
CoordinatorAgent - центральный управляющий агент
"""
import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentMessage


class CoordinatorAgent(BaseAgent):
    """Агент-координатор, управляет выполнением плана"""

    def __init__(self):
        super().__init__(
            name="CoordinatorAgent",
            role="Центральный управляющий агент и носитель стратегии",
            description="""Я интерпретирую пользовательский запрос на уровне цели,
формирую план выполнения, принимаю решения о вызове агентов,
обрабатываю обратную связь от CriticAgent и формирую финальный ответ."""
        )

        # План выполнения
        self.execution_plan: List[Dict] = []
        self.current_step = 0

    async def process(self, message: AgentMessage, context: Dict = None) -> AgentMessage:
        """Обработка запроса и формирование плана"""
        self.add_to_history(message)

        user_query = message.content.get("query", "")
        user_context = message.content.get("context", {})

        # Используем LLM для анализа запроса и формирования плана
        analysis_result = await self._analyze_query_with_llm(user_query, user_context)

        # Формируем план выполнения на основе анализа
        self.execution_plan = self._create_execution_plan(analysis_result)
        self.current_step = 0

        # Логируем решение
        print(f"  CoordinatorAgent создал план: {json.dumps(self.execution_plan, indent=2, ensure_ascii=False)}")

        # Создаем сообщение для следующего агента
        next_agent = self.execution_plan[0]["agent"]
        next_task = self.execution_plan[0]["task"]

        return AgentMessage(
            sender=self.name,
            recipient=next_agent,
            content={
                "query": user_query,
                "task": next_task,
                "context": user_context,
                "plan_step": 0,
                "execution_plan": self.execution_plan,
                "analysis": analysis_result
            },
            conversation_id=message.conversation_id
        )

    # coordinator_agent.py (обновление метода _analyze_query_with_llm)
    async def _analyze_query_with_llm(self, query: str, context: Dict) -> Dict[str, Any]:
        """Анализ запроса с использованием LLM (GigaChat)"""
        from config.agent_config import GIGACHAT_PROMPTS

        prompt_template = GIGACHAT_PROMPTS["CoordinatorAgent"]["query_analysis"]
        prompt = prompt_template.format(query=query, context=json.dumps(context, ensure_ascii=False))

        try:
            # Используем LLM через базовый класс
            result = await self.call_llm(
                prompt=prompt,
                system_prompt=GIGACHAT_PROMPTS["CoordinatorAgent"]["system"],
                return_json=True
            )

            # Если LLM вернул результат, используем его
            if isinstance(result, dict) and "error" not in result:
                return result

        except Exception as e:
            print(f"❌ CoordinatorAgent: Ошибка LLM анализа: {e}")

        # Fallback на эмуляцию если LLM не сработал
        return await self._emulate_llm_analysis(query, context)

    async def _emulate_llm_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Эмуляция анализа запроса (старая логика)"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["найти", "поиск", "искать"]):
            return {
                "query_type": "search",
                "priority": "find_materials",
                "complexity": "medium",
                "suggested_agents": ["SearchAgent", "AnalysisAgent", "CriticAgent", "RecommendationAgent"],
                "reasoning": "Пользователь хочет найти учебные материалы, требуется поиск и анализ"
            }
        # ... остальная эмуляционная логика
    def _create_execution_plan(self, analysis: Dict) -> List[Dict]:
        """Создание плана выполнения на основе анализа"""
        plan = []

        # Динамически определяем план на основе анализа
        if analysis["query_type"] == "search":
            plan = [
                {
                    "agent": "SearchAgent",
                    "task": "Найти релевантные учебные материалы по запросу",
                    "expected_output": "Ранжированный список кандидатов с обоснованием"
                },
                {
                    "agent": "AnalysisAgent",
                    "task": "Проанализировать найденные материалы",
                    "expected_output": "Структурированный анализ каждого кандидата"
                },
                {
                    "agent": "CriticAgent",
                    "task": "Проверить качество анализа",
                    "expected_output": "Оценка качества с замечаниями"
                },
                {
                    "agent": "RecommendationAgent",
                    "task": "Сформировать финальные рекомендации",
                    "expected_output": "Объяснимые рекомендации с альтернативами"
                }
            ]
        elif analysis["query_type"] == "analysis":
            plan = [
                {
                    "agent": "AnalysisAgent",
                    "task": "Глубокий анализ предоставленного контента",
                    "expected_output": "Детальный отчет с оценками"
                },
                {
                    "agent": "CriticAgent",
                    "task": "Критическая проверка анализа",
                    "expected_output": "Замечания и рекомендации по улучшению"
                },
                {
                    "agent": "RecommendationAgent",
                    "task": "Формирование выводов",
                    "expected_output": "Итоговые выводы и рекомендации"
                }
            ]

        return plan

    async def process_agent_response(self, message: AgentMessage) -> AgentMessage:
        """Обработка ответа от другого агента"""
        self.add_to_history(message)

        # Проверяем, нужно ли вызывать следующего агента
        if self.current_step < len(self.execution_plan) - 1:
            # Переходим к следующему шагу
            self.current_step += 1
            next_step = self.execution_plan[self.current_step]

            return AgentMessage(
                sender=self.name,
                recipient=next_step["agent"],
                content={
                    "previous_results": message.content,
                    "task": next_step["task"],
                    "plan_step": self.current_step,
                    "original_query": self.history[0].content.get("query", "")
                },
                conversation_id=message.conversation_id
            )
        else:
            # План завершен, возвращаем финальный результат
            return AgentMessage(
                sender=self.name,
                recipient="User",
                content={
                    "final_result": message.content,
                    "execution_summary": {
                        "total_steps": len(self.execution_plan),
                        "completed_steps": self.current_step + 1,
                        "plan": self.execution_plan
                    }
                },
                conversation_id=message.conversation_id
            )

    async def process_critic_feedback(self, feedback: AgentMessage) -> AgentMessage:
        """Обработка обратной связи от CriticAgent"""
        print(f"  CoordinatorAgent получил обратную связь от CriticAgent")

        # Анализируем обратную связь с помощью LLM
        needs_rerun = await self._evaluate_feedback_with_llm(feedback)

        if needs_rerun:
            # Принимаем решение о повторном выполнении
            rerun_decision = await self._decide_rerun_with_llm(feedback)

            print(f"   CoordinatorAgent решил: {rerun_decision['action']}")
            print(f"   Причина: {rerun_decision['reason']}")

            # Формируем сообщение для повторного выполнения
            target_agent = rerun_decision.get("target_agent", "AnalysisAgent")

            return AgentMessage(
                sender=self.name,
                recipient=target_agent,
                content={
                    "task": "Повторить анализ с учетом замечаний",
                    "feedback": feedback.content,
                    "corrections_needed": rerun_decision.get("corrections", []),
                    "original_results": feedback.content.get("original_results", {})
                },
                conversation_id=feedback.conversation_id
            )

        # Если повтор не требуется, продолжаем план
        return await self.process_agent_response(feedback)

    async def _evaluate_feedback_with_llm(self, feedback: AgentMessage) -> bool:
        """Оценка необходимости повторного выполнения"""
        # Эмуляция LLM анализа
        feedback_content = feedback.content

        if feedback_content.get("needs_correction", False):
            issues = feedback_content.get("issues", [])
            if issues and len(issues) > 0:
                # Если есть серьезные проблемы, требуется повтор
                return True

        return False

    async def _decide_rerun_with_llm(self, feedback: AgentMessage) -> Dict[str, Any]:
        """Принятие решения о повторном выполнении"""
        # Эмуляция LLM принятия решения
        feedback_content = feedback.content

        return {
            "action": "rerun_analysis",
            "target_agent": "AnalysisAgent",
            "reason": "CriticAgent выявил существенные проблемы в анализе",
            "corrections": feedback_content.get("corrections_needed", ["Улучшить анализ релевантности"])
        }