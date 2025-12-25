import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, Optional, List
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriticAgent:
    """
    Агент-критик - контролирует качество и корректность решений.
    """

    def __init__(self, gigachat_client):
        self.client = gigachat_client
        logger.info("CriticAgent инициализирован")

    def process(self, query: str = None, analysis_results: Dict = None,
                original_query: str = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Универсальный метод обработки для критика.
        """
        logger.info("CriticAgent: проверка качества результатов")

        # Получаем данные из разных источников
        actual_results = analysis_results
        actual_query = original_query or query

        # Если analysis_results нет в параметрах, ищем в context
        if actual_results is None and context:
            if 'analysis_results' in context:
                actual_results = context.get('analysis_results')
                logger.info("Найдены результаты анализа в контексте")
            elif 'previous_results' in context:
                # Ищем в предыдущих результатах
                prev_results = context.get('previous_results', {})
                execution_steps = prev_results.get('execution_steps', [])
                for step in execution_steps:
                    if step.get('agent') == 'AnalysisAgent':
                        actual_results = step.get('result', {})
                        logger.info("Найдены результаты анализа в предыдущих шагах")
                        break

        if actual_query is None and context:
            actual_query = context.get('original_query') or context.get('query')

        if not actual_results or not actual_query:
            logger.warning("Недостаточно данных для проверки")
            return {
                "status": "error",
                "error": "Недостаточно данных для проверки",
                "missing": {
                    "results": actual_results is None,
                    "query": actual_query is None
                }
            }

        try:
            # Вызываем существующую логику
            critique = self._perform_critique(actual_results, actual_query, context)

            # Принятие решения
            decision = self._make_decision(critique, actual_results)

            # Формирование ответа
            response = self._create_response(
                original_query=actual_query,
                analysis_results=actual_results,
                critique=critique,
                decision=decision,
                context=context
            )

            return response

        except Exception as e:
            logger.error(f"Ошибка в CriticAgent.process: {str(e)}")
            return self._create_error_response(actual_query, str(e))

    # Добавьте вспомогательный метод
    def _perform_critique_and_decision(self, analysis_results: Dict, original_query: str,
                                       context: Optional[Dict]) -> Dict[str, Any]:
        """
        Выполнение критического анализа и принятие решения.
        """
        # Полная проверка с помощью GigaChat
        critique = self._perform_critique(analysis_results, original_query, context)

        # Принятие решения
        decision = self._make_decision(critique, analysis_results)

        # Формирование ответа
        response = self._create_response(
            original_query=original_query,
            analysis_results=analysis_results,
            critique=critique,
            decision=decision,
            context=context
        )

        return response
    
    def _perform_critique(self, results: Dict, query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Выполнение критического анализа с помощью GigaChat.
        """
        system_prompt = """
        Ты - строгий критик и эксперт по контролю качества.
        
        Твои задачи:
        1. ПРОВЕРИТЬ РЕЛЕВАНТНОСТЬ: насколько результаты соответствуют запросу
        2. ОЦЕНИТЬ ДОСТАТОЧНОСТЬ: хватает ли информации для ответа
        3. ВЫЯВИТЬ ОШИБКИ: логические несоответствия, пробелы, неточности
        4. ПРОВЕРИТЬ ОБОСНОВАННОСТЬ: есть ли доказательства и обоснования
        5. ОЦЕНИТЬ ПОЛНОТУ: охвачены ли все аспекты запроса
        
        Будь строгим, но конструктивным. Указывай конкретные проблемы.
        """
        
        user_prompt = f"""
        ОРИГИНАЛЬНЫЙ ЗАПРОС: "{query}"
        
        РЕЗУЛЬТАТЫ АНАЛИЗА ДЛЯ ПРОВЕРКИ:
        {json.dumps(results, ensure_ascii=False, indent=2)}
        
        ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ: {context if context else "не предоставлен"}
        
        Проведи полную критическую проверку.
        
        Формат ответа JSON:
        {{
            "relevance_score": 0-10,
            "relevance_issues": ["проблема1", "проблема2"],
            "sufficiency_score": 0-10,
            "sufficiency_issues": ["чего не хватает"],
            "errors_found": [
                {{
                    "type": "тип ошибки",
                    "description": "описание",
                    "location": "где найдена",
                    "severity": "низкая/средняя/высокая"
                }}
            ],
            "completeness_score": 0-10,
            "missing_aspects": ["аспект1", "аспект2"],
            "justification_evaluation": {
                "score": 0-10,
                "strengths": ["сильная сторона1"],
                "weaknesses": ["слабая сторона1"]
            },
            "overall_assessment": "текстовая оценка",
            "specific_feedback": "конкретные замечания для исправления"
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=3000
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка в CriticAgent.process: {str(e)}")
            raise
    
    def _make_decision(self, critique: Dict, results: Dict) -> Dict[str, Any]:
        """
        Принятие решения на основе критического анализа.
        Решение принимает LLM.
        """
        system_prompt = """
        На основе критического анализа прими решение:
        - ПРИНЯТЬ: результаты достаточны и корректны
        - ОТКЛОНИТЬ: результаты непригодны для использования
        - ЗАПРОСИТЬ_ПОВТОР: нужен повторный анализ с учетом замечаний
        
        Критерии решения:
        1. Если relevance_score < 5 -> ОТКЛОНИТЬ
        2. Если есть ошибки высокой серьезности -> ЗАПРОСИТЬ_ПОВТОР
        3. Если sufficiency_score < 6 -> ЗАПРОСИТЬ_ПОВТОР
        4. В остальных случаях -> ПРИНЯТЬ
        
        Обязательно обоснуй решение.
        """
        
        user_prompt = f"""
        КРИТИЧЕСКИЙ АНАЛИЗ:
        {json.dumps(critique, ensure_ascii=False, indent=2)}
        
        ПРИМИ РЕШЕНИЕ о дальнейших действиях.
        
        Формат ответа JSON:
        {{
            "decision": "ACCEPT|REJECT|REQUEST_REANALYSIS",
            "confidence": 0.0-1.0,
            "reasoning": "подробное обоснование",
            "required_corrections": ["что именно исправить"],
            "priority": "низкий/средний/высокий",
            "estimated_improvement": "как улучшится результат"
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1000
            )
            return response
        except Exception as e:
            return {
                "decision": "REQUEST_REANALYSIS",
                "confidence": 0.5,
                "reasoning": f"Ошибка при принятии решения: {e}",
                "required_corrections": ["Проверить корректность анализа"],
                "priority": "высокий",
                "estimated_improvement": "Исправление ошибок системы"
            }
    
    def _create_response(self, original_query: str, analysis_results: Dict, 
                        critique: Dict, decision: Dict, context: Optional[Dict]) -> Dict[str, Any]:
        """Формирование финального ответа критика"""
        
        # Генерация объяснения для пользователя
        explanation = self._generate_user_explanation(critique, decision)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "agent": "CriticAgent",
            "original_query": original_query,
            "critique_results": critique,
            "decision": decision,
            "user_explanation": explanation,
            "impact_on_flow": self._assess_impact(decision),
            "next_steps_recommendation": self._recommend_next_steps(decision),
            "context_used": context,
            "gigachat_used": True
        }
    
    def _generate_user_explanation(self, critique: Dict, decision: Dict) -> str:
        """Генерация понятного объяснения для пользователя"""
        system_prompt = """
        Объясни пользователю результаты проверки качества простым языком.
        Будь честным, но тактичным.
        """
        
        user_prompt = f"""
        РЕЗУЛЬТАТЫ ПРОВЕРКИ КАЧЕСТВА:
        {json.dumps(critique, ensure_ascii=False, indent=2)}
        
        ПРИНЯТОЕ РЕШЕНИЕ: {decision.get('decision')}
        
        Объясни:
        1. Как прошла проверка
        2. Какие проблемы найдены (если есть)
        3. Что это значит для результата
        4. Что будет дальше
        
        Используй простой, понятный язык.
        """
        
        try:
            response = self.client.chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=800
            )
            return response
        except Exception as e:
            return f"Проверка качества завершена. Решение: {decision.get('decision')}. Подробности временно недоступны."
    
    def _assess_impact(self, decision: Dict) -> Dict[str, Any]:
        """Оценка влияния решения на ход выполнения"""
        decision_type = decision.get("decision", "REQUEST_REANALYSIS")
        
        impacts = {
            "ACCEPT": {
                "flow_change": "Продолжить выполнение",
                "time_impact": "нет",
                "quality_impact": "результаты приняты",
                "risk_level": "низкий"
            },
            "REJECT": {
                "flow_change": "Начать заново",
                "time_impact": "значительный",
                "quality_impact": "результаты отклонены",
                "risk_level": "высокий"
            },
            "REQUEST_REANALYSIS": {
                "flow_change": "Повторить анализ",
                "time_impact": "умеренный",
                "quality_impact": "требуются исправления",
                "risk_level": "средний"
            }
        }
        
        return impacts.get(decision_type, impacts["REQUEST_REANALYSIS"])
    
    def _recommend_next_steps(self, decision: Dict) -> List[str]:
        """Рекомендации следующих шагов"""
        decision_type = decision.get("decision", "REQUEST_REANALYSIS")
        
        if decision_type == "ACCEPT":
            return [
                "Передать результаты RecommendationAgent",
                "Сформировать финальный ответ"
            ]
        elif decision_type == "REJECT":
            return [
                "Уведомить CoordinatorAgent о необходимости нового плана",
                "Проанализировать причины неудачи"
            ]
        else:  # REQUEST_REANALYSIS
            return [
                "Вернуть результаты AnalysisAgent с замечаниями",
                "Указать конкретные аспекты для исправления",
                "Запросить повторный анализ с учетом фидбека"
            ]
    
    def _create_error_response(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Создание ответа об ошибке"""
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "agent": "CriticAgent",
            "original_query": query,
            "error": error_msg,
            "decision": {
                "decision": "REQUEST_REANALYSIS",
                "reasoning": f"Ошибка в CriticAgent: {error_msg}"
            },
            "user_explanation": f"При проверке качества возникла ошибка. Система запросит повторный анализ.",
            "gigachat_used": True
        }


# Пример использования
if __name__ == "__main__":
    from gigachat_client import GigaChatClient
    
    # Инициализация клиента
    client = GigaChatClient(verify_ssl=False)
    
    # Создание агента
    critic = CriticAgent(client)
    
    # Тестовые данные
    test_analysis = {
        "relevance_score": 7,
        "summary": "Анализ машинного обучения для студентов",
        "key_points": ["Обучение с учителем", "Нейронные сети"],
        "limitations": ["Не рассмотрены продвинутые темы"]
    }
    
    # Проверка
    result = critic.process(
        analysis_results=test_analysis,
        original_query="машинное обучение для начинающих"
    )
    
    print(f"Решение критика: {result['decision']['decision']}")
    print(f"Объяснение: {result['user_explanation']}")