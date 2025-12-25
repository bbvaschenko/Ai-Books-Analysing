
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serialize_for_json(obj: Any) -> Any:
    """
    Рекурсивная сериализация объектов для JSON.
    Обрабатывает циклические ссылки и сложные типы.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Для объектов с __dict__
        return serialize_for_json(obj.__dict__)
    else:
        # Для всего остального - строковое представление
        return str(obj)


class SafeJSONEncoder(json.JSONEncoder):
    """Безопасный JSON энкодер, который обрабатывает циклические ссылки"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen = set()

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            obj_id = id(obj)
            if obj_id in self._seen:
                return f"<циклическая ссылка на {type(obj).__name__}>"
            self._seen.add(obj_id)
            result = {k: self.default(v) for k, v in obj.__dict__.items()}
            self._seen.remove(obj_id)
            return result
        else:
            return str(obj)


class CoordinatorAgent:
    """
    Центральный управляющий агент - носитель стратегии.
    Принимает все решения с помощью GigaChat.
    """

    def __init__(self, gigachat_client, verify_ssl: bool = False):
        self.client = gigachat_client
        self.agent_registry = {}
        self.conversation_history = []

    def register_agent(self, name: str, agent_instance):
        """Регистрация агента в системе"""
        self.agent_registry[name] = agent_instance
        logger.info(f"Зарегистрирован агент: {name}")

    def process_query(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обрабатывает пользовательский запрос, координируя работу всех агентов.
        Все решения принимаются LLM.
        """
        logger.info(f"CoordinatorAgent: обработка запроса - {user_query}")

        try:
            # Начало плана выполнения
            execution_plan = self._create_execution_plan(user_query, context)

            results = {
                "query": user_query,
                "timestamp": datetime.now().isoformat(),
                "execution_steps": [],
                "final_result": None,
                "plan": execution_plan
            }

            # Выполнение плана
            current_step = 0
            max_steps = len(execution_plan.get("steps", [])) + 5  # Защита от бесконечного цикла

            while current_step < len(execution_plan["steps"]) and current_step < max_steps:
                step = execution_plan["steps"][current_step]

                logger.info(f"Шаг {current_step + 1}/{len(execution_plan['steps'])}: {step['action']}")

                # Выполнение шага
                step_result = self._execute_step(step, user_query, context, results)

                # Сериализуем результат для хранения в истории
                serialized_result = serialize_for_json(step_result)

                results["execution_steps"].append({
                    "step": current_step + 1,
                    "action": step["action"],
                    "agent": step["agent"],
                    "result": serialized_result
                })

                # Принятие решения о следующем шаге с помощью LLM
                # Если произошла ошибка, переходим к следующему шагу
                if "error" in step_result:
                    logger.warning(f"Ошибка на шаге {current_step}: {step_result['error']}")
                    current_step += 1
                    continue

                # Если это последний шаг, завершаем
                if current_step >= len(execution_plan["steps"]) - 1:
                    break

                # Решаем, продолжать ли
                try:
                    next_decision = self._decide_next_step(
                        user_query, results, current_step, execution_plan
                    )

                    if next_decision["decision"] == "continue":
                        current_step += 1
                    elif next_decision["decision"] == "jump":
                        target = next_decision.get("target_step", current_step + 1)
                        if 0 <= target < len(execution_plan["steps"]):
                            current_step = target
                        else:
                            current_step += 1
                    elif next_decision["decision"] == "retry":
                        # Оставляем на том же шаге, но добавляем фидбек
                        step["feedback"] = next_decision.get("feedback", "")
                        # Добавляем счетчик попыток
                        if "retry_count" not in step:
                            step["retry_count"] = 0
                        step["retry_count"] += 1

                        if step["retry_count"] > 2:  # Максимум 3 попытки
                            logger.warning(f"Превышено максимальное количество попыток для шага {current_step}")
                            current_step += 1
                    elif next_decision["decision"] == "complete":
                        break
                    elif next_decision["decision"] == "loop":
                        # Запуск нового цикла анализа
                        new_step = self._create_correction_step(
                            next_decision.get("feedback", "")
                        )
                        execution_plan["steps"].insert(current_step + 1, new_step)
                        current_step += 1

                except Exception as e:
                    logger.error(f"Ошибка принятия решения: {e}. Продолжаем выполнение.")
                    current_step += 1

                # Обновление контекста
                context = self._update_context(context, results, next_decision if 'next_decision' in locals() else {
                    "decision": "continue"})

            # Формирование финального ответа
            final_response = self._create_final_response(user_query, results, context)

            # Сериализуем финальный результат
            results["final_result"] = serialize_for_json(final_response)

            return serialize_for_json(results)

        except Exception as e:
            logger.error(f"Ошибка в process_query: {e}")
            return serialize_for_json({
                "error": str(e),
                "query": user_query,
                "timestamp": datetime.now().isoformat()
            })

    def _create_execution_plan(self, query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Создание плана выполнения с помощью GigaChat.
        LLM решает, каких агентов вызывать и в каком порядке.
        """
        system_prompt = """
        Ты - главный координатор агентной системы. Твоя задача:

        1. ПРОАНАЛИЗИРОВАТЬ ЗАПРОС пользователя
        2. РАЗРАБОТАТЬ ПЛАН ВЫПОЛНЕНИЯ
        3. ПРИНЯТЬ РЕШЕНИЯ О ПОСЛЕДОВАТЕЛЬНОСТИ АГЕНТОВ

        Доступные агенты:
        - SearchAgent: формирование стратегии поиска
        - AnalysisAgent: анализ найденного контента (только короткий анализ, не глубокий)
        - CriticAgent: контроль качества результатов
        - RecommendationAgent: формирование финальных рекомендаций

        ВАЖНО: Поиск должен осуществляться ТОЛЬКО по тегам или по резюме книг.
        Прямой глубокий анализ книги (полный анализ содержания) НЕ ДОСТУПЕН.

        В плане должен быть описан каждый шаг с указанием:
        - Какого агента вызвать
        - С какой целью
        - Какие параметры передать

        Важно: Не используй готовые шаблоны. Анализируй конкретный запрос.
        """

        user_prompt = f"""
        ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

        КОНТЕКСТ: {json.dumps(context, ensure_ascii=False) if context else "Нет"}

        Разработай оптимальный план выполнения. Учти:
        1. Сложность запроса
        2. Доступные варианты поиска (ТОЛЬКО по тегам или по резюме)
        3. Необходимость проверки качества

        ОГРАНИЧЕНИЕ: Прямой глубокий анализ книги (полное чтение и анализ содержания) НЕВОЗМОЖЕН.
        Используй только метаданные: теги и резюме книг.

        Ответ в формате JSON:
        {{
            "plan_summary": "краткое описание плана",
            "steps": [
                {{
                    "step_number": 1,
                    "agent": "имя_агента",
                    "action": "конкретное действие",
                    "parameters": {{}},
                    "expected_outcome": "чего ожидаем"
                }}
            ],
            "expected_iterations": 1,
            "risk_assessment": "оценка рисков",
            "search_method": "tags|summary"  # Укажи метод поиска: теги или резюме
        }}
        """

        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000
            )

            # Логирование принятого плана
            logger.info(f"Создан план выполнения: {response.get('plan_summary', 'без описания')}")
            logger.info(f"Выбран метод поиска: {response.get('search_method', 'tags')}")
            return response

        except Exception as e:
            logger.error(f"Ошибка создания плана: {e}")
            return self._create_default_plan(query)

    def _execute_step(self, step: Dict, query: str, context: Dict, results: Dict) -> Dict:
        """Выполнение одного шага плана"""
        agent_name = step.get("agent")

        if agent_name not in self.agent_registry:
            return {"error": f"Агент {agent_name} не найден"}

        agent = self.agent_registry[agent_name]

        try:
            # Получаем сигнатуру метода агента
            import inspect
            sig = inspect.signature(agent.process)
            expected_params = list(sig.parameters.keys())

            # Подготовка параметров
            params = {}

            # Всегда передаем query и context, если агент их ожидает
            if 'query' in expected_params:
                params['query'] = query

            if 'context' in expected_params:
                params['context'] = context if context is not None else {}

            # ДЛЯ RecommendationAgent
            if agent_name == "RecommendationAgent":
                # Находим результаты анализа из предыдущих шагов
                analysis_results = {}
                for step_result in results.get("execution_steps", []):
                    if step_result.get("agent") == "AnalysisAgent":
                        analysis_results = step_result.get("result", {})
                        break

                params['analysis_results'] = analysis_results
                params['user_context'] = context if context is not None else {}

                # Если query не входит в expected_params, но нужен, добавляем
                if 'query' in expected_params:
                    params['query'] = query

            # ДЛЯ CriticAgent
            elif agent_name == "CriticAgent":
                # Находим результаты анализа
                for step_result in results.get("execution_steps", []):
                    if step_result.get("agent") == "AnalysisAgent":
                        params['analysis_results'] = step_result.get("result", {})
                        break

                params['original_query'] = query
                params['context'] = context if context is not None else {}

            # ДЛЯ AnalysisAgent - ОГРАНИЧИВАЕМ: передаем только метаданные, а не полный текст книги
            elif agent_name == "AnalysisAgent":
                # Передаем только метаданные книги (название, теги, резюме)
                if context and isinstance(context, dict):
                    # Извлекаем только метаданные, не полный текст
                    metadata = {
                        "title": context.get('book_info', {}).get('title', ''),
                        "tags": context.get('book_info', {}).get('tags', []),
                        "summary_preview": context.get('book_info', {}).get('summary_preview', '')[:500],
                        # Только превью
                        "topic": context.get('topic', None),
                        "target_audience": context.get('target_audience', None)
                    }
                    params['content'] = json.dumps(metadata, ensure_ascii=False)
                else:
                    params['content'] = "Метаданные книги отсутствуют"

            # Дополнительные параметры из шага
            step_params = step.get("parameters", {})
            for param_name, param_value in step_params.items():
                if param_name in expected_params:
                    params[param_name] = param_value

            # Вызов агента
            result = agent.process(**params)

            # Логирование
            logger.info(f"Агент {agent_name} выполнил шаг: {step.get('action')}")

            return result

        except Exception as e:
            logger.error(f"Ошибка выполнения шага {agent_name}: {e}")
            return {"error": str(e), "step": step}

    def _decide_next_step(self, query: str, results: Dict, current_step: int,
                          plan: Dict) -> Dict[str, Any]:
        """
        Принятие решения о следующем шаге с помощью LLM.
        Все решения принимает нейросеть.
        """
        system_prompt = """
        Ты - стратег, принимающий решения о ходе выполнения.
        На основе текущих результатов решай:
        1. Продолжить выполнение плана
        2. Перейти к другому шагу
        3. Повторить текущий шаг
        4. Завершить выполнение
        5. Запустить новый цикл анализа

        ВАЖНО: Помни, что полный анализ книги (чтение всего текста) невозможен.
        Работаем только с метаданными: тегами и резюме.

        Обосновывай каждое решение.
        """

        user_prompt = f"""
        ЗАПРОС: "{query}"

        ТЕКУЩИЙ ШАГ: {current_step + 1}

        РЕЗУЛЬТАТЫ ТЕКУЩЕГО ШАГА:
        {json.dumps(results['execution_steps'][-1] if results['execution_steps'] else {},
                    ensure_ascii=False, indent=2)}

        ОБЩИЕ РЕЗУЛЬТАТЫ:
        Количество выполненных шагов: {len(results['execution_steps'])}

        ОРИГИНАЛЬНЫЙ ПЛАН:
        {json.dumps(plan, ensure_ascii=False, indent=2)}

        ПРИМИ РЕШЕНИЕ о следующем действии.
        Помни: полный анализ книги НЕВОЗМОЖЕН, только работа с метаданными.

        Формат ответа:
        {{
            "decision": "continue|jump|retry|complete|loop",
            "target_step": номер_шага (только для jump),
            "feedback": "обоснование решения",
            "reasoning": "подробное объяснение",
            "constraint_note": "напоминание о невозможности полного анализа книги"
        }}
        """

        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1000
            )

            logger.info(f"Решение принято: {response.get('decision')} - {response.get('feedback', '')}")
            return response

        except Exception as e:
            logger.error(f"Ошибка принятия решения: {e}")
            return {"decision": "continue", "feedback": "Ошибка принятия решения"}

    def _create_final_response(self, query: str, results: Dict, context: Dict) -> Dict:
        """Формирование финального ответа пользователю"""
        system_prompt = """
        Ты - финальный координатор. Собери все результаты работы агентов
        и создай связный, полезный ответ для пользователя.

        ВАЖНО: Учитывай, что полный анализ книги не проводился.
        Работа велась только с метаданными: тегами и резюме книг.
        Укажи это как ограничение в ответе.
        """

        user_prompt = f"""
        ИСХОДНЫЙ ЗАПРОС: "{query}"

        РЕЗУЛЬТАТЫ РАБОТЫ СИСТЕМЫ:
        {json.dumps(results, ensure_ascii=False, indent=2)}

        ОГРАНИЧЕНИЕ: Полный анализ книги (чтение всего текста) не проводился.
        Использовались только метаданные: теги и резюме книг.

        Создай финальный ответ, который:
        1. Отвечает на исходный запрос с учетом ограничений
        2. Обобщает найденную информацию на основе метаданных
        3. Приводит рекомендации, основанные на тегах и резюме
        4. Объясняет, как был получен результат (поиск по тегам/резюме)
        5. Честно указывает на ограничения метода

        Формат ответа:
        {{
            "answer": "основной текст ответа",
            "summary": "краткое резюме",
            "recommendations": [],
            "sources_used": ["теги", "резюме"],
            "confidence": 0.0,
            "limitations": [
                "Анализ основан только на метаданных (тегах и резюме)",
                "Полный текст книги не анализировался"
            ],
            "methodology": "Поиск по тегам и резюме книг"
        }}
        """

        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1500
            )
            return response
        except Exception as e:
            return {
                "answer": f"Ошибка формирования ответа: {str(e)}",
                "summary": "Не удалось обработать результаты",
                "recommendations": [],
                "sources_used": [],
                "confidence": 0.0,
                "limitations": ["Ошибка системы", "Анализ основан только на метаданных"],
                "methodology": "Поиск по тегам и резюме книг"
            }

    def _create_default_plan(self, query: str) -> Dict:
        """Создание плана по умолчанию"""
        return {
            "plan_summary": "Стандартный план выполнения (только метаданные)",
            "steps": [
                {
                    "step_number": 1,
                    "agent": "SearchAgent",
                    "action": "Анализ запроса и формирование стратегии поиска по метаданным",
                    "parameters": {},
                    "expected_outcome": "Стратегия поиска по тегам или резюме"
                },
                {
                    "step_number": 2,
                    "agent": "AnalysisAgent",
                    "action": "Краткий анализ на основе метаданных (тегов и резюме)",
                    "parameters": {},
                    "expected_outcome": "Структурированный анализ метаданных"
                },
                {
                    "step_number": 3,
                    "agent": "CriticAgent",
                    "action": "Проверка качества результатов",
                    "parameters": {},
                    "expected_outcome": "Оценка качества и рекомендации"
                },
                {
                    "step_number": 4,
                    "agent": "RecommendationAgent",
                    "action": "Формирование финальных рекомендаций",
                    "parameters": {},
                    "expected_outcome": "Итоговый ответ пользователю"
                }
            ],
            "expected_iterations": 1,
            "risk_assessment": "Средний - анализ только по метаданным",
            "search_method": "tags"
        }

    def _update_context(self, context: Dict, results: Dict, decision: Dict) -> Dict:
        """Обновление контекста выполнения"""
        if context is None:
            context = {}

        context.update({
            "last_decision": decision,
            "execution_steps": len(results["execution_steps"]),
            "current_state": "in_progress" if decision.get("decision") != "complete" else "completed",
            "methodology_note": "Анализ только по метаданным (теги/резюме)"
        })

        return context

    def _create_correction_step(self, feedback: str) -> Dict:
        """Создание шага коррекции на основе фидбека"""
        return {
            "step_number": 0,
            "agent": "AnalysisAgent",
            "action": f"Корректировка анализа метаданных на основе фидбека: {feedback}",
            "parameters": {
                "correction_feedback": feedback,
                "constraint_note": "Работаем только с метаданными (теги/резюме)"
            },
            "expected_outcome": "Исправленный анализ метаданных с учетом замечаний"
        }

    def get_execution_log(self) -> List[Dict]:
        """Получение лога выполнения"""
        return self.conversation_history

