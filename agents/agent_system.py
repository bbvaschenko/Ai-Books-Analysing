"""
Система управления мульти-агентной архитектурой
"""
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from coordinator_agent import CoordinatorAgent
from search_agent import SearchAgent
from analysis_agent import AnalysisAgent
from critic_agent import CriticAgent
from recommendation_agent import RecommendationAgent
from base_agent import AgentMessage


class AgentSystem:
    """Система управления взаимодействием агентов"""

    def __init__(self, library_core):
        print(" Инициализация системы агентов...")

        self.library_core = library_core

        # Инициализация всех агентов
        self.agents = {
            "CoordinatorAgent": CoordinatorAgent(),
            "SearchAgent": SearchAgent(library_core),
            "AnalysisAgent": AnalysisAgent(library_core),
            "CriticAgent": CriticAgent(),
            "RecommendationAgent": RecommendationAgent()
        }

        self.conversations: Dict[str, Dict] = {}  # conversation_id -> conversation_data
        self.message_log: List[Dict] = []

        print(f"✅ Система агентов инициализирована: {len(self.agents)} агентов готовы к работе")

    async def process_query(self, user_query: str, context: Dict = None) -> Dict[str, Any]:
        """Обработка пользовательского запроса через систему агентов"""
        conversation_id = str(uuid.uuid4())

        print(f"\n{'=' * 60}")
        print(f"   Обработка запроса через систему агентов")
        print(f"   ID разговора: {conversation_id}")
        print(f"   Запрос: '{user_query}'")
        print(f"{'=' * 60}")

        # Инициализируем разговор
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "query": user_query,
            "context": context or {},
            "start_time": datetime.now().isoformat(),
            "messages": [],
            "current_agent": "CoordinatorAgent",
            "status": "in_progress"
        }

        # Начинаем с CoordinatorAgent
        coordinator = self.agents["CoordinatorAgent"]

        initial_message = AgentMessage(
            sender="User",
            recipient="CoordinatorAgent",
            content={
                "query": user_query,
                "context": context or {},
                "conversation_id": conversation_id
            },
            conversation_id=conversation_id,
            timestamp=datetime.now().timestamp()
        )

        # Логируем начало
        self._log_message(initial_message, conversation_id)

        # Запускаем обработку
        try:
            result = await self._execute_agent_workflow(
                coordinator,
                initial_message,
                conversation_id
            )

            # Завершаем разговор
            self.conversations[conversation_id]["status"] = "completed"
            self.conversations[conversation_id]["end_time"] = datetime.now().isoformat()
            self.conversations[conversation_id]["result"] = result

            print(f"\n✅ Запрос обработан успешно")
            print(f"   Сообщений в логе: {len(self.conversations[conversation_id]['messages'])}")
            print(f"   Статус: {self.conversations[conversation_id]['status']}")

            return result

        except Exception as e:
            print(f"\n❌ Ошибка при обработке запроса: {e}")
            self.conversations[conversation_id]["status"] = "error"
            self.conversations[conversation_id]["error"] = str(e)

            return {
                "error": str(e),
                "conversation_id": conversation_id,
                "status": "error"
            }

    async def _execute_agent_workflow(self, start_agent, start_message: AgentMessage,
                                      conversation_id: str) -> Dict[str, Any]:
        """Выполнение workflow агентов"""
        current_message = start_message
        current_agent_name = start_message.recipient

        max_steps = 20  # Защита от бесконечного цикла
        step = 0

        while step < max_steps:
            step += 1
            print(f"\n Шаг {step}: {current_agent_name}")

            # Получаем текущего агента
            current_agent = self.agents.get(current_agent_name)
            if not current_agent:
                raise ValueError(f"Агент {current_agent_name} не найден")

            # Обрабатываем сообщение агентом
            response = await current_agent.process(current_message)

            # Логируем ответ
            self._log_message(response, conversation_id)

            # Обновляем историю разговора
            self.conversations[conversation_id]["messages"].append({
                "step": step,
                "from": response.sender,
                "to": response.recipient,
                "type": response.message_type,
                "timestamp": datetime.now().isoformat()
            })

            # Проверяем, завершен ли workflow
            if response.recipient == "User":
                print(f" Достигнут конечный пользователь")
                return response.content

            # Обрабатываем специальные случаи
            if current_agent_name == "CoordinatorAgent" and response.recipient != "User":
                # Coordinator решает, что делать дальше
                pass

            if current_agent_name == "CriticAgent":
                # Critic всегда отправляет Coordinator
                response.recipient = "CoordinatorAgent"

            # Переходим к следующему агенту
            current_message = response
            current_agent_name = response.recipient

            # Небольшая задержка для эмуляции обработки
            await asyncio.sleep(0.1)

        raise RuntimeError(f"Превышено максимальное количество шагов ({max_steps})")

    def _log_message(self, message: AgentMessage, conversation_id: str):
        """Логирование сообщения"""
        log_entry = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "sender": message.sender,
            "recipient": message.recipient,
            "message_type": message.message_type,
            "content_keys": list(message.content.keys()) if message.content else []
        }

        self.message_log.append(log_entry)

        # Выводим в консоль для отладки
        print(f"   {message.sender} → {message.recipient}: {message.message_type}")

    def get_conversation_log(self, conversation_id: str) -> List[Dict]:
        """Получение лога разговора"""
        return [
            log for log in self.message_log
            if log.get("conversation_id") == conversation_id
        ]

    def get_agent_status(self) -> Dict[str, Any]:
        """Получение статуса всех агентов"""
        status = {}

        for name, agent in self.agents.items():
            status[name] = {
                "message_count": len(agent.history),
                "last_active": "now",
                "status": "active"
            }

        return status

    def clear_conversations(self):
        """Очистка всех разговоров"""
        self.conversations.clear()
        self.message_log.clear()

        for agent in self.agents.values():
            agent.clear_history()

        print("🧹 Все разговоры очищены")


# Синхронная обертка для интеграции с существующим кодом
class SyncAgentSystem:
    """Синхронная обертка для системы агентов"""

    def __init__(self, library_core):
        self.async_system = AgentSystem(library_core)

    def process_query(self, user_query: str, context: Dict = None) -> Dict[str, Any]:
        """Синхронная обработка запроса"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.async_system.process_query(user_query, context)
            )
            return result
        finally:
            loop.close()