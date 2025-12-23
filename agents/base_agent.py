"""
Базовый класс для всех LLM агентов с GigaChat
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import asyncio

try:
    from giga_client import SyncGigaChatClient
    GIGACHAT_AVAILABLE = True
except ImportError:
    print("⚠️  GigaChat клиент не найден. Установите зависимости.")
    GIGACHAT_AVAILABLE = False
    SyncGigaChatClient = None


@dataclass
class AgentMessage:
    """Сообщение между агентами"""
    sender: str
    recipient: str
    content: Dict[str, Any]
    message_type: str = "request"
    timestamp: float = None
    conversation_id: str = None


class BaseAgent(ABC):
    """Абстрактный базовый класс для всех агентов с GigaChat"""

    def __init__(self, name: str, role: str, description: str):
        self.name = name
        self.role = role
        self.description = description
        self.system_prompt = self._create_system_prompt()
        self.history: List[AgentMessage] = []

        self.gigachat_client = None
        self._init_gigachat()

    def _init_gigachat(self):
        """Инициализация GigaChat клиента"""
        if not GIGACHAT_AVAILABLE:
            print(f"⚠️  {self.name}: GigaChat недоступен")
            return

        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()

            client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
            auth_data = os.getenv("GIGACHAT_AUTH_DATA")

            if client_secret and auth_data:
                self.gigachat_client = SyncGigaChatClient(
                    client_secret=client_secret,
                    auth_data=auth_data
                )
                print(f"✅ {self.name}: GigaChat клиент инициализирован")
            else:
                print(f"⚠️  {self.name}: Не установлены GIGACHAT_CLIENT_SECRET или GIGACHAT_AUTH_DATA")
        except Exception as e:
            print(f"⚠️  {self.name}: Ошибка инициализации GigaChat: {e}")

    def _create_system_prompt(self) -> str:
        """Создает системный промпт для агента"""
        return f"""Ты - {self.name}, {self.role}.

{self.description}

ПРАВИЛА:
1. Всегда возвращай ответ в формате JSON
2. Объясняй свои решения
3. Будь точным и конкретным
4. Учитывай контекст разговора
5. Отвечай на русском языке
"""

    async def call_llm(self, prompt: str, system_prompt: str = None, return_json: bool = True) -> Any:
        """
        Вызов GigaChat LLM
        """
        if self.gigachat_client:
            try:
                result = self.gigachat_client.analyze_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt or self.system_prompt,
                    return_json=return_json
                )
                return result
            except Exception as e:
                print(f"❌ {self.name}: Ошибка GigaChat: {e}")
                return self._fallback_response(prompt, return_json)
        else:
            return self._fallback_response(prompt, return_json)

    def _fallback_response(self, prompt: str, return_json: bool = True) -> Any:
        """Fallback ответ при недоступности GigaChat"""
        if return_json:
            return {
                "agent": self.name,
                "status": "fallback",
                "reason": "GigaChat недоступен",
                "recommendation": "Проверьте настройки API ключа"
            }
        else:
            return f"{self.name}: GigaChat недоступен. Проверьте настройки."

    @abstractmethod
    async def process(self, message: AgentMessage, context: Dict = None) -> AgentMessage:
        """Обработка сообщения агентом"""
        pass

    def add_to_history(self, message: AgentMessage):
        """Добавление сообщения в историю"""
        self.history.append(message)

    def get_conversation_history(self, conversation_id: str = None) -> List[AgentMessage]:
        """Получение истории разговора"""
        if conversation_id:
            return [msg for msg in self.history if msg.conversation_id == conversation_id]
        return self.history

    def clear_history(self):
        """Очистка истории"""
        self.history.clear()