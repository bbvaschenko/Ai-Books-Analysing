"""
Логирование взаимодействия агентов
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class AgentLogger:
    """Логирование взаимодействия между агентами"""

    def __init__(self, log_dir: str = "logs/agents"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Настройка файлового логгера
        self.logger = logging.getLogger("AgentSystem")
        self.logger.setLevel(logging.INFO)

        # Формат логов
        formatter = logging.Formatter(
            '%(asctime)s - %(agent)s - %(levelname)s - %(message)s'
        )

        # Файловый хендлер
        log_file = self.log_dir / f"agent_system_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_agent_message(self, sender: str, recipient: str,
                          message_type: str, conversation_id: str,
                          content_keys: list):
        """Логирование сообщения между агентами"""
        self.logger.info(
            f"{sender} → {recipient}: {message_type}",
            extra={
                "agent": sender,
                "recipient": recipient,
                "conversation_id": conversation_id,
                "content_keys": content_keys
            }
        )

    def log_agent_decision(self, agent: str, decision: str,
                           reasoning: str, conversation_id: str):
        """Логирование решения агента"""
        self.logger.info(
            f"Решение: {decision}",
            extra={
                "agent": agent,
                "decision": decision,
                "reasoning": reasoning[:100],  # Ограничиваем длину
                "conversation_id": conversation_id
            }
        )

    def log_conversation_start(self, conversation_id: str, query: str):
        """Логирование начала разговора"""
        self.logger.info(
            f"Начало разговора: {query[:50]}...",
            extra={
                "agent": "System",
                "conversation_id": conversation_id,
                "query_length": len(query)
            }
        )

    def log_conversation_end(self, conversation_id: str, status: str,
                             message_count: int):
        """Логирование завершения разговора"""
        self.logger.info(
            f"Завершение разговора: {status}",
            extra={
                "agent": "System",
                "conversation_id": conversation_id,
                "status": status,
                "message_count": message_count
            }
        )

    def save_conversation_json(self, conversation_data: Dict[str, Any]):
        """Сохранение разговора в JSON файл"""
        conversation_id = conversation_data.get("id", "unknown")
        filename = self.log_dir / f"conversation_{conversation_id}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"Разговор сохранен в {filename}",
            extra={"agent": "System"}
        )