import os
import json
import base64
import uuid
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import warnings
import urllib3

# Отключаем предупреждения о небезопасных SSL соединениях
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()


class GigaChatClient:
    """Улучшенный клиент для работы с GigaChat API"""

    def __init__(self, verify_ssl: bool = False):
        self.client_id = os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
        self.verify_ssl = verify_ssl

        if not self.client_id or not self.client_secret:
            raise ValueError("Установите GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET в .env файле")

        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        self.access_token = None
        self._authenticate()

    def _authenticate(self):
        """Аутентификация и получение токена"""
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {"scope": "GIGACHAT_API_PERS"}

        try:
            response = requests.post(
                self.auth_url,
                headers=headers,
                data=data,
                verify=self.verify_ssl,
                timeout=10
            )

            if response.status_code == 200:
                self.access_token = response.json()["access_token"]
                print("✅ Аутентификация успешна")
            else:
                raise Exception(f"Ошибка аутентификации: {response.text}")

        except Exception as e:
            print(f"⚠️ Предупреждение: {e}")
            response = requests.post(
                self.auth_url,
                headers=headers,
                data=data,
                verify=False,
                timeout=10
            )
            if response.status_code == 200:
                self.access_token = response.json()["access_token"]
                print("✅ Аутентификация успешна (с отключенной проверкой SSL)")
            else:
                raise Exception(f"Ошибка аутентификации: {response.text}")

    def chat_json(self, prompt: str, system_prompt: str = None, temperature: float = 0.3, max_tokens: int = 2000) -> \
    Dict[str, Any]:
        """Отправка запроса к GigaChat с ожиданием JSON-ответа"""
        if not self.access_token:
            self._authenticate()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Добавляем требование возвращать JSON
        json_prompt = f"{prompt}\n\nОтвет предоставь в формате JSON."
        messages.append({"role": "user", "content": json_prompt})

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "GigaChat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}  # Запрашиваем JSON-ответ
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=30
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Пытаемся распарсить JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Если не получилось распарсить, возвращаем как текст
                    return {"raw_response": content}
            else:
                raise Exception(f"Ошибка API ({response.status_code}): {response.text}")

        except Exception as e:
            print(f"⚠️ Ошибка при запросе: {e}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=30
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"raw_response": content}
            else:
                raise Exception(f"Ошибка API ({response.status_code}): {response.text}")

    def chat(self, prompt: str, system_prompt: str = None, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """Отправка запроса к GigaChat (текстовый ответ)"""
        result = self.chat_json(prompt, system_prompt, temperature, max_tokens)
        if isinstance(result, dict) and "raw_response" in result:
            return result["raw_response"]
        elif isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            return str(result)