"""
Клиент для работы с GigaChat API (исправленная версия с правильной авторизацией)
"""
import os
import json
import asyncio
import uuid
import base64
from typing import Dict, Any, List, Optional
import aiohttp


class GigaChatClient:
    """Клиент для взаимодействия с GigaChat API с правильной авторизацией"""

    def __init__(self,
                 client_secret: str = None,
                 auth_data: str = None,
                 auth_url: str = None,
                 api_url: str = None):
        """
        Инициализация клиента GigaChat с правильной авторизацией

        Args:
            client_secret: Client Secret из личного кабинета
            auth_data: Данные для авторизации в формате Base64 (username:password)
            auth_url: URL для авторизации
            api_url: URL API
        """
        # Получаем данные из переменных окружения
        self.client_secret = client_secret or os.getenv("GIGACHAT_CLIENT_SECRET")
        self.auth_data = auth_data or os.getenv("GIGACHAT_AUTH_DATA")
        self.auth_url = auth_url or os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
        self.api_url = api_url or os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1")

        # Проверяем наличие обязательных данных
        if not self.client_secret:
            raise ValueError("Не указан Client Secret. Укажите через GIGACHAT_CLIENT_SECRET")
        if not self.auth_data:
            raise ValueError("Не указаны данные авторизации. Укажите через GIGACHAT_AUTH_DATA")

        self.access_token = None
        self.token_expires = 0
        self.rq_uid = str(uuid.uuid4())

        print(f"✅ GigaChatClient инициализирован с RqUID: {self.rq_uid}")

    def _generate_rq_uid(self) -> str:
        """Генерация нового RqUID"""
        self.rq_uid = str(uuid.uuid4())
        return self.rq_uid

    async def get_access_token(self) -> str:
        """Получение access token для GigaChat API с правильной авторизацией"""
        # Проверяем, есть ли действующий токен
        if self.access_token and asyncio.get_event_loop().time() < self.token_expires:
            return self.access_token

        # Генерируем новый RqUID для запроса
        current_rq_uid = self._generate_rq_uid()

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': current_rq_uid,
            'Authorization': f'Basic {self.auth_data}'  # Используем Basic авторизацию
        }

        data = {
            'scope': 'GIGACHAT_API_PERS'
        }

        try:
            print(f"🔐 Запрашиваю токен с RqUID: {current_rq_uid}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.auth_url,
                    headers=headers,
                    data=data,
                    ssl=False
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        self.access_token = result.get('access_token')

                        # Токен обычно действует 30 минут (1800 секунд)
                        self.token_expires = asyncio.get_event_loop().time() + 1700  # 10 секунд запаса

                        print(f"✅ Токен получен успешно, действителен до: {self.token_expires}")
                        return self.access_token
                    else:
                        error_text = await response.text()
                        print(f"❌ Ошибка авторизации: {response.status}")
                        print(f"   Заголовки: {headers}")
                        print(f"   Ответ: {error_text}")
                        raise Exception(f"Ошибка авторизации: {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Ошибка получения токена: {e}")

    async def chat_completion(self,
                             messages: List[Dict[str, str]],
                             model: str = "GigaChat",
                             temperature: float = 1.0,
                             top_p: float = 0.1,
                             max_tokens: int = 512,
                             stream: bool = False) -> Dict[str, Any]:
        """
        Асинхронное завершение чата через GigaChat

        Args:
            messages: Список сообщений в формате [{"role": "user", "content": "текст"}]
            model: Модель GigaChat (GigaChat, GigaChat-Pro, GigaChat-Plus)
            temperature: Температура генерации
            top_p: Параметр top_p
            max_tokens: Максимальное количество токенов
            stream: Использовать ли streaming

        Returns:
            Ответ от API
        """
        access_token = await self.get_access_token()

        url = f"{self.api_url}/chat/completions"

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "stream": stream,
            "max_tokens": max_tokens,
            "repetition_penalty": 1
        }

        try:
            print(f"🤖 Отправляю запрос к модели {model}...")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, ssl=False) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Ответ получен успешно")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ Ошибка API: {response.status}")
                        print(f"   Ответ: {error_text}")
                        raise Exception(f"Ошибка API: {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Ошибка вызова GigaChat: {e}")

    async def generate_text(self,
                           prompt: str,
                           system_prompt: str = None,
                           **kwargs) -> str:
        """
        Генерация текста по промпту

        Args:
            prompt: Пользовательский промпт
            system_prompt: Системный промпт (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Сгенерированный текст
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        result = await self.chat_completion(messages, **kwargs)

        # Извлекаем текст ответа
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Неожиданный формат ответа: {result}")

    async def analyze_with_llm(self,
                              prompt: str,
                              system_prompt: str = None,
                              return_json: bool = False) -> Any:
        """
        Анализ с помощью LLM с возможностью возврата JSON

        Args:
            prompt: Промпт для анализа
            system_prompt: Системный промпт
            return_json: Нужно ли парсить JSON

        Returns:
            Результат анализа
        """
        if return_json:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nВАЖНО: Ответ должен быть строго в формате JSON."
            else:
                system_prompt = "Ответ должен быть строго в формате JSON."

        result_text = await self.generate_text(prompt, system_prompt, temperature=0.7)

        if return_json:
            try:
                # Пытаемся найти JSON в тексте ответа
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # Если не нашли чистый JSON, пытаемся парсить весь текст
                    return json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON: {e}")
                print(f"📝 Ответ от LLM: {result_text}")
                # Пробуем исправить JSON
                try:
                    # Убираем возможные лишние символы
                    cleaned = result_text.strip()
                    if cleaned.startswith('```json'):
                        cleaned = cleaned[7:]
                    if cleaned.endswith('```'):
                        cleaned = cleaned[:-3]
                    return json.loads(cleaned)
                except:
                    return {"error": "Не удалось распарсить JSON", "raw_response": result_text}

        return result_text


class SyncGigaChatClient:
    """Синхронная обертка для GigaChat клиента"""

    def __init__(self, client_secret: str = None, auth_data: str = None, **kwargs):
        self.async_client = GigaChatClient(client_secret, auth_data, **kwargs)
        self.loop = None

    def _ensure_loop(self):
        """Создает event loop если его нет"""
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def get_access_token(self) -> str:
        """Синхронное получение токена"""
        self._ensure_loop()
        return self.loop.run_until_complete(
            self.async_client.get_access_token()
        )

    def generate_text(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Синхронная генерация текста"""
        self._ensure_loop()
        return self.loop.run_until_complete(
            self.async_client.generate_text(prompt, system_prompt, **kwargs)
        )

    def analyze_with_llm(self, prompt: str, system_prompt: str = None, return_json: bool = False) -> Any:
        """Синхронный анализ с LLM"""
        self._ensure_loop()
        return self.loop.run_until_complete(
            self.async_client.analyze_with_llm(prompt, system_prompt, return_json)
        )

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Синхронное завершение чата"""
        self._ensure_loop()
        return self.loop.run_until_complete(
            self.async_client.chat_completion(messages, **kwargs)
        )