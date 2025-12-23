"""
Тестирование интеграции GigaChat (исправленная версия)
"""
import os
import json
import base64
from dotenv import load_dotenv
from agents.giga_client import SyncGigaChatClient


def prepare_auth_data(client_id: str, client_secret: str) -> str:
    """
    Подготовка данных для Basic авторизации

    Args:
        client_id: Client ID из личного кабинета
        client_secret: Client Secret из личного кабинета

    Returns:
        Base64 строка для заголовка Authorization
    """
    # Формируем строку "client_id:client_secret"
    auth_string = f"{client_id}:{client_secret}"
    # Кодируем в Base64
    auth_bytes = auth_string.encode('utf-8')
    auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
    return auth_b64


def test_gigachat_connection():
    """Тестирование подключения к GigaChat с правильной авторизацией"""
    load_dotenv()

    # Получаем данные из переменных окружения
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    client_id = os.getenv("GIGACHAT_CLIENT_ID")

    if not client_secret:
        print("❌ GIGACHAT_CLIENT_SECRET не найден в .env файле")
        return False

    if not client_id:
        print("❌ GIGACHAT_CLIENT_ID не найден в .env файле")
        return False

    print("🧪 Тестирование подключения к GigaChat...")

    try:
        # Подготавливаем данные для Basic авторизации
        auth_data = prepare_auth_data(client_id, client_secret)

        # Создаем клиент
        client = SyncGigaChatClient(
            client_secret=client_secret,
            auth_data=auth_data
        )

        print("🔑 Ключи найдены, тестирую получение токена...")

        # Пробуем получить токен
        token = client.get_access_token()

        if token:
            print(f"✅ Токен получен успешно!")
            print(f"🔐 Токен (первые 20 символов): {token[:20]}...")
        else:
            print("❌ Не удалось получить токен")
            return False

        # Тестируем простой запрос
        print("\n🤖 Тестирую простой запрос...")

        response = client.generate_text(
            prompt="Привет! Ответь коротко: как дела?",
            system_prompt="Ты полезный помощник. Отвечай кратко и вежливо."
        )

        print(f"✅ Запрос выполнен успешно!")
        print(f"📝 Ответ: {response}")

        # Тестируем JSON ответ
        print("\n📊 Тестирую JSON ответ...")

        json_response = client.analyze_with_llm(
            prompt="""Проанализируй запрос 'учебник по математике для студентов' 
            и верни в формате JSON с полями: query_type, main_topic, difficulty_level, target_audience.""",
            system_prompt="Ты аналитик учебной библиотеки. Всегда отвечай в формате JSON.",
            return_json=True
        )

        print(f"✅ JSON ответ получен:")
        print(json.dumps(json_response, ensure_ascii=False, indent=2))

        return True

    except Exception as e:
        print(f"❌ Ошибка подключения к GigaChat: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_env_file():
    """Проверка и создание правильного .env файла"""
    print("\n📁 Проверяю файл .env...")

    env_path = ".env"
    example_env = """# GigaChat API настройки
# Получите эти данные в личном кабинете: https://developers.sber.ru/studio

# Client ID и Client Secret из раздела "Доступы" или "API ключи"
GIGACHAT_CLIENT_ID=ваш_client_id_здесь
GIGACHAT_CLIENT_SECRET=ваш_client_secret_здесь

# Обычно эти URL не нужно менять
GIGACHAT_AUTH_URL=https://ngw.devices.sberbank.ru:9443/api/v2/oauth
GIGACHAT_API_URL=https://gigachat.devices.sberbank.ru/api/v1

# Опционально: модель по умолчанию
# GIGACHAT_MODEL=GigaChat  # Бесплатная (10K токенов)
# GIGACHAT_MODEL=GigaChat-Pro  # Бесплатная (50K токенов)
# GIGACHAT_MODEL=GigaChat-Plus  # Платная
"""

    if not os.path.exists(env_path):
        print("⚠️  Файл .env не найден. Создаю шаблон...")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(example_env)
        print("✅ Файл .env создан. Заполните его своими данными!")
        return False

    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "GIGACHAT_CLIENT_ID" not in content or "GIGACHAT_CLIENT_SECRET" not in content:
        print("⚠️  В файле .env не найдены необходимые переменные.")
        print("Добавьте следующие строки в ваш .env файл:")
        print("\nGIGACHAT_CLIENT_ID=ваш_client_id_здесь")
        print("GIGACHAT_CLIENT_SECRET=ваш_client_secret_здесь\n")
        return False

    print("✅ Файл .env выглядит корректно")
    return True


def test_direct_requests():
    """Тестирование прямых запросов через requests"""
    print("\n🔧 Тестирование прямых запросов...")

    load_dotenv()

    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("❌ Не найдены CLIENT_ID или CLIENT_SECRET")
        return

    import requests

    # 1. Получение токена
    auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    # Подготавливаем Basic авторизацию
    import base64
    auth_string = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_string.encode()).decode()

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(os.urandom(16).hex()),  # Генерируем случайный RqUID
        'Authorization': f'Basic {auth_b64}'
    }

    payload = 'scope=GIGACHAT_API_PERS'

    try:
        print("🔐 Получаю токен...")
        response = requests.post(auth_url, headers=headers, data=payload, verify=False)

        print(f"📊 Статус: {response.status_code}")

        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            print(f"✅ Токен получен: {access_token[:20]}...")

            # 2. Тестируем запрос к API
            api_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

            api_headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }

            api_payload = {
                "model": "GigaChat",
                "messages": [
                    {
                        "role": "user",
                        "content": "Привет! Как дела?"
                    }
                ],
                "temperature": 1,
                "top_p": 0.1,
                "n": 1,
                "stream": False,
                "max_tokens": 512,
                "repetition_penalty": 1
            }

            print("🤖 Отправляю запрос к API...")
            api_response = requests.post(api_url, headers=api_headers, json=api_payload, verify=False)

            print(f"📊 Статус API: {api_response.status_code}")

            if api_response.status_code == 200:
                result = api_response.json()
                print(f"✅ API ответ получен!")
                if "choices" in result:
                    answer = result["choices"][0]["message"]["content"]
                    print(f"📝 Ответ: {answer}")
                return True
            else:
                print(f"❌ Ошибка API: {api_response.text}")
        else:
            print(f"❌ Ошибка получения токена: {response.text}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

    return False


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ GIGACHAT (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print("=" * 60)

    # 1. Проверяем .env файл
    if not check_env_file():
        print("\n⚠️  Заполните файл .env и запустите тест снова.")
        print("=" * 60)
        exit(1)

    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК ТЕСТОВ")
    print("=" * 60)

    # 2. Тестируем прямые запросы
    if test_direct_requests():
        print("\n✅ Прямые запросы работают!")
    else:
        print("\n❌ Прямые запросы не работают")

    # 3. Тестируем через клиент
    print("\n" + "=" * 60)
    print("🧪 ТЕСТИРУЮ ЧЕРЕЗ КЛИЕНТ")
    print("=" * 60)

    if test_gigachat_connection():
        print("\n✅ Все тесты пройдены успешно!")
    else:
        print("\n❌ Тесты не пройдены")

    print("\n" + "=" * 60)
    print("📋 ИНСТРУКЦИЯ ПО ЗАПУСКУ:")
    print("=" * 60)
    print("1. Убедитесь, что в .env файле есть:")
    print("   - GIGACHAT_CLIENT_ID (ваш Client ID)")
    print("   - GIGACHAT_CLIENT_SECRET (ваш Client Secret)")
    print("2. Установите зависимости: pip install -r requirements.txt")
    print("3. Запустите тест: python test_gigachat.py")
    print("4. Если тест пройден, запустите бота: python main.py")
    print("5. Выберите режим с агентами")
    print("=" * 60)