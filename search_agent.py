import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Агент поиска - использует GigaChat API для формирования и исполнения стратегии поиска.
    Все решения принимает нейросеть GigaChat.
    """
    
    def __init__(self, gigachat_client):
        """
        Инициализация агента поиска.
        
        Args:
            gigachat_client: Клиент GigaChat API
        """
        self.client = gigachat_client

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обработка поискового запроса с использованием GigaChat API для всех решений.

        Args:
            query: Поисковый запрос пользователя
            context: Дополнительный контекст (опционально)

        Returns:
            Структурированный ответ с решениями от GigaChat
        """
        logger.info(f"SearchAgent: обработка запроса - {query[:100]}...")

        try:
            # Ограничиваем длину запроса для избежания ошибки 413
            MAX_QUERY_LENGTH = 1000
            truncated_query = query[:MAX_QUERY_LENGTH] + "..." if len(query) > MAX_QUERY_LENGTH else query

            # 1. Полный анализ запроса и формирование стратегии с помощью GigaChat
            analysis_result = self._perform_complete_analysis(truncated_query, context)

            # 2. Формирование финального ответа
            response = self._create_response(
                query=truncated_query,
                analysis_result=analysis_result,
                context=context
            )

            return response

        except Exception as e:
            logger.error(f"Ошибка в SearchAgent: {e}")
            return self._create_error_response(query[:100], str(e))
    def _perform_complete_analysis(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Выполнение полного анализа запроса с помощью GigaChat API.
        Все решения принимает нейросеть.
        """
        # Ограничиваем длину запроса и контекста
        MAX_CONTEXT_LENGTH = 500
        limited_context = {}
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    limited_context[key] = value[:MAX_CONTEXT_LENGTH]
                else:
                    limited_context[key] = value

        system_prompt = """
        Ты - экспертная система поиска информации. Твоя задача:

        1. ПРОАНАЛИЗИРОВАТЬ ЗАПРОС:
           - Выделить ключевые темы и концепции
           - Определить возможные синонимы
           - Определить целевую аудиторию
           - Определить уровень сложности
           - Определить потребности в информации

        2. СФОРМУЛИРОВАТЬ СТРАТЕГИЮ ПОИСКА:
           - Выбрать оптимальный алгоритм поиска из доступных
           - Обосновать выбор алгоритма
           - Определить параметры поиска

        3. ПЕРЕФОРМУЛИРОВАТЬ ЗАПРОС:
           - Создать 3-5 альтернативных формулировок
           - Объяснить, почему эти формулировки могут улучшить поиск

        4. РЕКОМЕНДОВАТЬ КРИТЕРИИ РАНЖИРОВАНИЯ:
           - Определить, как ранжировать результаты поиска
           - Объяснить критерии релевантности

        Доступные алгоритмы поиска:
        - keyword_search: точный поиск по ключевым словам
        - semantic_search: семантический поиск по смыслу
        - concept_expansion: расширение концепций и поиск связанных тем
        - alternative_queries: использование альтернативных формулировок
        - filtered_search: поиск с фильтрами по параметрам

        Ответ предоставь в формате JSON со следующей структурой:
        {
            "query_analysis": {
                "key_topics": [],
                "synonyms": [],
                "target_audience": "",
                "complexity_level": "",
                "information_needs": []
            },
            "search_strategy": {
                "chosen_algorithm": "",
                "algorithm_reason": "",
                "search_parameters": {}
            },
            "reformulated_queries": [],
            "ranking_criteria": {
                "primary_criteria": [],
                "secondary_criteria": [],
                "explanation": ""
            },
            "search_instructions": ""
        }
        """

        # Ограничиваем длину промпта
        MAX_PROMPT_LENGTH = 3000
        user_prompt = f"""
        ЗАПРОС ПОЛЬЗОВАТЕЛЯ НА ПОИСК КНИГИ: "{query[:500]}"

        ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ: {limited_context if limited_context else "не предоставлен"}

        ПРОАНАЛИЗИРУЙ этот запрос и предоставь полный план поиска в указанном формате JSON.

        Важно: Все решения должны быть обоснованы. Не используй готовые шаблоны - 
        анализируй конкретный запрос и принимай решения на его основе.
        """[:MAX_PROMPT_LENGTH]

        try:
            # Получаем полный анализ от GigaChat
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000  # Уменьшаем токены
            )

            # Обработка ответа
            if isinstance(response, dict) and "raw_response" in response:
                try:
                    return json.loads(response["raw_response"])
                except json.JSONDecodeError as e:
                    logger.error(f"Не удалось распарсить JSON ответ: {e}")
                    return self._extract_from_text(response["raw_response"])
            elif isinstance(response, dict):
                return response
            else:
                return self._extract_from_text(str(response))

        except Exception as e:
            logger.error(f"Ошибка при анализе запроса GigaChat: {e}")
            raise
    
    def _extract_from_text(self, text_response: str) -> Dict[str, Any]:
        """
        Попытка извлечь структурированные данные из текстового ответа.
        """
        try:
            # Пытаемся найти JSON в тексте
            import re
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Если не удалось, создаем базовую структуру
        return {
            "query_analysis": {
                "key_topics": ["анализ выполнен GigaChat", "структура не распознана"],
                "synonyms": [],
                "target_audience": "general",
                "complexity_level": "intermediate",
                "information_needs": ["получение структурированной информации"]
            },
            "search_strategy": {
                "chosen_algorithm": "semantic_search",
                "algorithm_reason": "семантический поиск для понимания контекста",
                "search_parameters": {}
            },
            "reformulated_queries": [],
            "ranking_criteria": {
                "primary_criteria": ["релевантность запросу"],
                "secondary_criteria": ["авторитетность источника"],
                "explanation": "Ранжирование на основе релевантности"
            },
            "search_instructions": "Выполнить поиск по выбранному алгоритму"
        }
    
    def _create_response(self, query: str, analysis_result: Dict[str, Any], 
                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Формирование финального ответа на основе анализа от GigaChat.
        """
        # Генерация объяснения от GigaChat
        explanation = self._generate_explanation(query, analysis_result)
        
        # Формирование инструкций для выполнения поиска
        search_instructions = self._generate_search_instructions(analysis_result)
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "agent": "SearchAgent",
            "original_query": query,
            "context": context,
            "analysis_result": analysis_result,
            "search_instructions": search_instructions,
            "explanation": explanation,
            "summary": {
                "algorithm_chosen": analysis_result.get("search_strategy", {}).get("chosen_algorithm", "не определен"),
                "query_reformulations": len(analysis_result.get("reformulated_queries", [])),
                "gigachat_used": True,
                "all_decisions_by_llm": True
            }
        }
    
    def _generate_explanation(self, query: str, analysis_result: Dict[str, Any]) -> str:
        """
        Генерация объяснения от GigaChat о принятых решениях.
        """
        system_prompt = """
        Ты - эксперт по поиску информации. Объясни пользователю, как был проанализирован его запрос 
        и какие решения были приняты для оптимизации поиска. Будь краток, информативен и полезен.
        
        Ответ должен быть в формате plain text, без JSON.
        """
        
        user_prompt = f"""
        ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"
        
        РЕЗУЛЬТАТЫ АНАЛИЗА GIGACHAT:
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        
        Объясни пользователю:
        1. Как был проанализирован его запрос
        2. Почему выбрана именно эта стратегия поиска
        3. Как будут использованы альтернативные формулировки
        4. Как будут ранжированы результаты
        
        Будь конкретным и ссылайся на особенности его запроса.
        """
        
        try:
            response = self.client.chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500
            )
            return response
        except Exception as e:
            logger.warning(f"Ошибка генерации объяснения: {e}")
            return "Анализ выполнен системой GigaChat. Все решения приняты нейросетью на основе вашего запроса."
    
    def _generate_search_instructions(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация конкретных инструкций для выполнения поиска.
        Все инструкции формируются GigaChat.
        """
        system_prompt = """
        На основе анализа поискового запроса сформулируй конкретные инструкции 
        для системы поиска. Инструкции должны быть четкими и исполняемыми.
        
        Ответ предоставь в формате JSON:
        {
            "algorithm_to_execute": "",
            "exact_parameters": {},
            "query_variants": [],
            "filters_to_apply": {},
            "sorting_order": ""
        }
        """
        
        user_prompt = f"""
        АНАЛИЗ ПОИСКОВОГО ЗАПРОСА:
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        
        Сформулируй конкретные инструкции для системы поиска на основе этого анализа.
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            
            if isinstance(response, dict) and "raw_response" in response:
                try:
                    return json.loads(response["raw_response"])
                except:
                    return self._create_default_instructions(analysis_result)
            elif isinstance(response, dict):
                return response
            else:
                return self._create_default_instructions(analysis_result)
                
        except Exception as e:
            logger.warning(f"Ошибка генерации инструкций: {e}")
            return self._create_default_instructions(analysis_result)
    
    def _create_default_instructions(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Создание инструкций по умолчанию на основе анализа."""
        strategy = analysis_result.get("search_strategy", {})
        
        return {
            "algorithm_to_execute": strategy.get("chosen_algorithm", "semantic_search"),
            "exact_parameters": strategy.get("search_parameters", {}),
            "query_variants": analysis_result.get("reformulated_queries", []),
            "filters_to_apply": {
                "audience": analysis_result.get("query_analysis", {}).get("target_audience"),
                "complexity": analysis_result.get("query_analysis", {}).get("complexity_level")
            },
            "sorting_order": "relevance"
        }
    
    def _create_error_response(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Создание ответа об ошибке."""
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "agent": "SearchAgent",
            "original_query": query,
            "error": error_msg,
            "explanation": f"Ошибка при обработке запроса GigaChat: {error_msg}",
            "gigachat_used": True
        }


# Пример использования агента
if __name__ == "__main__":
    # Инициализация клиента GigaChat
    try:
        from gigachat_client import GigaChatClient
        client = GigaChatClient(verify_ssl=False)
    except ImportError:
        print("Ошибка: Не найден файл gigachat_client.py")
        exit(1)
    except Exception as e:
        print(f"Ошибка инициализации GigaChatClient: {e}")
        exit(1)
    
    # Создание агента поиска
    search_agent = SearchAgent(client)
    
    # Тестовый запрос
    test_query = "ищу учебник по матанализу для начинающих с примерами и задачами"
    
    print(f"🔍 Запрос пользователя: {test_query}")
    print("=" * 80)
    
    # Обработка запроса
    try:
        result = search_agent.process(test_query)
        
        # Вывод результатов
        print(f"✅ Статус: {result['status']}")
        print(f"🕐 Время обработки: {result['timestamp']}")
        print(f"🤖 Все решения приняты GigaChat: {result['summary']['all_decisions_by_llm']}")
        print("\n" + "=" * 80)
        
        # Объяснение от GigaChat
        print("💬 ОБЪЯСНЕНИЕ ОТ GIGACHAT:")
        print(result['explanation'])
        
        print("\n" + "=" * 80)
        
        # Ключевые решения
        analysis = result['analysis_result']
        
        print("📊 АНАЛИЗ ЗАПРОСА:")
        query_analysis = analysis.get('query_analysis', {})
        for key, value in query_analysis.items():
            if isinstance(value, list):
                print(f"  • {key}: {', '.join(map(str, value[:5]))}")
            else:
                print(f"  • {key}: {value}")
        
        print("\n🎯 ВЫБРАННАЯ СТРАТЕГИЯ ПОИСКА:")
        strategy = analysis.get('search_strategy', {})
        print(f"  • Алгоритм: {strategy.get('chosen_algorithm', 'не определен')}")
        print(f"  • Обоснование: {strategy.get('algorithm_reason', 'не указано')}")
        
        print("\n🔄 ПЕРЕФОРМУЛИРОВАННЫЕ ЗАПРОСЫ:")
        reformulated = analysis.get('reformulated_queries', [])
        for i, q in enumerate(reformulated[:3], 1):
            print(f"  {i}. {q}")
        
        print("\n📋 ИНСТРУКЦИИ ДЛЯ ВЫПОЛНЕНИЯ ПОИСКА:")
        instructions = result['search_instructions']
        print(f"  • Алгоритм для выполнения: {instructions.get('algorithm_to_execute', 'не определен')}")
        print(f"  • Варианты запросов: {len(instructions.get('query_variants', []))}")
        print(f"  • Фильтры: {instructions.get('filters_to_apply', {})}")
        
        print("\n" + "=" * 80)
        
        # Сохранение результатов в файл
        output_file = "gigachat_search_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 Полный анализ сохранен в файл: {output_file}")
        
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")