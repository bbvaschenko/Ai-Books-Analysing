"""
SearchAgent - агент формирования и исполнения стратегии поиска
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentMessage


class SearchAgent(BaseAgent):
    """Агент поиска учебных материалов"""

    def __init__(self, library_core):
        super().__init__(
            name="SearchAgent",
            role="Агент формирования и исполнения стратегии поиска информации",
            description="""Я анализирую цель запроса, определяю ключевые темы и синонимы,
выбираю оптимальную стратегию поиска и возвращаю ранжированный список кандидатов с обоснованием."""
        )

        self.library_core = library_core

    async def process(self, message: AgentMessage, context: Dict = None) -> AgentMessage:
        """Обработка поискового запроса"""
        self.add_to_history(message)

        query = message.content.get("query", "")
        task = message.content.get("task", "")

        # Анализируем запрос с помощью LLM
        query_analysis = await self._analyze_search_query_with_llm(query)

        # Расширяем запрос синонимами
        expanded_queries = self._expand_query_with_synonyms(query_analysis)

        # Выполняем поиск по всем вариантам
        search_results = []
        for expanded_query in expanded_queries:
            results = await self._search_in_library(expanded_query)
            search_results.extend(results)

        # Ранжируем и фильтруем результаты
        ranked_results = await self._rank_results_with_llm(search_results, query_analysis)

        # Формируем ответ с обоснованием
        response_content = {
            "original_query": query,
            "query_analysis": query_analysis,
            "expanded_queries": expanded_queries,
            "found_candidates": len(search_results),
            "ranked_results": ranked_results[:10],  # Топ-10
            "search_strategy_explanation": query_analysis.get("strategy_explanation", ""),
            "reasoning": f"Найдено {len(search_results)} кандидатов. Использована стратегия: {query_analysis.get('search_strategy', 'комбинированная')}"
        }

        print(f"  SearchAgent нашел {len(search_results)} кандидатов по запросу: '{query}'")

        return AgentMessage(
            sender=self.name,
            recipient=message.sender,  # Отвечаем отправителю (обычно CoordinatorAgent)
            content=response_content,
            conversation_id=message.conversation_id
        )

    async def _analyze_search_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Анализ поискового запроса с помощью LLM"""
        # Эмуляция LLM анализа
        query_lower = query.lower()

        # Определяем тип запроса
        query_type = "general"
        if any(word in query_lower for word in ["учебник", "пособие", "книга"]):
            query_type = "textbook"
        elif any(word in query_lower for word in ["задача", "упражнение", "пример"]):
            query_type = "exercises"
        elif any(word in query_lower for word in ["курс", "лекция", "презентация"]):
            query_type = "course_material"

        # Определяем стратегию поиска
        strategies = []
        if len(query.split()) > 3:
            strategies.append("точное_соответствие")
            strategies.append("по_ключевым_словам")
        else:
            strategies.append("расширенный_поиск")
            strategies.append("синонимы")

        return {
            "query_type": query_type,
            "main_topics": self._extract_topics(query),
            "possible_synonyms": self._generate_synonyms(query),
            "search_strategy": strategies[0],
            "strategy_explanation": f"Используется стратегия '{strategies[0]}', потому что {'запрос сложный' if len(query.split()) > 3 else 'запрос простой'}",
            "complexity": "high" if len(query.split()) > 2 else "medium"
        }

    def _extract_topics(self, query: str) -> List[str]:
        """Извлечение ключевых тем из запроса"""
        # Простая эмуляция извлечения тем
        topics = []
        words = query.lower().split()

        math_keywords = ["математик", "алгебр", "геометри", "тригонометр", "уравнен", "формул"]
        for word in words:
            for keyword in math_keywords:
                if keyword in word:
                    topics.append(word)
                    break

        if not topics:
            topics = [query[:20] + "..."] if len(query) > 20 else [query]

        return topics

    def _generate_synonyms(self, query: str) -> List[str]:
        """Генерация синонимов для запроса"""
        synonym_map = {
            "математика": ["алгебра", "геометрия", "высшая математика", "матан"],
            "учебник": ["пособие", "книга", "руководство", "методичка"],
            "задача": ["упражнение", "пример", "проблема", "задание"],
            "программирование": ["кодирование", "разработка", "софт"],
            "физика": ["механика", "термодинамика", "оптика"]
        }

        synonyms = [query]
        query_lower = query.lower()

        for key, values in synonym_map.items():
            if key in query_lower:
                synonyms.extend(values)

        # Добавляем варианты с разными окончаниями
        if "учебник" in query_lower:
            synonyms.append("учебное пособие")
            synonyms.append("учебный материал")

        return list(set(synonyms))

    def _expand_query_with_synonyms(self, query_analysis: Dict) -> List[str]:
        """Расширение запроса синонимами"""
        base_query = query_analysis.get("main_topics", ["математика"])[0]
        synonyms = query_analysis.get("possible_synonyms", [])

        expanded = [base_query]
        expanded.extend(synonyms)

        # Добавляем комбинации
        if len(synonyms) > 1:
            for i in range(min(3, len(synonyms))):
                for j in range(i + 1, min(5, len(synonyms))):
                    expanded.append(f"{synonyms[i]} {synonyms[j]}")

        return list(set(expanded))[:10]  # Не более 10 вариантов

    async def _search_in_library(self, query: str) -> List[Dict]:
        """Поиск в библиотеке через library_core"""
        try:
            # Используем существующую функцию поиска
            results = self.library_core.search_books(query, top_k=20)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "book_id": result.get("book_id", ""),
                    "filename": result.get("filename", ""),
                    "area": result.get("area", ""),
                    "score": result.get("score", 0),
                    "matching_tags": result.get("matching_tags", []),
                    "ai_summary": result.get("ai_summary", ""),
                    "query_match": query,
                    "match_reason": f"Совпадение по тегам: {', '.join(result.get('matching_tags', [])[:3])}"
                })

            return formatted_results

        except Exception as e:
            print(f"❌ Ошибка поиска в библиотеке: {e}")
            return []

    async def _rank_results_with_llm(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """Ранжирование результатов с помощью LLM"""
        if not results:
            return []

        # Эмуляция LLM ранжирования
        query_type = query_analysis.get("query_type", "general")
        main_topics = query_analysis.get("main_topics", [])

        for result in results:
            # Простая эвристическая оценка
            score = result.get("score", 0)

            # Учитываем соответствие теме
            topic_match = 0
            for topic in main_topics:
                if topic.lower() in result.get("filename", "").lower():
                    topic_match += 1
                if topic.lower() in result.get("area", "").lower():
                    topic_match += 1

            # Комбинированный рейтинг
            combined_score = score * 0.7 + topic_match * 0.3
            result["combined_score"] = combined_score
            result["ranking_reason"] = f"Базовый score: {score}, соответствие теме: {topic_match}/10"

        # Сортировка по комбинированному score
        return sorted(results, key=lambda x: x.get("combined_score", 0), reverse=True)