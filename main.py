import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

# Импорт всех агентов
from gigachat_client import GigaChatClient
from coordinator_agent import CoordinatorAgent
from search_agent import SearchAgent
from analysis_agent import AnalysisAgent
from critic_agent import CriticAgent
from recommendation_agent import RecommendationAgent
from search_tags import TagSearch
from summary_agent import SummaryAgent
from pdf_processor import PDFProcessor
from data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartLibrarySystem:
    """
    Главная система Умной Библиотеки.
    Координирует работу всех компонентов.
    """
    
    def __init__(self, verify_ssl: bool = False):
        # Инициализация базовых компонентов
        logger.info("Инициализация Smart Library System...")
        
        # Клиент GigaChat
        self.gigachat = GigaChatClient(verify_ssl=verify_ssl)
        
        # Менеджер данных
        self.data_manager = DataManager()
        
        # Инициализация агентов
        self._init_agents()
        
        # Инициализация инструментов
        self._init_tools()
        
        logger.info("✅ Система инициализирована")
    
    def _init_agents(self):
        """Инициализация всех агентов"""
        # Создание агентов
        self.coordinator = CoordinatorAgent(self.gigachat)
        self.search_agent = SearchAgent(self.gigachat)
        self.analysis_agent = AnalysisAgent(verify_ssl=False)
        self.critic_agent = CriticAgent(self.gigachat)
        self.recommendation_agent = RecommendationAgent(self.gigachat)
        self.summary_agent = SummaryAgent(self.gigachat)
        
        # Регистрация агентов в координаторе
        self.coordinator.register_agent("SearchAgent", self.search_agent)
        self.coordinator.register_agent("AnalysisAgent", self.analysis_agent)
        self.coordinator.register_agent("CriticAgent", self.critic_agent)
        self.coordinator.register_agent("RecommendationAgent", self.recommendation_agent)
        
        logger.info("✅ Агенты инициализированы")
    
    def _init_tools(self):
        """Инициализация инструментов"""
        self.pdf_processor = PDFProcessor()
        self.tag_search = TagSearch()
        
        logger.info("✅ Инструменты инициализированы")

    def process_user_query(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        logger.info(f"Обработка запроса: {user_query}")

        # Гарантируем, что context - словарь
        if context is None:
            context = {}

        # Добавляем обязательные поля
        context['original_query'] = user_query

        try:
            # Сначала ищем книги
            search_results = self.search_books(user_query)

            # Если нашли книги, готовим контент для анализа
            if search_results.get('results_count', 0) > 0:
                # Берем первую книгу
                first_book = search_results['results'][0]

                # Получаем текст книги
                book_text = self._get_book_text(first_book)

                if book_text:
                    # Добавляем в контекст для анализа
                    context['content'] = book_text[:10000]  # Ограничиваем длину
                    context['topic'] = user_query
                    context['target_audience'] = 'студенты'

                    # Сохраняем информацию о книге
                    context['book_info'] = {
                        'title': first_book.get('title', ''),
                        'book_id': first_book.get('book_id', ''),
                        'file_path': first_book.get('file_path', '')
                    }

            # Обработка через координатор
            start_time = datetime.now()

            result = self.coordinator.process_query(user_query, context)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Добавление метаданных
            result["system_metadata"] = {
                "processing_time_seconds": processing_time,
                "agents_used": list(self.coordinator.agent_registry.keys()),
                "timestamp": datetime.now().isoformat(),
                "system_version": "1.0",
                "books_found": search_results.get('results_count', 0)
            }

            # Логирование
            self._log_query(user_query, result, processing_time)

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return self._create_error_response(user_query, str(e))
    
    def upload_book(self, pdf_path: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Загрузка и обработка новой книги.
        
        Args:
            pdf_path: Путь к PDF файлу
            title: Название книги (если не указано, берется из файла)
            
        Returns:
            Результат обработки
        """
        logger.info(f"Загрузка книги: {pdf_path}")
        
        try:
            # Валидация файла
            if not os.path.exists(pdf_path):
                return {"error": f"Файл не найден: {pdf_path}"}
            
            if not pdf_path.lower().endswith('.pdf'):
                return {"error": "Поддерживаются только PDF файлы"}
            
            # Извлечение текста
            pdf_result = self.pdf_processor.extract_text(pdf_path)
            if not pdf_result.get("success", False):
                return {"error": f"Ошибка обработки PDF: {pdf_result.get('error', 'неизвестно')}"}
            
            # Определение названия
            book_title = title or pdf_result.get("metadata", {}).get("title") or os.path.basename(pdf_path)
            
            # Тегирование книги
            from book_tagger import AdvancedBookTagger
            tagger = AdvancedBookTagger()
            
            tagging_result = tagger.analyze_book(
                text=pdf_result.get("text", ""),
                title=book_title,
                metadata=pdf_result.get("metadata", {})
            )
            
            # Создание резюме
            summary_result = self.summary_agent.create_summary(
                content=pdf_result.get("text", ""),
                title=book_title,
                summary_type="detailed"
            )
            
            # Сохранение данных
            book_id = f"book_{datetime.now().timestamp()}"
            
            # Подготовка данных для сохранения
            book_data = self._prepare_book_data(
                book_id=book_id,
                pdf_path=pdf_path,
                pdf_result=pdf_result,
                tagging_result=tagging_result,
                summary_result=summary_result,
                title=book_title
            )
            
            # Сохранение тегов
            self.data_manager.save_book_tags(book_data)
            
            # Сохранение резюме
            summary_path = self.data_manager.save_summary(summary_result, book_id)
            
            # Обновление записи с путем к резюме
            book_data["summary_file"] = summary_path
            self.data_manager.save_book_tags(book_data)  # Обновляем
            
            # Добавление в поисковую систему тегов
            self.tag_search.add_book_tags(book_data)
            
            result = {
                "success": True,
                "book_id": book_id,
                "title": book_title,
                "file_path": pdf_path,
                "processing_summary": {
                    "text_extracted": pdf_result.get("text_length", 0),
                    "pages_processed": pdf_result.get("processed_pages", 0),
                    "tags_generated": len(book_data.get("all_tags", [])),
                    "summary_created": "summary_text" in summary_result
                },
                "files_created": {
                    "tags": self.data_manager.tags_file,
                    "summary": summary_path
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Книга загружена: {book_title}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка загрузки книги: {e}")
            return {"error": str(e), "file": pdf_path}
    
    def _prepare_book_data(self, book_id: str, pdf_path: str, pdf_result: Dict, 
                          tagging_result: Any, summary_result: Dict, title: str) -> Dict[str, Any]:
        """Подготовка данных книги для сохранения"""
        # Извлечение тегов из результата тегирования
        # tagging_result - это кортеж, нужно правильно извлечь данные
        # В зависимости от структуры tagging_result
        
        all_tags = []
        tag_categories = {}
        
        if isinstance(tagging_result, tuple) and len(tagging_result) > 3:
            # Предполагаем структуру из AdvancedBookTagger
            tags_by_category = tagging_result[3]  # Кортеж пар (категория, теги)
            for category, tags in tags_by_category:
                tag_categories[category] = tags
                all_tags.extend(tags)
        
        book_data = {
            "book_id": book_id,
            "title": title,
            "original_filename": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "file_size": pdf_result.get("file_size", 0),
            "page_count": pdf_result.get("total_pages", 0),
            "all_tags": all_tags,
            "tags_confidence": 0.8,  # Примерное значение
            "summary_file": "",  # Будет добавлено позже
            "processing_status": "completed",
            "notes": f"Обработано {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        # Добавляем категоризированные теги
        book_data.update(tag_categories)
        
        return book_data
    
    def search_books(self, query: str, search_type: str = "auto") -> Dict[str, Any]:
        """
        Поиск книг.
        
        Args:
            query: Поисковый запрос
            search_type: auto/tags/summary/direct
            
        Returns:
            Результаты поиска
        """
        logger.info(f"Поиск книг: {query} (тип: {search_type})")
        
        try:
            # Автоматический выбор типа поиска
            if search_type == "auto":
                search_type = self._choose_search_type(query)
            
            # Выполнение поиска
            if search_type == "tags":
                results = self.tag_search.search_by_tags([query], operator="OR")
            elif search_type == "summary":
                results = self._search_in_summaries(query)
            elif search_type == "direct":
                results = self._direct_analysis_search(query)
            else:
                results = []
            
            # Анализ результатов
            if results:
                analyzed_results = self.analysis_agent.analyze_multiple_contents(
                    contents=[{"id": r.get("book_id", ""), "title": r.get("title", ""), 
                              "text": self._get_book_text(r)} for r in results],
                    topic=query
                )
            else:
                analyzed_results = {"error": "Ничего не найдено"}
            
            return {
                "search_type": search_type,
                "query": query,
                "results_count": len(results),
                "results": results[:10],  # Ограничиваем
                "analysis": analyzed_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return {"error": str(e), "query": query}
    
    def _choose_search_type(self, query: str) -> str:
        """
        Автоматический выбор типа поиска.
        Использует GigaChat для принятия решения.
        """
        system_prompt = """
        Выбери оптимальный тип поиска для запроса пользователя.
        
        Доступные типы:
        1. tags - поиск по тегам (быстрый, для конкретных тем)
        2. summary - поиск по резюме (более глубокий, для сложных запросов)
        3. direct - прямой анализ (самый точный, но медленный)
        
        Выбирай так, чтобы минимизировать время поиска.
        """
        
        user_prompt = f"""
        ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"
        
        Выбери тип поиска и кратко обоснуй.
        
        Формат ответа:
        {{
            "search_type": "tags|summary|direct",
            "reasoning": "почему выбрал этот тип",
            "expected_time": "быстро/средне/медленно",
            "expected_accuracy": "низкая/средняя/высокая"
        }}
        """
        
        try:
            response = self.gigachat.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=300
            )
            return response.get("search_type", "tags")
        except Exception as e:
            logger.warning(f"Ошибка выбора типа поиска: {e}. Использую tags.")
            return "tags"
    
    def _search_in_summaries(self, query: str) -> List[Dict]:
        """Поиск по резюме книг"""
        # Получаем все книги
        all_books = self.data_manager.get_all_books()
        results = []
        
        for book in all_books:
            summary = self.data_manager.get_summary(book.get('book_id', ''))
            if summary and query.lower() in summary.lower():
                results.append(book)
        
        return results
    
    def _direct_analysis_search(self, query: str) -> List[Dict]:
        """Прямой анализ для поиска"""
        # Получаем все книги
        all_books = self.data_manager.get_all_books()
        results = []
        
        # Для простоты берем только первые 5 книг (прямой анализ требует времени)
        for book in all_books[:5]:
            book_info = self._get_book_text(book)
            if book_info and query.lower() in book_info.lower():
                results.append(book)
        
        return results
    
    def _get_book_text(self, book: Dict) -> str:
        """Получение текста книги"""
        # Здесь должна быть логика получения полного текста книги
        # Для демо возвращаем заглушку
        return book.get('title', '') + " " + book.get('keywords', '')
    
    def _log_query(self, query: str, result: Dict, processing_time: float):
        """Логирование запроса"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "processing_time": processing_time,
            "result_summary": result.get("final_result", {}).get("executive_summary", ""),
            "success": "error" not in result
        }
        
        log_file = "query_log.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка логирования: {e}")
    
    def _create_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Создание ответа об ошибке"""
        return {
            "error": error,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "suggestions": [
                "Попробуйте переформулировать запрос",
                "Проверьте соединение с GigaChat",
                "Обратитесь к администратору системы"
            ]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        stats = self.data_manager.get_statistics()
        
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "agents": {
                "coordinator": "active",
                "search": "active",
                "analysis": "active",
                "critic": "active",
                "recommendation": "active"
            },
            "database": {
                "books_count": stats.get("total_books", 0),
                "summaries_count": stats.get("total_summaries", 0),
                "tags_count": stats.get("total_tags", 0)
            }
        }


def main():
    """Главная функция для демонстрации работы системы"""
    print("=" * 80)
    print("🤖 УМНАЯ БИБЛИОТЕКА - Полная система")
    print("=" * 80)
    
    try:
        # Инициализация системы
        print("\n🔄 Инициализация системы...")
        library = SmartLibrarySystem(verify_ssl=False)
        
        # Проверка статуса
        status = library.get_system_status()
        print(f"✅ Система готова. Книг в базе: {status['database']['books_count']}")
        
        while True:
            print("\n" + "=" * 80)
            print("МЕНЮ:")
            print("1. Поиск книг")
            print("2. Загрузить новую книгу (PDF)")
            print("3. Полный анализ запроса")
            print("4. Статус системы")
            print("5. Выход")
            
            choice = input("\nВыберите действие (1-5): ").strip()
            
            if choice == "1":
                query = input("Введите поисковый запрос: ").strip()
                if query:
                    results = library.search_books(query)
                    print(f"\n📚 Найдено книг: {results.get('results_count', 0)}")
                    for i, book in enumerate(results.get('results', [])[:5], 1):
                        print(f"{i}. {book.get('title', 'Без названия')}")
            
            elif choice == "2":
                file_path = input("Введите путь к PDF файлу: ").strip()
                if os.path.exists(file_path):
                    result = library.upload_book(file_path)
                    if "success" in result and result["success"]:
                        print(f"✅ Книга загружена: {result.get('title')}")
                    else:
                        print(f"❌ Ошибка: {result.get('error', 'неизвестно')}")
                else:
                    print("❌ Файл не найден")
            
            elif choice == "3":
                query = input("Введите запрос для полного анализа: ").strip()
                if query:
                    result = library.process_user_query(query)
                    if "final_result" in result:
                        final = result["final_result"]
                        print(f"\n📋 ОТВЕТ:")
                        print(final.get("executive_summary", ""))
                        print(f"\n⏱️ Время обработки: {result['system_metadata']['processing_time_seconds']:.2f} сек")
                    else:
                        print("❌ Не удалось получить ответ")
            
            elif choice == "4":
                status = library.get_system_status()
                print("\n📊 СТАТУС СИСТЕМЫ:")
                print(f"Книг в базе: {status['database']['books_count']}")
                print(f"Резюме создано: {status['database']['summaries_count']}")
                print(f"Тегов сгенерировано: {status['database']['tags_count']}")
                print(f"Все агенты: {'активны' if all(a == 'active' for a in status['agents'].values()) else 'есть проблемы'}")
            
            elif choice == "5":
                print("👋 Выход из системы...")
                break
            
            else:
                print("❌ Неверный выбор")
    
    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("Проверьте настройки в .env файле")


if __name__ == "__main__":
    main()