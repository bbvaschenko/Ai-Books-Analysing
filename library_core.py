"""
Основной модуль для работы с системой агентов
"""
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from autonomous_classifier import AutonomousEducationalClassifier

# Импорты для системы агентов
from agent_system import SyncAgentSystem
AGENTS_AVAILABLE = True

from dotenv import load_dotenv
load_dotenv()


@dataclass
class BookData:
    """Структура для хранения данных о книге"""
    book_number: int
    book_id: str
    filename: str
    area: str
    tags: Dict[str, List[str]]
    text: Optional[str] = None


class BookAnalyzer:
    """Класс для анализа книг с системой агентов"""

    def __init__(self, excel_file: str = "analyzed_books.xlsx", use_agents: bool = True):
        # В этой версии use_agents всегда True
        self.use_agents = True

        self.excel_file = os.path.abspath(excel_file)
        print(f"📁 Файл для сохранения данных: {self.excel_file}")

        # Инициализация классификатора
        print("Инициализация классификатора учебной литературы...")
        self.educational_classifier = AutonomousEducationalClassifier()
        print("✅ Классификатор готов к работе")

        # Инициализация системы агентов
        print("🤖 Инициализация системы агентов...")
        try:
            self.agent_system = SyncAgentSystem(self)
            print("✅ Система из 5 агентов готова к работе")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации системы агентов: {e}")
            self.use_agents = False
            self.agent_system = None
            print(f"⚠️  Система агентов будет работать в ограниченном режиме")

        # Проверяем и создаем файл Excel если его нет
        self._ensure_excel_file()

    def _ensure_excel_file(self):
        """Создает файл Excel с правильной структурой если его нет"""
        if not os.path.exists(self.excel_file):
            print(f"📄 Создаю новый файл Excel: {self.excel_file}")

            df = pd.DataFrame(columns=[
                "Номер книги", "ID книги", "Имя файла", "Область знаний", "Текст (фрагмент)",
                "Разделы", "Предметы", "Классы", "Авторы", "Темы"
            ])

            df.to_excel(self.excel_file, index=False)
            print(f"✅ Файл Excel создан успешно")
        else:
            print(f"✅ Файл Excel уже существует: {self.excel_file}")

    def get_next_book_number(self) -> int:
        """Получение следующего номера книги"""
        if not os.path.exists(self.excel_file):
            return 1

        try:
            df = pd.read_excel(self.excel_file)
            if df.empty or 'Номер книги' not in df.columns:
                return 1

            max_number = df['Номер книги'].max()
            if pd.isna(max_number):
                return 1
            return int(max_number) + 1
        except Exception as e:
            print(f"⚠️ Ошибка при чтении номера книги: {e}")
            return 1

    def analyze_book(self, pdf_path: str) -> Optional[BookData]:
        """Анализ книги из PDF файла"""
        if not os.path.exists(pdf_path):
            print(f"❌ Файл не найден: {pdf_path}")
            return None

        # Извлечение текста с помощью PyMuPDF
        import fitz
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"❌ Ошибка при чтении PDF: {e}")
            return None

        if len(text) < 200:
            print("❌ Недостаточно текста для анализа.")
            return None

        # Проверка на учебную литературу
        print("\n🔍 Проверка на учебную литературу...")
        check_result = self.educational_classifier.check_if_educational(text)

        print(f"   Результат: {'✅ УЧЕБНАЯ' if check_result['is_educational'] else '❌ НЕ учебная'}")
        print(f"   Уверенность: {check_result['confidence']:.0%}")

        if not check_result['is_educational']:
            print(f"\n❌ Файл не является учебной литературой!")
            return None

        print(f"\n✅ Проверка пройдена успешно!")

        # Здесь можно добавить анализ через систему агентов
        # Пока сохраняем базовую информацию

        book_number = self.get_next_book_number()
        book_id = f"{book_number:04d}"

        # Для простоты используем базовые теги
        found_tags = {
            "предметы": ["математика"],
            "классы": ["университетский"],
            "темы": ["учебный материал"]
        }

        book_data = BookData(
            book_number=book_number,
            book_id=book_id,
            filename=os.path.basename(pdf_path),
            area="математика",
            tags=found_tags,
            text=text[:500] + "..." if len(text) > 500 else text
        )

        # Сохраняем данные в Excel
        try:
            self.save_to_database(book_data)
        except Exception as e:
            print(f"❌ Ошибка при сохранении в Excel: {e}")
            return None

        return book_data

    def save_to_database(self, book_data: BookData):
        """Сохранение данных книги в Excel"""
        print("\n💾 Сохранение данных в Excel...")

        book_dict = {
            "Номер книги": book_data.book_number,
            "ID книги": book_data.book_id,
            "Имя файла": book_data.filename,
            "Область знаний": book_data.area,
            "Текст (фрагмент)": book_data.text
        }

        for category, tags in book_data.tags.items():
            column_name = category.capitalize()
            book_dict[column_name] = ', '.join(tags) if tags else ""

        new_row_df = pd.DataFrame([book_dict])

        try:
            if os.path.exists(self.excel_file):
                existing_df = pd.read_excel(self.excel_file)

                if 'ID книги' in existing_df.columns:
                    if book_data.book_id in existing_df['ID книги'].values:
                        print(f"⚠️  Книга с ID {book_data.book_id} уже существует")
                        mask = existing_df['ID книги'] == book_data.book_id
                        existing_df.loc[mask, list(book_dict.keys())] = pd.Series(book_dict)
                        df = existing_df
                    else:
                        df = pd.concat([existing_df, new_row_df], ignore_index=True)
                else:
                    df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                df = new_row_df

            df.to_excel(self.excel_file, index=False)
            print(f"✅ Данные сохранены в файл: {self.excel_file}")
            print(f"📊 Всего записей в базе: {len(df)}")

        except Exception as e:
            print(f"❌ Ошибка при сохранении в Excel: {e}")
            raise

    def search_books_with_agents(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Поиск книг с использованием системы агентов"""
        if not self.use_agents or not self.agent_system:
            print("❌ Система агентов недоступна, используем базовый поиск")
            return {
                "results": [],
                "method": "basic_search",
                "error": "Система агентов не инициализирована"
            }

        print(f"🔍 Поиск через систему агентов: '{query}'")

        try:
            result = self.agent_system.process_query(query, context or {})

            # Форматируем результат
            formatted_results = self._format_agent_results(result)

            return {
                "results": formatted_results,
                "agent_system_result": result,
                "method": "agent_system",
                "conversation_id": result.get("conversation_id")
            }

        except Exception as e:
            print(f"❌ Ошибка системы агентов: {e}")
            return {
                "results": [],
                "method": "error",
                "error": str(e)
            }

    def _format_agent_results(self, agent_result: Dict) -> List[Dict]:
        """Форматирование результатов системы агентов"""
        recommendations = agent_result.get("recommendations", {})
        top_recs = recommendations.get("top_recommendations", [])

        formatted = []

        for rec in top_recs:
            formatted.append({
                "book_id": rec.get("id", f"agent_rec_{len(formatted)}"),
                "filename": rec.get("name", "Рекомендация системы"),
                "area": rec.get("area", "не определено"),
                "score": rec.get("relevance_score", 0.5),
                "matching_tags": rec.get("details", {}).get("key_points", []),
                "explanation": rec.get("explanation", "Рекомендовано системой агентов")
            })

        return formatted