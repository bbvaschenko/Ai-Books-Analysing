import json
import pandas as pd
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import openpyxl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Менеджер данных для хранения и управления информацией о книгах.
    Работает с Excel (теги) и текстовыми файлами (резюме).
    """
    
    def __init__(self, base_dir: str = "uploads"):
        self.base_dir = base_dir
        self.tags_file = os.path.join(base_dir, "book_tags.xlsx")
        self.summaries_dir = os.path.join(base_dir, "summaries")
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        
        # Создание директорий
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        
        # Инициализация файлов
        self._init_files()
    
    def _init_files(self):
        """Инициализация файлов данных"""
        # Инициализация Excel файла для тегов
        if not os.path.exists(self.tags_file):
            self._create_tags_template()
        
        # Инициализация метаданных
        if not os.path.exists(self.metadata_file):
            self._create_metadata_template()
    
    def _create_tags_template(self):
        """Создание шаблона Excel файла для тегов"""
        df = pd.DataFrame(columns=[
            'book_id', 'title', 'original_filename', 'file_path',
            'upload_timestamp', 'file_size', 'page_count',
            'academic_subjects', 'genres', 'audience', 
            'complexity_level', 'time_period', 'format',
            'language', 'keywords', 'tags_confidence',
            'summary_file', 'processing_status', 'notes'
        ])
        
        df.to_excel(self.tags_file, index=False)
        logger.info(f"Создан шаблон файла тегов: {self.tags_file}")
    
    def _create_metadata_template(self):
        """Создание шаблона метаданных"""
        metadata = {
            "system_info": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Метаданные системы управления книгами"
            },
            "statistics": {
                "total_books": 0,
                "total_tags": 0,
                "total_summaries": 0,
                "last_update": datetime.now().isoformat()
            },
            "categories": {
                "academic_subjects": [],
                "genres": [],
                "audience_levels": [],
                "complexity_levels": [],
                "formats": []
            },
            "processing_history": []
        }
        
        self.save_metadata(metadata)
    
    def save_book_tags(self, book_data: Dict[str, Any]) -> bool:
        """
        Сохранение тегов книги в Excel.
        
        Args:
            book_data: Данные о книге и тегах
            
        Returns:
            Успех операции
        """
        try:
            # Загрузка существующих данных
            df = pd.read_excel(self.tags_file)
            
            # Создание новой записи
            new_record = {
                'book_id': book_data.get('book_id', f"book_{datetime.now().timestamp()}"),
                'title': book_data.get('title', ''),
                'original_filename': book_data.get('original_filename', ''),
                'file_path': book_data.get('file_path', ''),
                'upload_timestamp': datetime.now().isoformat(),
                'file_size': book_data.get('file_size', 0),
                'page_count': book_data.get('page_count', 0),
                'academic_subjects': '|'.join(book_data.get('academic_subjects', [])),
                'genres': '|'.join(book_data.get('genres', [])),
                'audience': '|'.join(book_data.get('audience', [])),
                'complexity_level': book_data.get('complexity_level', ''),
                'time_period': book_data.get('time_period', ''),
                'format': book_data.get('format', ''),
                'language': book_data.get('language', ''),
                'keywords': '|'.join(book_data.get('keywords', [])),
                'tags_confidence': book_data.get('tags_confidence', 0),
                'summary_file': book_data.get('summary_file', ''),
                'processing_status': 'completed',
                'notes': book_data.get('notes', '')
            }
            
            # Добавление записи
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            
            # Сохранение
            df.to_excel(self.tags_file, index=False)
            
            logger.info(f"Теги сохранены для книги: {book_data.get('title')}")
            self._update_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения тегов: {e}")
            return False
    
    def save_summary(self, summary_data: Dict[str, Any], book_id: str) -> str:
        """
        Сохранение резюме в текстовый файл.
        
        Args:
            summary_data: Данные резюме
            book_id: ID книги
            
        Returns:
            Путь к сохраненному файлу
        """
        try:
            # Создание имени файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{book_id}_{timestamp}.txt"
            filepath = os.path.join(self.summaries_dir, filename)
            
            # Форматирование содержимого
            content = self._format_summary_content(summary_data)
            
            # Сохранение
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Резюме сохранено: {filepath}")
            
            # Обновление статистики
            self._update_statistics()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Ошибка сохранения резюме: {e}")
            return ""
    
    def _format_summary_content(self, summary_data: Dict[str, Any]) -> str:
        """Форматирование содержимого резюме"""
        content = []
        
        # Заголовок
        if 'title' in summary_data:
            content.append(f"РЕЗЮМЕ: {summary_data['title']}")
            content.append("=" * 80)
        
        # Основной текст
        if 'summary_text' in summary_data:
            content.append("\nОСНОВНОЕ СОДЕРЖАНИЕ:")
            content.append(summary_data['summary_text'])
        
        # Ключевые идеи
        if 'key_ideas' in summary_data:
            content.append("\n\nКЛЮЧЕВЫЕ ИДЕИ:")
            for i, idea in enumerate(summary_data['key_ideas'], 1):
                content.append(f"{i}. {idea.get('idea', '')}")
                if 'importance' in idea:
                    content.append(f"   Важность: {idea['importance']}")
        
        # TLDR
        if 'tldr' in summary_data:
            content.append("\n\nКРАТКОЕ РЕЗЮМЕ (TLDR):")
            content.append(summary_data['tldr'])
        
        # Метаданные
        content.append("\n\n" + "=" * 80)
        content.append("МЕТАДАННЫЕ:")
        content.append(f"Создано: {datetime.now().isoformat()}")
        if 'summary_type' in summary_data:
            content.append(f"Тип резюме: {summary_data['summary_type']}")
        if 'content_length' in summary_data:
            content.append(f"Длина исходного текста: {summary_data['content_length']} символов")
        
        return '\n'.join(content)
    
    def get_book_by_id(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Получение информации о книге по ID"""
        try:
            df = pd.read_excel(self.tags_file)
            book_row = df[df['book_id'] == book_id]
            
            if book_row.empty:
                return None
            
            return book_row.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Ошибка получения книги: {e}")
            return None
    
    def search_books(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Поиск книг по критериям.
        
        Args:
            search_criteria: Критерии поиска
            
        Returns:
            Список найденных книг
        """
        try:
            df = pd.read_excel(self.tags_file)
            results = []
            
            for _, row in df.iterrows():
                if self._matches_criteria(row.to_dict(), search_criteria):
                    results.append(row.to_dict())
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска книг: {e}")
            return []
    
    def _matches_criteria(self, book: Dict, criteria: Dict) -> bool:
        """Проверка соответствия книги критериям поиска"""
        for key, value in criteria.items():
            if key not in book:
                continue
            
            book_value = book[key]
            
            # Обработка строк с разделителями
            if isinstance(book_value, str) and '|' in book_value:
                book_values = book_value.split('|')
                if isinstance(value, list):
                    if not any(v in book_values for v in value):
                        return False
                else:
                    if value not in book_values:
                        return False
            else:
                if book_value != value:
                    return False
        
        return True
    
    def get_all_books(self) -> List[Dict[str, Any]]:
        """Получение всех книг"""
        try:
            df = pd.read_excel(self.tags_file)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Ошибка получения всех книг: {e}")
            return []
    
    def get_summary(self, book_id: str) -> Optional[str]:
        """Получение резюме книги"""
        try:
            book = self.get_book_by_id(book_id)
            if not book or 'summary_file' not in book:
                return None
            
            summary_file = book['summary_file']
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    return f.read()
            
            return None
        except Exception as e:
            logger.error(f"Ошибка получения резюме: {e}")
            return None
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Сохранение метаданных системы"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных: {e}")
    
    def load_metadata(self) -> Dict[str, Any]:
        """Загрузка метаданных системы"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки метаданных: {e}")
            return {}
    
    def _update_statistics(self):
        """Обновление статистики"""
        try:
            metadata = self.load_metadata()
            
            # Подсчет книг
            df = pd.read_excel(self.tags_file)
            total_books = len(df)
            
            # Подсчет резюме
            summary_files = [f for f in os.listdir(self.summaries_dir) 
                           if f.endswith('.txt')]
            total_summaries = len(summary_files)
            
            # Подсчет тегов
            total_tags = 0
            for _, row in df.iterrows():
                for col in ['academic_subjects', 'genres', 'keywords']:
                    if isinstance(row[col], str):
                        total_tags += len(row[col].split('|'))
            
            metadata['statistics'] = {
                "total_books": total_books,
                "total_tags": total_tags,
                "total_summaries": total_summaries,
                "last_update": datetime.now().isoformat()
            }
            
            # Добавление в историю
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "books_count": total_books,
                "summaries_count": total_summaries,
                "tags_count": total_tags
            }
            metadata['processing_history'].append(history_entry)
            
            # Ограничение истории
            if len(metadata['processing_history']) > 100:
                metadata['processing_history'] = metadata['processing_history'][-100:]
            
            self.save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики"""
        metadata = self.load_metadata()
        return metadata.get('statistics', {})
    
    def export_data(self, output_dir: str) -> Dict[str, str]:
        """
        Экспорт всех данных.
        
        Returns:
            Словарь с путями к экспортированным файлам
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Экспорт тегов
            tags_export = os.path.join(output_dir, f"tags_export_{timestamp}.xlsx")
            df = pd.read_excel(self.tags_file)
            df.to_excel(tags_export, index=False)
            
            # Экспорт метаданных
            metadata_export = os.path.join(output_dir, f"metadata_export_{timestamp}.json")
            metadata = self.load_metadata()
            with open(metadata_export, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Экспорт резюме
            summaries_export = os.path.join(output_dir, f"summaries_export_{timestamp}.zip")
            # Здесь можно добавить архивирование
            
            logger.info(f"Данные экспортированы в {output_dir}")
            
            return {
                "tags": tags_export,
                "metadata": metadata_export,
                "summaries_dir": self.summaries_dir
            }
            
        except Exception as e:
            logger.error(f"Ошибка экспорта данных: {e}")
            return {}


# Пример использования
if __name__ == "__main__":
    # Создание менеджера данных
    manager = DataManager("test_uploads")
    
    # Пример сохранения книги
    book_data = {
        'book_id': 'test_book_001',
        'title': 'Тестовая книга по Python',
        'original_filename': 'python_book.pdf',
        'file_path': '/uploads/python_book.pdf',
        'academic_subjects': ['программирование', 'математика'],
        'genres': ['учебная литература', 'программирование'],
        'audience': ['студенты', 'начинающие'],
        'complexity_level': 'начальный',
        'keywords': ['python', 'программирование', 'обучение'],
        'tags_confidence': 0.85
    }
    
    # Сохранение тегов
    success = manager.save_book_tags(book_data)
    print(f"Теги сохранены: {success}")
    
    # Пример сохранения резюме
    summary_data = {
        'title': 'Тестовая книга по Python',
        'summary_text': 'Эта книга представляет собой введение в программирование на Python...',
        'key_ideas': [
            {'idea': 'Основы синтаксиса Python', 'importance': 'высокая'},
            {'idea': 'Работа с данными', 'importance': 'средняя'}
        ],
        'tldr': 'Книга для начинающих программистов на Python.'
    }
    
    summary_path = manager.save_summary(summary_data, 'test_book_001')
    print(f"Резюме сохранено: {summary_path}")
    
    # Получение статистики
    stats = manager.get_statistics()
    print(f"Статистика: {stats}")
    
    # Поиск книг
    search_results = manager.search_books({
        'academic_subjects': 'программирование',
        'complexity_level': 'начальный'
    })
    print(f"Найдено книг: {len(search_results)}")