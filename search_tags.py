import json
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TagSearch:
    """
    Система поиска книг по тегам.
    Работает с Excel-таблицей тегов.
    """
    
    def __init__(self, tags_file: str = "book_tags.xlsx"):
        self.tags_file = tags_file
        self.tags_df = None
        self._load_tags()
    
    def _load_tags(self):
        """Загрузка тегов из Excel файла"""
        try:
            if os.path.exists(self.tags_file):
                self.tags_df = pd.read_excel(self.tags_file)
                logger.info(f"Загружено {len(self.tags_df)} записей тегов")
            else:
                logger.warning(f"Файл тегов {self.tags_file} не найден. Создаю пустую таблицу.")
                self.tags_df = pd.DataFrame(columns=[
                    'book_id', 'title', 'file_path', 'upload_date',
                    'academic_subjects', 'genres', 'audience', 
                    'complexity_levels', 'time_periods', 'formats',
                    'all_tags', 'tag_confidence'
                ])
        except Exception as e:
            logger.error(f"Ошибка загрузки тегов: {e}")
            self.tags_df = pd.DataFrame()
    
    def search_by_tags(self, tags: List[str], operator: str = "OR", 
                      min_confidence: float = 0.3) -> List[Dict]:
        """
        Поиск книг по тегам.
        
        Args:
            tags: Список тегов для поиска
            operator: "OR" (хотя бы один тег) или "AND" (все теги)
            min_confidence: Минимальная уверенность тега
            
        Returns:
            Список найденных книг
        """
        if self.tags_df.empty:
            return []
        
        results = []
        
        for _, row in self.tags_df.iterrows():
            book_tags = self._parse_tags(row.get('all_tags', ''))
            tag_confidences = self._parse_confidences(row.get('tag_confidence', ''))
            
            matches = 0
            matched_tags = []
            
            for search_tag in tags:
                for book_tag, confidence in zip(book_tags, tag_confidences):
                    if search_tag.lower() in book_tag.lower() and confidence >= min_confidence:
                        matches += 1
                        matched_tags.append(book_tag)
                        break
            
            # Проверка условия поиска
            if operator == "OR" and matches > 0:
                results.append(self._create_result(row, matched_tags, matches))
            elif operator == "AND" and matches == len(tags):
                results.append(self._create_result(row, matched_tags, matches))
        
        # Сортировка по релевантности
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def search_by_category(self, category: str, value: str, 
                          min_confidence: float = 0.3) -> List[Dict]:
        """
        Поиск по конкретной категории тегов.
        """
        if self.tags_df.empty or category not in self.tags_df.columns:
            return []
        
        results = []
        
        for _, row in self.tags_df.iterrows():
            category_tags = self._parse_tags(row.get(category, ''))
            
            for tag in category_tags:
                if value.lower() in tag.lower():
                    results.append(self._create_result(row, [tag], 1))
                    break
        
        return results
    
    def search_similar(self, book_id: str, top_n: int = 5) -> List[Dict]:
        """
        Поиск похожих книг.
        """
        if self.tags_df.empty:
            return []
        
        # Находим целевую книгу
        target_book = self.tags_df[self.tags_df['book_id'] == book_id]
        if target_book.empty:
            return []
        
        target_tags = set(self._parse_tags(target_book.iloc[0].get('all_tags', '')))
        
        similarities = []
        
        for _, row in self.tags_df.iterrows():
            if row['book_id'] == book_id:
                continue
            
            book_tags = set(self._parse_tags(row.get('all_tags', '')))
            
            # Вычисление сходства (коэффициент Жаккара)
            if target_tags and book_tags:
                intersection = len(target_tags.intersection(book_tags))
                union = len(target_tags.union(book_tags))
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0:
                    similarities.append((row, similarity))
        
        # Сортировка и выбор топ-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for row, similarity in similarities[:top_n]:
            result = self._create_result(row, [], 0)
            result['similarity_score'] = similarity
            results.append(result)
        
        return results
    
    def add_book_tags(self, book_data: Dict):
        """
        Добавление тегов новой книги.
        """
        new_row = {
            'book_id': book_data.get('book_id', f"book_{datetime.now().timestamp()}"),
            'title': book_data.get('title', ''),
            'file_path': book_data.get('file_path', ''),
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'academic_subjects': '|'.join(book_data.get('academic_subjects', [])),
            'genres': '|'.join(book_data.get('genres', [])),
            'audience': '|'.join(book_data.get('audience', [])),
            'complexity_levels': '|'.join(book_data.get('complexity_levels', [])),
            'time_periods': '|'.join(book_data.get('time_periods', [])),
            'formats': '|'.join(book_data.get('formats', [])),
            'all_tags': '|'.join(book_data.get('all_tags', [])),
            'tag_confidence': '|'.join(map(str, book_data.get('tag_confidence', [])))
        }
        
        self.tags_df = pd.concat([self.tags_df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_tags()
        
        logger.info(f"Добавлены теги для книги: {book_data.get('title')}")
    
    def _parse_tags(self, tags_str: str) -> List[str]:
        """Парсинг строки тегов"""
        if pd.isna(tags_str):
            return []
        return [tag.strip() for tag in str(tags_str).split('|') if tag.strip()]
    
    def _parse_confidences(self, conf_str: str) -> List[float]:
        """Парсинг строки уверенностей"""
        if pd.isna(conf_str):
            return []
        return [float(c.strip()) for c in str(conf_str).split('|') if c.strip()]
    
    def _create_result(self, row: pd.Series, matched_tags: List[str], 
                      match_count: int) -> Dict:
        """Создание результата поиска"""
        return {
            'book_id': row.get('book_id', ''),
            'title': row.get('title', ''),
            'file_path': row.get('file_path', ''),
            'upload_date': row.get('upload_date', ''),
            'matched_tags': matched_tags,
            'match_count': match_count,
            'relevance_score': match_count / (len(matched_tags) + 1) if matched_tags else 0,
            'all_categories': {
                'academic_subjects': self._parse_tags(row.get('academic_subjects', '')),
                'genres': self._parse_tags(row.get('genres', '')),
                'audience': self._parse_tags(row.get('audience', '')),
                'complexity_levels': self._parse_tags(row.get('complexity_levels', '')),
                'time_periods': self._parse_tags(row.get('time_periods', '')),
                'formats': self._parse_tags(row.get('formats', ''))
            }
        }
    
    def _save_tags(self):
        """Сохранение тегов в Excel"""
        try:
            self.tags_df.to_excel(self.tags_file, index=False)
            logger.info(f"Теги сохранены в {self.tags_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения тегов: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Статистика базы тегов"""
        if self.tags_df.empty:
            return {"total_books": 0, "total_tags": 0}
        
        total_tags = sum(len(self._parse_tags(tags)) for tags in self.tags_df['all_tags'])
        
        return {
            "total_books": len(self.tags_df),
            "total_tags": total_tags,
            "unique_tags": len(set(tag for tags in self.tags_df['all_tags'] 
                                  for tag in self._parse_tags(tags))),
            "categories_coverage": {
                col: self.tags_df[col].notna().sum() 
                for col in ['academic_subjects', 'genres', 'audience', 
                           'complexity_levels', 'time_periods', 'formats']
            }
        }
    
    def export_to_json(self, output_file: str = "tags_export.json"):
        """Экспорт тегов в JSON"""
        try:
            data = self.tags_df.to_dict('records')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Теги экспортированы в {output_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}")
            return False


# Пример использования
if __name__ == "__main__":
    # Создание поисковой системы
    tag_search = TagSearch()
    
    # Поиск по тегам
    results = tag_search.search_by_tags(
        tags=["математика", "алгебра"],
        operator="OR"
    )
    
    print(f"Найдено книг: {len(results)}")
    for i, book in enumerate(results[:3], 1):
        print(f"{i}. {book['title']} (совпадений: {book['match_count']})")
        print(f"   Теги: {', '.join(book['matched_tags'][:3])}")
        print()
    
    # Статистика
    stats = tag_search.get_statistics()
    print("СТАТИСТИКА:")
    print(f"Книг в базе: {stats['total_books']}")
    print(f"Всего тегов: {stats['total_tags']}")