"""
Основной модуль для обработки PDF и работы с эмбеддингами
"""
import re
import os
import torch
import torch.nn.functional as F
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Импортируем AI агента
from ai_agent import EducationalAIAgent, BookAnalysis as AIBookAnalysis


@dataclass
class BookData:
    """Структура для хранения данных о книге"""
    book_number: int
    book_id: str
    filename: str
    area: str
    tags: Dict[str, List[str]]
    text: Optional[str] = None
    embedding: Optional[torch.Tensor] = None
    ai_analysis: Optional[AIBookAnalysis] = None  # Добавляем поле для AI анализа


class AutonomousEducationalClassifier:
    """Автономный классификатор учебной литературы"""

    def __init__(self):
        """Инициализация автономного классификатора"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Извлечение признаков из текста"""
        features = {
            'text_length': len(text),
            'paragraph_count': len(re.split(r'\n\s*\n', text)),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_sentence_length': 0,
            'vocabulary_richness': 0
        }

        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = len(text) / features['sentence_count']

        # Анализ богатства словарного запаса
        words = re.findall(r'\b[а-яА-ЯёЁ]{3,}\b', text.lower())
        if words:
            unique_words = set(words)
            features['vocabulary_richness'] = len(unique_words) / len(words) if len(words) > 0 else 0

        return features

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Анализ структуры текста"""
        structure = {
            'has_numerical_sections': False,
            'has_definitions': False,
            'has_examples': False,
            'has_exercises': False,
            'has_references': False,
            'has_tables_figures': False,
            'section_hierarchy_depth': 0
        }

        # Анализ структуры по заголовкам
        headings = re.findall(r'(?:Глава|Раздел|§|Тема|Параграф|Часть)\s+[^\n]+', text)
        if headings:
            structure['has_numerical_sections'] = True
            structure['section_hierarchy_depth'] = min(3, len(headings) // 2)

        # Поиск определений
        definition_patterns = [
            r'Определение\s*[0-9]*[:.]?\s*[^\n]+',
            r'\bопределим\b.*?как\b',
            r'\bназывается\b.*?\bесли\b'
        ]
        for pattern in definition_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure['has_definitions'] = True
                break

        # Поиск примеров
        example_patterns = [
            r'Пример\s*[0-9]*[:.]',
            r'Рассмотрим\s+пример',
            r'В\s+качестве\s+примера'
        ]
        for pattern in example_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure['has_examples'] = True
                break

        # Поиск упражнений
        exercise_patterns = [
            r'Задача\s*[0-9]*[:.]',
            r'Упражнение\s*[0-9]*[:.]',
            r'Контрольный\s+вопрос',
            r'Самостоятельная\s+работа'
        ]
        for pattern in exercise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure['has_exercises'] = True
                break

        # Поиск ссылок
        reference_patterns = [
            r'\[[0-9]+\]',
            r'\([А-Яа-я]+\s*,\s*\d{4}\)',
            r'Список\s+литературы',
            r'Библиография'
        ]
        for pattern in reference_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure['has_references'] = True
                break

        # Поиск таблиц и рисунков
        table_figure_patterns = [
            r'Таблица\s*[0-9]*',
            r'Рис\.\s*[0-9]*',
            r'Схема\s*[0-9]*',
            r'График\s*[0-9]*'
        ]
        for pattern in table_figure_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure['has_tables_figures'] = True
                break

        return structure

    def _analyze_mathematical_content(self, text: str) -> Dict[str, Any]:
        """Анализ математического содержания"""
        math_analysis = {
            'has_formulas': False,
            'has_equations': False,
            'has_proofs': False,
            'has_theorems': False,
            'formula_density': 0,
            'math_keyword_count': 0
        }

        # Математические ключевые слова
        math_keywords = [
            'уравнение', 'формула', 'теорема', 'доказательство', 'решение',
            'вычислить', 'рассчитать', 'функция', 'производная', 'интеграл',
            'матрица', 'вектор', 'вероятность', 'статистика', 'алгоритм'
        ]

        # Подсчет математических ключевых слов
        text_lower = text.lower()
        math_analysis['math_keyword_count'] = sum(
            1 for keyword in math_keywords if keyword in text_lower
        )

        # Поиск формул и уравнений
        formula_patterns = [
            r'\$[^$]+\$',  # LaTeX
            r'\\[(\[]?[^\\]*?\\[\])]?',  # Математические выражения
            r'[A-Za-zА-Яа-яα-ωΑ-Ω]+\s*=\s*[^=\n]{3,}',  # Равенства с содержанием
            r'\b\w+\s*[+\-*/^=<>≤≥≠]\s*\w+\b',  # Математические операции
        ]

        formula_count = 0
        for pattern in formula_patterns:
            matches = re.findall(pattern, text)
            formula_count += len(matches)
            if matches:
                math_analysis['has_formulas'] = True
                if '=' in pattern or '≠' in pattern or '≤' in pattern or '≥' in pattern:
                    math_analysis['has_equations'] = True

        # Рассчитываем плотность формул
        if len(text) > 0:
            math_analysis['formula_density'] = formula_count / (len(text) / 1000)

        # Поиск доказательств и теорем
        proof_theorem_patterns = [
            r'Теорема\s*[0-9]*[:.]',
            r'Доказательство\.',
            r'Лемма\s*[0-9]*[:.]',
            r'Следствие\s*[0-9]*[:.]',
            r'докажем\b', r'доказать\b'
        ]

        for pattern in proof_theorem_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if 'теорема' in pattern.lower() or 'лемма' in pattern.lower() or 'следствие' in pattern.lower():
                    math_analysis['has_theorems'] = True
                if 'доказа' in pattern.lower():
                    math_analysis['has_proofs'] = True

        return math_analysis

    def check_if_educational(self, text: str) -> Dict[str, Any]:
        """Проверка, является ли текст учебной литературой"""
        # Для быстрой проверки анализируем только часть текста
        if len(text) > 2000:
            analysis_text = text[:1500]
        else:
            analysis_text = text

        # Извлекаем признаки
        features = self._extract_text_features(analysis_text)
        structure = self._analyze_text_structure(analysis_text)
        math_content = self._analyze_mathematical_content(analysis_text)

        # Рассчитываем общий скоринг
        structural_score = (
            (1.0 if structure['has_numerical_sections'] else 0) * 0.25 +
            (1.0 if structure['has_definitions'] else 0) * 0.20 +
            (1.0 if structure['has_examples'] else 0) * 0.15 +
            (1.0 if structure['has_exercises'] else 0) * 0.20 +
            (1.0 if structure['has_references'] else 0) * 0.10 +
            (1.0 if structure['has_tables_figures'] else 0) * 0.10
        )

        mathematical_score = (
            (1.0 if math_content['has_formulas'] else 0) * 0.30 +
            (1.0 if math_content['has_equations'] else 0) * 0.25 +
            (1.0 if math_content['has_proofs'] else 0) * 0.15 +
            (1.0 if math_content['has_theorems'] else 0) * 0.15 +
            min(1.0, math_content['formula_density'] / 5.0) * 0.15
        )

        formal_score = min(1.0, features['vocabulary_richness'] * 1.5)

        # Взвешенная итоговая оценка
        total_score = (
            structural_score * 0.40 +
            mathematical_score * 0.35 +
            formal_score * 0.25
        )

        # Определяем, является ли учебной литературой
        is_educational = total_score >= 0.5

        # Генерируем детали
        details = []
        if structure['has_numerical_sections']:
            details.append("имеет структурированные разделы")
        if structure['has_definitions']:
            details.append("содержит определения терминов")
        if structure['has_exercises']:
            details.append("включает упражнения и задачи")
        if math_content['has_formulas']:
            details.append("содержит математические формулы")

        if len(details) == 0:
            details.append("не обнаружено характерных признаков учебника")

        return {
            'is_educational': is_educational,
            'confidence': round(total_score, 2),
            'total_score': round(total_score, 2),
            'structural_score': round(structural_score, 2),
            'mathematical_score': round(mathematical_score, 2),
            'formal_score': round(formal_score, 2),
            'details': details[:3],
            'recommendation': '✅ Рекомендуется для библиотеки' if is_educational else '❌ Не является учебной литературой'
        }


class PDFProcessor:
    """Класс для обработки PDF файлов"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста от лишних пробелов"""
        return re.sub(r'\s+', ' ', text.strip())

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Извлечение текста из PDF файла"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return PDFProcessor.clean_text(text)
        except Exception as e:
            raise Exception(f"Ошибка при чтении PDF: {str(e)}")

    @staticmethod
    def validate_text(text: str, min_length: int = 200) -> bool:
        """Проверка достаточности текста для анализа"""
        return text and len(text.strip()) >= min_length


class EmbeddingModel:
    """Класс для работы с моделями эмбеддингов"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        """Инициализация модели для эмбеддингов"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Для поиска по тегам
        self.search_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def get_text_embedding(self, text: str, chunk_size: int = 512) -> torch.Tensor:
        """Получение эмбеддинга для текста"""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu())

        return torch.mean(torch.stack(embeddings), dim=0)

    def get_tag_embeddings(self, tags: List[str]) -> torch.Tensor:
        """Получение эмбеддингов для списка тегов"""
        tag_embeddings = []

        for tag in tags:
            inputs = self.tokenizer(
                tag,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=10
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            tag_embedding = outputs.last_hidden_state.mean(dim=1)
            tag_embeddings.append(tag_embedding.cpu())

        return torch.cat(tag_embeddings)

    def find_similar_tags(self, query: str, all_tags: List[str], top_k: int = 8) -> List[str]:
        """Поиск тегов, похожих на запрос"""
        query = query.lower()
        query = re.sub(r'[^\w\sа-яё-]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()

        if len(query) < 3:
            return []

        # Получаем эмбеддинг запроса
        query_embedding = self.search_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Получаем эмбеддинги всех тегов
        tag_embeddings = []
        for tag in all_tags:
            tag_emb = self.search_model.encode(
                tag,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            tag_embeddings.append(tag_emb)

        tag_embeddings = np.array(tag_embeddings)

        # Вычисляем схожесть
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            tag_embeddings
        )[0]

        # Сортируем по убыванию схожести
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        return [all_tags[i] for i in sorted_indices if similarities[i] > 0.3]


class TagManager:
    """Менеджер для работы с тегами"""

    def __init__(self, tags_directory: str = "tags"):
        self.tags_directory = tags_directory
        self.tags_dict = self.load_tags()
        self.section_to_area = self.load_area_mapping()

    def load_tags(self) -> Dict[str, List[str]]:
        """Загрузка тегов из файлов"""
        tags_dict = {}

        if not os.path.exists(self.tags_directory):
            os.makedirs(self.tags_directory)
            print(f"Создана директория {self.tags_directory}")
            return tags_dict

        for filename in os.listdir(self.tags_directory):
            if filename.endswith(".txt") and filename != "области_знаний.txt":
                category = filename[:-4]  # убираем расширение .txt
                filepath = os.path.join(self.tags_directory, filename)

                with open(filepath, 'r', encoding='utf-8') as f:
                    tags = [line.strip() for line in f if line.strip()]
                    tags_dict[category] = tags
                    print(f"Загружено {len(tags)} тегов из категории '{category}'")

        return tags_dict

    def load_area_mapping(self, mapping_file: str = "tags/области_знаний.txt") -> Dict[str, str]:
        """Загрузка соответствий разделов и областей знаний"""
        section_to_area = {}

        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        section, area = line.split(':', 1)
                        section_to_area[section.strip()] = area.strip()
            print(f"Загружено {len(section_to_area)} соответствий разделов и областей")
        else:
            print(f"Файл с соответствиями {mapping_file} не найден")

        return section_to_area

    def get_all_tags_flat(self) -> List[str]:
        """Получение всех тегов в одном списке"""
        all_tags = []
        for tags_list in self.tags_dict.values():
            all_tags.extend(tags_list)
        return all_tags

    def infer_area_from_sections(self, section_tags_found: List[str]) -> List[str]:
        """Определение области знаний на основе найденных разделов"""
        area_counter = {}
        for section in section_tags_found:
            area = self.section_to_area.get(section)
            if area:
                area_counter[area] = area_counter.get(area, 0) + 1

        if area_counter:
            sorted_areas = sorted(area_counter.items(), key=lambda x: -x[1])
            return [sorted_areas[0][0]]
        return ["не определено"]


class BookAnalyzer:
    """Основной класс для анализа книг"""

    def __init__(self, excel_file: str = "analyzed_books.xlsx"):
        self.excel_file = excel_file
        self.tag_manager = TagManager()
        self.embedding_model = EmbeddingModel()
        self.pdf_processor = PDFProcessor()

        # Инициализация автономного классификатора учебной литературы
        self.educational_checker = AutonomousEducationalClassifier()

        # Инициализация AI агента
        print("Инициализация AI агента для расширенного анализа...")
        self.ai_agent = EducationalAIAgent()
        print("AI агент готов к работе")

    def get_next_book_number(self) -> int:
        """Получение следующего номера книги"""
        if not os.path.exists(self.excel_file):
            return 1

        try:
            df = pd.read_excel(self.excel_file)
            if df.empty or 'Номер книги' not in df.columns:
                return 1
            return df['Номер книги'].max() + 1
        except:
            return 1

    def find_tags_for_text(self, text_embedding: torch.Tensor,
                          tag_list: List[str], top_k: int = 3) -> List[str]:
        """Поиск тегов для текста"""
        if not tag_list:
            return []

        tag_embeddings = self.embedding_model.get_tag_embeddings(tag_list)
        similarities = F.cosine_similarity(text_embedding, tag_embeddings)
        top_indices = similarities.topk(min(top_k, len(tag_list))).indices
        return [tag_list[i] for i in top_indices]

    def analyze_book(self, pdf_path: str) -> Optional[BookData]:
        """Анализ книги из PDF файла"""
        print(f"Начинаю анализ файла: {pdf_path}")

        if not os.path.exists(pdf_path):
            print(f"Файл не найден: {pdf_path}")
            return None

        # Извлечение текста
        print("Извлечение текста из PDF...")
        raw_text = self.pdf_processor.extract_text(pdf_path)

        if not self.pdf_processor.validate_text(raw_text):
            print("Недостаточно текста для анализа.")
            return None

        # 🔍 АВТОНОМНАЯ ПРОВЕРКА НА УЧЕБНУЮ ЛИТЕРАТУРУ
        print("🔍 Автономная проверка на учебную литературу...")
        educational_check = self.educational_checker.check_if_educational(raw_text)

        print(f"   Оценка учебности: {educational_check['total_score']:.2f}")
        print(f"   Структурный скор: {educational_check['structural_score']:.2f}")
        print(f"   Математический скор: {educational_check['mathematical_score']:.2f}")
        print(f"   Формальность: {educational_check['formal_score']:.2f}")

        if not educational_check['is_educational']:
            print(f"❌ Файл не является учебной литературой!")
            print(f"   Причина: {educational_check['recommendation']}")
            print(f"   Детали: {', '.join(educational_check['details'])}")
            return None

        print(f"✅ Автономная проверка пройдена успешно")
        print(f"   Рекомендация: {educational_check['recommendation']}")

        # Генерация эмбеддингов
        print("Генерация эмбеддингов по тексту...")
        text_embedding = self.embedding_model.get_text_embedding(raw_text)

        # Анализ тегов по категориям
        print("Анализ тегов...")
        found_tags = {}

        for category, tags in self.tag_manager.tags_dict.items():
            if category == "области_знаний":
                continue

            found_tags[category] = self.find_tags_for_text(text_embedding, tags, top_k=3)
            print(f"{category}: {', '.join(found_tags[category])}")

        # Определение области знаний
        found_area = self.tag_manager.infer_area_from_sections(
            found_tags.get("разделы", [])
        )

        # AI анализ содержания книги
        print("Запуск расширенного AI анализа...")
        ai_analysis = None
        try:
            # Анализируем только часть текста для производительности
            analysis_text = raw_text[:10000]  # Первые 10000 символов
            ai_analysis = self.ai_agent.analyze_book_content(analysis_text, found_tags)
            print("AI анализ завершен успешно")

            # Выводим результаты AI анализа
            print("\n=== РЕЗУЛЬТАТЫ AI АНАЛИЗА ===")
            print(f"Краткое резюме: {ai_analysis.summary[:200]}...")
            print(f"Уровень сложности: {ai_analysis.difficulty_level}")
            print(f"Области математики: {', '.join(ai_analysis.mathematical_areas)}")
            print(f"Ключевые темы: {', '.join(ai_analysis.key_topics[:5])}")
            if ai_analysis.recommendations:
                print(f"Рекомендации: {ai_analysis.recommendations[0]}")
            print("=" * 40)

        except Exception as e:
            print(f"Ошибка при AI анализе: {str(e)}")
            print("Продолжаю без расширенного анализа...")

        # Создание данных книги
        book_number = self.get_next_book_number()
        book_id = f"{book_number:04d}"

        return BookData(
            book_number=book_number,
            book_id=book_id,
            filename=os.path.basename(pdf_path),
            area=', '.join(found_area),
            tags=found_tags,
            text=raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
            embedding=text_embedding,
            ai_analysis=ai_analysis
        )

    def save_to_database(self, book_data: BookData):
        """Сохранение данных книги в Excel"""
        # Подготовка данных для сохранения
        book_dict = {
            "Номер книги": book_data.book_number,
            "ID книги": book_data.book_id,
            "Имя файла": book_data.filename,
            "Область знаний": book_data.area,
            "Текст (фрагмент)": book_data.text
        }

        # Добавляем теги по категориям
        for category, tags in book_data.tags.items():
            book_dict[category.capitalize()] = ', '.join(tags)

        # Добавляем данные AI анализа, если они есть
        if book_data.ai_analysis:
            ai_data = book_data.ai_analysis

            # Преобразуем BookAnalysis в словарь
            ai_dict = asdict(ai_data)

            # Добавляем только основные поля
            book_dict["AI_Резюме"] = ai_dict['summary'][:500]  # Ограничиваем длину

            if ai_dict['sections_summary']:
                sections_str = ' | '.join(
                    f"{k}: {v[:100]}"
                    for k, v in list(ai_dict['sections_summary'].items())[:3]
                )
                book_dict["AI_Разделы"] = sections_str[:300]

            book_dict["AI_Ключевые_темы"] = ', '.join(ai_dict['key_topics'][:5])
            book_dict["AI_Уровень_сложности"] = ai_dict['difficulty_level']
            book_dict["AI_Предварительные_знания"] = ', '.join(ai_dict['prerequisites'][:3])
            book_dict["AI_Области_математики"] = ', '.join(ai_dict['mathematical_areas'])
            book_dict["AI_Рекомендации"] = ' | '.join(ai_dict['recommendations'][:3])

        # Сохранение в Excel
        df = pd.DataFrame([book_dict])

        if os.path.exists(self.excel_file):
            existing_df = pd.read_excel(self.excel_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        # Сохраняем в Excel
        df.to_excel(self.excel_file, index=False)
        print(f"Данные сохранены в файл: {self.excel_file}")

        # Также сохраняем расширенный отчет в отдельный файл
        if book_data.ai_analysis:
            self._save_ai_report(book_data)

    def _save_ai_report(self, book_data: BookData):
        """Сохранение расширенного отчета AI анализа"""
        report_dir = "ai_reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        report_file = os.path.join(report_dir, f"{book_data.book_id}_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"=== ОТЧЕТ AI АНАЛИЗА КНИГИ ===\n\n")
            f.write(f"ID книги: {book_data.book_id}\n")
            f.write(f"Название: {book_data.filename}\n")
            f.write(f"Область знаний: {book_data.area}\n\n")

            if book_data.ai_analysis:
                ai = book_data.ai_analysis

                f.write("КРАТКОЕ РЕЗЮМЕ:\n")
                f.write(f"{ai.summary}\n\n")

                f.write("УРОВЕНЬ СЛОЖНОСТИ:\n")
                f.write(f"{ai.difficulty_level}\n\n")

                if ai.mathematical_areas:
                    f.write("ОБЛАСТИ МАТЕМАТИКИ:\n")
                    for area in ai.mathematical_areas:
                        f.write(f"- {area}\n")
                    f.write("\n")

                if ai.key_topics:
                    f.write("КЛЮЧЕВЫЕ ТЕМЫ:\n")
                    for topic in ai.key_topics[:10]:
                        f.write(f"- {topic}\n")
                    f.write("\n")

                if ai.prerequisites:
                    f.write("НЕОБХОДИМЫЕ ПРЕДВАРИТЕЛЬНЫЕ ЗНАНИЯ:\n")
                    for prereq in ai.prerequisites:
                        f.write(f"- {prereq}\n")
                    f.write("\n")

                if ai.sections_summary:
                    f.write("АНАЛИЗ РАЗДЕЛОВ:\n")
                    for section, desc in list(ai.sections_summary.items())[:5]:
                        f.write(f"{section}:\n")
                        f.write(f"{desc[:200]}...\n\n")

                if ai.recommendations:
                    f.write("РЕКОМЕНДАЦИИ ПО ИЗУЧЕНИЮ:\n")
                    for rec in ai.recommendations:
                        f.write(f"- {rec}\n")

        print(f"Расширенный отчет сохранен: {report_file}")

    def search_books(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск книг по запросу"""
        if not os.path.exists(self.excel_file):
            return []

        try:
            df = pd.read_excel(self.excel_file)

            # Поиск по тегам
            all_tags = self.tag_manager.get_all_tags_flat()
            similar_tags = self.embedding_model.find_similar_tags(query, all_tags)

            if not similar_tags:
                return []

            # Ищем книги, содержащие похожие теги
            results = []
            for _, row in df.iterrows():
                score = 0
                book_tags = []

                # Собираем все теги книги из разных категорий
                for col in df.columns:
                    if col not in ["Номер книги", "ID книги", "Имя файла",
                                 "Область знаний", "Текст (фрагмент)"] and not col.startswith("AI_"):
                        if pd.notna(row[col]):
                            book_tags.extend(str(row[col]).split(', '))

                # Проверяем совпадения
                for tag in similar_tags:
                    if tag in book_tags:
                        score += 1

                if score > 0:
                    result = {
                        'score': score,
                        'book_id': row['ID книги'],
                        'filename': row['Имя файла'],
                        'area': row['Область знаний'],
                        'matching_tags': [t for t in similar_tags if t in book_tags]
                    }

                    # Добавляем AI данные, если они есть
                    if 'AI_Резюме' in df.columns and pd.notna(row['AI_Резюме']):
                        result['ai_summary'] = str(row['AI_Резюме'])[:150] + "..."

                    if 'AI_Уровень_сложности' in df.columns and pd.notna(row['AI_Уровень_сложности']):
                        result['ai_difficulty'] = str(row['AI_Уровень_сложности'])

                    results.append(result)

            # Сортируем по релевантности
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"Ошибка при поиске: {str(e)}")
            return []

    def get_book_details(self, book_id: str) -> Optional[Dict]:
        """Получение детальной информации о книге"""
        if not os.path.exists(self.excel_file):
            return None

        try:
            df = pd.read_excel(self.excel_file)
            book_row = df[df['ID книги'] == book_id]

            if book_row.empty:
                return None

            row = book_row.iloc[0]

            # Формируем детальную информацию
            details = {
                'book_id': row['ID книги'],
                'filename': row['Имя файла'],
                'area': row['Область знаний'],
                'tags': {}
            }

            # Собираем теги
            for col in df.columns:
                if col not in ["Номер книги", "ID книги", "Имя файла",
                             "Область знаний", "Текст (фрагмент)"] and not col.startswith("AI_"):
                    if pd.notna(row[col]):
                        details['tags'][col] = str(row[col]).split(', ')

            # Добавляем AI данные
            ai_fields = [col for col in df.columns if col.startswith("AI_")]
            if ai_fields:
                details['ai_analysis'] = {}
                for field in ai_fields:
                    if pd.notna(row[field]):
                        details['ai_analysis'][field.replace("AI_", "")] = str(row[field])

            return details

        except Exception as e:
            print(f"Ошибка при получении деталей книги: {str(e)}")
            return None