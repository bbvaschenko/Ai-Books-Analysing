"""
AI агент для автономного анализа учебной литературы
"""
import re
import torch
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np


class AutonomousEducationalClassifier:
    """Автономный классификатор учебной литературы с использованием Qwen"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        """Инициализация автономного классификатора"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device} для автономной проверки учебности")

        try:
            # Загружаем основную модель для эмбеддингов
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            # Загружаем текстовую модель для анализа (более легкую версию)
            try:
                # Пробуем загрузить текстовую модель для классификации
                self.text_model_name = "cointegrated/rubert-tiny2"
                self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
                self.text_model = AutoModel.from_pretrained(self.text_model_name).to(self.device)
                self.text_model.eval()
            except:
                print("⚠️  Не удалось загрузить текстовую модель, использую основную")
                self.text_tokenizer = self.tokenizer
                self.text_model = self.model

            print("✅ Модели загружены успешно")

        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            self.tokenizer = None
            self.model = None
            self.text_tokenizer = None
            self.text_model = None

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Автономное извлечение признаков из текста"""
        features = {
            'text_length': len(text),
            'paragraph_count': len(re.split(r'\n\s*\n', text)),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_sentence_length': 0,
            'vocabulary_richness': 0,
            'formality_score': 0,
            'structure_score': 0
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
        """Автономный анализ структуры текста"""
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
        """Автономный анализ математического содержания"""
        math_analysis = {
            'has_formulas': False,
            'has_equations': False,
            'has_proofs': False,
            'has_theorems': False,
            'formula_density': 0,
            'math_keyword_count': 0
        }

        # Математические ключевые слова (минимальный набор для инициализации)
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

    def _get_semantic_embedding(self, text: str, max_length: int = 512) -> Optional[np.ndarray]:
        """Получение семантического эмбеддинга текста"""
        if self.model is None or self.tokenizer is None:
            return None

        try:
            # Ограничиваем текст для быстрой обработки
            if len(text) > 2000:
                text = text[:2000]

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем средний пул
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()[0]

        except Exception as e:
            print(f"Ошибка при получении эмбеддинга: {e}")
            return None

    def _generate_self_learning_features(self, text: str) -> Dict[str, float]:
        """Генерация признаков через самообучение на лету"""
        # Эмбеддинг текста
        embedding = self._get_semantic_embedding(text)

        # Анализ признаков
        features = self._extract_text_features(text)
        structure = self._analyze_text_structure(text)
        math_content = self._analyze_mathematical_content(text)

        # Объединяем все признаки
        all_features = {}

        # Базовые признаки текста
        all_features['text_length_norm'] = min(1.0, features['text_length'] / 5000)
        all_features['vocabulary_richness'] = features['vocabulary_richness']

        # Структурные признаки
        structure_features = [
            'has_numerical_sections',
            'has_definitions',
            'has_examples',
            'has_exercises',
            'has_references',
            'has_tables_figures'
        ]

        for feature in structure_features:
            all_features[feature] = 1.0 if structure[feature] else 0.0

        all_features['section_depth_norm'] = structure['section_hierarchy_depth'] / 3.0

        # Математические признаки
        math_features = [
            'has_formulas',
            'has_equations',
            'has_proofs',
            'has_theorems'
        ]

        for feature in math_features:
            all_features[feature] = 1.0 if math_content[feature] else 0.0

        all_features['math_keyword_density'] = min(1.0, math_content['math_keyword_count'] / 10.0)
        all_features['formula_density_norm'] = min(1.0, math_content['formula_density'] / 5.0)

        # Если есть эмбеддинг, добавляем некоторые его статистики
        if embedding is not None:
            all_features['embedding_norm'] = float(np.linalg.norm(embedding))
            all_features['embedding_mean'] = float(np.mean(embedding))
            all_features['embedding_std'] = float(np.std(embedding))

        return all_features

    def _make_autonomous_decision(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Автономное принятие решения на основе признаков"""
        # Взвешенная оценка различных аспектов
        weights = {
            'structural': 0.35,  # Структура учебника
            'mathematical': 0.30,  # Математическое содержание
            'formal': 0.20,  # Формальность языка
            'compositional': 0.15  # Композиционные особенности
        }

        # Рассчитываем оценку для каждого аспекта

        # 1. Структурная оценка
        structural_score = (
                                   features.get('has_numerical_sections', 0) * 0.3 +
                                   features.get('has_definitions', 0) * 0.2 +
                                   features.get('has_examples', 0) * 0.15 +
                                   features.get('has_exercises', 0) * 0.2 +
                                   features.get('has_references', 0) * 0.15
                           ) * weights['structural']

        # 2. Математическая оценка
        mathematical_score = (
                                     features.get('has_formulas', 0) * 0.25 +
                                     features.get('has_equations', 0) * 0.25 +
                                     features.get('has_proofs', 0) * 0.15 +
                                     features.get('has_theorems', 0) * 0.15 +
                                     features.get('formula_density_norm', 0) * 0.2
                             ) * weights['mathematical']

        # 3. Оценка формальности
        formal_score = (
                               features.get('vocabulary_richness', 0) * 0.4 +
                               features.get('section_depth_norm', 0) * 0.3 +
                               features.get('has_tables_figures', 0) * 0.3
                       ) * weights['formal']

        # 4. Композиционная оценка
        compositional_score = (
                                      features.get('text_length_norm', 0) * 0.6 +
                                      min(1.0, features.get('math_keyword_density', 0) * 2) * 0.4
                              ) * weights['compositional']

        # Итоговая оценка
        total_score = structural_score + mathematical_score + formal_score + compositional_score

        # Автономное определение порога
        # Анализируем распределение оценок
        sub_scores = [structural_score, mathematical_score, formal_score, compositional_score]
        score_variance = np.var(sub_scores) if len(sub_scores) > 1 else 0

        # Динамический порог на основе согласованности оценок
        if score_variance < 0.05:  # Оценки согласованы
            threshold = 0.45
        else:  # Оценки противоречивы
            threshold = 0.55

        # Принимаем решение
        is_educational = total_score >= threshold

        # Рассчитываем уверенность
        confidence = min(0.95, total_score * 1.2)

        # Определяем тип контента на основе доминирующего аспекта
        content_type = "не определено"
        if is_educational:
            if mathematical_score > structural_score and mathematical_score > formal_score:
                content_type = "учебный математический"
            elif structural_score > mathematical_score:
                content_type = "структурированный учебный"
            else:
                content_type = "формальный учебный"
        else:
            if total_score < 0.3:
                content_type = "развлекательный/художественный"
            elif mathematical_score < 0.1:
                content_type = "ненормативный/неформальный"
            else:
                content_type = "смешанный/неопределенный"

        return {
            'is_educational': is_educational,
            'confidence': round(confidence, 3),
            'total_score': round(total_score, 3),
            'content_type': content_type,
            'sub_scores': {
                'structural': round(structural_score, 3),
                'mathematical': round(mathematical_score, 3),
                'formal': round(formal_score, 3),
                'compositional': round(compositional_score, 3)
            },
            'threshold_used': round(threshold, 3),
            'score_variance': round(score_variance, 3)
        }

    def analyze_autonomously(self, text: str, fast_check: bool = True) -> Dict[str, Any]:
        """
        Автономный анализ текста на учебность

        Args:
            text: Текст для анализа
            fast_check: Если True, анализирует только часть текста

        Returns:
            Dict с результатами автономного анализа
        """
        if fast_check and len(text) > 1500:
            # Для быстрой проверки анализируем начало и конец текста
            analysis_text = text[:1000] + text[-500:] if len(text) > 1500 else text[:1000]
        else:
            analysis_text = text

        print("🔍 Начинаю автономный анализ текста...")

        # 1. Генерируем признаки
        features = self._generate_self_learning_features(analysis_text)

        # 2. Принимаем автономное решение
        decision = self._make_autonomous_decision(features)

        # 3. Формируем детализированный отчет
        report = {
            'decision': decision,
            'analysis_metadata': {
                'text_length_analyzed': len(analysis_text),
                'total_text_length': len(text),
                'fast_check_used': fast_check,
                'features_extracted': len(features)
            },
            'key_findings': self._extract_key_findings(analysis_text, features),
            'recommendation': self._generate_recommendation(decision, features)
        }

        return report

    def _extract_key_findings(self, text: str, features: Dict[str, float]) -> List[str]:
        """Извлечение ключевых находок из анализа"""
        findings = []

        # Проверяем ключевые структуры
        if features.get('has_numerical_sections', 0) > 0.5:
            findings.append("Обнаружена четкая структура с нумерованными разделами")

        if features.get('has_definitions', 0) > 0.5:
            findings.append("Найдены формальные определения терминов")

        if features.get('has_exercises', 0) > 0.5:
            findings.append("Присутствуют упражнения и задачи")

        if features.get('has_formulas', 0) > 0.5:
            findings.append("Содержит математические формулы и уравнения")

        if features.get('has_references', 0) > 0.5:
            findings.append("Есть ссылки на литературу и источники")

        # Анализ сложности
        if features.get('formula_density_norm', 0) > 0.7:
            findings.append("Высокая плотность математических выражений")

        if features.get('section_depth_norm', 0) > 0.7:
            findings.append("Глубокая иерархическая структура")

        # Если находок мало, добавляем общие
        if len(findings) < 2:
            if len(text) > 1000:
                findings.append("Достаточный объем текста для анализа")
            if features.get('vocabulary_richness', 0) > 0.6:
                findings.append("Богатый и разнообразный словарный запас")

        return findings[:5]  # Ограничиваем 5 находками

    def _generate_recommendation(self, decision: Dict[str, Any], features: Dict[str, float]) -> str:
        """Генерация рекомендации на основе решения"""
        if decision['is_educational']:
            confidence = decision['confidence']

            if confidence > 0.8:
                return "✅ Высокая вероятность учебной литературы. Рекомендуется полный анализ."
            elif confidence > 0.6:
                return "✅ Умеренная вероятность учебной литературы. Рекомендуется анализ с проверкой."
            else:
                return "⚠️  Низкая вероятность учебной литературы. Требуется дополнительная проверка."
        else:
            if decision['total_score'] < 0.3:
                return "❌ Вероятно, не учебная литература. Рекомендуется отклонить."
            elif decision['total_score'] < 0.5:
                return "⚠️  Сомнительная учебная ценность. Требуется ручная проверка."
            else:
                return "⚠️  Пограничный случай. Рекомендуется экспертная оценка."

    def batch_analyze(self, texts: List[str], max_workers: int = 2) -> List[Dict[str, Any]]:
        """Пакетный анализ нескольких текстов"""
        results = []

        for i, text in enumerate(texts):
            print(f"Анализ текста {i + 1}/{len(texts)}...")
            result = self.analyze_autonomously(text, fast_check=True)
            results.append(result)

        return results


# Утилитарные функции для интеграции
def create_quick_classifier():
    """Создание быстрого классификатора для интеграции"""
    return AutonomousEducationalClassifier()


def check_if_educational(text: str, classifier: Optional[AutonomousEducationalClassifier] = None) -> Dict[str, Any]:
    """
    Упрощенная функция проверки учебности

    Args:
        text: Текст для проверки
        classifier: Опциональный классификатор

    Returns:
        Упрощенный результат проверки
    """
    if classifier is None:
        classifier = AutonomousEducationalClassifier()

    result = classifier.analyze_autonomously(text, fast_check=True)

    # Упрощенный формат ответа для интеграции
    simplified_result = {
        'is_educational': result['decision']['is_educational'],
        'confidence': result['decision']['confidence'],
        'content_type': result['decision']['content_type'],
        'recommendation': result['recommendation']
    }

    return simplified_result