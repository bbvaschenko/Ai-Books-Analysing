"""
Автономный классификатор учебной литературы
Основной критерий: наличие математических формул = учебная литература
"""
import re
import torch
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel
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

    def _analyze_mathematical_content(self, text: str) -> Dict[str, Any]:
        """Анализ математического содержания - упрощенный вариант"""
        math_analysis = {
            'has_formulas': False,
            'has_equations': False,
            'formula_count': 0
        }

        # Паттерны для поиска формул
        formula_patterns = [
            r'\$[^$]+\$',  # LaTeX формулы между долларами
            r'\\[(\[]?[^\\]*?\\[\])]?',  # LaTeX команды
            r'\b\w+\s*=\s*[^=\n]{3,}',  # Равенства с содержанием
        ]

        formula_count = 0
        for pattern in formula_patterns:
            matches = re.findall(pattern, text)
            formula_count += len(matches)
            if matches:
                math_analysis['has_formulas'] = True
                math_analysis['has_equations'] = True

        math_analysis['formula_count'] = formula_count

        return math_analysis

    def _check_simple_criteria(self, text: str) -> Dict[str, Any]:
        """Упрощенная проверка критериев учебности"""
        text_lower = text.lower()

        # Простые критерии
        criteria = {
            # 1. МАТЕМАТИЧЕСКИЕ ФОРМУЛЫ - ГЛАВНЫЙ КРИТЕРИЙ
            'has_mathematics': False,

            # 2. Очевидные учебные маркеры
            'has_obvious_edu_markers': False,

            # 3. Минимальное содержание
            'has_minimal_content': len(text) > 200
        }

        # КРИТЕРИЙ 1: ПРОВЕРКА МАТЕМАТИЧЕСКИХ ФОРМУЛ
        math_content = self._analyze_mathematical_content(text)
        if math_content['has_formulas'] or math_content['formula_count'] > 0:
            criteria['has_mathematics'] = True

        # КРИТЕРИЙ 2: ОЧЕВИДНЫЕ УЧЕБНЫЕ МАРКЕРЫ
        obvious_edu_markers = [
            'учебник', 'пособие', 'задачник', 'практикум', 'лекция',
            'глава', 'раздел', 'тема', 'задача', 'упражнение'
        ]

        found_markers = []
        for marker in obvious_edu_markers:
            if marker in text_lower:
                found_markers.append(marker)

        if len(found_markers) > 2:  # Если найдено хотя бы 3 очевидных маркера
            criteria['has_obvious_edu_markers'] = True

        return criteria

    def check_if_educational(self, text: str) -> Dict[str, Any]:
        """Проверка учебности текста"""
        if len(text) < 100:
            return {
                'is_educational': False,
                'confidence': 0.0,
                'reason': 'Текст слишком короткий для анализа'
            }

        # Для быстрой проверки анализируем только часть текста
        if len(text) > 3000:
            analysis_text = text[:2000]
        else:
            analysis_text = text

        # Упрощенная проверка критериев
        criteria = self._check_simple_criteria(analysis_text)

        # Простые правила принятия решения
        is_educational = False
        confidence = 0.0
        reason = ""

        # ПРАВИЛО 1: Если есть математические формулы -> УЧЕБНАЯ
        if criteria['has_mathematics']:
            is_educational = True
            confidence = 0.9
            reason = "Содержит математические формулы"

        # ПРАВИЛО 2: Если есть очевидные учебные маркеры -> УЧЕБНАЯ
        elif criteria['has_obvious_edu_markers']:
            is_educational = True
            confidence = 0.8
            reason = "Содержит явные учебные маркеры"

        # ПРАВИЛО 3: Если недостаточно данных -> НЕ учебная
        elif not criteria['has_minimal_content']:
            is_educational = False
            confidence = 0.7
            reason = "Недостаточно данных для анализа"

        else:
            # ПРАВИЛО 4: Во всех остальных случаях -> проверяем базовые вещи
            if 'математика' in analysis_text.lower() or len(re.findall(r'\d+', analysis_text)) > 10:
                is_educational = True
                confidence = 0.6
                reason = "Возможна учебная литература по математике"
            else:
                is_educational = False
                confidence = 0.7
                reason = "Не обнаружено признаков учебной литературы"

        return {
            'is_educational': is_educational,
            'confidence': round(confidence, 2),
            'reason': reason,
            'criteria_met': {
                'has_mathematics': criteria['has_mathematics'],
                'has_obvious_edu_markers': criteria['has_obvious_edu_markers'],
                'text_length': len(analysis_text)
            }
        }

    def analyze_quick(self, text: str) -> Dict[str, Any]:
        """Быстрая проверка учебности текста"""
        print("🔍 Быстрая проверка на учебную литературу...")

        result = self.check_if_educational(text)

        status = "✅ УЧЕБНАЯ" if result['is_educational'] else "❌ НЕ учебная"

        report = {
            'status': status,
            'is_educational': result['is_educational'],
            'confidence': result['confidence'],
            'reason': result['reason'],
            'criteria': result['criteria_met']
        }

        print(f"   Результат: {status}")
        print(f"   Уверенность: {result['confidence']:.0%}")
        print(f"   Причина: {result['reason']}")
        if result['criteria_met']['has_mathematics']:
            print(f"   🔢 Обнаружены математические формулы")

        return report


# Простая функция для интеграции
def quick_educational_check(text: str) -> bool:
    """Проверяет, является ли текст учебной литературой"""
    if len(text) < 150:
        return False

    # Проверка математических формул
    formula_patterns = [
        r'\$[^$]+\$',  # LaTeX формулы
        r'\\[(\[]?[^\\]*?\\[\])]?',  # LaTeX команды
    ]

    has_formulas = False
    for pattern in formula_patterns:
        if re.search(pattern, text):
            has_formulas = True
            break

    # Если есть формулы -> сразу УЧЕБНАЯ
    if has_formulas:
        return True

    # Проверка очевидных учебных маркеров
    text_lower = text.lower()
    obvious_markers = [
        'учебник', 'пособие', 'задачник', 'практикум',
        'глава', 'раздел', 'тема', 'задача'
    ]

    marker_count = sum(1 for marker in obvious_markers if marker in text_lower)

    # Если найдено достаточно маркеров -> УЧЕБНАЯ
    if marker_count >= 2:
        return True

    return False


# Альтернативная простая версия
class SimpleClassifier:
    """ПРОСТОЙ классификатор"""

    @staticmethod
    def is_educational(text: str) -> bool:
        """ПРОСТАЯ ПРОВЕРКА"""
        if len(text) < 100:
            return False

        # 1. Проверяем формулы LaTeX
        if re.search(r'\$[^$]+\$', text) or re.search(r'\\[(\[]', text):
            return True

        # 2. Проверяем равенства (математические)
        if len(re.findall(r'\w+\s*=\s*\w+', text)) > 3:
            return True

        # 3. Проверяем очевидные учебные слова
        text_lower = text.lower()
        educational_words = ['учебник', 'пособие', 'задачник', 'глава', 'раздел']

        for word in educational_words:
            if word in text_lower:
                return True

        # 4. Проверяем "математика" и числа
        if 'математика' in text_lower and len(re.findall(r'\d+', text)) > 5:
            return True

        return False