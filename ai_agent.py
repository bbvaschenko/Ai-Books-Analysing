"""
AI агент для расширенного анализа учебной литературы
"""
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BookAnalysis:
    """Результаты расширенного анализа книги"""
    summary: str  # Краткое резюме книги
    sections_summary: Dict[str, str]  # Резюме по разделам
    key_topics: List[str]  # Ключевые темы
    difficulty_level: str  # Уровень сложности
    prerequisites: List[str]  # Необходимые предварительные знания
    mathematical_areas: List[str]  # Области математики
    recommendations: List[str]  # Рекомендации по изучению


class MathDomainAnalyzer:
    """Класс для анализа областей математики по задачам"""

    def __init__(self):
        """Инициализация анализатора математических областей"""
        # Детальная классификация областей математики
        self.math_domains = {
            'алгебра': {
                'keywords': ['уравнение', 'система уравнений', 'полином', 'многочлен',
                           'квадратное уравнение', 'линейное уравнение', 'алгебраическое',
                           'факторизация', 'разложение', 'тождество', 'неравенство'],
                'subdomains': ['элементарная алгебра', 'абстрактная алгебра', 'линейная алгебра']
            },
            'геометрия': {
                'keywords': ['треугольник', 'окружность', 'угол', 'площадь', 'объем',
                           'теорема Пифагора', 'подобие', 'конгруэнтность', 'вектор',
                           'координаты', 'расстояние', 'прямая', 'плоскость'],
                'subdomains': ['планиметрия', 'стереометрия', 'аналитическая геометрия']
            },
            'тригонометрия': {
                'keywords': ['синус', 'косинус', 'тангенс', 'котангенс', 'тригонометрическое тождество',
                           'тригонометрическое уравнение', 'окружность', 'радиан', 'градус'],
                'subdomains': ['тригонометрические функции', 'тригонометрические уравнения']
            },
            'математический анализ': {
                'keywords': ['производная', 'интеграл', 'предел', 'функция', 'дифференцирование',
                           'интегрирование', 'ряд', 'непрерывность', 'дифференциальное уравнение'],
                'subdomains': ['дифференциальное исчисление', 'интегральное исчисление', 'ряды']
            },
            'дифференциальные уравнения': {
                'keywords': ['диф. уравнение', 'ОДУ', 'ДУ', 'начальное условие', 'краевая задача',
                           'линейное ДУ', 'нелинейное ДУ', 'система ДУ', 'частная производная'],
                'subdomains': ['обыкновенные ДУ', 'уравнения в частных производных']
            },
            'теория вероятностей': {
                'keywords': ['вероятность', 'случайная величина', 'распределение', 'мат. ожидание',
                           'дисперсия', 'комбинаторика', 'теорема Байеса', 'независимость',
                           'условная вероятность'],
                'subdomains': ['комбинаторика', 'теория вероятностей', 'математическая статистика']
            },
            'теория чисел': {
                'keywords': ['делимость', 'простое число', 'НОД', 'НОК', 'сравнение по модулю',
                           'диофантово уравнение', 'теорема Ферма', 'алгоритм Евклида'],
                'subdomains': ['элементарная теория чисел', 'аналитическая теория чисел']
            },
            'математическая логика': {
                'keywords': ['множество', 'предикат', 'квантор', 'теорема', 'доказательство',
                           'аксиома', 'логическая операция', 'импликация', 'эквивалентность'],
                'subdomains': ['теория множеств', 'математическая логика']
            }
        }

        # Паттерны для поиска математического контента
        self.math_patterns = {
            'equation': r'\$[^$]+\$|\\[(\[]?[^\\]*?\\[\])]?|[A-Za-zА-Яа-я]+\s*=\s*[^=\n]+',
            'problem': r'Задача\s+\d+[\.:]?|Упражнение\s+\d+[\.:]?|Пример\s+\d+[\.:]?',
            'theorem': r'Теорема\s+\d+[\.:]?|Лемма\s+\d+[\.:]?|Следствие\s+\d+[\.:]?',
            'definition': r'Определение\s+\d+[\.:]?'
        }

    def extract_math_content(self, text: str) -> Dict[str, Any]:
        """Извлечение математического контента из текста"""
        math_content = {
            'equations': [],
            'problems': [],
            'theorems': [],
            'definitions': [],
            'domains_detected': set(),
            'complexity_indicators': []
        }

        # Поиск уравнений
        equation_matches = re.findall(r'\$(.*?)\$', text, re.DOTALL)
        math_content['equations'] = [eq.strip() for eq in equation_matches if eq.strip()]

        # Поиск задач и примеров
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Берем несколько строк после заголовка
                start_pos = match.end()
                end_pos = text.find('\n\n', start_pos)
                if end_pos == -1:
                    end_pos = min(start_pos + 500, len(text))

                content = text[start_pos:end_pos].strip()
                if content:
                    if pattern_name == 'problem':
                        math_content['problems'].append(content[:200])
                    elif pattern_name == 'theorem':
                        math_content['theorems'].append(content[:200])
                    elif pattern_name == 'definition':
                        math_content['definitions'].append(content[:200])

        # Анализ для определения областей математики
        text_lower = text.lower()

        for domain, info in self.math_domains.items():
            domain_score = 0

            # Проверка ключевых слов
            for keyword in info['keywords']:
                if keyword in text_lower:
                    domain_score += 1

            # Проверка поддоменов
            for subdomain in info['subdomains']:
                if subdomain in text_lower:
                    domain_score += 2

            # Если набрали достаточно баллов, добавляем домен
            if domain_score >= 2:
                math_content['domains_detected'].add(domain)

                # Анализ сложности для этого домена
                complexity = self.analyze_domain_complexity(domain, text)
                if complexity:
                    math_content['complexity_indicators'].append({
                        'domain': domain,
                        'complexity': complexity
                    })

        return math_content

    def analyze_domain_complexity(self, domain: str, text: str) -> Optional[str]:
        """Анализ сложности математического контента в конкретной области"""
        complexity_indicators = {
            'базовая': ['простое', 'элементарное', 'основное', 'начальное'],
            'средняя': ['сложное', 'продвинутое', 'углубленное', 'специальное'],
            'высокая': ['университетский', 'теоретический', 'исследовательский', 'доказательство']
        }

        text_lower = text.lower()
        domain_keywords = self.math_domains.get(domain, {}).get('keywords', [])

        # Подсчет различных типов математического контента
        equation_count = len(re.findall(r'\$.*?\$', text))
        theorem_count = len(re.findall(r'Теорема|Лемма|Следствие', text, re.IGNORECASE))
        proof_count = len(re.findall(r'Доказательство|Proof', text, re.IGNORECASE))

        total_score = equation_count + theorem_count * 2 + proof_count * 3

        # Анализ по ключевым словам сложности
        complexity_score = 0
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    if level == 'базовая':
                        complexity_score += 1
                    elif level == 'средняя':
                        complexity_score += 2
                    elif level == 'высокая':
                        complexity_score += 3

        # Определение итогового уровня сложности
        final_score = total_score + complexity_score

        if final_score > 15:
            return "высокая сложность"
        elif final_score > 8:
            return "средняя сложность"
        else:
            return "базовая сложность"

    def identify_math_areas(self, math_content: Dict[str, Any]) -> List[Dict[str, str]]:
        """Идентификация областей математики на основе извлеченного контента"""
        areas = []

        for domain in math_content['domains_detected']:
            # Собираем доказательства принадлежности к области
            evidence = []

            if domain == 'алгебра':
                evidence.extend(self._find_algebra_evidence(math_content))
            elif domain == 'геометрия':
                evidence.extend(self._find_geometry_evidence(math_content))
            elif domain == 'математический анализ':
                evidence.extend(self._find_calculus_evidence(math_content))
            elif domain == 'теория вероятностей':
                evidence.extend(self._find_probability_evidence(math_content))

            # Находим уровень сложности для этой области
            complexity = next(
                (item['complexity'] for item in math_content['complexity_indicators']
                 if item['domain'] == domain),
                "не определено"
            )

            areas.append({
                'domain': domain,
                'complexity': complexity,
                'evidence': evidence[:3]  # Берем только 3 доказательства
            })

        return areas

    def _find_algebra_evidence(self, math_content: Dict) -> List[str]:
        """Поиск доказательств алгебраического содержания"""
        evidence = []
        all_text = ' '.join(math_content['problems'] + math_content['theorems'])

        if any(keyword in all_text.lower() for keyword in ['уравнение', 'система', 'полином']):
            evidence.append("Содержит алгебраические уравнения")

        if any('=' in eq for eq in math_content['equations']):
            evidence.append("Использует алгебраические выражения")

        if 'неравенство' in all_text.lower():
            evidence.append("Рассматривает алгебраические неравенства")

        return evidence

    def _find_geometry_evidence(self, math_content: Dict) -> List[str]:
        """Поиск доказательств геометрического содержания"""
        evidence = []
        all_text = ' '.join(math_content['problems'] + math_content['theorems'])

        geometry_keywords = ['треугольник', 'окружность', 'угол', 'площадь', 'объем']
        found_keywords = [kw for kw in geometry_keywords if kw in all_text.lower()]

        if found_keywords:
            evidence.append(f"Использует геометрические понятия: {', '.join(found_keywords[:2])}")

        if any('теорема Пифагора' in text.lower() for text in math_content['theorems']):
            evidence.append("Содержит теорему Пифагора")

        return evidence

    def _find_calculus_evidence(self, math_content: Dict) -> List[str]:
        """Поиск доказательств анализа"""
        evidence = []
        all_text = ' '.join(math_content['problems'] + math_content['theorems'])

        calculus_keywords = ['производная', 'интеграл', 'предел', 'дифференцирование']
        found_keywords = [kw for kw in calculus_keywords if kw in all_text.lower()]

        if found_keywords:
            evidence.append(f"Содержит элементы анализа: {', '.join(found_keywords)}")

        # Проверка наличия обозначений производных/интегралов в уравнениях
        for eq in math_content['equations']:
            if '\\frac{d}{dx}' in eq or '\\int' in eq:
                evidence.append("Использует обозначения математического анализа")
                break

        return evidence

    def _find_probability_evidence(self, math_content: Dict) -> List[str]:
        """Поиск доказательств теории вероятностей"""
        evidence = []
        all_text = ' '.join(math_content['problems'] + math_content['theorems'])

        prob_keywords = ['вероятность', 'случайная величина', 'распределение']
        found_keywords = [kw for kw in prob_keywords if kw in all_text.lower()]

        if found_keywords:
            evidence.append(f"Содержит элементы теории вероятностей: {', '.join(found_keywords)}")

        return evidence


class EducationalAIAgent:
    """AI агент для анализа учебной литературы с легкими моделями"""

    def __init__(self):
        """Инициализация AI агента с легкими моделями"""
        print("Инициализация AI агента для анализа учебной литературы...")

        # Инициализация математического анализатора
        self.math_analyzer = MathDomainAnalyzer()

        # Используем более легкую модель Sentence Transformers для суммаризации
        try:
            # Модель для суммаризации текста - используем легкую версию
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("Модель для эмбеддингов и суммаризации загружена")
        except Exception as e:
            print(f"Не удалось загрузить модель эмбеддингов: {e}")
            self.embedding_model = None

        # База знаний по образовательным уровням
        self.education_knowledge = {
            'школьный': {
                'keywords': ['школа', 'класс', 'учебник', 'задачник', 'упражнение'],
                'levels': ['начальная школа', 'средняя школа', 'старшая школа']
            },
            'вузовский': {
                'keywords': ['вуз', 'университет', 'курс лекций', 'пособие', 'практикум'],
                'levels': ['бакалавриат', 'магистратура', 'аспирантура']
            },
            'специальный': {
                'keywords': ['монография', 'исследование', 'научная работа', 'диссертация'],
                'levels': ['научный', 'исследовательский', 'профессиональный']
            }
        }

        print("AI агент успешно инициализирован с легкими моделями")

    def generate_text_summary(self, text: str, max_length: int = 150) -> str:
        """Генерация краткого резюме текста с использованием эмбеддингов"""
        if not self.embedding_model:
            # Если модель не загружена, используем простой метод
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            if len(sentences) >= 3:
                summary_sentences = [
                    sentences[0],
                    sentences[len(sentences) // 2],
                    sentences[-1]
                ]
                summary = ' '.join(summary_sentences)
                return summary[:max_length] + '...' if len(summary) > max_length else summary
            else:
                return text[:max_length] + '...' if len(text) > max_length else text

        try:
            # Разбиваем текст на предложения
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            if len(sentences) < 3:
                return text[:max_length] + '...' if len(text) > max_length else text

            # Получаем эмбеддинги предложений
            sentence_embeddings = self.embedding_model.encode(sentences)

            # Вычисляем центроид (средний вектор) всего текста
            text_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)

            # Находим наиболее репрезентативные предложения
            similarities = cosine_similarity(text_embedding, sentence_embeddings)[0]
            top_indices = np.argsort(similarities)[-3:][::-1]  # Топ-3 самых похожих

            # Собираем резюме из ключевых предложений
            summary_sentences = [sentences[i] for i in sorted(top_indices)]
            summary = ' '.join(summary_sentences)

            return summary[:max_length] + '...' if len(summary) > max_length else summary

        except Exception as e:
            print(f"Ошибка при генерации резюме: {e}")
            # Возвращаем начало текста в случае ошибки
            return text[:max_length] + '...' if len(text) > max_length else text

    def analyze_book_content(self, text: str, tags: Dict[str, List[str]]) -> BookAnalysis:
        """
        Основной метод анализа содержания книги

        Args:
            text: Текст книги
            tags: Теги, найденные ранее

        Returns:
            BookAnalysis: Результаты анализа
        """
        print("Начинаю анализ содержания книги...")

        # Извлекаем математический контент
        math_content = self.math_analyzer.extract_math_content(text[:5000])  # Анализируем первые 5000 символов

        # Определяем области математики
        math_areas_info = self.math_analyzer.identify_math_areas(math_content)
        math_areas = [area['domain'] for area in math_areas_info]

        # Генерируем резюме книги
        summary = self._generate_summary(text, tags, math_areas_info)

        # Анализируем разделы
        sections_summary = self._analyze_sections(text)

        # Определяем ключевые темы
        key_topics = self._extract_key_topics(text, tags)

        # Определяем уровень сложности
        difficulty_level = self._assess_difficulty(text, math_content)

        # Определяем необходимые предварительные знания
        prerequisites = self._identify_prerequisites(text, math_areas)

        # Формируем рекомендации
        recommendations = self._generate_recommendations(
            difficulty_level, math_areas, tags
        )

        return BookAnalysis(
            summary=summary,
            sections_summary=sections_summary,
            key_topics=key_topics,
            difficulty_level=difficulty_level,
            prerequisites=prerequisites,
            mathematical_areas=math_areas,
            recommendations=recommendations
        )

    def _generate_summary(self, text: str, tags: Dict, math_areas: List[Dict]) -> str:
        """Генерация краткого резюме книги"""
        # Используем теги для формирования базового описания
        summary_parts = []

        # Добавляем информацию из тегов
        if 'предметы' in tags:
            subjects = ', '.join(tags['предметы'][:3])
            summary_parts.append(f"Учебное пособие по {subjects}.")

        if 'классы' in tags:
            grades = ', '.join(tags['классы'])
            summary_parts.append(f"Предназначено для {grades} классов.")

        # Добавляем информацию о математических областях
        if math_areas:
            domains = ', '.join([area['domain'] for area in math_areas[:2]])
            summary_parts.append(f"Основное внимание уделяется {domains}.")

            # Добавляем информацию о сложности
            complexities = [area['complexity'] for area in math_areas if area['complexity'] != 'не определено']
            if complexities:
                avg_complexity = max(set(complexities), key=complexities.count)
                summary_parts.append(f"Уровень сложности: {avg_complexity}.")

        # Генерируем резюме из текста
        if len(text) > 500:
            generated_summary = self.generate_text_summary(text[:2000], 100)
            if generated_summary:
                summary_parts.append(generated_summary)

        # Если резюме пустое, создаем базовое
        if not summary_parts:
            if math_areas:
                domains = ', '.join([area['domain'] for area in math_areas[:2]])
                summary_parts.append(f"Учебный материал с фокусом на {domains}.")
            else:
                summary_parts.append("Учебное пособие с математическим содержанием.")

        return ' '.join(summary_parts)

    def _analyze_sections(self, text: str) -> Dict[str, str]:
        """Анализ основных разделов книги"""
        sections = {}

        # Ищем заголовки глав и разделов
        patterns = [
            (r'Глава\s+\d+[\.\s]+([^\n]+)', 'глава'),
            (r'Раздел\s+\d+[\.\s]+([^\n]+)', 'раздел'),
            (r'§\s*\d+[\.\s]+([^\n]+)', 'параграф'),
            (r'\n([А-Я][А-ЯА-Яа-я\s]{5,})\n', 'подраздел')
        ]

        for pattern, section_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                title = match.group(1).strip()
                if len(title) > 4 and len(title) < 100:  # Фильтруем слишком короткие/длинные
                    # Анализируем содержание раздела (первые 500 символов после заголовка)
                    start_pos = match.end()
                    end_pos = text.find('\n\n', start_pos)
                    if end_pos == -1:
                        end_pos = min(start_pos + 500, len(text))

                    content = text[start_pos:end_pos].strip()
                    if content:
                        # Создаем краткое описание раздела
                        if len(content) > 200:
                            # Генерируем резюме для раздела
                            section_desc = self.generate_text_summary(content, 100)
                        else:
                            section_desc = content

                        sections[f"{section_type}: {title}"] = section_desc

        # Если не нашли разделов, создаем обобщенные
        if not sections:
            # Делим текст на части и анализируем
            parts = text.split('\n\n')
            for i, part in enumerate(parts[:5]):
                if len(part) > 200:
                    key_topics = self._extract_keywords(part)
                    if key_topics:
                        sections[f"Тема {i+1}"] = f"Рассматривает: {', '.join(key_topics[:3])}"

        return sections

    def _extract_key_topics(self, text: str, tags: Dict) -> List[str]:
        """Извлечение ключевых тем"""
        key_topics = []

        # Добавляем теги как ключевые темы
        for category, tag_list in tags.items():
            if tag_list and category not in ['классы', 'авторы']:
                key_topics.extend(tag_list[:2])

        # Извлекаем дополнительные темы из текста
        if self.embedding_model:
            try:
                # Ищем часто встречающиеся термины
                words = re.findall(r'\b[А-Яа-я]{5,}\b', text.lower())
                word_freq = {}
                for word in words:
                    if word not in ['которые', 'который', 'которые', 'также', 'очень']:
                        word_freq[word] = word_freq.get(word, 0) + 1

                # Берем самые частые слова как дополнительные темы
                frequent_words = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
                for word, freq in frequent_words:
                    if freq > 3:  # Слово встречается минимум 3 раза
                        key_topics.append(word)
            except:
                pass

        # Удаляем дубликаты и ограничиваем количество
        unique_topics = []
        seen = set()
        for topic in key_topics:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                unique_topics.append(topic)

        return unique_topics[:10]  # Не более 10 тем

    def _assess_difficulty(self, text: str, math_content: Dict) -> str:
        """Оценка уровня сложности"""
        # Анализируем математическое содержание
        if math_content['complexity_indicators']:
            complexities = [item['complexity'] for item in math_content['complexity_indicators']]
            if 'высокая сложность' in complexities:
                return "высокий"
            elif 'средняя сложность' in complexities:
                return "средний"

        # Анализируем общий текст
        text_lower = text.lower()

        # Ключевые слова для определения сложности
        basic_keywords = ['основы', 'введение', 'начальный', 'базовый', 'элементарный']
        advanced_keywords = ['углубленный', 'продвинутый', 'специальный', 'теоретический', 'исследование']

        basic_count = sum(1 for word in basic_keywords if word in text_lower)
        advanced_count = sum(1 for word in advanced_keywords if word in text_lower)

        if advanced_count > basic_count:
            return "высокий"
        elif basic_count > advanced_count:
            return "начальный"
        else:
            return "средний"

    def _identify_prerequisites(self, text: str, math_areas: List[str]) -> List[str]:
        """Определение необходимых предварительных знаний"""
        prerequisites = []

        # Базовые предпосылки в зависимости от областей математики
        area_prerequisites = {
            'алгебра': ['арифметика', 'основы алгебры'],
            'геометрия': ['планиметрия', 'пространственное мышление'],
            'математический анализ': ['алгебра', 'тригонометрия', 'пределы'],
            'тригонометрия': ['геометрия', 'алгебра'],
            'теория вероятностей': ['комбинаторика', 'статистика'],
            'дифференциальные уравнения': ['математический анализ', 'алгебра'],
            'теория чисел': ['алгебра', 'математическая логика']
        }

        # Добавляем предпосылки для каждой области
        for area in math_areas:
            if area in area_prerequisites:
                prerequisites.extend(area_prerequisites[area])

        # Удаляем дубликаты
        return list(set(prerequisites))[:5]

    def _generate_recommendations(self, difficulty: str, math_areas: List[str],
                                 tags: Dict) -> List[str]:
        """Генерация рекомендаций по изучению"""
        recommendations = []

        # Рекомендации по уровню сложности
        if difficulty == 'начальный':
            recommendations.append("Подходит для начального изучения темы")
            recommendations.append("Рекомендуется для самостоятельной работы")
        elif difficulty == 'средний':
            recommendations.append("Требует базовых знаний по предмету")
            recommendations.append("Подходит для углубленного изучения")
        elif difficulty == 'высокий':
            recommendations.append("Требует серьезной математической подготовки")
            recommendations.append("Рекомендуется с преподавателем")

        # Рекомендации по областям математики
        if math_areas:
            areas_str = ', '.join(math_areas[:2])
            recommendations.append(f"Особое внимание уделено: {areas_str}")

        # Рекомендации на основе тегов
        if 'классы' in tags:
            grades = tags['классы']
            if any(int(g) >= 10 for g in grades if g.isdigit()):
                recommendations.append("Подходит для старшеклассников и студентов")

        return recommendations[:5]

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        # Удаляем стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'у', 'о', 'при'}

        # Ищем существительные и прилагательные
        words = re.findall(r'\b[А-Яа-я]{4,}\b', text.lower())

        # Фильтруем стоп-слова и считаем частоту
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Сортируем по частоте и возвращаем топ-5
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [word for word, freq in sorted_words[:5]]