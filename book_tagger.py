import re
import json
import math
from typing import List, Dict, Set, Tuple, Any
from collections import Counter
import logging
from datetime import datetime
import string

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedBookTagger:
    """Продвинутый класс для семантического тегирования книг"""

    def __init__(self, config_path: str = None):
        """Инициализация с загрузкой конфигурации"""
        self.config_path = config_path
        self.taxonomy = self._load_taxonomy()
        self._prepare_regex_patterns()
        self.stats = {
            'books_processed': 0,
            'total_tags_assigned': 0,
            'tag_distribution': Counter()
        }

        # Параметры тегирования
        self.params = {
            'max_tags_per_category': 3,
            'min_confidence': 0.05,
            'top_n_overall': 10,
            'word_window_size': 10,
            'use_tfidf_weighting': True,
            'enable_synonym_expansion': True,
            'enable_composite_keywords': True,
            'language': 'ru'  # 'ru' или 'en'
        }

    def _load_taxonomy(self) -> Dict[str, Any]:
        """Загружает расширенную таксономию тегов"""

        # Если указан путь к конфигурации, пытаемся загрузить из файла
        if self.config_path:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить конфигурацию: {e}. Используем встроенную таксономию.")

        # Встроенная расширенная таксономия
        return {
            "academic_subjects": {
                "математика": {
                    "keywords": [
                        "математика", "алгебра", "геометрия", "анализ", "исчисление",
                        "дифференциальный", "интеграл", "производная", "функция", "уравнение",
                        "неравенство", "система", "матрица", "вектор", "пространство",
                        "теория вероятностей", "статистика", "дискретная математика",
                        "комбинаторика", "графы", "логика", "число", "формула", "теорема",
                        "доказательство", "аксиома", "лемма", "следствие", "корень",
                        "предел", "ряд", "фурье", "лаплас", "комплексное число"
                    ],
                    "subcategories": {
                        "алгебра": ["многочлен", "уравнение", "система", "матрица", "определитель"],
                        "геометрия": ["треугольник", "окружность", "угол", "площадь", "объем"],
                        "анализ": ["предел", "производная", "интеграл", "ряд", "функция"],
                        "статистика": ["вероятность", "распределение", "выборка", "дисперсия", "корреляция"],
                        "теория чисел": ["простое число", "делимость", "конгруэнция", "факторизация"]
                    }
                },
                "программирование": {
                    "keywords": [
                        "программирование", "код", "алгоритм", "структура данных", "язык программирования",
                        "python", "java", "javascript", "c++", "c#", "ruby", "php", "go", "rust", "kotlin",
                        "swift", "typescript", "scala", "haskell", "lisp", "perl", "bash", "sql",
                        "база данных", "nosql", "mongodb", "postgresql", "mysql", "sqlite", "redis",
                        "api", "rest", "graphql", "soap", "веб-сервис", "микросервис",
                        "фреймворк", "библиотека", "django", "flask", "spring", "react", "angular", "vue",
                        "объектно-ориентированный", "функциональный", "процедурный", "декларативный",
                        "компилятор", "интерпретатор", "отладка", "тестирование", "разработка", "деплоймент"
                    ],
                    "subcategories": {
                        "веб-разработка": ["html", "css", "javascript", "react", "angular", "vue", "node.js"],
                        "мобильная разработка": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
                        "игры": ["unity", "unreal engine", "directx", "opengl", "игровой движок"],
                        "искусственный интеллект": ["нейронная сеть", "машинное обучение", "тензор",
                                                    "обработка данных"],
                        "безопасность": ["криптография", "шифрование", "хэширование", "аутентификация", "авторизация"]
                    }
                },
                "физика": {
                    "keywords": [
                        "физика", "механика", "термодинамика", "оптика", "электричество",
                        "магнетизм", "квантовая", "атомная", "ядерная", "релятивистская",
                        "кинематика", "динамика", "статика", "гидродинамика", "аэродинамика",
                        "закон Ньютона", "закон сохранения", "энергия", "мощность", "сила",
                        "давление", "температура", "энтропия", "волна", "частица", "поле",
                        "гравитация", "черная дыра", "космология", "астрофизика", "плазма"
                    ]
                },
                "химия": {
                    "keywords": [
                        "химия", "органическая", "неорганическая", "аналитическая", "физическая",
                        "биохимия", "электрохимия", "квантовая химия", "стереохимия", "термохимия",
                        "атом", "молекула", "элемент", "соединение", "реакция", "катализатор",
                        "периодическая система", "валентность", "оксид", "кислота", "основание",
                        "соль", "раствор", "концентрация", "pH", "титрование", "хроматография"
                    ]
                },
                "биология": {
                    "keywords": [
                        "биология", "генетика", "эволюция", "экология", "ботаника",
                        "зоология", "микробиология", "анатомия", "физиология", "цитология",
                        "гистология", "эмбриология", "биохимия", "молекулярная биология",
                        "клетка", "ДНК", "РНК", "белок", "фермент", "метаболизм",
                        "фотосинтез", "дыхание", "размножение", "наследственность", "мутация"
                    ]
                },
                "медицина": {
                    "keywords": [
                        "медицина", "анатомия", "физиология", "патология", "фармакология",
                        "хирургия", "терапия", "диагностика", "профилактика", "реабилитация",
                        "болезнь", "симптом", "диагноз", "лечение", "лекарство", "вакцина",
                        "иммунитет", "инфекция", "вирус", "бактерия", "паразит", "аллергия"
                    ]
                },
                "экономика": {
                    "keywords": [
                        "экономика", "финансы", "бухгалтерия", "маркетинг", "менеджмент",
                        "предприятие", "бизнес", "рынок", "спрос", "предложение", "цена",
                        "стоимость", "прибыль", "убыток", "инвестиции", "акции", "облигации",
                        "кредит", "депозит", "инфляция", "валюта", "биржа", "банк", "налоги",
                        "бюджет", "отчетность", "аудит", "консалтинг", "стартап", "венчурный"
                    ]
                },
                "история": {
                    "keywords": [
                        "история", "археология", "палеонтология", "цивилизация", "империя",
                        "война", "революция", "реформа", "монархия", "республика", "демократия",
                        "феодализм", "капитализм", "социализм", "коммунизм", "средневековье",
                        "возрождение", "просвещение", "индустриализация", "глобализация"
                    ]
                },
                "философия": {
                    "keywords": [
                        "философия", "метафизика", "эпистемология", "этика", "эстетика",
                        "логика", "диалектика", "идеализм", "материализм", "экзистенциализм",
                        "стоицизм", "утопия", "утопия", "свобода", "сознание", "бытие",
                        "истина", "добро", "красота", "справедливость", "мораль", "нравственность"
                    ]
                },
                "психология": {
                    "keywords": [
                        "психология", "психоанализ", "бихевиоризм", "когнитивная", "гуманистическая",
                        "личность", "сознание", "подсознание", "эмоция", "мотивация", "интеллект",
                        "память", "восприятие", "мышление", "речь", "общение", "стресс",
                        "тревога", "депрессия", "терапия", "консультация", "тест", "опросник"
                    ]
                },
                "литературоведение": {
                    "keywords": [
                        "литература", "проза", "поэзия", "драма", "роман", "повесть", "рассказ",
                        "стихотворение", "поэма", "комедия", "трагедия", "сатира", "фарс",
                        "сюжет", "композиция", "персонаж", "образ", "символ", "метафора",
                        "аллегория", "ирония", "гипербола", "литота", "эпитет", "сравнение"
                    ]
                },
                "лингвистика": {
                    "keywords": [
                        "лингвистика", "языкознание", "грамматика", "синтаксис", "морфология",
                        "фонетика", "фонология", "семантика", "прагматика", "диалектология",
                        "словообразование", "этимология", "орфография", "пунктуация",
                        "перевод", "билингвизм", "полиглот", "языковая семья", "диалект"
                    ]
                },
                "искусство": {
                    "keywords": [
                        "искусство", "живопись", "скульптура", "архитектура", "музыка",
                        "театр", "кино", "фотография", "дизайн", "мода", "графика",
                        "авангард", "классицизм", "романтизм", "импрессионизм", "экспрессионизм",
                        "сюрреализм", "модернизм", "постмодернизм", "инсталляция", "перформанс"
                    ]
                },
                "юриспруденция": {
                    "keywords": [
                        "право", "юриспруденция", "закон", "конституция", "кодекс",
                        "суд", "прокуратура", "адвокат", "нотариус", "договор",
                        "сделка", "собственность", "наследство", "развод", "алименты",
                        "преступление", "наказание", "свидетель", "подозреваемый", "обвиняемый"
                    ]
                },
                "социология": {
                    "keywords": [
                        "социология", "общество", "социальная структура", "социальный институт",
                        "социальная мобильность", "стратификация", "социальный конфликт",
                        "социальное неравенство", "социальный контроль", "девиация",
                        "социальная группа", "сообщество", "организация", "бюрократия"
                    ]
                },
                "политология": {
                    "keywords": [
                        "политология", "политика", "государство", "власть", "демократия",
                        "автократия", "тоталитаризм", "либерализм", "консерватизм", "социализм",
                        "коммунизм", "национализм", "фашизм", "анархизм", "партия",
                        "выборы", "референдум", "парламент", "правительство", "оппозиция"
                    ]
                },
                "география": {
                    "keywords": [
                        "география", "физическая география", "экономическая география",
                        "политическая география", "ландшафтоведение", "картография",
                        "геодезия", "климатология", "метеорология", "океанология",
                        "геология", "палеогеография", "демография", "урбанистика"
                    ]
                }
            },

            "genres": {
                "художественная литература": {
                    "keywords": ["роман", "повесть", "рассказ", "новелла", "эпопея", "сказка"],
                    "subgenres": {
                        "фантастика": ["космос", "инопланетянин", "путешествие во времени", "киборг", "робот"],
                        "фэнтези": ["магия", "дракон", "эльф", "гном", "волшебство", "заклинание"],
                        "детектив": ["убийство", "преступление", "следователь", "улика", "разгадка"],
                        "триллер": ["опасность", "погоня", "напряжение", "тайна", "смерть"],
                        "романтика": ["любовь", "отношения", "чувства", "сердце", "страсть"],
                        "исторический роман": ["исторический", "эпоха", "древний", "средневековье", "рыцарь"],
                        "биография": ["жизнь", "личность", "судьба", "мемуары", "автобиография"]
                    }
                },
                "научная литература": {
                    "keywords": ["исследование", "эксперимент", "теория", "гипотеза", "открытие"],
                    "types": ["монография", "диссертация", "статья", "обзор", "методичка"]
                },
                "популярная наука": {
                    "keywords": ["научно-популярный", "доступно", "просто", "интересно", "увлекательно"]
                },
                "справочная литература": {
                    "keywords": ["справочник", "энциклопедия", "словарь", "каталог", "руководство"]
                },
                "учебная литература": {
                    "keywords": ["учебник", "пособие", "задачник", "практикум", "тетрадь", "конспект"]
                },
                "деловая литература": {
                    "keywords": ["бизнес", "управление", "маркетинг", "финансы", "переговоры", "лидерство"]
                },
                "психология и саморазвитие": {
                    "keywords": ["саморазвитие", "мотивация", "успех", "счастье", "привычки", "продуктивность"]
                },
                "религия и духовность": {
                    "keywords": ["религия", "вера", "бог", "молитва", "медитация", "духовность"]
                }
            },

            "time_periods": {
                "античность": ["древний", "античный", "греция", "рим", "философ"],
                "средневековье": ["средневековье", "рыцарь", "замок", "феодал", "инквизиция"],
                "возрождение": ["возрождение", "ренессанс", "гуманизм", "леонардо", "микеланджело"],
                "новое время": ["новое время", "просвещение", "революция", "наполеон", "абсолютизм"],
                "xx век": ["xx век", "война", "революция", "индустриализация", "технология"],
                "современность": ["современный", "информационный", "глобальный", "цифровой", "интернет"]
            },

            "audience": {
                "дети": ["детский", "малыш", "игра", "сказка", "мультфильм"],
                "подростки": ["подросток", "школа", "дружба", "первая любовь", "бунт"],
                "студенты": ["студент", "вуз", "сессия", "лекция", "экзамен"],
                "специалисты": ["специалист", "эксперт", "профессионал", "квалификация", "опыт"],
                "широкая аудитория": ["для всех", "популярный", "доступный", "интересный", "увлекательный"]
            },

            "complexity_levels": {
                "начальный": ["простой", "основы", "введение", "для начинающих", "шаг за шагом"],
                "средний": ["продолжающий", "углубленный", "детальный", "анализ", "исследование"],
                "продвинутый": ["продвинутый", "экспертный", "специализированный", "теория", "доказательство"],
                "академический": ["академический", "научный", "исследовательский", "диссертация", "монография"]
            },

            "formats": {
                "учебник": ["учебник", "пособие", "курс", "лекции"],
                "монография": ["монография", "исследование", "диссертация", "труд"],
                "справочник": ["справочник", "энциклопедия", "словарь", "каталог"],
                "сборник": ["сборник", "антология", "хрестоматия", "альманах"],
                "художественное произведение": ["роман", "повесть", "рассказ", "поэма"],
                "биография": ["биография", "автобиография", "мемуары", "жизнеописание"],
                "практическое руководство": ["руководство", "инструкция", "методика", "рекомендации"]
            },

            "writing_styles": {
                "научный": ["научный", "академический", "исследовательский", "теоретический"],
                "публицистический": ["публицистический", "журналистский", "эссе", "очерк"],
                "художественный": ["художественный", "литературный", "поэтический", "образный"],
                "деловой": ["деловой", "официальный", "документальный", "отчетный"],
                "разговорный": ["разговорный", "неформальный", "простой", "доступный"]
            },

            "special_features": {
                "иллюстрированное": ["иллюстрация", "рисунок", "фотография", "диаграмма", "график"],
                "интерактивное": ["интерактивный", "упражнение", "задание", "тест", "опрос"],
                "с аудио": ["аудио", "озвучка", "подкаст", "запись", "звук"],
                "с видео": ["видео", "фильм", "ролик", "анимация", "мультимедиа"],
                "электронное": ["электронный", "цифровой", "онлайн", "веб", "приложение"]
            },

            "geographical_regions": {
                "россия": ["россия", "русский", "москва", "петербург", "сибирь"],
                "европа": ["европа", "европейский", "франция", "германия", "англия"],
                "азия": ["азия", "азиатский", "китай", "япония", "индия"],
                "америка": ["америка", "американский", "сша", "канада", "латинская"],
                "африка": ["африка", "африканский", "египет", "сахара", "саванна"]
            },

            "languages": {
                "русский": ["русский", "россия", "славянский", "кириллица"],
                "английский": ["английский", "англия", "сша", "great britain"],
                "немецкий": ["немецкий", "германия", "deutsch", "берлин"],
                "французский": ["французский", "франция", "paris", "франк"],
                "испанский": ["испанский", "испания", "мадрид", "латинская америка"]
            },

            "additional_categories": {
                "технологии": ["технология", "инновация", "гаджет", "девайс", "программное обеспечение"],
                "экология": ["экология", "окружающая среда", "природа", "загрязнение", "устойчивое развитие"],
                "здоровье": ["здоровье", "медицина", "питание", "фитнес", "спорт"],
                "путешествия": ["путешествие", "туризм", "страна", "культура", "достопримечательность"],
                "кулинария": ["кулинария", "рецепт", "блюдо", "готовка", "продукт"]
            }
        }

    def _prepare_regex_patterns(self):
        """Подготавливает регулярные выражения для всех категорий и тегов"""
        self.patterns = {}

        for category, tags in self.taxonomy.items():
            self.patterns[category] = {}
            for tag_name, tag_config in tags.items():
                if isinstance(tag_config, dict) and 'keywords' in tag_config:
                    keywords = tag_config['keywords']
                    # Добавляем подкатегории если есть
                    if 'subcategories' in tag_config:
                        for subcat, sub_keywords in tag_config['subcategories'].items():
                            keywords.extend(sub_keywords)
                    if 'subgenres' in tag_config:
                        for subgenre, sub_keywords in tag_config['subgenres'].items():
                            keywords.extend(sub_keywords)

                    # Создаем паттерн
                    pattern_str = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
                    self.patterns[category][tag_name] = re.compile(pattern_str, re.IGNORECASE)

    def analyze_book(self, text: str, title: str = None, metadata: Dict = None) -> Tuple[
        str, Dict, Dict, Tuple[Tuple[str, List[str]], ...], Dict, List[Dict], Dict, Dict, str, str]:
        """
        Полный анализ книги с тегированием по всем категориям

        Args:
            text: Текст книги
            title: Название книги
            metadata: Дополнительные метаданные

        Returns:
            Кортеж с результатами анализа:
            (
                title: str,
                metadata: Dict,
                text_statistics: Dict,
                tags_by_category: Tuple[Tuple[str, List[str]], ...],  # Кортеж пар (категория, теги)
                tag_details: Dict,
                top_tags: List[Dict],
                tag_profile: Dict,
                recommendations: Dict,
                processing_timestamp: str,
                tagger_version: str
            )
        """
        logger.info(f"Начинаю анализ книги: {title or 'Без названия'}")

        # Статистика текста
        text_stats = self._analyze_text_statistics(text)

        # Извлечение тегов по всем категориям
        all_tags = {}
        tag_details = {}

        for category in self.taxonomy.keys():
            category_tags = self._extract_tags_from_category(text, category)
            if category_tags:
                all_tags[category] = [tag['tag'] for tag in category_tags]
                tag_details[category] = category_tags

        # Топ общих тегов (из всех категорий)
        top_tags = self._get_top_tags_overall(tag_details)

        # Построение тегового профиля
        tag_profile = self._build_tag_profile(tag_details)

        # Рекомендации на основе тегов
        recommendations = self._generate_recommendations(tag_profile, text_stats)

        # Преобразуем словарь тегов в кортеж пар (категория, теги)
        tags_tuple = tuple((category, tags) for category, tags in all_tags.items())

        # Формируем результат в виде кортежа
        result = (
            title or "Неизвестно",  # 0: title
            metadata or {},  # 1: metadata
            text_stats,  # 2: text_statistics
            tags_tuple,  # 3: tags_by_category (кортеж пар)
            tag_details,  # 4: tag_details
            top_tags,  # 5: top_tags
            tag_profile,  # 6: tag_profile
            recommendations,  # 7: recommendations
            datetime.now().isoformat(),  # 8: processing_timestamp
            "2.0"  # 9: tagger_version
        )

        # Обновляем статистику
        self._update_statistics(result)

        logger.info(f"Анализ завершен. Найдено тегов: {sum(len(tags) for _, tags in tags_tuple)}")
        return result

    def _extract_tags_from_category(self, text: str, category: str) -> List[Dict[str, Any]]:
        """
        Извлекает теги из конкретной категории
        """
        text_lower = text.lower()
        tags_with_scores = []

        if category not in self.patterns:
            return []

        patterns = self.patterns[category]
        total_words = len(text_lower.split())

        if total_words == 0:
            return []

        for tag_name, pattern in patterns.items():
            # Находим все совпадения
            matches = pattern.findall(text_lower)

            if not matches:
                continue

            # Уникальные совпадения
            unique_matches = set(matches)
            match_count = len(unique_matches)

            # Подсчитываем общее количество вхождений
            total_occurrences = len(matches)

            # Рассчитываем различные метрики
            frequency = total_occurrences / total_words * 1000  # на 1000 слов

            # TF (Term Frequency)
            tf = total_occurrences / total_words if total_words > 0 else 0

            # Рассчитываем плотность (сколько разных ключевых слов найдено)
            unique_density = match_count / 10  # нормализуем

            # Рассчитываем уверенность
            base_confidence = min(total_occurrences / 5, 1.0) * 0.6
            density_confidence = min(unique_density, 1.0) * 0.4
            confidence = base_confidence + density_confidence

            # Учитываем частоту
            frequency_bonus = min(frequency / 10, 0.3)
            confidence = min(confidence + frequency_bonus, 1.0)

            if confidence >= self.params['min_confidence']:
                tag_info = {
                    "tag": tag_name,
                    "confidence": round(confidence, 3),
                    "match_count": total_occurrences,
                    "unique_matches": len(unique_matches),
                    "frequency_per_1000_words": round(frequency, 2),
                    "tf_score": round(tf, 5),
                    "found_keywords": list(unique_matches)[:10],
                    "category": category
                }

                tags_with_scores.append(tag_info)

        # Сортируем по уверенности и берем топ-N
        tags_with_scores.sort(key=lambda x: x["confidence"], reverse=True)
        return tags_with_scores[:self.params['max_tags_per_category']]

    def _get_top_tags_overall(self, tag_details: Dict) -> List[Dict[str, Any]]:
        """
        Возвращает топ тегов из всех категорий
        """
        all_tags = []
        for category, tags in tag_details.items():
            for tag in tags:
                # Взвешиваем уверенность в зависимости от категории
                category_weight = self._get_category_weight(category)
                weighted_confidence = tag['confidence'] * category_weight

                all_tags.append({
                    "tag": tag['tag'],
                    "confidence": tag['confidence'],
                    "weighted_confidence": round(weighted_confidence, 3),
                    "category": category,
                    "match_count": tag['match_count']
                })

        # Сортируем по взвешенной уверенности
        all_tags.sort(key=lambda x: x['weighted_confidence'], reverse=True)
        return all_tags[:self.params['top_n_overall']]

    def _get_category_weight(self, category: str) -> float:
        """
        Возвращает вес категории для взвешивания тегов
        """
        weights = {
            'academic_subjects': 1.2,
            'genres': 1.1,
            'time_periods': 1.0,
            'audience': 0.9,
            'complexity_levels': 1.0,
            'formats': 0.8,
            'writing_styles': 0.9,
            'special_features': 0.7,
            'geographical_regions': 0.8,
            'languages': 0.8,
            'additional_categories': 0.9
        }
        return weights.get(category, 1.0)

    def _build_tag_profile(self, tag_details: Dict) -> Dict[str, Any]:
        """
        Строит профиль тегов книги
        """
        profile = {
            "primary_subjects": [],
            "genres": [],
            "target_audience": None,
            "complexity": None,
            "format": None,
            "time_period": None,
            "geographic_focus": None,
            "writing_style": None
        }

        # Находим наиболее вероятные значения для каждой характеристики
        for category, tags in tag_details.items():
            if not tags:
                continue

            if category == 'academic_subjects':
                profile['primary_subjects'] = [t['tag'] for t in tags[:3]]
            elif category == 'genres':
                profile['genres'] = [t['tag'] for t in tags[:2]]
            elif category == 'audience' and tags:
                profile['target_audience'] = tags[0]['tag']
            elif category == 'complexity_levels' and tags:
                profile['complexity'] = tags[0]['tag']
            elif category == 'formats' and tags:
                profile['format'] = tags[0]['tag']
            elif category == 'time_periods' and tags:
                profile['time_period'] = tags[0]['tag']
            elif category == 'geographical_regions' and tags:
                profile['geographic_focus'] = tags[0]['tag']
            elif category == 'writing_styles' and tags:
                profile['writing_style'] = tags[0]['tag']

        return profile

    def _generate_recommendations(self, tag_profile: Dict, stats: Dict) -> Dict[str, List[str]]:
        """
        Генерирует рекомендации на основе тегового профиля
        """
        recommendations = {
            "similar_books": [],
            "complementary_topics": [],
            "reading_tips": []
        }

        # Рекомендации по похожим книгам
        subjects = tag_profile.get('primary_subjects', [])
        genres = tag_profile.get('genres', [])

        if subjects and genres:
            for subject in subjects[:2]:
                for genre in genres[:2]:
                    recommendations["similar_books"].append(
                        f"Книги по теме '{subject}' в жанре '{genre}'"
                    )

        # Комплементарные темы
        subject_complements = {
            "математика": ["физика", "информатика", "статистика"],
            "программирование": ["математика", "алгоритмы", "инженерия"],
            "история": ["политология", "социология", "культурология"],
            "философия": ["психология", "литература", "религия"],
            "психология": ["социология", "философия", "медицина"]
        }

        for subject in subjects:
            if subject in subject_complements:
                recommendations["complementary_topics"].extend(subject_complements[subject])

        # Советы по чтению
        complexity = tag_profile.get('complexity')
        if complexity == 'начальный':
            recommendations["reading_tips"].append(
                "Подходит для начинающих. Рекомендуется читать последовательно, делая заметки."
            )
        elif complexity == 'продвинутый':
            recommendations["reading_tips"].append(
                "Для понимания материала могут потребоваться предварительные знания. "
                "Рекомендуется иметь под рукой справочную литературу."
            )

        if stats['estimated_complexity'] == 'сложный':
            recommendations["reading_tips"].append(
                "Текст имеет высокую сложность. Рекомендуется читать небольшими порциями."
            )

        # Убираем дубликаты
        for key in recommendations:
            recommendations[key] = list(set(recommendations[key]))

        return recommendations

    def _analyze_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Анализирует статистику текста
        """
        # Очистка текста
        text_clean = re.sub(r'[^\w\s.,!?;:()\-—]', ' ', text, flags=re.UNICODE)

        # Разделение на слова и предложения
        words = text_clean.split()
        sentences = re.split(r'[.!?]+', text_clean)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Базовые метрики
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)

        # Уникальные слова
        unique_words = len(set(w.lower() for w in words))

        # Статистика по длине слов
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / word_count if word_count > 0 else 0

        # Статистика по предложениям
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / sentence_count if sentence_count > 0 else 0

        # Определяем уровень сложности
        if avg_sentence_length < 10:
            complexity = "простой"
        elif avg_sentence_length < 20:
            complexity = "средний"
        elif avg_sentence_length < 30:
            complexity = "сложный"
        else:
            complexity = "очень сложный"

        # Частота знаков препинания
        punctuation_counts = {}
        for punct in ['.', ',', '!', '?', ';', ':']:
            punctuation_counts[punct] = text.count(punct)

        # Доля уникальных слов
        unique_ratio = unique_words / word_count if word_count > 0 else 0

        return {
            "total_words": word_count,
            "total_sentences": sentence_count,
            "total_characters": char_count,
            "unique_words": unique_words,
            "unique_word_ratio": round(unique_ratio * 100, 2),
            "average_word_length": round(avg_word_length, 1),
            "average_sentence_length": round(avg_sentence_length, 1),
            "estimated_complexity": complexity,
            "punctuation_distribution": punctuation_counts,
            "vocabulary_richness": self._calculate_vocabulary_richness(words)
        }

    def _calculate_vocabulary_richness(self, words: List[str]) -> str:
        """
        Оценивает богатство словарного запаса
        """
        if not words:
            return "не определено"

        unique_ratio = len(set(w.lower() for w in words)) / len(words)

        if unique_ratio > 0.7:
            return "очень богатый"
        elif unique_ratio > 0.5:
            return "богатый"
        elif unique_ratio > 0.3:
            return "средний"
        else:
            return "ограниченный"

    def _update_statistics(self, result: Tuple):
        """
        Обновляет статистику теггера
        """
        self.stats['books_processed'] += 1

        # Достаем теги из кортежа (элемент с индексом 3)
        tags_by_category = result[3]
        total_tags = sum(len(tags) for _, tags in tags_by_category)
        self.stats['total_tags_assigned'] += total_tags

        # Обновляем распределение тегов
        for category, tags in tags_by_category:
            for tag in tags:
                self.stats['tag_distribution'][f"{category}:{tag}"] += 1

    def analyze_file(self, file_path: str, title: str = None, metadata: Dict = None) -> Tuple:
        """
        Анализирует книгу из файла
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Добавляем информацию о файле
            file_info = {
                "file_path": file_path,
                "file_size_bytes": len(text.encode('utf-8')),
                "encoding": "utf-8"
            }

            if metadata is None:
                metadata = {}
            metadata['file_info'] = file_info

            result = self.analyze_book(text, title, metadata)
            return result

        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
            return ("Ошибка", {"error": str(e)}, {}, (), {}, [], {}, {}, datetime.now().isoformat(), "2.0")

    def batch_analyze(self, books: List[Dict]) -> List[Tuple]:
        """
        Анализирует несколько книг
        """
        results = []
        for i, book in enumerate(books):
            logger.info(f"Обработка книги {i + 1}/{len(books)}: {book.get('title', f'Книга {i + 1}')}")

            if 'text' in book:
                result = self.analyze_book(
                    book['text'],
                    book.get('title'),
                    book.get('metadata', {})
                )
            elif 'file_path' in book:
                result = self.analyze_file(
                    book['file_path'],
                    book.get('title'),
                    book.get('metadata', {})
                )
            else:
                result = ("Ошибка", {"error": "Не указан текст или путь к файлу"}, {}, (), {}, [], {}, {},
                          datetime.now().isoformat(), "2.0")

            results.append(result)

        return results

    def save_analysis(self, analysis: Tuple, output_file: str):
        """
        Сохраняет анализ в JSON файл
        """
        try:
            # Преобразуем кортеж в словарь для сохранения в JSON
            analysis_dict = {
                "title": analysis[0],
                "metadata": analysis[1],
                "text_statistics": analysis[2],
                "tags_by_category": dict(analysis[3]),  # Преобразуем кортеж обратно в словарь
                "tag_details": analysis[4],
                "top_tags": analysis[5],
                "tag_profile": analysis[6],
                "recommendations": analysis[7],
                "processing_timestamp": analysis[8],
                "tagger_version": analysis[9]
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_dict, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Анализ сохранен в {output_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении: {e}")
            return False

    def save_statistics(self, output_file: str = "tagger_statistics.json"):
        """
        Сохраняет статистику теггера
        """
        stats_report = {
            "statistics": self.stats,
            "parameters": self.params,
            "categories_count": len(self.taxonomy),
            "total_tags_available": sum(len(tags) for tags in self.taxonomy.values()),
            "report_timestamp": datetime.now().isoformat()
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats_report, f, ensure_ascii=False, indent=2)
            logger.info(f"Статистика сохранена в {output_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении статистики: {e}")
            return False

    def export_taxonomy(self, output_file: str = "taxonomy_export.json"):
        """
        Экспортирует таксономию тегов
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.taxonomy, f, ensure_ascii=False, indent=2)
            logger.info(f"Таксономия экспортирована в {output_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при экспорте таксономии: {e}")
            return False


# Утилитарные функции
def create_sample_texts():
    """Создает образцы текстов для тестирования"""
    return {
        "math_textbook": """
        УЧЕБНИК ПО ВЫСШЕЙ МАТЕМАТИКЕ
        Автор: И.И. Иванов
        Издательство: "Наука", 2020

        Глава 1. Введение в математический анализ

        1.1. Основные понятия
        Математический анализ — раздел математики, изучающий функции, пределы, производные и интегралы.

        Определение 1.1: Функция f(x) называется непрерывной в точке x0, если...

        Теорема 1.1 (Основная теорема анализа): Если функция f непрерывна на отрезке [a, b], то...

        Доказательство: Рассмотрим разбиение отрезка...

        Пример 1.1: Найти производную функции f(x) = x^2 + 3x - 5.
        Решение: f'(x) = 2x + 3.

        Упражнения:
        1. Вычислите предел: lim(x→0) (sin x)/x
        2. Найдите производную функции: f(x) = e^x * ln(x)
        3. Решите дифференциальное уравнение: y' + 2y = x

        Контрольные вопросы:
        1. Сформулируйте определение производной.
        2. Что такое интеграл Римана?
        3. Объясните геометрический смысл определенного интеграла.

        Приложение: Таблица интегралов и производных
        """,

        "programming_book": """
        PYTHON ДЛЯ НАЧИНАЮЩИХ
        Практическое руководство по программированию

        Глава 1. Основы Python
        Python — интерпретируемый язык программирования высокого уровня с динамической типизацией.

        Пример кода:
        def hello_world():
            print("Hello, World!")

        Глава 2. Структуры данных
        2.1. Списки (lists)
        Список — упорядоченная изменяемая коллекция элементов.

        Пример:
        numbers = [1, 2, 3, 4, 5]
        numbers.append(6)

        2.2. Словари (dictionaries)
        Словарь — неупорядоченная коллекция пар ключ-значение.

        Пример:
        person = {"name": "John", "age": 30, "city": "Moscow"}

        Упражнения:
        1. Напишите функцию для вычисления факториала.
        2. Создайте класс "Студент" с атрибутами имя, возраст, оценки.
        3. Реализуйте алгоритм бинарного поиска.

        Глава 3. Веб-разработка на Django
        Django — фреймворк для быстрой разработки веб-приложений.

        Пример модели:
        class Book(models.Model):
            title = models.CharField(max_length=200)
            author = models.ForeignKey(Author, on_delete=models.CASCADE)
            published_date = models.DateField()

        Приложение: Шпаргалка по синтаксису Python
        """,

        "history_book": """
        ИСТОРИЯ РОССИИ XX ВЕКА
        Монография профессора П.П. Петрова

        Введение
        XX век стал переломным периодом в истории России, отмеченным революциями, войнами и социальными преобразованиями.

        Глава 1. Революция 1917 года
        Февральская революция привела к падению монархии, Октябрьская революция установила советскую власть.

        Глава 2. Великая Отечественная война 1941-1945
        Война против нацистской Германии стала испытанием для советского народа.

        Глава 3. Распад СССР
        В 1991 году Советский Союз прекратил свое существование, образовалось Содружество Независимых Государств.

        Заключение
        История России XX века демонстрирует сложный путь развития от империи через советский период к современной федерации.

        Приложение: Хронологическая таблица событий
        Библиография: Список использованных источников
        """
    }


def print_analysis_results(results: Tuple):
    """Красиво выводит результаты анализа"""
    print("\n" + "=" * 80)
    print(f"📚 АНАЛИЗ КНИГИ: {results[0]}")
    print("=" * 80)

    # Статистика текста
    stats = results[2]
    print(f"\n📊 СТАТИСТИКА ТЕКСТА:")
    print(f"   • Слов: {stats.get('total_words', 0):,}")
    print(f"   • Уникальных слов: {stats.get('unique_words', 0):,} ({stats.get('unique_word_ratio', 0)}%)")
    print(f"   • Предложений: {stats.get('total_sentences', 0):,}")
    print(f"   • Уровень сложности: {stats.get('estimated_complexity', 'неизвестно')}")
    print(f"   • Богатство словаря: {stats.get('vocabulary_richness', 'неизвестно')}")

    # Топ теги
    top_tags = results[5]
    if top_tags:
        print(f"\n🏷️  ТОП-ТЕГОВ:")
        for i, tag in enumerate(top_tags[:10], 1):
            print(f"   {i:2}. [{tag['category']}] {tag['tag']} - {tag['confidence']:.1%}")

    # Теги по категориям (из кортежа)
    tags_by_cat = results[3]
    if tags_by_cat:
        print(f"\n📂 ТЕГИ ПО КАТЕГОРИЯМ:")
        for category, tags in tags_by_cat:
            if tags:
                tag_names = ', '.join(tags[:3])
                print(f"   • {category}: {tag_names}")

    # Профиль книги
    profile = results[6]
    print(f"\n📋 ПРОФИЛЬ КНИГИ:")
    if profile.get('primary_subjects'):
        print(f"   • Основные темы: {', '.join(profile['primary_subjects'])}")
    if profile.get('genres'):
        print(f"   • Жанры: {', '.join(profile['genres'])}")
    if profile.get('target_audience'):
        print(f"   • Целевая аудитория: {profile['target_audience']}")
    if profile.get('complexity'):
        print(f"   • Уровень сложности: {profile['complexity']}")
    if profile.get('format'):
        print(f"   • Формат: {profile['format']}")

    print("=" * 80)


# Пример использования
def main():
    """Пример использования теггера"""

    print("📚 ПРОДВИНУТЫЙ ТЕГГЕР КНИГ")
    print("=" * 60)

    # Создаем экземпляр теггера
    tagger = AdvancedBookTagger()

    # Создаем образцы текстов
    print("\n🔄 Создаю тестовые тексты...")
    samples = create_sample_texts()

    # Анализируем книгу по математике
    print("\n🔍 Анализирую учебник по математике...")
    result = tagger.analyze_book(
        text=samples['math_textbook'],
        title='Учебник по высшей математике',
        metadata={'автор': 'И.И. Иванов', 'год': 2020}
    )

    # Выводим результаты
    print_analysis_results(result)

    # Показываем структуру возвращаемого кортежа
    print("\n📦 СТРУКТУРА ВОЗВРАЩАЕМОГО КОРТЕЖА:")
    print(f"  0. Название: {result[0]}")
    print(f"  1. Метаданные: {list(result[1].keys())}")
    print(f"  2. Статистика текста: {list(result[2].keys())}")

    # Теги по категориям (кортеж пар)
    tags_tuple = result[3]
    print(f"  3. Теги по категориям (кортеж из {len(tags_tuple)} пар):")
    for category, tags in tags_tuple:
        print(f"     • {category}: {', '.join(tags)}")

    print(f"  4. Детали тегов: словарь с {len(result[4])} категориями")
    print(f"  5. Топ тегов: список из {len(result[5])} элементов")
    print(f"  6. Профиль книги: {list(result[6].keys())}")
    print(f"  7. Рекомендации: {list(result[7].keys())}")
    print(f"  8. Время обработки: {result[8]}")
    print(f"  9. Версия теггера: {result[9]}")

    # Пример доступа к тегам по категориям
    print("\n🔍 ПРИМЕР ДОСТУПА К ТЕГАМ:")
    for category, tags in tags_tuple:
        if tags:
            print(f"  Категория '{category}': {tags}")

    # Пример анализа нескольких книг
    print("\n📚 АНАЛИЗ НЕСКОЛЬКИХ КНИГ:")
    books = [
        {
            'title': 'Python для начинающих',
            'text': samples['programming_book']
        },
        {
            'title': 'История России XX века',
            'text': samples['history_book']
        }
    ]

    batch_results = tagger.batch_analyze(books)

    for i, res in enumerate(batch_results):
        print(f"\nКнига {i + 1}: {res[0]}")
        # Доступ к тегам по категориям
        tags = res[3]
        if tags:
            print(f"  Категории с тегами: {len(tags)}")
            for category, tag_list in tags:
                if tag_list:
                    print(f"    {category}: {tag_list}")


if __name__ == "__main__":
    main()