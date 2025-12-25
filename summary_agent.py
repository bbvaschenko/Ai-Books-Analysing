import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryAgent:
    """
    Агент для создания резюме книг и контента.
    Использует GigaChat для генерации качественных резюме.
    """
    
    def __init__(self, gigachat_client):
        self.client = gigachat_client
    
    def create_summary(self, content: str, title: str = "", 
                      summary_type: str = "detailed") -> Dict[str, Any]:
        """
        Создание резюме контента.
        
        Args:
            content: Текст для резюмирования
            title: Название контента
            summary_type: Тип резюме (brief/detailed/academic)
        
        Returns:
            Структурированное резюме
        """
        logger.info(f"Создание резюме: {title}")
        
        try:
            # Выбор промпта в зависимости от типа
            if summary_type == "brief":
                return self._create_brief_summary(content, title)
            elif summary_type == "academic":
                return self._create_academic_summary(content, title)
            else:  # detailed
                return self._create_detailed_summary(content, title)
                
        except Exception as e:
            logger.error(f"Ошибка создания резюме: {e}")
            return self._create_error_summary(title, str(e))
    
    def _create_detailed_summary(self, content: str, title: str) -> Dict[str, Any]:
        """Создание детального резюме"""
        system_prompt = """
        Ты - эксперт по созданию детальных и структурированных резюме.
        
        Создай резюме, которое включает:
        1. Ключевые идеи и тезисы
        2. Основные аргументы
        3. Важные примеры и иллюстрации
        4. Структуру содержания
        5. Практическую значимость
        
        Резюме должно быть полезным для быстрого понимания содержания.
        """
        
        user_prompt = f"""
        НАЗВАНИЕ: {title if title else "Без названия"}
        
        СОДЕРЖАНИЕ:
        {content[:5000]}  # Ограничиваем для экономии токенов
        
        Создай детальное резюме этого материала.
        
        Формат JSON:
        {{
            "title": "название",
            "content_length": число_символов,
            "summary_type": "детальное",
            "key_ideas": [
                {{
                    "idea": "ключевая идея",
                    "importance": "высокая/средняя/низкая",
                    "evidence": "подтверждение из текста"
                }}
            ],
            "main_arguments": [
                "аргумент1",
                "аргумент2"
            ],
            "structure_overview": {{
                "main_sections": [
                    {{
                        "section": "раздел",
                        "purpose": "цель",
                        "key_points": ["пункт1", "пункт2"]
                    }}
                ],
                "logical_flow": "описание логики изложения"
            }},
            "practical_applications": [
                "применение1",
                "применение2"
            ],
            "target_audience": "для кого предназначен материал",
            "complexity_level": "начальный/средний/продвинутый",
            "time_to_read": "оценочное время чтения полного текста",
            "summary_text": "полный текст резюме",
            "tldr": "краткое резюме в 1-2 предложениях"
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=2000
            )
            
            # Добавляем метаданные
            response["generation_timestamp"] = datetime.now().isoformat()
            response["original_content_length"] = len(content)
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка создания детального резюме: {e}")
            raise
    
    def _create_brief_summary(self, content: str, title: str) -> Dict[str, Any]:
        """Создание краткого резюме"""
        system_prompt = """
        Создай краткое, но информативное резюме.
        Уложись в 3-5 ключевых пунктов.
        """
        
        user_prompt = f"""
        НАЗВАНИЕ: {title if title else "Без названия"}
        
        СОДЕРЖАНИЕ:
        {content[:3000]}
        
        Создай краткое резюме.
        
        Формат JSON:
        {{
            "title": "название",
            "brief_summary": "краткое резюме текстом",
            "key_points": ["пункт1", "пункт2", "пункт3"],
            "main_takeaway": "главный вывод",
            "useful_for": ["ситуация1", "ситуация2"],
            "summary_length": "короткое/среднее"
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=500
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка создания краткого резюме: {e}")
            raise
    
    def _create_academic_summary(self, content: str, title: str) -> Dict[str, Any]:
        """Создание академического резюме"""
        system_prompt = """
        Ты - академический рецензент. Создай резюме в академическом стиле.
        Включи: гипотезы, методологию, результаты, выводы.
        """
        
        user_prompt = f"""
        АКАДЕМИЧЕСКИЙ МАТЕРИАЛ:
        Название: {title}
        
        Содержание:
        {content[:4000]}
        
        Создай академическое резюме.
        
        Формат JSON:
        {{
            "title": "название",
            "research_question": "исследовательский вопрос",
            "hypotheses": ["гипотеза1", "гипотеза2"],
            "methodology": "методология исследования",
            "key_findings": [
                {{
                    "finding": "находка",
                    "significance": "значимость",
                    "evidence": "доказательства"
                }}
            ],
            "conclusions": ["вывод1", "вывод2"],
            "limitations": ["ограничение1", "ограничение2"],
            "future_research": "направления будущих исследований",
            "academic_contribution": "вклад в науку",
            "citations_style": "стиль цитирования если есть"
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1500
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка создания академического резюме: {e}")
            raise
    
    def compare_summaries(self, summaries: List[Dict]) -> Dict[str, Any]:
        """
        Сравнение нескольких резюме.
        """
        system_prompt = """
        Сравни несколько резюме и выдели их общие и отличительные черты.
        """
        
        user_prompt = f"""
        РЕЗЮМЕ ДЛЯ СРАВНЕНИЯ:
        {json.dumps(summaries, ensure_ascii=False, indent=2)}
        
        Сравни эти резюме и создай сравнительный анализ.
        
        Формат JSON:
        {{
            "common_themes": ["тема1", "тема2"],
            "unique_aspects": [
                {{
                    "summary_index": 1,
                    "unique_features": ["особенность1", "особенность2"]
                }}
            ],
            "coverage_comparison": {{
                "most_comprehensive": "индекс самого полного резюме",
                "most_concise": "индекс самого краткого",
                "balance_score": [0.5, 0.7, ...]  # оценка баланса каждого
            }},
            "recommended_use": [
                {{
                    "summary_index": 1,
                    "best_for": "для чего лучше всего",
                    "why": "почему"
                }}
            ],
            "quality_assessment": [
                {{
                    "summary_index": 1,
                    "clarity": 0-10,
                    "completeness": 0-10,
                    "accuracy": 0-10,
                    "overall": 0-10
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.error(f"Ошибка сравнения резюме: {e}")
            raise
    
    def _create_error_summary(self, title: str, error: str) -> Dict[str, Any]:
        """Создание резюме об ошибке"""
        return {
            "title": title,
            "error": error,
            "summary_type": "error",
            "key_ideas": [],
            "summary_text": f"Не удалось создать резюме: {error}",
            "generation_timestamp": datetime.now().isoformat()
        }
    
    def save_summary_to_file(self, summary: Dict, output_path: str) -> bool:
        """
        Сохранение резюме в файл.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Резюме сохранено в {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения резюме: {e}")
            return False


# Пример использования
if __name__ == "__main__":
    from gigachat_client import GigaChatClient
    
    # Инициализация
    client = GigaChatClient(verify_ssl=False)
    summary_agent = SummaryAgent(client)
    
    # Тестовый контент
    test_content = """
    Машинное обучение — это подраздел искусственного интеллекта, 
    который изучает методы построения алгоритмов, способных обучаться на основе данных.
    
    Основные типы:
    1. Обучение с учителем - алгоритм обучается на размеченных данных
    2. Обучение без учителя - алгоритм ищет паттерны в данных без меток
    3. Обучение с подкреплением - алгоритм учится через взаимодействие со средой
    
    Применение: распознавание изображений, обработка естественного языка.
    """
    
    # Создание резюме
    summary = summary_agent.create_summary(
        content=test_content,
        title="Введение в машинное обучение",
        summary_type="detailed"
    )
    
    print("=" * 80)
    print(f"РЕЗЮМЕ: {summary.get('title')}")
    print("=" * 80)
    
    if "error" not in summary:
        print(f"Ключевые идеи: {len(summary.get('key_ideas', []))}")
        for idea in summary.get('key_ideas', [])[:3]:
            print(f"  • {idea.get('idea', '')}")
        
        print(f"\nTLDR: {summary.get('tldr', '')}")
        print(f"\nПолное резюме: {summary.get('summary_text', '')[:200]}...")
    else:
        print(f"Ошибка: {summary.get('error')}")