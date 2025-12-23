"""
Пакет агентов для умной библиотеки
Содержит 5 LLM агентов для интеллектуального поиска и анализа
"""

from agents.base_agent import BaseAgent, AgentMessage
from coordinator_agent import CoordinatorAgent
from search_agent import SearchAgent
from agents.analysis_agent import AnalysisAgent
from critic_agent import CriticAgent
from recommendation_agent import RecommendationAgent

__all__ = [
    'BaseAgent',
    'AgentMessage',
    'CoordinatorAgent',
    'SearchAgent',
    'AnalysisAgent',
    'CriticAgent',
    'RecommendationAgent',
    'AgentSystem',
    'SyncAgentSystem'
]

__version__ = '1.0.0'