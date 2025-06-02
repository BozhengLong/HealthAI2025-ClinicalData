"""
HealthAI Knowledge Distillation Module
提供多种知识蒸馏策略和API蒸馏支持
"""

from .distillation_engine import DistillationEngine
from .api_distillation import APIDistillationEngine

__all__ = [
    'DistillationEngine',
    'APIDistillationEngine'
]

__version__ = "1.0.0" 