"""
HealthAI SFT (Supervised Fine-Tuning) Module
提供医疗模型微调、诊断生成和推理的核心功能
"""

from .finetune import HealthAIFineTuner
from .generate_diagnosis import DiagnosisGenerator  
from .inference import HealthAIInference

__all__ = [
    'HealthAIFineTuner',
    'DiagnosisGenerator', 
    'HealthAIInference'
]

__version__ = "1.0.0" 