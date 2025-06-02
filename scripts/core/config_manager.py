"""
HealthAI项目配置管理模块
提供统一的配置文件管理、默认设置等功能
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

from .error_handler import ConfigError

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型相关配置"""
    base_model_path: str = "qwen/Qwen2___5-7B-Instruct"
    max_length: int = 4096
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    save_steps: int = 200
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    load_in_8bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_gradient_checkpointing: bool = True
    temperature: float = 2.0
    alpha: float = 0.5

@dataclass
class APIConfig:
    """API相关配置"""
    api_key: Optional[str] = None
    api_base: str = "https://api.siliconflow.cn/v1/chat/completions"
    model_name: str = "qwen2.5-72b-instruct"
    max_retries: int = 3
    retry_delay: float = 1.2
    checkpoint_interval: int = 50

@dataclass
class DataConfig:
    """数据相关配置"""
    train_file: str = "train_labeled_by_qwen.jsonl"
    test_file: str = "camp_data_step_2_without_answer.jsonl"
    output_file: str = "predictions.jsonl"
    checkpoint_file: str = "checkpoint.jsonl"
    distillation_file: str = "train_distillation_data.jsonl"

@dataclass
class InferenceConfig:
    """推理相关配置"""
    max_samples: int = 10
    max_length: int = 512
    temperature: float = 0.0
    timeout: int = 10
    load_in_4bit: bool = True
    batch_size: int = 1

@dataclass
class ProjectConfig:
    """项目主配置类"""
    model: ModelConfig
    api: APIConfig
    data: DataConfig
    inference: InferenceConfig
    
    def __post_init__(self):
        """配置验证"""
        # 确保API密钥设置
        if self.api.api_key is None:
            self.api.api_key = os.getenv("OPENAI_API_KEY")
        
        # 确保API基础URL设置
        if not self.api.api_base:
            self.api.api_base = os.getenv("OPENAI_API_BASE", self.api.api_base)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        if config_dir is None:
            # 默认配置目录
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils import get_project_root
            config_dir = Path(get_project_root()) / "config"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "project_config.yaml"
        
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> ProjectConfig:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径，默认使用项目配置文件
            
        Returns:
            项目配置对象
        """
        if config_file is None:
            config_file = self.config_file
        else:
            config_file = Path(config_file)
        
        if not config_file.exists():
            logger.info(f"配置文件不存在，使用默认配置: {config_file}")
            return self.get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    data = json.load(f)
                else:  # 默认使用YAML
                    data = yaml.safe_load(f)
            
            # 将字典转换为配置对象
            config = self._dict_to_config(data)
            logger.info(f"成功加载配置文件: {config_file}")
            return config
            
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {str(e)}") from e
    
    def save_config(self, config: ProjectConfig, config_file: Optional[Union[str, Path]] = None):
        """
        保存配置到文件
        
        Args:
            config: 项目配置对象
            config_file: 配置文件路径，默认使用项目配置文件
        """
        if config_file is None:
            config_file = self.config_file
        else:
            config_file = Path(config_file)
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = self._config_to_dict(config)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:  # 默认使用YAML
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"配置文件保存成功: {config_file}")
            
        except Exception as e:
            raise ConfigError(f"保存配置文件失败: {str(e)}") from e
    
    def get_default_config(self) -> ProjectConfig:
        """获取默认配置"""
        return ProjectConfig(
            model=ModelConfig(),
            api=APIConfig(),
            data=DataConfig(),
            inference=InferenceConfig()
        )
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ProjectConfig:
        """将字典转换为配置对象"""
        try:
            model_config = ModelConfig(**data.get('model', {}))
            api_config = APIConfig(**data.get('api', {}))
            data_config = DataConfig(**data.get('data', {}))
            inference_config = InferenceConfig(**data.get('inference', {}))
            
            return ProjectConfig(
                model=model_config,
                api=api_config,
                data=data_config,
                inference=inference_config
            )
        except Exception as e:
            raise ConfigError(f"配置格式错误: {str(e)}") from e
    
    def _config_to_dict(self, config: ProjectConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        return {
            'model': asdict(config.model),
            'api': asdict(config.api),
            'data': asdict(config.data),
            'inference': asdict(config.inference)
        }
    
    def update_config(self, updates: Dict[str, Any], config_file: Optional[Union[str, Path]] = None) -> ProjectConfig:
        """
        更新配置
        
        Args:
            updates: 要更新的配置项
            config_file: 配置文件路径
            
        Returns:
            更新后的配置对象
        """
        config = self.load_config(config_file)
        
        # 更新配置
        for section, values in updates.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logger.warning(f"未知配置项: {section}.{key}")
            else:
                logger.warning(f"未知配置段: {section}")
        
        # 保存更新后的配置
        self.save_config(config, config_file)
        return config

def load_config(config_file: Optional[Union[str, Path]] = None) -> ProjectConfig:
    """便利函数：加载配置"""
    manager = ConfigManager()
    return manager.load_config(config_file)

def get_default_config() -> ProjectConfig:
    """便利函数：获取默认配置"""
    manager = ConfigManager()
    return manager.get_default_config()

def create_sample_config():
    """创建示例配置文件"""
    manager = ConfigManager()
    default_config = manager.get_default_config()
    
    # 创建示例配置文件
    sample_file = manager.config_dir / "sample_config.yaml"
    manager.save_config(default_config, sample_file)
    
    logger.info(f"示例配置文件已创建: {sample_file}")
    return sample_file

def get_config_for_script(script_name: str, config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    为特定脚本获取相关配置
    
    Args:
        script_name: 脚本名称 ('train', 'inference', 'generate', 'distill')
        config_file: 配置文件路径
        
    Returns:
        该脚本相关的配置字典
    """
    config = load_config(config_file)
    
    if script_name in ['train', 'distill', 'finetune']:
        return {
            **asdict(config.model),
            **asdict(config.data),
            'api_config': asdict(config.api)
        }
    elif script_name in ['inference', 'infer']:
        return {
            **asdict(config.inference),
            **asdict(config.data),
            'model_config': asdict(config.model)
        }
    elif script_name in ['generate', 'label']:
        return {
            **asdict(config.api),
            **asdict(config.data)
        }
    else:
        # 返回所有配置
        return {
            'model': asdict(config.model),
            'api': asdict(config.api),
            'data': asdict(config.data),
            'inference': asdict(config.inference)
        } 