"""
HealthAI项目统一模型加载模块
提供统一的模型加载、检查点管理等功能
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from .error_handler import ModelLoadError, handle_memory_error, log_exception

logger = logging.getLogger(__name__)

class UnifiedModelLoader:
    """统一的模型加载器，支持各种模型类型和配置"""
    
    def __init__(self):
        self.loaded_models = {}
        self.loaded_tokenizers = {}
    
    @handle_memory_error
    @log_exception
    def load_model(self, 
                   model_type: str,
                   base_model_path: str,
                   adapter_path: Optional[str] = None,
                   load_in_8bit: bool = False,
                   load_in_4bit: bool = False,
                   device_map: str = "auto",
                   torch_dtype: torch.dtype = torch.float16,
                   use_cache: bool = True) -> Tuple[Any, Any]:
        """
        统一的模型加载方法
        
        Args:
            model_type: 模型类型 ('sft', 'distilled', 'base')
            base_model_path: 基础模型路径
            adapter_path: 适配器路径（可选）
            load_in_8bit: 是否使用8位量化
            load_in_4bit: 是否使用4位量化
            device_map: 设备映射
            torch_dtype: 数据类型
            use_cache: 是否启用缓存
            
        Returns:
            (model, tokenizer) 元组
        """
        # 生成缓存键
        cache_key = f"{model_type}_{base_model_path}_{adapter_path}"
        
        # 检查是否已加载
        if cache_key in self.loaded_models:
            logger.info(f"使用缓存的模型: {cache_key}")
            return self.loaded_models[cache_key], self.loaded_tokenizers[cache_key]
        
        logger.info(f"开始加载模型: {model_type}")
        logger.info(f"基础模型路径: {base_model_path}")
        if adapter_path:
            logger.info(f"适配器路径: {adapter_path}")
        
        # 加载分词器
        tokenizer = self._load_tokenizer(base_model_path, adapter_path)
        
        # 设置量化配置
        quantization_config = self._get_quantization_config(load_in_8bit, load_in_4bit)
        
        # 加载基础模型
        model = self._load_base_model(
            base_model_path, 
            quantization_config, 
            device_map, 
            torch_dtype
        )
        
        # 如果有适配器，加载适配器
        if adapter_path and os.path.exists(adapter_path):
            model = self._load_adapter(model, adapter_path)
        
        # 配置模型
        model.config.use_cache = use_cache
        model.eval()
        
        # 缓存模型
        self.loaded_models[cache_key] = model
        self.loaded_tokenizers[cache_key] = tokenizer
        
        logger.info("模型加载完成")
        return model, tokenizer
    
    def _load_tokenizer(self, base_model_path: str, adapter_path: Optional[str] = None) -> Any:
        """加载分词器"""
        # 对于LoRA适配器，始终从基础模型路径加载分词器
        # checkpoint目录通常只包含适配器权重，不包含完整的分词器文件
        tokenizer_path = base_model_path
        
        logger.info(f"从基础模型路径加载分词器: {tokenizer_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 确保pad_token设置正确
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info("设置pad_token_id为eos_token_id")
            
            return tokenizer
            
        except Exception as e:
            raise ModelLoadError(f"分词器加载失败: {str(e)}") from e
    
    def _get_quantization_config(self, load_in_8bit: bool, load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if load_in_4bit:
            logger.info("使用4位量化")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            logger.info("使用8位量化")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            return None
    
    def _load_base_model(self, 
                        model_path: str, 
                        quantization_config: Optional[BitsAndBytesConfig],
                        device_map: str,
                        torch_dtype: torch.dtype) -> Any:
        """加载基础模型"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            logger.info("基础模型加载成功")
            return model
            
        except Exception as e:
            raise ModelLoadError(f"基础模型加载失败: {str(e)}") from e
    
    def _load_adapter(self, model: Any, adapter_path: str) -> Any:
        """加载LoRA适配器"""
        try:
            # 检查是否为LoRA适配器
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                logger.info(f"加载LoRA适配器: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info("LoRA适配器加载成功")
            else:
                logger.warning(f"适配器路径不包含adapter_config.json: {adapter_path}")
            
            return model
            
        except Exception as e:
            raise ModelLoadError(f"适配器加载失败: {str(e)}") from e
    
    def clear_cache(self):
        """清理模型缓存"""
        self.loaded_models.clear()
        self.loaded_tokenizers.clear()
        torch.cuda.empty_cache()
        logger.info("模型缓存已清理")

def get_latest_checkpoint(model_dir: Union[str, Path]) -> Path:
    """
    获取最新的checkpoint目录
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        最新checkpoint的路径
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise ModelLoadError(f"模型目录不存在: {model_dir}")
    
    # 查找checkpoint目录
    checkpoint_dirs = sorted([
        d for d in model_dir.iterdir() 
        if d.is_dir() and d.name.startswith("checkpoint-")
    ], key=lambda d: int(d.name.split("-")[1]), reverse=True)
    
    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[0]
        logger.info(f"找到最新checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    # 如果没有找到checkpoint目录，检查是否有其他模型文件
    model_files = ["pytorch_model.bin", "model.safetensors", "adapter_model.safetensors"]
    if any((model_dir / f).exists() for f in model_files):
        logger.info(f"使用模型目录本身: {model_dir}")
        return model_dir
    
    raise ModelLoadError(f"在 {model_dir} 中未找到有效的模型文件或checkpoint")

def load_model_safely(model_path: str, 
                     model_type: str = "base",
                     adapter_path: Optional[str] = None,
                     **kwargs) -> Tuple[Any, Any]:
    """
    安全加载模型的便利函数
    
    Args:
        model_path: 模型路径
        model_type: 模型类型
        adapter_path: 适配器路径
        **kwargs: 其他参数
        
    Returns:
        (model, tokenizer) 元组
    """
    loader = UnifiedModelLoader()
    return loader.load_model(
        model_type=model_type,
        base_model_path=model_path,
        adapter_path=adapter_path,
        **kwargs
    )

def auto_detect_model_type(model_path: str) -> str:
    """
    自动检测模型类型
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型类型 ('sft', 'distilled', 'base')
    """
    model_path = Path(model_path)
    
    # 检查是否有LoRA适配器
    if (model_path / "adapter_config.json").exists():
        # 检查配置文件中是否包含蒸馏相关信息
        config_path = model_path / "adapter_config.json"
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 简单的启发式检测
            if "distill" in str(config).lower():
                return "distilled"
            else:
                return "sft"
        except Exception:
            return "sft"
    
    # 检查是否有checkpoint目录
    checkpoint_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        latest_checkpoint = get_latest_checkpoint(model_path)
        return auto_detect_model_type(str(latest_checkpoint))
    
    return "base"

def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型信息字典
    """
    model_path = Path(model_path)
    
    info = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "type": "unknown",
        "has_adapter": False,
        "has_checkpoints": False,
        "latest_checkpoint": None,
        "files": []
    }
    
    if model_path.exists():
        info["type"] = auto_detect_model_type(str(model_path))
        info["has_adapter"] = (model_path / "adapter_config.json").exists()
        
        checkpoint_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        info["has_checkpoints"] = len(checkpoint_dirs) > 0
        
        if info["has_checkpoints"]:
            info["latest_checkpoint"] = str(get_latest_checkpoint(model_path))
        
        info["files"] = [f.name for f in model_path.iterdir() if f.is_file()]
    
    return info 