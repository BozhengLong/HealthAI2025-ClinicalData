"""
HealthAI项目统一错误处理模块
提供自定义异常类和常用的安全处理函数
"""

import json
import logging
import time
import traceback
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# 自定义异常类
class HealthAIException(Exception):
    """HealthAI项目的基础异常类"""
    pass

class ModelLoadError(HealthAIException):
    """模型加载相关错误"""
    pass

class DataValidationError(HealthAIException):
    """数据验证相关错误"""
    pass

class APIError(HealthAIException):
    """API调用相关错误"""
    pass

class ConfigError(HealthAIException):
    """配置相关错误"""
    pass

def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    安全的JSON解析函数
    
    Args:
        text: 要解析的JSON字符串
        default: 解析失败时的默认返回值
        
    Returns:
        解析后的对象，或默认值
    """
    if not text or not text.strip():
        logger.warning("尝试解析空字符串为JSON")
        return default
    
    try:
        # 首先尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        try:
            import re
            # 寻找JSON对象
            json_pattern = r'({[\s\S]*})'
            matches = re.findall(json_pattern, text)
            
            # 按长度排序，尝试最长的匹配
            matches.sort(key=len, reverse=True)
            
            for match in matches:
                try:
                    result = json.loads(match)
                    logger.info("成功从文本中提取JSON")
                    return result
                except json.JSONDecodeError:
                    continue
            
            # 如果所有尝试都失败
            logger.error(f"JSON解析失败，文本内容: {text[:200]}...")
            return default
            
        except Exception as e:
            logger.error(f"JSON解析过程中发生错误: {str(e)}")
            return default

def safe_api_call(func: Callable, *args, max_retries: int = 3, 
                 retry_delay: float = 1.0, **kwargs) -> Any:
    """
    安全的API调用包装器，支持重试机制
    
    Args:
        func: 要调用的函数
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        *args, **kwargs: 传递给函数的参数
        
    Returns:
        函数调用结果
        
    Raises:
        APIError: 当所有重试都失败时
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # 指数退避
                logger.info(f"等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
    
    # 所有重试都失败
    error_msg = f"API调用在 {max_retries} 次尝试后仍然失败: {str(last_exception)}"
    logger.error(error_msg)
    raise APIError(error_msg) from last_exception

def validate_data_format(data: Dict, required_fields: list, 
                        data_id: str = "unknown") -> bool:
    """
    验证数据格式是否符合要求
    
    Args:
        data: 要验证的数据字典
        required_fields: 必需字段列表
        data_id: 数据标识符（用于日志）
        
    Returns:
        验证是否通过
        
    Raises:
        DataValidationError: 当验证失败时
    """
    missing_fields = []
    empty_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif not data[field] or (isinstance(data[field], str) and not data[field].strip()):
            empty_fields.append(field)
    
    if missing_fields:
        error_msg = f"数据 {data_id} 缺少必需字段: {missing_fields}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)
    
    if empty_fields:
        error_msg = f"数据 {data_id} 字段为空: {empty_fields}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)
    
    return True

def log_exception(func: Callable) -> Callable:
    """
    装饰器：自动记录异常信息
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行时发生异常: {str(e)}")
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            raise
    return wrapper

def handle_memory_error(func: Callable) -> Callable:
    """
    装饰器：处理内存相关错误
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 检查是否为内存相关错误
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logger.error("GPU内存不足，尝试清理缓存...")
                try:
                    import torch
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    logger.info("缓存清理完成，建议减少批处理大小或序列长度")
                except ImportError:
                    pass
                raise ModelLoadError(f"内存不足: {str(e)}") from e
            else:
                raise
    return wrapper

def create_error_context(operation: str, **context) -> Dict:
    """
    创建错误上下文信息，便于调试
    
    Args:
        operation: 正在执行的操作
        **context: 上下文信息
        
    Returns:
        格式化的错误上下文
    """
    return {
        "operation": operation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context": context
    }

# 通用的数据字段验证函数
def validate_medical_data(data: Dict, data_id: str = "unknown") -> bool:
    """验证医学数据的通用格式"""
    required_fields = ['feature_content']
    return validate_data_format(data, required_fields, data_id)

def validate_training_data(data: Dict, data_id: str = "unknown") -> bool:
    """验证训练数据的格式"""
    required_fields = ['feature_content', 'diseases', 'reason']
    return validate_data_format(data, required_fields, data_id)

def validate_distillation_data(data: Dict, data_id: str = "unknown") -> bool:
    """验证蒸馏数据的格式"""
    required_fields = ['feature_content', 'disease_probabilities', 'reason']
    return validate_data_format(data, required_fields, data_id) 