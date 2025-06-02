## Author: Bozheng Long
## Date Created: 2025-03-16
## Last Modified: 2025-03-16
## Description: 工具函数模块，包含路径处理、日志设置等通用功能。


import os
import logging
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def get_project_root():
    """返回项目根目录的Path对象"""
    return PROJECT_ROOT

def get_data_path(filename=None):
    """
    获取数据目录或特定数据文件的路径
    
    Args:
        filename: 可选，数据文件名
        
    Returns:
        Path对象，指向数据目录或特定数据文件
    """
    if filename:
        return DATA_DIR / filename
    return DATA_DIR

def get_model_path(model_name=None):
    """
    获取模型目录或特定模型的路径
    
    Args:
        model_name: 可选，模型名称或子目录
        
    Returns:
        Path对象，指向模型目录或特定模型
    """
    if model_name:
        return MODELS_DIR / model_name
    return MODELS_DIR

def setup_logging(log_file=None, level=logging.INFO):
    """
    设置日志记录
    
    Args:
        log_file: 可选，日志文件路径
        level: 日志级别，默认INFO
        
    Returns:
        logger对象
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger()

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径，可以是字符串或Path对象
    """
    Path(directory).mkdir(parents=True, exist_ok=True) 