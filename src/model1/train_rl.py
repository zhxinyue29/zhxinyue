import math
from transformers import AutoConfig, AutoTokenizer
import os
import re
from collections import OrderedDict
from venv import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import yaml                                                                    
import logging
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F 
from contextlib import contextmanager
import inspect
import warnings
from ..model2.inference import DeepSeekPredictor
from transformers import AutoTokenizer, AutoModel  
from sklearn.metrics import mean_squared_error, mean_absolute_error


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 使用第一个GPU
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("警告: 没有可用的GPU，将使用CPU")
class SafetyModule(nn.Module):  # 🔴 关键：继承nn.Module
    def __init__(self, input_size=773):
        super().__init__()
        self.protector = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    
    def forward(self, x):
        return self.protector(x)

config_path = '/home/liyakun/twitter-stock-prediction/configs/model1.yaml'
def load_model(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            print("配置为空或格式有误")
            return None
        return config
    except Exception as e:
        print("读取配置失败：", e)
        return None
config = load_model(config_path)

try:
    model_path = config['env']['prediction_model_path']
    print("模型路径：", model_path)
except KeyError:
    print("配置文件中未找到 prediction_model_path 字段。")
    # 你可以设置默认路径或抛出异常
if not config:
    raise RuntimeError("配置加载失败！")

# 提取参数和模型路径



class ConfigLoader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

def setup_logger(log_file: str) -> logging.Logger:
    """设置日志记录器"""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    
    # 清除之前的handlers避免重复
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def create_output_directories(config: Dict[str, Any]) -> None:
    """创建所有必要的输出目录"""
    paths = config['paths']
    # 模型输出目录 (model1_output/)
    output_dir = Path(paths.get('processed_output_dir', 'data/processed/model1_output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 评估结果目录 (model1_eval/)
    eval_dir = Path(paths.get('eval_output_dir', 'results/model1_eval'))
    eval_dir.mkdir(parents=True, exist_ok=True)
    # 模型保存目录
    model_save_dir = Path(config['paths']['output_model']).parent
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # 日志文件目录
    log_dir = Path(config['paths']['log_file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)

def convert_to_serializable(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def save_structured_data(data: pd.DataFrame, config: Dict[str, Any], filename: str = "processed_data.csv") -> Path:
    """
    保存结构化输出数据到model1_output目录
    """
    output_dir = Path(config['paths'].get('processed_output_dir', 'data/processed/model1_output'))
    output_path = output_dir / filename 
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    # 添加日期列作为结构化数据的一部分
    if 'date' not in data.columns:
        if 'timestamp' in data.columns:
            data['date'] = pd.to_datetime(data['timestamp']).dt.date
        else:
            data['date'] = pd.Timestamp.now().strftime("%Y-%m-%d")
    data.to_csv(output_path, index=False)
    logger = logging.getLogger("training")
    logger.info(f"✅ 保存结构化输出数据到: {output_path}")
    return output_path

def save_evaluation_results(results: Dict[str, Any], config: Dict[str, Any], filename: str = "evaluation_results.json") -> Tuple[Path, Path]:
    """
    保存评估结果到model1_eval目录
    """
    eval_dir = Path(config['paths'].get('eval_output_dir', 'results/model1_eval'))
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / filename
    results_serializable = convert_to_serializable(results)
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    logger = logging.getLogger("training")
    logger.info(f"✅ 保存评估结果到: {output_path}")
    # 同时保存CSV格式便于分析
    csv_path = output_path.with_suffix('.csv')
    eval_df = pd.DataFrame([results])
    eval_df.to_csv(csv_path, index=False)
    logger.info(f"✅ 同时保存CSV格式评估结果: {csv_path}")
    
    return output_path, csv_path

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """计算夏普比率"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / (excess_returns.std() + 1e-8)

def max_drawdown(values: np.ndarray) -> float:
    """计算最大回撤"""
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return np.min(drawdown)
class SafetyController:
    def __init__(self, model, optimizer, config, logger):
        self.model = model
        self.optimizer = optimizer
        self.config = config['training']['stability']
        self.logger = logger
        self.epoch_backup_count = 0
        # 创建备份目录
        self.backup_dir = Path(f"{config['paths']['log_file']}_safety_backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        # 学习率备份
        self.original_lr = optimizer.param_groups[0]['lr']
    
    def check_inputs(self, inputs, targets):
        """输入数据健康检查"""
        nan_count = torch.isnan(inputs).sum().item() + torch.isnan(targets).sum().item()
        inf_count = torch.isinf(inputs).sum().item() + torch.isinf(targets).sum().item()
        
        if nan_count > self.config.get('max_nan_allowed', 0):
            self.logger.warning(f"跳过含NaN数据的批次 (NaN数量: {nan_count})")
            return False
        
        if inf_count > self.config.get('max_inf_allowed', 0):
            self.logger.warning(f"跳过含Inf数据的批次 (Inf数量: {inf_count})")
            return False 
        return True
    
    def protect_outputs(self, outputs):
        """模型输出防护"""
        if torch.isnan(outputs).any():
            self.logger.warning("⚠️ 检测到模型输出包含NaN值，执行修复...")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
          # 修复NaN/Inf需要特殊处理（使用torch.where可导）
        outputs = torch.where(
       torch.isnan(outputs) | torch.isinf(outputs),
       torch.zeros_like(outputs),
       outputs
   )    
        # 防止指数爆炸
        outputs = torch.clamp(outputs, 
                              min=self.config['output_clip_range'][0], 
                              max=self.config['output_clip_range'][1]) 
        return outputs
    
    def check_gradients(self):
        """梯度健康检查和修复"""
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        problematic_layers = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad              
                # 修复NaN/Inf梯度
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    problematic_layers.append(name)
                    param.grad = torch.where(
                        torch.isnan(grad) | torch.isinf(grad),
                        torch.zeros_like(grad),
                        grad
                    )            
                # 梯度裁剪
                torch.clamp_(param.grad, -self.config['gradient_clip_value'], 
                            self.config['gradient_clip_value'])
        
        # 全局梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm) 
        if problematic_layers:
            self.logger.warning(f"修复梯度异常层: {problematic_layers[:3]}{'...' if len(problematic_layers)>3 else ''}")
        
        return len(problematic_layers) == 0
    
    def create_backup(self, epoch):
        """创建安全恢复点"""
        if epoch % self.config.get('backup_interval', 5) == 0:
            backup_path = self.backup_dir / f"epoch_{epoch}_safety_backup.pt"
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epoch': epoch
            }, backup_path)
            
            # 保存最近的5个备份
            backup_files = sorted(self.backup_dir.glob("*.pt"), key=os.path.getmtime)
            if len(backup_files) > 5:
                os.remove(backup_files[0])
    
    def recover(self, current_epoch):
        """从错误中恢复"""
        recovery_type = self.config.get('recovery_strategy', 'backoff')    
        if recovery_type == 'backoff':
            # 学习率退避
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.config.get('lr_backoff_factor', 0.5)            
            self.logger.warning(f"采用学习率退避策略，新学习率: {self.optimizer.param_groups[0]['lr']:.2e}")            
            # 尝试恢复最近的备份
            backup_files = sorted(self.backup_dir.glob("*.pt"), key=os.path.getmtime)
            if backup_files:
                latest_backup = backup_files[-1]
                checkpoint = torch.load(latest_backup)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.logger.warning(f"从备份恢复: {latest_backup.name} (epoch {checkpoint['epoch']})")
                return checkpoint['epoch']  # 返回恢复到的epoch
            
        elif recovery_type == 'reset':
            # 完全重置模型
            self.logger.error("执行模型完全重置!")            
            # 重新初始化模型
            for module in self.model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()            
            # 重置优化器
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.original_lr            
            return 0  # 从头开始训练
        
        return current_epoch
    
    @contextmanager
    def safe_forward_context(self):
        """安全前向传播上下文管理器"""
        try:
            # 启用PyTorch异常检测
            torch.autograd.set_detect_anomaly(True)
            yield
        except RuntimeError as e:
            self.logger.error(f"前向传播异常: {str(e)}")
            # 提取异常位置
            stack = inspect.stack()
            caller = stack[1]  # 调用safe_forward_context的函数
            self.logger.error(f"异常位置: {caller.filename}:{caller.lineno}")
            raise
        finally:
            torch.autograd.set_detect_anomaly(False)

# === 新增: 梯度监控器 ===
class GradientMonitor:
    def __init__(self, model, layers_to_watch=None):
        self.model = model
        self.layers = layers_to_watch or self._identify_critical_layers()
        self.handles = []
        self.logger = logging.getLogger("gradient_monitor")
        self.reset_stats()
    
    def _identify_critical_layers(self):
        """识别关键层（最后一层和含LayerNorm的层）"""
        critical_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                critical_layers.append(name)
            if 'out' in name or 'final' in name or 'proj' in name:
                critical_layers.append(name)
        return list(set(critical_layers))
    
    def reset_stats(self):
        self.max_grad = 0.0
        self.max_grad_layer = None
        self.nan_count = 0
    
    def _hook_fn(self, module, grad_input, grad_output, layer_name):
        # 检查NaN和Inf
        for g in grad_output:
            if g is not None:
                if torch.isnan(g).any():
                    self.nan_count += torch.isnan(g).sum().item()
                if torch.isinf(g).any():
                    self.nan_count += torch.isinf(g).sum().item()        
        # 记录最大梯度
        for g in grad_output:
            if g is not None:
                grad_norm = torch.norm(g)
                if grad_norm > self.max_grad:
                    self.max_grad = grad_norm.item()
                    self.max_grad_layer = layer_name
    
    def attach(self):
        """附加梯度钩子"""
        self.reset_stats()
        for name in self.layers:
            module = self._get_module(name)
            hook = lambda m, gi, go, n=name: self._hook_fn(m, gi, go, n)
            handle = module.register_full_backward_hook(hook)
            self.handles.append(handle)
    
    def detach(self):
        """移除梯度钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def _get_module(self, name):
        """通过名称获取模块"""
        names = name.split('.')
        module = self.model
        for n in names:
            module = getattr(module, n)
        return module
    
    def report(self, batch_idx, epoch):
        """生成梯度报告"""
        report = f"梯度监控 - Epoch {epoch} Batch {batch_idx}:\n"
        report += f"  🚩 最大梯度: {self.max_grad:.2e} ({self.max_grad_layer})\n"
        report += f"  ⚠️ 异常梯度计数: {self.nan_count}"
        
        if self.nan_count > 0:
            self.logger.warning(report)
        elif batch_idx % 10 == 0:
            self.logger.info(report)


# === 新增: SmoothL1损失函数 ===
class RLoss(nn.Module):
    def __init__(self, supervised_criterion, base_loss_weight=0.5):
        super().__init__()
        self.base_loss_weight = base_loss_weight
        self.supervised_criterion = supervised_criterion
    
    def forward(self, model_outputs, targets, reward):
        supervised_loss = self.supervised_criterion(model_outputs, targets)
        # 3. 分情况处理的策略损失
        policy_loss, match_rate = self.calculate_policy_loss(
            model_outputs, 
            targets,
            reward,
        )
        print("监督损失 grad_fn:",supervised_loss.grad_fn)
        print("策略损失 grad_fn:", policy_loss.grad_fn) 
        # 4. 动态调整权重
        volatility = torch.abs(targets).mean()
        current_weight = self.dynamic_weight_adjust(volatility)

        # 5. 混合损失 - 综合所有成分
        total_loss = (
            current_weight * supervised_loss +
            (1 - current_weight) * 0.7 * policy_loss
        )
        # 返回损失和相关指标
        return {
            "total_loss": total_loss,
            "supervised_loss": supervised_loss,
            "policy_loss": policy_loss,
            "weight": current_weight,
            "mean_reward": reward.mean(),
            "match_rate": match_rate
        }

    def calculate_policy_loss(self, model_outputs, targets, reward):
        logits = model_outputs["logits"]        
        # 确保处理批量数据
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  
        # 对特征维度应用softmax (dim=1)
        action_probs = F.softmax(logits, dim=1)        
        # 每个样本选择最大概率动作
        position_direction = torch.argmax(action_probs, dim=1)  # [batch_size]        
        # 风险判断 - 确保targets适当形状
        if targets.dim() > 1:
            targets = targets.squeeze()
        risk_mask = targets < -0.05  # [batch_size]        
        # 有效动作设置
        valid_actions = torch.zeros_like(position_direction)
        valid_actions[risk_mask] = 3        
        # 计算匹配率
        directional_match = (position_direction == valid_actions).float()        
        # 安全选择概率（避免in-place操作）
        batch_indices = torch.arange(logits.size(0))
        chosen_probs = action_probs[batch_indices, position_direction]
        log_probs = torch.log(chosen_probs + 1e-8)        
        # 策略梯度损失
        with torch.no_grad():
            advantage = reward * (1.0 + 0.5 * directional_match)
        
        rl_loss = -torch.mean(log_probs * advantage)
        
        return rl_loss, directional_match.mean()
    
    def dynamic_weight_adjust(self, volatility):
        """根据市场波动率调整监督损失权重 (完全保留)"""
        return torch.clamp(0.65 - volatility * 25, min=0.2, max=0.7).item()

def get_rl_weight(epoch):
    """根据训练进度返回监督损失权重"""
    # 第一阶段：侧重监督学习（epoch 0-9）
    if epoch < 10:
        return 0.7
    # 第二阶段：平衡学习（epoch 10-19）
    elif epoch < 20:
        # 线性过渡：0.7 → 0.5
        return 0.7 - 0.2 * (epoch - 9) / 10
    # 第三阶段：侧重强化学习（epoch ≥20）
    else:
        return 0.3
    
class SafeSmoothL1Loss(nn.Module):
    """数值稳定的SmoothL1损失"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
     
    def forward(self, model_outputs, target):
        if isinstance(model_outputs, dict):
        # 尝试常见键名（根据实际模型输出键名调整）
            if "logits" in model_outputs:
                input_tensor = model_outputs["logits"]
            elif "value" in model_outputs:
                input_tensor = model_outputs["value"]
            else:
                # 如果没有标准键名，使用第一个值
                input_tensor = next(iter(model_outputs.values()))
        else:
            input_tensor = model_outputs
        print("model_outputs",input_tensor)
        safe_input = torch.clamp(input_tensor, -50, 50)
        safe_target = torch.clamp(target, -50, 50)
        diff = safe_input - safe_target
        diff = torch.where(torch.isinf(diff), torch.zeros_like(diff), diff)
        diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
        
        abs_diff = torch.abs(diff)
        mask = abs_diff < self.beta        
        # L2部分: 0.5 * x^2 / beta
        l2_loss = 0.5 * torch.pow(diff, 2) / self.beta
        # L1部分: |x| - 0.5 * beta
        l1_loss = abs_diff - 0.5 * self.beta        
        loss = torch.where(mask, l2_loss, l1_loss)
        loss = loss.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn("检测到NaN/Inf损失值，执行修复!")
            return torch.zeros_like(loss)
            
        return loss
class CSVDataset(Dataset):
    def __init__(
        self,
        data_path: Optional[str] = None,
        filter_nan: bool = True,
        sequence_length: int = 30,
        feature_names: Optional[List[str]] = None,
        exclude_text_columns: bool = True,
        min_valid_sequences: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.sequence_length = sequence_length
        self.filter_nan = filter_nan
        self.feature_names = feature_names
        self.exclude_text_columns = exclude_text_columns
        self.min_valid_sequences = min_valid_sequences
        
        self.logger.info(f"🔍 加载数据文件: {data_path}")
        self.logger.info(f"序列长度: {sequence_length}, 过滤NaN: {filter_nan}, 最小序列数: {min_valid_sequences}")
        self.logger.info("=" * 50)
        
        # 错误日志文件路径
        error_log_dir = "logs"
        os.makedirs(error_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_log_path = os.path.join(error_log_dir, f"data_parse_errors_{timestamp}.log")
        
        # 1. 加载原始数据文本
        self.logger.info("📥 读取原始数据...")
        try:
            with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_lines = f.readlines()
        except Exception as e:
            self.logger.error(f"❌ 文件读取错误: {str(e)}")
            raise
        
        self.logger.info(f"📄 读取 {len(raw_lines)} 行数据")
        
        # 2. 提取结构化数据
        self.logger.info("\n⚙️ 解析结构化数据...")
        data_records = []
        valid_count = 0
        error_count = 0
        
        # 打印前5行用于调试
        self.logger.info(f"🔍 前5行原始数据内容:")
        for i, line in enumerate(raw_lines[:5], 1):
            self.logger.info(f"行 {i} 原始内容: {line.strip()[:200]}")
        
        with open(self.error_log_path, "w", encoding="utf-8") as error_log:
            self.logger.info(f"开始解析... (错误日志将保存到: {self.error_log_path})")
            
            for i, line in enumerate(raw_lines, 1):
                if i % 1000 == 0 or i <= 5 or i == len(raw_lines):
                    self.logger.info(f"处理中... 已处理: {i}/{len(raw_lines)} 行")
                
                try:
                    if not line.strip():
                        continue
                        
                    # 尝试解析行
                    record = self._parse_hydra_format(line)
                    
                    # DEBUG: 打印前5行的解析结果
                    if i <= 5:
                        self.logger.info(f"行 {i} 解析结果: {str(record)[:300]}")
                    
                    # 检查解析结果
                    if not isinstance(record, dict) or not record:
                        raise ValueError("解析结果为空或不是字典")
                        
                    # 🔑 键名规范化处理 (核心修改)
                    normalized_record = {}
                    for key, value in record.items():
                        # 标准化键名: 小写，去掉前后空格，替换空格为下划线
                        clean_key = str(key).strip().lower().replace(" ", "_")
                        normalized_record[clean_key] = value
                    
                    record = normalized_record
                    
                    # 检查必要字段
                    missing_fields = []
                    if 'timestamp' not in record:
                        missing_fields.append('timestamp')
                        record['timestamp'] = pd.Timestamp.now()
                    
                    if 'symbol' not in record:
                        missing_fields.append('symbol')
                        record['symbol'] = f"UNKNOWN_{i}"
                    
                    if 'sentiment' not in record:
                        missing_fields.append('sentiment')
                        record['sentiment'] = 0.0
                    
                    # 如果有缺失字段，记录警告
                    if missing_fields:
                        self.logger.debug(f"行 {i}: 缺失必要字段 {', '.join(missing_fields)}，已添加默认值")
                    
                    # 类型转换确保正确
                    try:
                        if isinstance(record['timestamp'], str):
                            record['timestamp'] = pd.to_datetime(record['timestamp'], errors='coerce')
                        if not isinstance(record['timestamp'], pd.Timestamp):
                            record['timestamp'] = pd.Timestamp.now()
                    except Exception:
                        record['timestamp'] = pd.Timestamp.now()
                    
                    try:
                        record['sentiment'] = float(record['sentiment'])
                    except:
                        record['sentiment'] = 0.0
                    
                    # 添加记录
                    data_records.append(record)
                    valid_count += 1
                    
                except Exception as e:
                    error_count += 1
                    # 记录详细错误信息
                    error_details = {
                        "error": str(e),
                        "line_number": i,
                        "raw_content": line.strip()[:500],
                        "parsed_result": str(record)[:300] if "record" in locals() else "未解析"
                    }
                    error_log.write(json.dumps(error_details, indent=2) + "\n\n")
                    
                    if error_count % 100 == 0:
                        self.logger.warning(f"解析错误数已达 {error_count} (最后错误: {str(e)})")
        
        self.logger.info(f"✅ 成功解析 {valid_count} 条记录")
        self.logger.info(f"⚠️ 解析错误: {error_count} 条 (详细日志: {self.error_log_path})")
        
        # 如果有效记录少于阈值，发出严重警告
        if valid_count < min_valid_sequences:
            self.logger.warning(f"⚠️ 有效记录数 ({valid_count}) 低于阈值 ({min_valid_sequences})")
        
        # 3. 创建DataFrame
        self.logger.info("\n📊 创建DataFrame...")
        self.df = pd.DataFrame(data_records) if data_records else pd.DataFrame()
        
        if self.df.empty:
            self.logger.warning("⚠️ 警告: DataFrame为空，添加占位符")
            self.df = pd.DataFrame({
                'timestamp': [pd.Timestamp.now() for _ in range(sequence_length)],
                'symbol': ['PLACEHOLDER' for _ in range(sequence_length)],
                'sentiment': [0.0 for _ in range(sequence_length)]
            })
        
        self.logger.info(f"📋 数据集包含 {self.df['symbol'].nunique()} 只股票/组")
        
        # 4. 处理文本列
        if exclude_text_columns:
            text_columns = [col for col in self.df.columns if col in ['text', 'content', 'message']]
            self.logger.info(f"📝 排除 {len(text_columns)} 个文本列: {text_columns}")
            self.df = self.df.drop(columns=text_columns, errors='ignore')
        
        # 5. 确定特征列
        self.feature_names = self._detect_features() if feature_names is None else feature_names
        if not self.feature_names:
            self.logger.warning("⚠️ 自动重新确定特征名称: 0 个特征")
        
        # 6. 清理和转换
        self.logger.info("\n🛠️ 数据清理和转换...")
        self._clean_and_transform()
        
        # 7. 创建序列
        self.logger.info("\n⏱️ 创建序列...")
        self._create_sequences()
        
        # 8. 过滤NaN序列
        self.logger.info("\n🧹 过滤NaN序列...")
        self._filter_nan_sequences()
        
        self.logger.info("\n🎉 数据集初始化完成")
        self.logger.info(f"📊 数据集统计:\n{self.get_data_report()}")

    def _parse_hydra_format(self, line: str) -> dict:
        """解析多种格式的时间序列数据，特别针对Python字典字符串格式"""
        try:
            # 清理输入
            clean_line = line.strip()
            
            # 新增：处理日志中错误类型2的情况 - 外部双引号包裹问题
            # 示例: "\"{'symbol': 'GOOG', ...}\""
            if clean_line.startswith('"') and clean_line.endswith('"'):
                inner = clean_line[1:-1]
                # 如果内容本身是有效的字典格式，尝试解析
                if inner.startswith("{") or inner.startswith("'{"):
                    clean_line = inner
            
            # 1. 尝试使用ast.literal_eval（最可能处理Python字典格式）
            try:
                import ast
                parsed = ast.literal_eval(clean_line)
                if isinstance(parsed, dict):
                    return parsed
            except (SyntaxError, ValueError, TypeError):
                pass
            
            # 新增：更全面的单引号转双引号处理
            # 处理类似: '{...}' 或 '{"key": "value"}' 格式
            if clean_line.startswith("{") or clean_line.startswith("{"):
                # 分步骤转换以确保安全
                normalized = clean_line
                # 替换单引号键和值
                normalized = re.sub(r"'\s*:\s*'", '": "', normalized)  # 键值对
                normalized = re.sub(r"'\s*,\s*'", '", "', normalized)  # 键值对之间
                normalized = normalized.replace("{'", '{"').replace("'}", '"}')  # 大括号
                normalized = normalized.replace(": '", ': "').replace("',", '",')  # 通用替换
                
                # 处理特殊字符
                normalized = normalized.replace("None", "null")
                normalized = normalized.replace("True", "true").replace("False", "false")
                
                # 尝试解析转换后的字符串
                try:
                    return json.loads(normalized)
                except json.JSONDecodeError:
                    pass
            
            # 2. 尝试直接解析为JSON
            try:
                return json.loads(clean_line)
            except json.JSONDecodeError:
                pass
            
            # 3. 尝试转换为JSON格式（保留原始逻辑）
            try:
                json_str = (
                    clean_line
                    .replace("'", '"')  # 替换单引号为双引号
                    .replace("None", "null")  # Python None → JSON null
                    .replace("True", "true")  # Python True → JSON true
                    .replace("False", "false")  # Python False → JSON false
                    .replace("\\\"", "'")  # 处理转义的双引号
                )
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # 4. 尝试处理嵌套引号（保留原始逻辑）
            try:
                # 使用正则表达式处理复杂引号结构
                pattern = r'("[^"]*"|\'[^\']*\')'
                fixed_line = re.sub(pattern, lambda m: m.group(0).replace('"', "!QUOTE!"), clean_line)
                fixed_line = fixed_line.replace("'", '"').replace('!QUOTE!', "'")
                return json.loads(fixed_line)
            except:
                pass
            if clean_line.startswith('{') and not clean_line.endswith('}'):
            # 尝试补全缺少的结束花括号
                fixed_line = clean_line.rstrip()  # 移除可能的空白字符
                if not fixed_line.endswith('}') and not fixed_line.endswith('"'):
                    # 添加缺失的结束结构
                    if fixed_line.rfind(',') > fixed_line.rfind(':'):
                        # 类似 'key' 这种不完整键的情况
                        fixed_line = fixed_line.rsplit(',', 1)[0] + '}'
                    else:
                        fixed_line += '}'
                        
                    # 尝试解析修复后的数据
                    try:
                        return self._parse_hydra_format(fixed_line)
                    except:
                        pass
            
            return {}
        
        except Exception as e:
            self.logger.debug(f"解析函数内部错误: {str(e)}")
            return {}

    def _detect_features(self) -> List[str]:
        """自动检测数值特征列"""
        if self.df.empty:
            return []
        
        # 排除非数值列和关键列
        non_features = ['symbol', 'timestamp', 'sentiment', 'text', 'content', 'message', 'id']
        numeric_cols = self.df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
        
        # 过滤非特征列
        features = [col for col in numeric_cols if col not in non_features]
        self.logger.info(f"📊 自动识别出 {len(features)} 个特征列: {features[:10]}{'...' if len(features) > 10 else ''}")
        return features

    def _clean_and_transform(self):
        """数据清理和转换"""
        # 1. 填充缺失值
        if self.df.empty:
            self.logger.warning("🧼 填充缺失值 (跳过，数据集为空)")
            return
            
        missing_values = self.df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.info(f"🧼 填充缺失值 (共 {missing_values} 个)")
            self.df = self.df.ffill().bfill().fillna(0)
        else:
            self.logger.info("🧼 无缺失值")

    def _create_sequences(self):
        """为每个股票/分组创建时间序列"""
        if self.df.empty:
            self.logger.warning("⚠️ 警告: 未创建任何序列，使用占位符")
            self.sequences = [np.zeros((self.sequence_length, 1))]  # 创建一个占位符序列
            self.targets = [0.0]
            return
        
        self.sequences = []
        self.targets = []
        
        # 按股票分组处理
        for symbol, group in self.df.groupby('symbol'):
            group = group.sort_values('timestamp')
            
            # 确保有足够的数据点
            if len(group) < self.sequence_length:
                self.logger.debug(f"跳过{symbol}: 数据点不足 ({len(group)} < {self.sequence_length})")
                continue
            
            # 提取特征和标签
            features = group[self.feature_names].values if self.feature_names else np.zeros((len(group), 1))
            sentiments = group['sentiment'].values
            
            # 创建序列
            for i in range(len(group) - self.sequence_length):
                seq = features[i:i+self.sequence_length]
                target = sentiments[i+self.sequence_length-1]
                self.sequences.append(seq)
                self.targets.append(target)
        
        if not self.sequences:
            self.logger.warning("⚠️ 警告: 未创建任何有效序列，使用占位符")
            self.sequences = [np.zeros((self.sequence_length, 1))]
            self.targets = [0.0]
            
        self.logger.info(f"✅ 创建了 {len(self.sequences)} 个时间序列 (来自 {self.df['symbol'].nunique()} 只股票)")
        self.logger.info(f"📐 序列形状: {self.sequences[0].shape}")

    def _filter_nan_sequences(self):
        """过滤包含NaN的序列"""
        if not self.filter_nan:
            self.logger.info("跳过过滤NaN序列")
            return
            
        original_count = len(self.sequences)
        valid_indices = [i for i, seq in enumerate(self.sequences) if not np.isnan(seq).any()]
        
        self.sequences = [self.sequences[i] for i in valid_indices]
        self.targets = [self.targets[i] for i in valid_indices]
        
        filtered_count = original_count - len(self.sequences)
        self.logger.info(f"✅ 所有 {len(self.sequences)} 个序列均有效 (过滤了 {filtered_count} 个无效序列)")
        
        # 如果序列数量不足，添加占位符
        if len(self.sequences) < self.min_valid_sequences:
            self.logger.warning(f"⚠️ 序列数量不足 ({len(self.sequences)} < {self.min_valid_sequences})，添加占位符")
            placeholder = np.zeros((self.sequence_length, len(self.feature_names) if self.feature_names else 1))
            self.sequences.append(placeholder)
            self.targets.append(0.0)

    def get_data_report(self) -> str:
        """生成数据集报告"""
        report = f"  序列数量: {len(self.sequences)}\n"
        if self.sequences:
            report += f"  序列长度: {self.sequence_length}\n"
            report += f"  特征维度: {self.sequences[0].shape[1]}\n"
            report += f"  序列形状: {self.sequences[0].shape}\n"
        
        if not self.df.empty:
            report += f"  数据起始时间: {self.df['timestamp'].min()}\n"
            report += f"  数据结束时间: {self.df['timestamp'].max()}\n"
            report += f"  股票数量: {self.df['symbol'].nunique()}\n"
            report += "  情感分布:\n"
            report += f"    中性: {np.sum(self.df['sentiment'] == 0)}\n"
            report += f"    积极: {np.sum(self.df['sentiment'] > 0)}\n"
            report += f"    消极: {np.sum(self.df['sentiment'] < 0)}"
        
        return report
    def __len__(self) -> int:
        """返回数据集中序列的数量"""
        return len(self.sequences)
    
    def __getitem__(self, index: int) -> tuple:
        """
        获取索引对应的数据样本
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (sequence_tensor, sentiment_target, metadata)
                   包含特征序列张量、情感目标值和其他元数据
        """
        # 1. 确保索引有效
        if index < 0 or index >= len(self.sequences):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self.sequences)-1}]")
        
        # 2. 获取序列特征和目标
        sequence = self.sequences[index]
        sentiment = self.targets[index]
        
        # 3. 转换为张量（非常重要！）
        try:
            # 确保数据类型正确 - 使用 float32 精度
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            sentiment_target = torch.tensor([sentiment], dtype=torch.float32)  # 使其变为张量形式
        except Exception as e:
            # 如果转换失败，记录错误并尝试处理
            self.logger.error(f"张量转换错误（索引 {index}）: {e}")
            # 回退方案：创建占位符序列
            sequence_tensor = torch.zeros(
                (self.sequence_length, len(self.feature_names) if self.feature_names else 1),
                dtype=torch.float32
            )
            sentiment_target = torch.tensor([0.0], dtype=torch.float32)
        
        # 4. 获取元数据（可选，但有价值）
        metadata = {
            'sequence_id': index,
            'features': self.feature_names,
            'sequence_length': self.sequence_length
        }
        
        return sequence_tensor, sentiment_target, metadata
# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class TransformerModel(nn.Module):
    """Deepseek 1.5B兼容模型（完全匹配预训练权重命名）"""
    def __init__(
        self,
        tokenizer_path: str = "",
        model_weights_path: Optional[str] = None,
        vocab_size: int = 51200,
        hidden_size: int = 1536,
        num_heads: int = 24,
        num_kv_heads: int = 4,
        hidden_size_mlp: int = 8960,
        num_layers: int = 30,
        output_size: int = 1,
        max_seq_len: int = 4096,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        # 保存模型配置
        self.config = {
            "tokenizer_path": tokenizer_path,
            "model_weights_path": model_weights_path,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "hidden_size_mlp": hidden_size_mlp,
            "num_layers": num_layers,
            "output_size": output_size,
            "max_seq_len": max_seq_len
        }
        
        # 1. Token嵌入（用于文本输入）
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # 新增：输入投影层（用于浮点数输入）
        self.input_projection = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        
        # 2. 旋转位置编码
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads, 
            max_seq_len=max_seq_len
        )
        
        # 3. Transformer层堆叠（使用修正后的块）
        self.layers = nn.ModuleList([
            DeepseekBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_size=hidden_size_mlp,
            )
            for _ in range(num_layers)
        ])
        
        # 4. 最终归一化层 - 匹配预训练权重命名: norm
        self.norm = DeepseekRMSNorm(hidden_size)
        
        # 5. 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
        # 设备管理
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # 记录模型信息
        self.log_model_info()
        
        # 加载预训练权重
        if model_weights_path:
            self._smart_load_weights(model_weights_path)
    
    def log_model_info(self):
        """记录模型参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔄 模型总参数量: {total_params:,}")
        print(f"🔄 可训练参数量: {trainable_params:,}")
        print(f"🔄 配置信息: {self.config}")
    
    def _smart_load_weights(self, weights_path: str):
        """智能加载模型权重，处理层级命名差异"""
        try:
            # 加载原始状态字典
            state_dict = torch.load(weights_path, map_location=self.device)
            
            print(f"🔄 原始权重包含 {len(state_dict)} 个参数")
            
            # 创建修正后的状态字典
            new_state_dict = {}
            for key, value in state_dict.items():
                # 应用重命名规则
                if key.startswith("layers."):
                    # 1. 处理归一化层
                    if "input_layernorm" in key:
                        new_key = key.replace("input_layernorm", "layers.{}.input_layernorm")
                    elif "post_attention_layernorm" in key:
                        new_key = key.replace("post_attention_layernorm", "layers.{}.post_attention_layernorm")
                    
                    # 2. 处理注意力层
                    elif "self_attn_q_proj" in key:
                        new_key = key.replace("self_attn_q_proj", "layers.{}.self_attn_q_proj")
                    elif "self_attn_k_proj" in key:
                        new_key = key.replace("self_attn_k_proj", "layers.{}.self_attn_k_proj")
                    elif "self_attn_v_proj" in key:
                        new_key = key.replace("self_attn_v_proj", "layers.{}.self_attn_v_proj")
                    elif "self_attn_o_proj" in key:
                        new_key = key.replace("self_attn_o_proj", "layers.{}.self_attn_o_proj")
                    
                    # 3. 处理MLP层
                    elif "mlp_gate_proj" in key:
                        new_key = key.replace("mlp_gate_proj", "layers.{}.mlp_gate_proj")
                    elif "mlp_up_proj" in key:
                        new_key = key.replace("mlp_up_proj", "layers.{}.mlp_up_proj")
                    elif "mlp_down_proj" in key:
                        new_key = key.replace("mlp_down_proj", "layers.{}.mlp_down_proj")
                    
                    # 应用层级索引
                    parts = key.split(".")
                    layer_idx = parts[1]  # 从 "layers.0..." 获取索引
                    new_key = new_key.format(layer_idx)
                else:
                    # 特殊处理非层级的键
                    new_key = key
                
                new_state_dict[new_key] = value
            
            # 加载修正后的权重
            result = self.load_state_dict(new_state_dict, strict=False)
            missing_keys = result.missing_keys
            unexpected_keys = result.unexpected_keys
            
            # 检查权重加载情况
            print(f"✅ 权重加载完成 ({len(state_dict) - len(missing_keys)}/{len(state_dict)} 匹配)")
            
            # 打印详细报告
            self.print_weight_report(state_dict, missing_keys, unexpected_keys)
            
        except Exception as e:
            print(f"❌ 加载权重失败: {str(e)}")
            raise
    
    def print_weight_report(self, state_dict, missing_keys, unexpected_keys):
        """打印详细的权重加载报告"""
        total = len(state_dict)
        loaded = total - len(missing_keys)
        missing_count = len(missing_keys)
        unused_count = len(unexpected_keys)
        
        print(f"\n{'='*60}")
        print("权重加载诊断报告")
        print(f"{'='*60}")
        print(f"✓ 权重文件包含 {total} 个参数")
        print(f"✓ 成功加载 {loaded} 个参数 ({loaded/total*100:.1f}%)")
        print(f"⚠️ 缺失 {missing_count} 个参数")
        print(f"⚠️ 未使用 {unused_count} 个参数")
        
        if missing_keys:
            print("\n关键缺失参数:")
            for i, key in enumerate(missing_keys[:10]):
                print(f"  {i+1}. {key}")
            if len(missing_keys) > 10:
                print(f"  ... 及 {len(missing_keys)-10} 个更多")
        
        if unexpected_keys:
            print("\n未使用的权重参数:")
            for i, key in enumerate(unexpected_keys[:10]):
                print(f"  {i+1}. {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... 及 {len(unexpected_keys)-10} 个更多")
        
        print(f"{'='*60}\n")
    
    def forward(
        self, 
        input_data: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        完整的前向传播实现
        
        参数:
            input_data: 输入张量, 形状 [batch_size, seq_len]
                        可以是整数Token(使用嵌入层)或浮点数(使用投影层)
            position_ids: 位置ID张量, 形状 [batch_size, seq_len]
            attention_mask: 注意力掩码, 形状 [batch_size, seq_len]
            
        返回:
            torch.Tensor: 输出预测结果, 形状 [batch_size, output_size]
        """
        # === 1. 输入形状检查和处理 ===
        # 确保输入是2D张量 [batch_size, seq_len]
        if input_data.dim() != 2:
            raise ValueError(f"输入张量必须是2D (batch_size, seq_len), 但是得到 {input_data.dim()}D")
        
        batch_size, seq_len = input_data.size()
        
        # === 2. 输入类型处理 ===
        # 确定输入类型（整数Token或浮点数）
        if input_data.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            # 整数输入：通过Token嵌入层
            inputs_embeds = self.embed_tokens(input_data)
        else:
            # 浮点数输入：通过投影层
            # 添加一个特征维度 [batch_size, seq_len, 1] 然后投影到隐藏维度
            inputs_embeds = self.input_projection(input_data.unsqueeze(-1))
        
        # === 3. 位置ID处理 ===
        if position_ids is None:
            # 创建默认位置ID: [0, 1, 2, ..., seq_len-1]
            position_ids = torch.arange(
                seq_len, 
                dtype=torch.long, 
                device=self.device
            ).unsqueeze(0).expand(batch_size, seq_len)
        
        # === 4. 注意力掩码处理 ===
        if attention_mask is None:
            # 创建默认全1掩码
            attention_mask = torch.ones(
                (batch_size, seq_len),
                device=self.device,
                dtype=torch.long
            )
        
        # 转换注意力掩码为浮点型并扩展维度
        # 原始形状: [batch_size, seq_len]
        # 扩展后: [batch_size, 1, 1, seq_len] (用于批处理和头维度)
        attn_mask = attention_mask.to(dtype=inputs_embeds.dtype)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # -> [batch_size, 1, 1, seq_len]
        
        # 应用负无穷值屏蔽需要关注的位置
        # 创建因果掩码（防止位置查看未来位置）
        causal_mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=self.device)
        ).to(inputs_embeds.dtype)
        
        # 合并注意力掩码和因果掩码
        combined_mask = attn_mask * causal_mask
        
        # 为屏蔽位置设置非常大的负值（用于softmax中的屏蔽）
        # 其中 combined_mask == 0 的位置需要被屏蔽
        neg_inf = -1e10
        attn_mask = combined_mask * (1.0 - neg_inf) + (1.0 - combined_mask) * neg_inf
        
        # === 5. 通过Transformer层 ===
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                x=hidden_states,
                rotary_emb=self.rotary_emb,  # 传递RotaryEmbedding实例
                position_ids=position_ids,
                attention_mask=attn_mask
            )
        
        # === 6. 最终归一化 ===
        hidden_states = self.norm(hidden_states)
        
        # === 7. 获取序列最后位置的表示 ===
        # 方法1：使用注意力掩码确定实际有效位置
        sequence_lengths = attention_mask.sum(dim=1) - 1  # 最后有效位置索引
        sequence_lengths = sequence_lengths.clamp(min=0)  # 确保非负
        
        # 创建批处理索引
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # 提取每个序列的最后有效隐藏状态
        last_hidden_states = hidden_states[batch_indices, sequence_lengths, :]
        
        # === 8. 输出预测 ===
        logits = self.output(last_hidden_states)
        
        return logits
    
class RotaryEmbedding(nn.Module):
    """旋转位置编码实现（保持不变）"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """应用旋转位置嵌入到输入张量"""
    seq_len = x.size(-2)
    cos = cos[:, :, :seq_len, :].to(x.device)
    sin = sin[:, :, :seq_len, :].to(x.device)
    x_rot = x[..., : x.shape[-1] // 2]
    x_pass = x[..., x.shape[-1] // 2 :]
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将输入张量的后一半旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
class DeepSeekCompatibleTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, num_kv_heads, head_dim, dim_feedforward, dropout):
        super().__init__()
        # 关键：使用统一维度定义
        self.self_attn = DeepseekAttention(
            embed_dim=d_model,
            num_heads=n_head,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=10000.0
        )
        
        # MLP使用dim_feedforward=8960
        self.mlp = SwishGLU(
            input_size=d_model,
            hidden_size=dim_feedforward  # 设为8960
        )
        
        self.input_norm = DeepseekRMSNorm(d_model, eps=1e-5)
        self.output_norm = DeepseekRMSNorm(d_model, eps=1e-5)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN结构
        # 第一个归一化
        x_normalized = self.input_norm(x)
        
        # 自注意力
        attn_output = self.self_attn(x_normalized)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # 残差连接
        
        # MLP前的第二个归一化
        x_normalized = self.output_norm(x)
        
        # 前馈网络
        mlp_output = self.mlp(x_normalized)
        mlp_output = self.dropout(mlp_output)
        return x + mlp_output  # 残差连接
class DeepseekRMSNorm(nn.Module):
    """Deepseek使用的RMSNorm（保持不变）"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
class SwishGLU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 关键：修正维度为8960
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, input_size, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        # SwishGLU激活
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
class DeepseekAttention(nn.Module):
    """Deepseek 1.5B 的自注意力模块 (分组查询注意力)"""
    def __init__(self, hidden_size, num_heads, num_kv_heads=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 处理分组查询注意力
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_repeats = num_heads // self.num_kv_heads
        
        # 确保隐藏大小可被头数整除
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除")
        
        self.head_dim = hidden_size // num_heads
        
        # 线性投影层
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 用于注意力掩码的常量
        self.register_buffer("mask_bias", torch.tril(torch.ones(1024, 1024)))
    
    def group_kv_repeat(self, kv):
        """为分组查询注意力重复键值对"""
        return (
            kv[0].repeat_interleave(self.num_kv_repeats, dim=2),
            kv[1].repeat_interleave(self.num_kv_repeats, dim=2)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 投影查询、键和值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 分割多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        q = rotary_emb(q, position_ids)
        k = rotary_emb(k, position_ids)
        
        # 处理分组查询注意力 (重复键值对)
        if self.num_kv_repeats > 1:
            k, v = self.group_kv_repeat((k, v))
        
        # 转置以便矩阵乘法 (批量大小, 头数, 序列长度, 头维度)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码 (因果掩码或自定义掩码)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(1)
        else:
            # 应用因果掩码
            causal_mask = self.mask_bias[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
            attn_scores = torch.where(
                causal_mask > 0, 
                attn_scores, 
                torch.tensor(-1e9, dtype=attn_scores.dtype, device=attn_scores.device)
            )
        
        # Softmax 归一化
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 注意力输出
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 最终投影
        return self.o_proj(attn_output)

class DeepseekMLP(nn.Module):
    """Deepseek 1.5B 的MLP模块 (SwiGLU激活)"""
    def __init__(self, hidden_size, mlp_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, mlp_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_size, bias=False)
        self.down_proj = nn.Linear(mlp_size, hidden_size, bias=False)
        
    def forward(self, x):
        """前向传播使用SwiGLU激活"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
class DeepseekBlock(nn.Module):
    """完全匹配预训练权重命名的Transformer块"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        # 检查头数和头维度是否有效
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        assert self.head_dim * num_heads == hidden_size, "头维度计算错误"
        assert num_kv_heads > 0 and num_kv_heads <= num_heads, "num_kv_heads必须在1和num_heads之间"
        
        # 1. 输入归一化 - 匹配预训练权重命名: input_layernorm
        self.input_layernorm = DeepseekRMSNorm(hidden_size)
        
        # 2. 注意力机制 - 完全匹配预训练权重命名
        self.self_attn_q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.self_attn_k_proj = nn.Linear(
            hidden_size, 
            num_kv_heads * self.head_dim,
            bias=True
        )
        self.self_attn_v_proj = nn.Linear(
            hidden_size, 
            num_kv_heads * self.head_dim,
            bias=True
        )
        self.self_attn_o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 3. MLP输入归一化 - 匹配预训练权重命名: post_attention_layernorm
        self.post_attention_layernorm = DeepseekRMSNorm(hidden_size)
        
        # 4. MLP网络 - 匹配预训练权重命名
        self.mlp_gate_proj = nn.Linear(hidden_size, mlp_size, bias=False)
        self.mlp_up_proj = nn.Linear(hidden_size, mlp_size, bias=False)
        self.mlp_down_proj = nn.Linear(mlp_size, hidden_size, bias=False)
        
        # 5. 注意力缩放因子
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)
        
        # 6. 注意力丢弃
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        rotary_emb: RotaryEmbedding,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 保存残差连接
        residual = x
        
        # === 1. 注意力归一化 ===
        x = self.input_layernorm(x)
        
        # === 2. 多头注意力机制 ===
        # 计算查询、键、值投影
        q = self.self_attn_q_proj(x)  # [batch, seq_len, hidden_size]
        k = self.self_attn_k_proj(x)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.self_attn_v_proj(x)  # [batch, seq_len, num_kv_heads * head_dim]
        
        # 重塑查询张量 [batch, seq_len, num_heads, head_dim]
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim)
        
        # 重塑键张量 [batch, seq_len, num_kv_heads, head_dim]
        k = k.view(k.size(0), k.size(1), self.num_kv_heads, self.head_dim)
        
        # 重塑值张量 [batch, seq_len, num_kv_heads, head_dim]
        v = v.view(v.size(0), v.size(1), self.num_kv_heads, self.head_dim)
        
        # === 3. 应用旋转位置编码 ===
        q = rotary_emb(q)
        k = rotary_emb(k)
        
        # 关键修复：处理查询头和键值头不匹配问题
        if self.num_kv_heads != self.num_heads:
            # 计算重复因子
            repeat_factor = self.num_heads // self.num_kv_heads
            
            # 重复键值头以匹配查询头数
            k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, seq_len, num_heads, head_dim]
            v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, seq_len, num_heads, head_dim]
        
        # === 4. 重塑张量用于注意力计算 ===
        # 转置张量维度 [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # === 5. 计算注意力分数 ===
        # 点积注意力: Q @ K^T
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        
        # 应用注意力掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # 应用掩码：将屏蔽位置设为大负数
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e10)
        
        # 计算注意力概率
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # === 6. 应用注意力权重 ===
        attn_output = torch.matmul(attn_probs, v)
        
        # === 7. 重塑注意力输出 ===
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(
            attn_output.size(0), 
            attn_output.size(1), 
            self.hidden_size
        )
        
        # === 8. 注意力输出投影 ===
        attn_output = self.self_attn_o_proj(attn_output)
        
        # === 9. 添加残差连接 ===
        x = residual + attn_output
        
        # === 10. MLP部分 ===
        residual = x
        x = self.post_attention_layernorm(x)
        
        # MLP层计算 (SwiGLU激活)
        gate = torch.sigmoid(self.mlp_gate_proj(x))
        up = self.mlp_up_proj(x)
        mlp_output = self.mlp_down_proj(gate * up)
        
        # 残差连接
        x = residual + mlp_output
        
        return x
class DeepseekLayer(nn.Module):
    """Deepseek Transformer模块的完整实现"""
    def __init__(self, hidden_size, num_heads, num_kv_heads, hidden_size_mlp):
        super().__init__()
        
        # 注意力层
        self.input_norm = DeepseekRMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads)
        
        # MLP层
        self.output_norm = DeepseekRMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size_mlp, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size_mlp, bias=False)
        self.down_proj = nn.Linear(hidden_size_mlp, hidden_size, bias=False)
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_dim = self.head_dim * num_kv_heads
        
        # 确保参数设置有效
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除"
            )

    def forward(self, x, attention_mask=None, position_ids=None):
        # 输入归一化
        residual = x
        x = self.input_norm(x)
        
        # 自注意力机制
        batch_size, seq_len, _ = x.size()
        
        # 投影查询、键和值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 分割多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)
        
        # 计算注意力分数
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(1)
        
        # Softmax 归一化
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 计算注意力输出
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)
        
        # 最终投影
        attn_output = self.o_proj(attn_output)
        
        # 残差连接
        x = residual + attn_output
        
        # MLP部分
        residual = x
        x = self.output_norm(x)
        
        # MLP层计算
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_output = self.down_proj(gate * up)
        
        # 残差连接
        x = residual + mlp_output
        
        return x
def compute_reward(model2_result, targets,lookback=5):
    """模拟实际交易效果的奖励"""
    # 获取模型预测序列
    predictions = model2_result[:, :lookback]
    
    # 构建模拟持仓
    portfolio = torch.zeros_like(targets)
    total_return = torch.zeros_like(targets)
    
    # 基于连续预测的仓位变化
    for t in range(1, lookback):
        # 基于预测信号调整仓位
        position_change = predictions[:, t] - predictions[:, t-1]
        portfolio += position_change
        
        # 计算每日收益
        daily_return = portfolio * targets[:, t]
        total_return += daily_return
        
        # 风险管理: 强制平仓规则
        portfolio = torch.where(torch.abs(portfolio) > 2, 
                                torch.clamp(portfolio, -1, 1), 
                                portfolio)
    
    # 最终组合价值作为奖励
    final_value = 1 + total_return
    reward = final_value - 1  # 组合收益作为奖励
    return reward

def get_actual_price(stock_code, date):
    # 文件路径示例
    csv_file = os.path.join('data', 'stock_prices', f'{stock_code}.csv')
    df = pd.read_csv(csv_file)

    # 假设csv有列：'date', 'close'
    row = df[df['date'] == date]
    if row.empty:
        print(f"未找到对应日期的股价：{date}")
        return None
    return float(row['close'].values[0])

def main(config_path: str):
    """主函数"""
    # 加载配置
    config_loader = ConfigLoader(config_path)
    config = config_loader.config
    config['training']['stability'] = config.get('stability', {
        'max_nan_allowed': 0,
        'output_clip_range': [-5.0, 5.0],
        'max_grad_norm': 1.0,
        'gradient_clip_value': 10000.0,
        'add_output_noise': True,
        'noise_std': 0.01,
        'backup_interval': 5,
        'recovery_strategy': 'backoff',
        'lr_backoff_factor': 0.5
    })
    # 设置日志记录器
    logger = setup_logger(config['paths']['log_file'])
    logger.info("🚀 开始运行训练脚本")
    logger.info(f"📄 使用配置文件: {config_path}")
    
    # 创建所有必要的输出目录
    create_output_directories(config)
    logger.info("📁 所有输出目录已创建")
    
    # 获取训练和测试数据路径
    train_dir = config['paths'].get('train_data_path')
    val_dir = config['paths'].get('val_data_path', None)  
    test_dir = config['paths'].get('test_data_path')
    
    if not train_dir or not os.path.exists(train_dir):
        raise RuntimeError(f"训练数据目录不存在或未定义：{train_dir}")
    if not test_dir or not os.path.exists(test_dir):
        raise RuntimeError(f"测试数据目录不存在或未定义：{test_dir}")
    
    seq_length = config['training'].get('sequence_length', 30)

    logger.info(f"📊 数据配置: 序列长度={seq_length}")
    
    # 创建训练数据集
    logger.info(f"🔍 加载训练数据集: {train_dir}")
    train_dataset = CSVDataset(
        data_path=train_dir,
        sequence_length=seq_length
    )
    device = torch.device("cuda:0")
    # 获取特征名称
    feature_names = train_dataset.feature_names
    logger.info(f"🔤 特征名称: {feature_names}")

    # 打印数据集信息
    train_dataset.get_data_report()
    
    # 处理验证数据集
    val_dataset = None
    if val_dir and os.path.exists(val_dir):
        logger.info(f"🔍 加载验证数据集: {val_dir}")
        val_dataset = CSVDataset(
            data_path=val_dir,
            sequence_length=seq_length
        )
        val_dataset.get_data_report()
    try:
        hidden_size = config['env']['hidden_size']
        num_layers = config['env']['num_layers']
        num_heads = config['env']['num_heads']
        model_path = config['env']['prediction_model_path']
        sample, _, _ = train_dataset[0]
        input_dim=sample.shape[-1]
    except KeyError as e:
        raise RuntimeError(f"配置缺少字段: {e}")
    # 根据配置路径加载模型
    safety = SafetyModule()
    if model_path:
        model2 = DeepSeekPredictor(device=device)
        safety = safety.to(device)
        print("加载的model2类型：", type(model2))
        assert model2 is not None, "模型加载失败，model2为None！"
    else:
        print("未提供模型路径，无法加载模型。")    
    # 创建模型
    try:
        # 直接从训练数据集获取输入维度
        sample, _, _ = train_dataset[0]
        input_dim = sample.shape[-1]  # 特征维度在最后一位
        logger.info(f"🧠 模型输入维度: {input_dim}")
    except Exception as e:
        logger.error(f"无法确定输入维度: {e}")
        raise
    
    # 创建或加载模型
    pretrained_path = config['paths'].get('output_model')
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"⬇️ 加载预训练模型: {pretrained_path}")
        try:
            model = TransformerModel(
                tokenizer_path= "/home/liyakun/LLaMA-Factory-main/deepseek1.5B",       # ✔️ 新的分词器参数
                model_weights_path=pretrained_path,   # ✔️ 新的权重路径参数
                vocab_size=151936,  # 修正：Deepseek 1.5B的实际词汇表大小是51200，不是32000
                hidden_size=1536,  # 使用正确的参数名（不是d_model或2048）
                num_heads=24,      # 使用正确的参数名（不是n_head或16）
                num_kv_heads=4,    # 关键：固定为4
                hidden_size_mlp=8960, # 使用正确的参数名（不是dim_feedforward或8192）
                num_layers=30,      # 1.5B模型的实际层数是30
                output_size=1
            ).to(device)
            logger.info(f"✅ 预训练模型加载成功")
            logger.info(f"🔄 模型架构:\n{model}")
        except Exception as e:
                logger.error(f"❌ 加载预训练模型失败: {str(e)}")
                exit(1)
    else:
        logger.error("❌ 找不到模型路径或路径无效")

    # 2. 加载预训练权重（关键步骤！）
    weights_path = pretrained_path 
    if os.path.exists(weights_path):
        logger.info(f"⬇️ 加载预训练权重: {weights_path}")
        try:
            # 加载保存的状态字典
            checkpoint = torch.load(weights_path, map_location=device)
            
            # 根据保存的方式处理
            state_dict = {}
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # 处理不匹配的键名（常见解决方案）
            new_state_dict = {}
            for k, v in state_dict.items():
                # 去掉可能的"module."前缀（多GPU训练保存的权重）
                name = k.replace("module.", "")
                new_state_dict[name] = v
                
            # 加载权重
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("✅ 权重加载成功")
        except Exception as e:
            logger.error(f"❌ 权重加载失败: {str(e)}")
            exit(1)
    else:
        logger.error(f"❌ 权重文件不存在: {weights_path}")
        exit(1)
    # 创建数据加载器
    batch_size = config['training'].get('batch_size', 32)
    logger.info(f"📦 创建数据加载器 (batch_size={batch_size})")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count())  # 限制工作线程数
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(2, os.cpu_count())  # 限制工作线程数
        )
        logger.info(f"✅ 训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    else:
        logger.warning("⚠️ 未创建验证数据加载器")
    
    # 准备优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        eps=1e-6  # 防止除零
    )

    max_grad_norm = config['training'].get('max_grad_norm', 1.0) 
    logger.info("ℹ️ 使用SmoothL1Loss作为损失函数，对异常值更鲁棒")
    # 创建安全控制器
    safety = SafetyController(model, optimizer, config, logger)

# 使用新损失函数
    loss_fn = RLoss(
    supervised_criterion=SafeSmoothL1Loss(beta=1.0).to(device),
    base_loss_weight=config['training']['base_loss_weight'],
).to(device)
    grad_monitor = GradientMonitor(model)
    grad_monitor.attach()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(config['training']['epochs']):
        new_weight = get_rl_weight(epoch)
        loss_fn.base_loss_weight = new_weight  # 动态调整权重
        safety.create_backup(epoch)
        model.train()
        epoch_loss = 0.0
        nan_batch_count = 0
        valid_batch_count = 0
            # 在每个epoch开始时输出
        print(f"\n【调试】开始第 {epoch+1} 轮训练")
        for batch_idx, (inputs, targets, dates) in enumerate(train_loader):
            print(f"\n【调试】第 {batch_idx+1} 批次")
            # print("batch_data 内容：", batch_data)123456
            print("原始inputs.shape:", inputs.shape)
            print("原始targets.shape:", targets.shape)
            # 转移到设备
            inputs = inputs.to(device)
            targets = targets.to(device)  
            print("inputs设备:", inputs.device)
            print("targets设备:", targets.device)   
           
            # 1. 输入数据检查
            if not safety.check_inputs(inputs, targets):
                nan_batch_count += 1
                continue
                           
            # 2. 在安全上下文中执行前向传播
            loss = None
            with safety.safe_forward_context():
                outputs = model(inputs)
                print("outputs形状")
                print(outputs)
                print("outputs的形状：", outputs.shape)
                print(torch.min(outputs).item(), torch.max(outputs).item())
                
                if outputs is None:
                    print("模型输出为None，跳过保护和loss计算")
                    # 可以定义跳出当前batch或终止
                    continue
                if model2 is None:
                    raise RuntimeError("模型未正确加载，不能进行预测！")
                model2_result = DeepSeekPredictor(model2, outputs)
                print("模型二推理结果：", model2_result)
                # 计算奖励（示例）
                direction_match = (torch.sign(model2_result) == torch.sign(targets)).float()
                magnitude_error = torch.abs(model2_result - targets)
                accuracy_reward = direction_match * 0.8
                
                # 幅度误差奖励（误差越小奖励越高）
                error_reward = torch.exp(-2 * magnitude_error) * 0.2
                reward = (accuracy_reward + error_reward)
                print(f"平均方向匹配率: {direction_match.mean().item():.4f}")
                print(f"平均误差奖励: {error_reward.mean().item():.4f}")
                print(f"最终平均奖励: {reward.mean().item():.4f}")
                protected_logits = safety.protect_outputs(outputs)
                print(f"logits保护后: min={protected_logits.min().item():.4f}, max={protected_logits.max().item():.4f}")
                protected_outputs = {'logits': protected_logits}
            # 6. 使用损失函数（保持原接口不变）
                loss_dict = loss_fn(
                    model_outputs=protected_outputs,
                    targets=targets,
                    reward=reward,
                )
                
                loss = loss_dict["total_loss"]
                print(f"总损失: {loss.item():.4f} | "
                    f"监督损失: {loss_dict['supervised_loss'].item():.4f} | "
                    f"策略损失: {loss_dict['policy_loss'].item():.4f} | "
                    f"匹配率: {loss_dict['match_rate'].item():.4f}")
                
            
            # 反向传播
            print("logits.requires_grad:", protected_logits.requires_grad)
            print("loss.requires_grad:", loss.requires_grad)
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_max = param.grad.abs().max().item()
                    print(f"层 {name} 梯度最大值: {grad_max}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            safety.check_gradients()
            optimizer.step()
            batch_reward = reward.mean().item()
            epoch_loss += loss.item()
            valid_batch_count += 1
            # 每10批次报告梯度状态
            if batch_idx % 10 == 0:
                grad_monitor.report(batch_idx, epoch)
        if val_loader:
            val_loss = 0.0
            model.eval()  # 切换为评估模式
            with torch.no_grad():  # 禁用梯度计算
                for val_inputs, val_targets, val_dates in val_loader:
                    # 将数据转移到设备
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    
                    # 验证前向传播
                    val_outputs = model(val_inputs)
                    
                    # 计算验证损失（仅基础监督损失）
                    val_loss_dict = loss_fn(
                        model_outputs={'logits': val_outputs},
                        targets=val_targets,
                        reward=None  # 验证时不使用奖励模型
                    )
                    
                    # 累计验证损失
                    val_loss += val_loss_dict["supervised_loss"].item()
            
            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} 验证损失: {avg_val_loss:.4f}")
            
            # ===== 3. 模型保存决策 =====
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = Path(config['paths']['output_model'])
                torch.save(model.state_dict(), model_save_path)
                print(f"💾 保存最佳模型到: {model_save_path} (验证损失: {avg_val_loss:.4f})")
            else:
                print(f"当前验证损失 {avg_val_loss:.4f} 比最佳 {best_val_loss:.4f} 差，不保存模型")
        # 移除梯度监控钩子
        grad_monitor.detach()
        # 报告当前epoch状态
        if valid_batch_count > 0:
            avg_loss = epoch_loss / valid_batch_count
            logger.info(f"🏁 Epoch {epoch+1} | 平均损失: {avg_loss:.6f} | 跳过批次: {nan_batch_count}")
        else:
            logger.error(f"⛔ Epoch {epoch+1} 没有有效训练批次，尝试恢复...")
            recovered_epoch = safety.recover(epoch)
            logger.warning(f"恢复至epoch {recovered_epoch}")
            epoch = recovered_epoch  # 重置epoch计数器

    model_save_path = Path(config['paths']['output_model']) 
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"💾 模型已保存到: {model_save_path}")
    # 测试和评估阶段
    logger.info("🧪 开始测试阶段...")

    # 1. 数据加载和准备
    if not test_dir or not test_dir.get('test'):
        logger.info("⚠️ 没有配置测试数据集，跳过测试阶段")
    else:
        # 安全加载测试数据集
        test_datasets = []
        for test_file in test_dir['test']:
            logger.info(f"📥 加载测试数据集: {test_file}")
            if not os.path.exists(test_file):
                logger.warning(f"⚠️ 测试文件不存在，跳过: {test_file}")
                continue
                
            try:
                test_dataset = CSVDataset(
                    data_path=test_file,
                    sequence_length=config['training'].get('sequence_length', 30),
                )
                test_datasets.append(test_dataset)
            except Exception as e:
                logger.error(f"❌ 加载测试数据失败: {test_file}, 错误: {str(e)}")
        
        # 处理没有有效测试数据的情况
        if not test_datasets:
            logger.warning("⚠️ 没有可用的测试数据，跳过测试阶段")
        else:
            # 组合测试数据集
            if len(test_datasets) > 1:
                final_test_dataset = torch.utils.data.ConcatDataset(test_datasets)
                logger.info(f"✅ 合并了 {len(test_datasets)} 个测试数据集，总样本数: {len(final_test_dataset)}")
            else:
                final_test_dataset = test_datasets[0]
            
            # 创建数据加载器
            test_loader = DataLoader(
                final_test_dataset, 
                batch_size=config['training']['batch_size'], 
                shuffle=False,
                num_workers=4
            )
            
            # 2. 测试过程
            model.eval()
            all_predictions = []
            all_targets = []
            all_dates = []
            all_rewards = []
            all_model2_preds = []  # 存储model2的预测结果
            all_feature_sequences = []  # 存储输入特征序列
            all_processed_features = []  # 存储处理后的特征
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    # 解包数据 (根据CSVDataset的__getitem__方法)
                    feature_sequences = batch_data[0]
                    processed_features = batch_data[1]
                    targets = batch_data[2]
                    dates = batch_data[3]
                    
                    # 转移到设备
                    processed_features = processed_features.to(device)
                    targets = targets.to(device)
                    
                    # 模型预测 (使用处理后的特征)
                    outputs = model(processed_features)
                    model2_preds = DeepSeekPredictor(model2, outputs)  # 使用model2进行预测
                    
                    # 收集数据
                    all_predictions.extend(outputs.squeeze().cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_dates.extend(dates)
                    all_model2_preds.extend(model2_preds.cpu().numpy())
                    
                    # 收集特征数据用于分析
                    all_feature_sequences.extend(feature_sequences.cpu().numpy())
                    all_processed_features.extend(processed_features.cpu().numpy())
                    
                    # 计算奖励
                    direction_match = (torch.sign(model2_preds) == torch.sign(targets)).float()
                    accuracy_reward = direction_match * 0.8
                    error_reward = torch.exp(-2 * torch.abs(model2_preds - targets)) * 0.2
                    batch_reward = (accuracy_reward + error_reward).cpu().numpy()
                    all_rewards.extend(batch_reward)
                    
                    # 每100批次报告一次进度
                    if batch_idx % 100 == 0:
                        logger.info(f"🔄 处理测试批次 {batch_idx+1}/{len(test_loader)}")
            
            # 3. 结果保存（结构化数据）
            results_df = pd.DataFrame({
                'date': all_dates,
                'prediction': all_predictions,  # 主模型预测
                'model2_prediction': all_model2_preds,  # model2的预测
                'target': all_targets,
                'error': np.abs(np.array(all_model2_preds) - np.array(all_targets)),
                'reward': all_rewards
            })

            # 添加特征数据到DataFrame
            if all_feature_sequences:
                # 特征序列通常是三维的 [batch, seq_len, features]
                # 转换为numpy数组以便高效处理
                feature_array = np.array(all_feature_sequences)
                
                # 获取特征数量
                num_features = feature_array.shape[-1]
                
                # 直接使用特征索引命名列
                for i in range(num_features):
                    # 取序列中最后一个时间步的值作为当前特征值
                    results_df[f'feature_{i}'] = feature_array[:, -1, i]

            # 保存预测结果
            output_path = save_structured_data(
                results_df, 
                config, 
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            logger.info(f"💾 保存预测结果到: {output_path}")

            # 4. 计算评估指标
            try:
                # 确保有足够的样本
                num_samples = len(all_targets)
                if num_samples < 10:
                    raise ValueError(f"样本数量不足: {num_samples}, 无法进行可靠的评估")
                
                # 基本统计指标
                mae = np.mean(np.abs(np.array(all_model2_preds) - np.array(all_targets)))
                rmse = np.sqrt(mean_squared_error(all_targets, all_model2_preds))
                
                # 金融相关指标
                sharpe_val = sharpe_ratio(np.array(all_model2_preds))
                drawdown = max_drawdown(np.array(all_model2_preds))
                
                # 奖励统计
                avg_test_reward = np.mean(all_rewards) if all_rewards else 0.0
                reward_std = np.std(all_rewards) if all_rewards else 0.0
                
                # 方向准确率
                predicted_signs = np.sign(np.array(all_model2_preds))
                target_signs = np.sign(np.array(all_targets))
                valid_idx = (predicted_signs != 0) & (target_signs != 0)  # 忽略零值
                direction_accuracy = np.mean(predicted_signs[valid_idx] == target_signs[valid_idx]) if any(valid_idx) else np.nan
                
                # 特征相关性分析 (使用特征索引)
                feature_correlations = {}
                for i in range(num_features):
                    col_name = f'feature_{i}'
                    # 过滤无穷大和NaN值
                    valid_idx = np.isfinite(results_df[col_name]) & np.isfinite(results_df['model2_prediction'])
                    
                    if np.sum(valid_idx) > 10:  # 确保有足够的数据点
                        correlation = np.corrcoef(
                            results_df.loc[valid_idx, col_name],
                            results_df.loc[valid_idx, 'model2_prediction']
                        )[0, 1]
                        feature_correlations[col_name] = correlation
                    else:
                        feature_correlations[col_name] = np.nan
                        logger.warning(f"无法计算特征'{col_name}'的相关性 - 有效数据点不足")
                
                # 组合评估结果
                eval_results = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "Direction_Accuracy": direction_accuracy,
                    "Sharpe_Ratio": sharpe_val,
                    "Max_Drawdown": drawdown,
                    "Avg_Reward": avg_test_reward,
                    "Reward_STD": reward_std,
                    "num_samples": num_samples,
                    "config_path": config_path,
                    "test_data_path": test_dir,
                    "feature_correlations": feature_correlations,
                    "test_start_date": min(all_dates) if all_dates else "N/A",
                    "test_end_date": max(all_dates) if all_dates else "N/A"
                }
                
                # 5. 保存评估结果
                eval_results_serializable = convert_to_serializable(eval_results)
                eval_dir = config['paths']['model1_eval'] if 'model1_eval' in config['paths'] else './eval_results'
                output_path = os.path.join(
                    eval_dir,
                    f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                os.makedirs(eval_dir, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(eval_results_serializable, f, indent=4)
                logger.info(f"📝 保存评估结果到: {output_path}")
                
                # 6. 打印关键指标
                logger.info("\n📊 测试结果摘要:")
                logger.info(f"样本数量: {num_samples}")
                if all_dates:
                    logger.info(f"时间范围: {min(all_dates)} 至 {max(all_dates)}")
                logger.info(f"平均绝对误差 (MAE): {mae:.6f}")
                logger.info(f"均方根误差 (RMSE): {rmse:.6f}")
                logger.info(f"方向准确率: {direction_accuracy*100 if not np.isnan(direction_accuracy) else 'N/A':.2f}%")
                logger.info(f"平均奖励: {avg_test_reward:.4f} ± {reward_std:.4f}")
                logger.info(f"夏普比率: {sharpe_val:.4f}")
                logger.info(f"最大回撤: {drawdown*100:.2f}%")
                
                # 打印特征相关性摘要
                if feature_correlations:
                    logger.info("\n🔍 特征预测值相关性:")
                    for feature, corr in feature_correlations.items():
                        logger.info(f"{feature}: {corr:.4f}")
                
                # 保存特征映射信息
                if test_datasets:
                    try:
                        # 获取特征映射
                        feature_mapping = test_datasets[0].get_feature_mapping() if isinstance(test_datasets, list) else test_datasets.get_feature_mapping()
                        
                        # 保存特征映射文件
                        mapping_path = output_path.replace('.json', '_feature_mapping.json')
                        with open(mapping_path, 'w') as f:
                            json.dump(feature_mapping, f, indent=4)
                        logger.info(f"📋 保存特征映射到: {mapping_path}")
                        
                        # 同时记录在评估结果中
                        eval_results['feature_mapping'] = feature_mapping
                        with open(output_path, 'w') as f:  # 重新写入包含映射信息的结果
                            json.dump(convert_to_serializable(eval_results), f, indent=4)
                    except Exception as e:
                        logger.warning(f"⚠️ 无法保存特征映射: {str(e)}")
                
            except ValueError as e:
                logger.warning(f"⚠️ {str(e)} - 跳过部分评估指标")
                # 创建最小错误报告
                error_report = {
                    "warning": str(e),
                    "num_samples": len(all_targets),
                    "test_files": test_dir['test'],
                    "partial_results": eval_results if 'eval_results' in locals() else None
                }
                error_path = os.path.join(
                    config['paths']['model1_eval'],
                    f"partial_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(error_path, 'w') as f:
                    json.dump(convert_to_serializable(error_report), f, indent=4)
                logger.warning(f"⚠️ 保存部分评估结果到: {error_path}")

            except Exception as e:
                logger.error(f"❌ 评估指标计算失败: {str(e)}", exc_info=True)
                # 创建最小化错误报告
                error_report = {
                    "error": str(e),
                    "num_samples": len(all_targets) if 'all_targets' in locals() else 0,
                    "test_files": test_dir['test']
                }
                error_path = os.path.join(
                    config['paths']['model1_eval'],
                    f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(error_path, 'w') as f:
                    json.dump(convert_to_serializable(error_report), f, indent=4)
                logger.error(f"💾 保存错误报告到: {error_path}")

            logger.info("✅ 测试阶段完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL交易系统训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", 
                        type=str, 
                        default="configs/model1.yaml",
                        help="配置文件路径，默认: configs/model1.yaml")
    
    args = parser.parse_args()
    main(args.config)
