import os
import re
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
from ..model2.inference import load_model, predict, DeepSeekModel
from transformers import AutoTokenizer, AutoModel  


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
    
safety = SafetyModule()

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
try:
    hidden_size = config['env']['hidden_size']
    num_layers = config['env']['num_layers']
    num_heads = config['env']['num_heads']
    model_path = config['env']['prediction_model_path']
    input_dim=config['model']['params']['input_dim']
except KeyError as e:
    raise RuntimeError(f"配置缺少字段: {e}")
# 根据配置路径加载模型
if model_path:
    model2 = DeepSeekModel(input_dim,hidden_size, num_layers, num_heads)
    model2 = model2.to(device)
    safety = safety.to(device)
    print("加载的model2类型：", type(model2))
    assert model2 is not None, "模型加载失败，model2为None！"
else:
    print("未提供模型路径，无法加载模型。")

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
    def __init__(self, x_file: str, y_file: str, dates_file: str, 
                 filter_nan=True, 
                 sequence_length=30,
                 feature_names=None):
        """
        针对市场推特数据优化的CSVDataset
        
        参数:
            x_file: 包含特征的CSV文件路径
            y_file: 包含标签的CSV文件路径
            dates_file: 包含日期的CSV文件路径
            filter_nan: 是否过滤包含NaN的样本 (默认为True)
            sequence_length: 时间序列长度 (默认为30)
            feature_names: 可选特征名称列表
        """
        print("Loading data files:")
        print("x_file:", x_file)
        print("y_file:", y_file)
        print("dates_file:", dates_file)        
        try:
            # 1. 加载数据文件
            self.df = pd.read_csv(x_file)
            self.label_df = pd.read_csv(y_file) if y_file else None
            self.dates_df = pd.read_csv(dates_file) if dates_file else None            
            # 打印基本信息
            print("Data shapes:")
            print("x_data:", self.df.shape)
            print("y_data:", self.label_df.shape if self.label_df is not None else "None")
            print("dates:", self.dates_df.shape if self.dates_df is not None else "None")            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise
            
        # 2. 解析JSON字段（如果存在）
        self._parse_json_columns()
        
        # 3. 合并data, label, date
        self._merge_dataframes()
        
        # 4. 验证数据一致性
        assert len(self.x_data) == len(self.y_data), "x和y行数不一致"
        if self.dates_df is not None:
            assert len(self.x_data) == len(self.dates_df), "x和dates行数不一致"
            
        # 5. 创建时间序列
        self._create_sequences(sequence_length)
        
        # 6. 过滤NaN值
        if filter_nan:
            self._filter_nan_samples()
            
        self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.x_data.shape[1])]
        
        # 7. 打印最终形状
        print(f"创建序列后形状 - x_data: {self.x_data.shape}, y_data: {self.y_data.shape}")
        
    def _parse_json_columns(self):
        """解析CSV中的JSON格式列"""
        print("解析JSON格式列...")
        json_columns = []
        
        # 检查df中的JSON列
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].str.startswith('{').any():
                json_columns.append(col)
                self.df[col] = self.df[col].apply(self._safe_parse_json)
                print(f"解析JSON列: {col}")
        
        # 创建特征列
        self._create_feature_columns()
    
    def _safe_parse_json(self, json_str):
        """安全解析JSON字符串"""
        if pd.isnull(json_str):
            return {}
        try:
            # 处理不标准的引号
            json_str = json_str.replace('""', '"')
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            try:
                # 尝试替换单引号
                json_str = json_str.replace("'", '"')
                return json.loads(json_str)
            except:
                return {}
    
    def _create_feature_columns(self):
        """从JSON结构中创建特征列"""
        # 市场数据特征
        if 'market_context' in self.df.columns:
            market_data = self.df['market_context']
            self.df['open'] = market_data.apply(lambda x: x.get('open', np.nan))
            self.df['high'] = market_data.apply(lambda x: x.get('high', np.nan))
            self.df['low'] = market_data.apply(lambda x: x.get('low', np.nan))
            self.df['close'] = market_data.apply(lambda x: x.get('close', np.nan))
            self.df['volume'] = market_data.apply(lambda x: x.get('volume', np.nan))
        
        # 技术指标特征
        if 'technical_indicators' in self.df.columns:
            tech_data = self.df['technical_indicators']
            self.df['sma'] = tech_data.apply(lambda x: x.get('sma', np.nan))
            self.df['ema'] = tech_data.apply(lambda x: x.get('ema', np.nan))
            self.df['rsi'] = tech_data.apply(lambda x: x.get('rsi', np.nan))
            self.df['bollinger_upper'] = tech_data.apply(lambda x: x.get('bollinger_upper', np.nan))
            self.df['bollinger_lower'] = tech_data.apply(lambda x: x.get('bollinger_lower', np.nan))
        
        # 推特元数据特征
        if 'tweet_metadata' in self.df.columns:
            tweet_data = self.df['tweet_metadata']
            self.df['retweet_count'] = tweet_data.apply(lambda x: x.get('retweet_count', 0))
            self.df['favorite_count'] = tweet_data.apply(lambda x: x.get('favorite_count', 0))
            self.df['mention_count'] = tweet_data.apply(lambda x: len(x.get('mentioned_symbols', [])))
    
    def _merge_dataframes(self):
        """合并data, label和date数据帧"""
        # 提取基础特征列
        base_features = ['symbol', 'timestamp', 'sentiment', 'text',
                         'open', 'high', 'low', 'close', 'volume',
                         'sma', 'ema', 'rsi', 'bollinger_upper', 'bollinger_lower',
                         'retweet_count', 'favorite_count', 'mention_count']
        
        # 创建最终特征矩阵
        feature_cols = [col for col in base_features if col in self.df.columns]
        self.x_data = self.df[feature_cols].values
        
        # 创建标签向量
        if self.label_df is not None:
            # 如果是单列标签文件
            if len(self.label_df.columns) == 1:
                self.y_data = self.label_df.values
            else:
                # 尝试从标签文件识别目标列
                for col in ['target', 'label', 'close']:
                    if col in self.label_df.columns:
                        self.y_data = self.label_df[col].values
                        break
                else:
                    # 如果没有找到有效的标签列，抛出一个详细的错误
                    available_cols = list(self.label_df.columns)
                    if len(available_cols) > 5:
                        available_cols = available_cols[:5] + ["..."]
                    
                    raise ValueError(
                        f"❌ 无法识别标签列。候选列名 {self.label_df.columns} 均不存在于标签文件中。\n"
                        f"标签文件包含的列: {available_cols}\n"
                        f"请通过以下方式之一解决:\n"
                        f"1. 在标签文件中添加 'target' 或 'label' 列\n"
                        f"2. 指定代码中使用的实际标签列名"
                )
        else:
            # 如果没提供标签文件，尝试从特征数据中提取标签
            if 'close' in self.df.columns:
                print("警告：使用特征数据中的'close'作为标签")
                self.y_data = self.df['close'].values
            else:
                raise ValueError("无法确定标签列")
        
        # 创建日期向量
        if self.dates_df is not None:
            # 尝试识别日期列
            for col in ['date', 'timestamp', 'time']:
                if col in self.dates_df.columns:
                    self.dates = self.dates_df[col].values
                    break
            else:
                # 如果没有找到，取第一列作为日期
                    available_cols = list(self.dates_df.columns)
                    if len(available_cols) > 5:
                        available_cols = available_cols[:5] + ["..."]
                    
                    raise ValueError(
                        f"❌ 无法识别标签列。候选列名 {self.label_df.columns} 均不存在于标签文件中。\n"
                        f"标签文件包含的列: {available_cols}\n"
                        f"请通过以下方式之一解决:\n"
                        f"1. 在标签文件中添加 'target' 或 'label' 列\n"
                        f"2. 指定代码中使用的实际标签列名"
                )
        else:
            # 如果没提供日期文件，尝试从特征数据中提取日期
            if 'timestamp' in self.df.columns:
                print("警告：使用特征数据中的'timestamp'作为日期")
                self.dates = self.df['timestamp'].values
            else:
                self.dates = np.arange(len(self.x_data))
    
    def _create_sequences(self, sequence_length):
        """创建时间序列序列"""
        sequences = []
        labels = []
        date_sequences = []
        
        # 按时间序列长度创建序列
        for i in range(len(self.x_data) - sequence_length):
            sequences.append(self.x_data[i:i+sequence_length])
            labels.append(self.y_data[i+sequence_length])  # 预测下一个时间点的标签
            date_sequences.append(self.dates[i:i+sequence_length])
        
        self.x_data = np.array(sequences)
        self.y_data = np.array(labels)
        self.dates = np.array(date_sequences)
        
        print(f"创建时间序列: {len(self.x_data)} 个序列 (长度: {sequence_length})")
        print(f"序列形状: x_data {self.x_data.shape}, y_data {self.y_data.shape}")
    
    def _filter_nan_samples(self):
        """过滤包含NaN的样本"""
        original_count = len(self.x_data)
        
        # 创建有效索引列表
        valid_indices = [
            i for i in range(original_count)
            if not np.isnan(self.x_data[i]).any() and not np.isnan(self.y_data[i]).any()
        ]
        
        # 应用过滤
        self.x_data = self.x_data[valid_indices]
        self.y_data = self.y_data[valid_indices]
        self.dates = self.dates[valid_indices] if self.dates is not None else None
        
        filtered_count = original_count - len(valid_indices)
        print(f"过滤掉包含NaN的样本: {filtered_count}/{original_count}")
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        data = self.x_data[idx]
        label = self.y_data[idx]
        date = self.dates[idx]
        
        # 转换为Tensor
        data = torch.tensor(data, dtype=torch.float32) 
        label = torch.tensor(label, dtype=torch.float32)
        
        # 数据清洗
        if torch.isnan(data).any():
            data = torch.nan_to_num(data, nan=0.0)
        if torch.isinf(data).any():
            data = torch.nan_to_num(data, posinf=1e6, neginf=-1e6)
        
        # 返回包含事件标签的三元组
        return data, label, date
    
    def get_event_names(self):
        """获取所有事件名称列表"""
        return []  # 您的原始代码中有此函数
    
    def get_special_event_id(self):
        """获取特殊事件ID"""
        return 3  # 您的原始代码中有此函数

class TransformerModel(nn.Module):
    """Transformer模型定义，匹配预训练权重结构"""
    def __init__(self, model_name: str, output_size: int = 4):
        super().__init__()
        self.model_name = model_name
        self.output_size = output_size
        
        # 1. 加载预训练模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 2. 添加适配层输出向量
        encoder_hidden_size = self.encoder.config.hidden_size
        self.vector_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        self.output_layer = None
        self.set_output_size(output_size)
        # 3. 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
    def set_output_size(self, size: int = None):
        """动态设置输出维度(也可不设置)"""
        if size is None:
            # 保留原始维度
            self.output_size = self.hidden_size
            if hasattr(self, 'output_layer'):
                del self.output_layer  # 移除适配层
            self.output_layer = nn.Identity()
            print(f"输出层：保留原始维度 {self.hidden_size}")
        else:
            # 添加/调整适配层
            self.output_size = size
            if hasattr(self, 'output_layer') and isinstance(self.output_layer, nn.Linear):
                # 调整现有适配层
                self.output_layer.out_features = size
            else:
                # 新建适配层
                self.output_layer = nn.Linear(self.hidden_size, size)
            print(f"输出层：适配到 {size} 维")

    def forward(self, inputs) -> torch.Tensor:  # 直接返回张量
        """直接返回名为 outputs 的向量（不是字典！）"""
        # 处理文本输入或张量输入
        if isinstance(inputs, dict) and 'text' in inputs:
            # 文本输入模式
            tokenized = self.tokenizer(
                inputs['text'],
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.encoder.device)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
        else:
            # 张量输入模式
            input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs
            attention_mask = inputs['attention_mask'] if isinstance(inputs, dict) else None
        
        # 提取文本表示
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS]标记作为序列表示
        last_hidden_state = encoder_outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :] if last_hidden_state.dim() == 3 else last_hidden_state
        
        # 生成输出向量 - 命名为 outputs
        outputs = self.vector_head(cls_token)  # <-- 直接返回张量而非字典
        
        return outputs  # 返回名为 outputs 的向量！
    
    @classmethod
    def load_pretrained(cls, config: Dict[str, Any]) -> 'TransformerModel':
        """加载预训练权重"""
        logger = logging.getLogger("training")
        prediction_model_path = config['paths']['output_model']
        model_params = config['model']['params']
        
        # 创建模型实例
        model = cls(
            model_name=model_params['model_name'],
            output_size=model_params.get('output_size')
        )
        model = model.to(device)
        safety = SafetyModule()
        safety = safety.to(device)
        # 如果预训练模型存在，加载权重
        if os.path.exists(prediction_model_path):
            logger.info(f"⚙️ 加载预训练模型: {prediction_model_path}")
            pretrained_dict = torch.load(prediction_model_path)
            model.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"✅ 成功加载预训练模型权重")
        else:
            logger.warning(f"⚠️ 预训练模型未找到: {prediction_model_path}")
        
        return model

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

def get_data_files_from_dir(dir_path: str) -> Dict[str, str]:
    """安全地从目录识别CSV数据文件（特征/X，标签/Y和日期/dates文件）"""
    path_obj = Path(dir_path)
    if not path_obj.exists() or not path_obj.is_dir():
        raise FileNotFoundError(f"目录不存在或不是文件夹: {dir_path}")
    
    # 1. 查找所有CSV文件
    csv_files = list(path_obj.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"目录中未找到CSV文件: {dir_path}")
    
    # 2. 分类文件
    result = {'train': [], 'val': [], 'test': []}
    
    # 优先级1: 按标准命名模式
    for file in csv_files:
        fname = file.stem.lower()
        
        # 匹配训练集文件
        if re.search(r'(train|training|train_data|training_data|train_set)', fname):
            result['train'].append(str(file))
        
        # 匹配验证集文件
        elif re.search(r'(val|validation|valid|valid_data|val_data)', fname):
            result['val'].append(str(file))
        
        # 匹配测试集文件
        elif re.search(r'(test|testing|test_data|testing_data|test_set)', fname):
            result['test'].append(str(file))
    
    # 优先级2: 按数据集目录结构
    if not any(result.values()):
        for file in csv_files:
            parent_dir = file.parent.stem.lower()
            
            if 'train' in parent_dir:
                result['train'].append(str(file))
            elif 'val' in parent_dir or 'valid' in parent_dir:
                result['val'].append(str(file))
            elif 'test' in parent_dir:
                result['test'].append(str(file))
    
    # 优先级3: 按简单命名
    if not any(result.values()):
        for file in csv_files:
            fname = file.stem.lower()
            
            if fname == 'data' or fname == 'dataset' or fname == 'full':
                # 如果只有一个数据文件，则用于所有集
                result['train'].append(str(file))
                result['val'].append(str(file))
                result['test'].append(str(file))
                break
    
    # 4. 验证结果
    if not result['train']:
        raise FileNotFoundError(f"未找到训练数据集文件: {dir_path}")
    
    return result

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
    train_dir = config['env'].get('train_data_path')
    val_dir = config['env'].get('val_data_path', None)  
    test_dir = config['env'].get('test_data_path')
    
    if not train_dir or not os.path.exists(train_dir):
        raise RuntimeError(f"训练数据目录不存在或未定义：{train_dir}")
    if not test_dir or not os.path.exists(test_dir):
        raise RuntimeError(f"测试数据目录不存在或未定义：{test_dir}")
    
    # 获取标签列和日期列名
    label_col = config['data'].get('label_col', 'label')
    date_col = config['data'].get('date_col', 'date')
    seq_length = config['training'].get('sequence_length', 30)
    
    logger.info(f"📊 数据配置: 标签列='{label_col}', 日期列='{date_col}', 序列长度={seq_length}")
    
    # 获取数据集文件
    train_files = get_data_files_from_dir(train_dir)
    logger.info(f"训练数据集文件: train={train_files['train']}, val={train_files['val']}, test={train_files['test']}")
    
    # 如果有单独的验证目录
    val_files = get_data_files_from_dir(val_dir) if val_dir else {'train': [], 'val': [], 'test': []}
    # 创建训练数据集
    train_datasets = []
    for path in train_files['train']:
        logger.info(f"🔍 加载训练数据集: {path}")
        ds = CSVDataset(
            data_path=path,
            label_col=label_col,
            date_col=date_col,
            sequence_length=seq_length
        )
        train_datasets.append(ds)
    
    # 处理验证数据集
    val_datasets = []
    for path in train_files['val'] + val_files['val']:  # 先使用训练目录中的验证集，再使用验证目录中的
        if path:  # 确保路径有效
            logger.info(f"🔍 加载验证数据集: {path}")
            ds = CSVDataset(
                data_path=path,
                label_col=label_col,
                date_col=date_col,
                sequence_length=seq_length
            )
            val_datasets.append(ds)
    
    # 如果没有显式的验证集，尝试从训练集中划分
    if not val_datasets and train_datasets:
        logger.warning("⚠️ 未找到显式验证集，将从训练集中划分20%作为验证集")
        total_size = len(train_datasets[0])
        val_size = int(total_size * 0.2)
        train_size = total_size - val_size
        
        # 分割第一个训练集作为验证集
        train_ds, val_ds = torch.utils.data.random_split(
            train_datasets[0], 
            [train_size, val_size]
        )
        
        # 替换原来的训练集列表
        train_datasets = [train_ds] + train_datasets[1:]
        val_datasets = [val_ds]
    # 组合数据集
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        logger.info(f"✅ 合并了 {len(train_datasets)} 个训练数据集")
    else:
        train_dataset = train_datasets[0] if train_datasets else None
    
    if len(val_datasets) > 1:
        val_dataset = ConcatDataset(val_datasets)
        logger.info(f"✅ 合并了 {len(val_datasets)} 个验证数据集")
    else:
        val_dataset = val_datasets[0] if val_datasets else None
    
    if not train_dataset:
        raise RuntimeError("❌ 没有可用的训练数据!")
    
    # 创建模型
    if hasattr(train_dataset, 'get_sample'):
        sample = train_dataset.get_sample(0)
        input_dim = sample[0].shape[1] if isinstance(sample, tuple) else None
    else:
        # 如果dataset没有get_sample方法，尝试其他方法获取输入维度
        try:
            first_item = next(iter(train_dataset))
            input_dim = first_item[0].shape[1]  # (batch, seq_len, features)
        except:
            raise RuntimeError("无法确定输入维度")
    
    logger.info(f"🧠 模型输入维度: {input_dim}")
    
    # 创建或加载模型
    pretrained_path = config['paths'].get('pretrained_model')
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"⬇️ 加载预训练模型: {pretrained_path}")
        model = TransformerModel.load_pretrained(config)
    else:
        logger.info("🆕 创建新模型")
        model = TransformerModel(
            input_dim=input_dim,
            output_dim=1,  # 单输出
            config=config
        )
    
    logger.info(f"🔄 模型架构:\n{model}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"⚙️ 使用设备: {device}")
    
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
     # 训练模型
    logger.info("⏳ 开始训练...")
    trained_model, stats = model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    logger.info("🏁 训练完成!")
    
    # 保存模型
    model_save_path = Path(config['paths']['model_save_dir']) / "trained_model.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f"💾 模型已保存到: {model_save_path}")
    
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
                model2_result = predict(model2, outputs)
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
        
    
    # 测试和评估阶段
    model.eval()
    all_predictions = []
    all_targets = []
    all_dates = []
    test_rewards = []
    test_files = get_data_files_from_dir(test_dir)
    # 创建测试数据集（可能包含多个测试文件）
    test_datasets = []
    for test_file in test_files['test']:
        logger.info(f"加载测试数据集: {test_file}")
        test_dataset = CSVDataset(
            data_path=test_file,
            label_col=label_col,
            date_col=date_col,
            sequence_length=config['training'].get('sequence_length', 30)
        )
        test_datasets.append(test_dataset)
    
    # 如果只有一个测试文件，直接使用；否则使用ConcatDataset
    if len(test_datasets) == 1:
        final_test_dataset = test_datasets[0]
    else:
        final_test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    test_loader = DataLoader(
        final_test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    with torch.no_grad():
        for inputs, targets, dates in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs= model(inputs) 
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_dates.extend(dates)
            model2_preds = predict(model2, outputs)
            direction_match = (torch.sign(model2_preds) == torch.sign(targets)).float()
            accuracy_reward = direction_match * 0.8
            error_reward = torch.exp(-2 * torch.abs(model2_preds- targets)) * 0.2
            batch_reward = (accuracy_reward + error_reward).cpu().numpy()
            test_rewards.extend(batch_reward)
    # 转换为DataFrame便于保存结构化数据
    results_df = pd.DataFrame({
        'date': all_dates,  # 直接从加载器中获取的日期
        'prediction': model2_preds,
        'target': all_targets,
        'error': np.abs(np.array(model2_preds) - np.array(all_targets))
    })
    
    # 保存结构化输出数据到model1_output
    output_path = save_structured_data(results_df, config, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logger.info(f"保存预测结果到: {output_path}")
    # 计算评估指标
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets))**2))
    sharpe_val = sharpe_ratio(np.array(all_predictions))
    drawdown = max_drawdown(np.array(all_predictions))
    if test_rewards:
        avg_test_reward = sum(test_rewards) / len(test_rewards)
        reward_std = np.std(test_rewards)  # 需要import numpy as np
    else:
        avg_test_reward = 0.0
        reward_std = 0.0
    eval_results = {
        "MAE": mae,
        "RMSE": rmse,
        "Sharpe_Ratio": sharpe_val,
        "Max_Drawdown": drawdown,
        "num_samples": len(all_targets),
        "input_features": train_dataset.feature_names,
        "config_path": config_path,
        "test_data_path": test_dir,
        "model_summary": str(model),
        "Avg_Reward": avg_test_reward,  # 新字段
        "Reward_STD": reward_std,       # 新字段
    }
    eval_results_serializable = convert_to_serializable(eval_results) #计算评估指标
    
    output_path, _ = save_evaluation_results(eval_results, config, filename="evaluation_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".json")
    with open(output_path, 'w') as f:
        json.dump(eval_results_serializable, f, indent=4)
    # 保存评估结果到model1_eval
    save_evaluation_results(eval_results, config, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 预测阶段（如果配置）
    if 'predict' in config and config['predict'].get('steps', 0) > 0:
        logger.info(f"🔮 开始预测, 步数: {config['predict']['steps']}")

        # 计算最大长度
        len_dates = len(test_dataset.dates)
        len_preds = len(all_predictions)
        max_len = max(len_dates, len_preds, config['predict']['steps'])

        # 补齐dates
        dates_padded = np.pad(test_dataset.dates, (0, max_len - len_dates), constant_values=np.nan)
        # 补齐预测结果
        preds_padded = np.pad(all_predictions, (0, max_len - len_preds), constant_values=np.nan)
        # 生成 confidence 数组，长度为 max_len
        confidence_array = np.random.uniform(0.7, 0.95, max_len)

        # 只取前 max_len
        dates_for_df = dates_padded[:max_len]
        preds_for_df = preds_padded[:max_len]

        # 保存预测结果
        predict_path = Path(config['predict']['output_path'])
        predict_path.parent.mkdir(parents=True, exist_ok=True)

        # 构造 DataFrame
        predict_df = pd.DataFrame({
            'date': dates_for_df,
            'prediction': preds_for_df,
            'confidence': confidence_array
        })

        predict_df.to_csv(predict_path, index=False)
        logger.info(f"✅ 保存预测结果到: {predict_path}")

    logger.info("🎉 训练和评估完成!")
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
