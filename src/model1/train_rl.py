import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
import random
import copy
import torch.nn.functional as F 
from contextlib import contextmanager
import inspect
import warnings
from ..model2.inference import load_model, predict, DeepSeekModel


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
        
            
        # 防止指数爆炸
        outputs = torch.clamp(outputs, 
                              min=self.config['output_clip_range'][0], 
                              max=self.config['output_clip_range'][1])
        
        # 添加噪声增强稳定性
        # if self.config.get('add_output_noise'):
        #     noise_std = self.config.get('noise_std', 1e-3)
        #     outputs += torch.randn_like(outputs) * noise_std
        
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
        self.supervised_criterion = supervised_criterion  # SafeSmoothL1Loss
    
    def forward(self, policy_outputs, model2_outputs, targets, reward):
        # 监督损失部分 - 保持基础预测能力
        supervised_loss = self.supervised_criterion(model2_outputs.squeeze(), targets)
         # 2. 专业的策略梯度损失 - 将奖励与动作概率关联
        # 将策略输出转换为概率分布
        action_probs = F.softmax(policy_outputs, dim=1)
        # 金融策略梯度损失 (方向性加权)
        position_direction = torch.argmax(action_probs, dim=1)  # 0: 做多, 1: 空仓, 2: 做空
        market_direction = (targets > 0).long()  # 市场方向 1: 上涨, 0: 下跌
        valid_actions = market_direction.clone()
        valid_actions[market_direction == 0] = 2  # 下跌时希望模型选做空
        valid_actions[market_direction == 1] = 0  # 上涨时希望模型选做多
        
        # 方向匹配度 (用于策略梯度加权)
        directional_match = (position_direction == valid_actions).float()
        
        # 计算对数概率 (用于策略梯度)
        chosen_action_probs = torch.gather(action_probs, 1, position_direction.unsqueeze(1)).squeeze()
        log_probs = torch.log(chosen_action_probs + 1e-8)
        
        # 金融特化的策略梯度损失
        rl_loss = -torch.mean(log_probs * reward * (1.0 + 0.5 * directional_match))
        
        # 3. 动态调整权重 (基于市场波动率)
        volatility = torch.abs(targets).mean()
        current_weight = self.dynamic_weight_adjust(volatility)
        
        # 4. 混合损失
        total_loss = current_weight * supervised_loss + (1 - current_weight) * rl_loss
        
        # 返回损失和相关指标
        return {
            "total_loss": total_loss,
            "supervised_loss": supervised_loss,
            "rl_loss": rl_loss,
            "weight": current_weight,
            "mean_reward": reward.mean(),
            "match_rate": directional_match.mean()
        }
    
    def dynamic_weight_adjust(self, volatility):
        """根据市场波动率调整监督损失权重"""
        # 波动率较高时降低监督权重 (0.2-0.8范围)
        return torch.clamp(0.7 - volatility * 20, min=0.2, max=0.8).item()

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
     
    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(f"输入和目标的形状不一致：{input.shape} vs {target.shape}")
        safe_input = torch.clamp(input, -50, 50)
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

class NPYDataset(Dataset):
    def __init__(self, x_file: str, y_file: str,dates_file: str,filter_nan=True,feature_names=None):
        print("Loading data files:")
        print("x_file:", x_file)
        print("y_file:", y_file)
        print("dates_file:", dates_file)
        self.x_data = np.load(x_file)
        self.y_data = np.load(y_file)
        self.dates = np.load(dates_file)  # [N]
        try:
            self.x_data = np.load(x_file)
            self.y_data = np.load(y_file)
            self.dates = np.load(dates_file) if dates_file else None
        except Exception as e:
            print("加载npz文件时出错:", e)
            raise
        print("x_data.shape:", self.x_data.shape)
        print("y_data.shape:", self.y_data.shape)
        if self.dates is not None:
            print("dates.shape:", self.dates.shape)
        assert len(self.x_data) == len(self.y_data), "x和y行数不一致"
        if self.dates is not None:
            assert len(self.x_data) == len(self.dates), "x和dates行数不一致"
        assert len(self.x_data) == len(self.y_data) == len(self.dates), "x, y, dates行数不一致"
        if filter_nan:
            self._filter_nan_samples()
        self.feature_names = feature_names
    
        # 如果没有传入，则可以尝试自动定义（比如：假设是所有特征名）
        if self.feature_names is None:
            # 这里可以自定义默认特征名
            self.feature_names = [f'Feature_{i}' for i in range(self.x_data.shape[1])]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        data = self.x_data[idx]
        label = self.y_data[idx]
        date = self.dates[idx]
        # 额外检查
        data = torch.tensor(data, dtype=torch.float32) 
        label = torch.tensor(label, dtype=torch.float32)
        if torch.isnan(data).any():
            # 处理NaN，比如用0替代
            data = torch.nan_to_num(data, nan=0.0)
        if torch.isinf(data).any():
            data = torch.nan_to_num(data, posinf=1e6, neginf=-1e6)
            print("inputs:",data.shape)
        return data, label, date
    
    def _filter_nan_samples(self):
        """过滤包含NaN的样本"""
        original_count = len(self.x_data)
        
        # 找到有效索引
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

class TransformerModel(nn.Module):
    """Transformer模型定义，匹配预训练权重结构"""
    def __init__(self, 
                 vocab_size: int,
                 input_dim: int, 
                 hidden_size: int, 
                 num_layers: int,
                 output_size: int,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 11008):
        super().__init__()
        
          # 直接用Linear层处理输入特征
        self.input_fc = nn.Linear(input_dim, hidden_size)
        
        # Transformer层定义 - 匹配"layers.x..."权重
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入保护
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 原始前向传播
        x = self.input_fc(x)
        for layer in self.layers:
            x = layer(x)
        
        outputs = self.out_proj(x)
        
        # 输出保护
        outputs = torch.clamp(outputs, min=-10.0, max=10.0)
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        return outputs

    
    @classmethod
    def load_pretrained(cls, config: Dict[str, Any]) -> 'TransformerModel':
        """加载预训练权重"""
        logger = logging.getLogger("training")
        prediction_model_path = config['paths']['output_model']
        model_params = config['model']['params']
        
        # 创建模型实例
        model = cls(
            input_dim=model_params['input_dim'],
            vocab_size=model_params['vocab_size'],
            hidden_size=model_params['hidden_size'],
            num_layers=model_params['num_layers'],
            output_size=model_params['action_space_size'],
            num_attention_heads=model_params.get('num_attention_heads', 8),
            intermediate_size=model_params.get('intermediate_size', 11008)
        )
        
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
    """安全地识别X/Y/dates文件，添加dates处理"""
    path_obj = Path(dir_path)
    files = {'x': None, 'y': None, 'dates': None}
    
    # 1. 查找所有.npy文件
    npy_files = list(path_obj.glob('*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"目录中未找到.npy文件: {dir_path}")
    
    # 2. 尝试优先匹配标准命名文件
    for f in npy_files:
        fname = f.name
        if fname in ['X.npy', 'x.npy']:
            files['x'] = str(f)
        elif fname in ['Y.npy', 'y.npy']:
            files['y'] = str(f)
        elif fname in ['dates.npy', 'date.npy', 'time.npy']:
            files['dates'] = str(f)
    
    # 3. 如果未找到标准文件，尝试按模式匹配
    if files['x'] is None:
        for f in npy_files:
            fname_lower = f.name.lower()
            if 'x' in fname_lower or 'feature' in fname_lower or 'input' in fname_lower:
                files['x'] = str(f)
                break
    
    if files['y'] is None:
        for f in npy_files:
            fname_lower = f.name.lower()
            if 'y' in fname_lower or 'target' in fname_lower or 'label' in fname_lower:
                files['y'] = str(f)
                break
    
    if files['dates'] is None:
        for f in npy_files:
            fname_lower = f.name.lower()
            if 'date' in fname_lower or 'time' in fname_lower or 'timestamp' in fname_lower:
                files['dates'] = str(f)
                break
    
    # 4. 必需的验证
    if files['x'] is None:
        raise FileNotFoundError(f"目录中未找到特征文件(X): {dir_path}")
    if files['y'] is None:
        raise FileNotFoundError(f"目录中未找到标签文件(Y): {dir_path}")
    
    return files

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
    # 获取目录下的x和y文件路径
    train_dir = config['env'].get('train_data_path')
    test_data_path = config['env'].get('test_data_path') 

    if not train_dir or not os.path.exists(train_dir):
        raise RuntimeError(f"训练数据目录不存在或未定义：{train_dir}")
    data_files = get_data_files_from_dir(train_dir)
    print("找到的文件：", data_files)
    if 'x' not in data_files or 'y' not in data_files:
        raise RuntimeError(f"未在目录中找到x或y.npy文件，文件列表：{data_files}")
    

    # 读取x和y
    x_path = data_files.get('x')
    print("x_path:", x_path)
    
    y_path = data_files.get('y')
    print("y_path:", y_path)
    dates_path = data_files.get('dates') 
    print("dates_path:",dates_path)



    if not x_path:
        raise RuntimeError("没有找到x.npy的路径")
    if not y_path:
        raise RuntimeError("没有找到y.npy的路径")
    
    # 创建或加载模型
    train_dataset =NPYDataset(x_path,y_path ,dates_path)
    input_dim = train_dataset.x_data.shape[1]
    model = TransformerModel.load_pretrained(config)
    logger.info(f"🔄 模型初始化完成, 结构: {model}")
    
    # 检查模型设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"⚙️ 模型运行在: {device}")
    
    # 加载数据集
    train_dir = config['env'].get('train_data_path')

    # 获取目录下的x和y文件路径
    data_files = get_data_files_from_dir(train_dir)

    # 读取x和y
    x_path = data_files.get('x')
    y_path = data_files.get('y')
    dates_path = data_files.get('dates')  

    if not x_path or not y_path:
        raise RuntimeError("未找到x或y文件，请确认目录中存在对应的csv文件。")

    # 使用x和y作为数据路径
    train_dataset =NPYDataset(x_path,y_path,dates_path)
    sequence_length = config['model']['params'].get('sequence_length', 30)
    
    # 训练数据集
    logger.info(f"📦 加载训练数据: {train_dir}")
    train_dataset = NPYDataset(x_path,y_path,dates_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    # 测试数据集
    test_dir = config['env'].get('test_data_path')
    test_data_files = get_data_files_from_dir(test_dir)

    x_test_path = test_data_files.get('x')
    y_test_path = test_data_files.get('y')
    dates_test_path = test_data_files.get('dates')

    if not x_test_path or not y_test_path:
        raise RuntimeError("未找到测试x或y文件，请确认目录中存在对应的.npy文件。")

    # 创建测试数据集
    test_dataset = NPYDataset(x_test_path,y_test_path ,dates_test_path)

    test_loader = DataLoader ( 
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
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
    
    # 创建梯度监控器
    grad_monitor = GradientMonitor(model)
    grad_monitor.attach()
    torch.autograd.set_detect_anomaly(True)
    supervised_criterion = SafeSmoothL1Loss(beta=1.0).to(device)
    rl_criterion = RLoss(supervised_criterion, base_loss_weight=0.5).to(device)  
    for epoch in range(config['training']['epochs']):
        new_weight = get_rl_weight(epoch)
        rl_criterion.update_weight(new_weight)
        safety.create_backup(epoch)
        model.train()
        epoch_loss = 0.0
        nan_batch_count = 0
        valid_batch_count = 0
            # 在每个epoch开始时输出
        print(f"\n【调试】开始第 {epoch+1} 轮训练")
        for batch_idx,batch_data in enumerate(train_loader):
            print(f"\n【调试】第 {batch_idx+1} 批次")
            # print("batch_data 内容：", batch_data)

            inputs, targets, dates = batch_data
            print("原始inputs.shape:", inputs.shape)
            print("原始targets.shape:", targets.shape)
            print("inputs设备:", inputs.device)
            print("targets设备:", targets.device)
            # 转移到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
                        
           
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
                direction_match = (torch.sign(model2_result[:,0]) == torch.sign(targets)).float()
                accuracy_reward = direction_match * 0.8
                error_reward = torch.exp(-2 * torch.abs(model2_result[:,0] - targets)) * 0.2
                reward = (accuracy_reward + error_reward).detach()
                print("模型输出（调试）：", outputs)
                print(f"平均奖励: {reward.mean().item():.4f}")
                outputs_protected = safety.protect_outputs(outputs)

                if outputs_protected is None:
                    print("保护后输出为None，跳过loss计算")
                    continue

                # 以保护后输出进行loss
                loss = rl_criterion(outputs_protected, model2_result, targets, reward)
                batch_reward = reward.mean().item()
                print("当前loss:", loss.item())
                
            
            # 反向传播
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
    with torch.no_grad():
        for inputs, targets, dates in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs= model(inputs)
            
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_dates.extend(dates)
            model2_preds = predict(model2, outputs)
            direction_match = (torch.sign(model2_preds[:,0]) == torch.sign(targets)).float()
            accuracy_reward = direction_match * 0.8
            error_reward = torch.exp(-2 * torch.abs(model2_preds[:,0] - targets)) * 0.2
            batch_reward = (accuracy_reward + error_reward).cpu().numpy()
            test_rewards.extend(batch_reward)
    # 转换为DataFrame便于保存结构化数据
    results_df = pd.DataFrame({
        'date': test_dataset.dates[:len(all_predictions)],
        'prediction': all_predictions,
        'target': all_targets,
        'error': np.abs(np.array(all_predictions) - np.array(all_targets))
    })
    
    # 保存结构化输出数据到model1_output
    output_path = save_structured_data(results_df, config, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
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
        "test_data_path": test_data_path,
        "model_summary": str(model),
        "Avg_Reward": avg_test_reward,  # 新字段
        "Reward_STD": reward_std,       # 新字段
    }
    eval_results_serializable = convert_to_serializable(eval_results) #计算评估指标
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets))**2))
    sharpe_val = sharpe_ratio(np.array(all_predictions))
    drawdown = max_drawdown(np.array(all_predictions))
    
    eval_results = {
        "MAE": mae,
        "RMSE": rmse,
        "Sharpe_Ratio": sharpe_val,
        "Max_Drawdown": drawdown,
        "num_samples": len(all_targets),
        "input_features": train_dataset.feature_names,
        "config_path": config_path,
        "test_data_path": test_data_path,
        "model_summary": str(model)
    }
    eval_results_serializable = convert_to_serializable(eval_results)
    
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