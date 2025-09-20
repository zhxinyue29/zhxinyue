import os
from transformers.tokenization_utils_base import BatchEncoding
from venv import logger
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset, random_split,Subset, DataLoader
import torch.nn.functional as F 
from contextlib import contextmanager
import inspect
from ..model2.inference import DeepSeekInfer
from transformers import AutoModel,  AutoModelForSequenceClassification,AutoConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoModelForCausalLM,AutoTokenizer,PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
def _auto_amp_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
USE_AUTOCAST = True                         # ✅ 全大写，后面一直用这个名
AMP_DTYPE   = _auto_amp_dtype()
USE_SCALER  = (AMP_DTYPE == torch.float16)  # 仅 FP16 需要 GradScaler
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 使用第一个GPU
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("警告: 没有可用的GPU，将使用CPU")
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
train_cfg = load_model(config_path)

try:
    model_path = train_cfg['env']['prediction_model_path']
    print("模型路径：", model_path)
except KeyError:
    print("配置文件中未找到 prediction_model_path 字段。")
    # 你可以设置默认路径或抛出异常
if not train_cfg:
    raise RuntimeError("配置加载失败！")

# 提取参数和模型路径

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> dict:
        p = Path(config_path)

        # 1) 绝对路径 & 存在：直接用
        if p.is_absolute() and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            # 可选：记录一下解析后的真实路径（不影响外部用法）
            self.config_path_resolved = str(p)
            print(f"[INFO] 解析配置：{self.config_path_resolved}")
            return cfg

        # 2) 相对路径：按多基准尝试
        script_dir = Path(__file__).resolve().parent
        # 依次尝试：CWD、脚本目录、以及脚本目录向上最多 5 层父目录
        bases = [Path.cwd(), script_dir]
        bases.extend(list(script_dir.parents)[:5])

        tried = []
        for base in bases:
            cand = (base / p).resolve()
            tried.append(str(cand))
            if cand.exists():
                with open(cand, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.config_path_resolved = str(cand)
                print(f"[INFO] 解析配置：{self.config_path_resolved}")
                return cfg

        # 3) 全都没命中：报错并把尝试过的路径列出来，方便排查
        msg = "未找到配置文件。尝试过：\n" + "\n".join(f"  - {x}" for x in tried)
        raise FileNotFoundError(msg)

def setup_logger(
    name: str,
    log_file: str,
    console_level: int = logging.WARNING,   # 关键：默认 WARNING 起
    file_level: int = logging.DEBUG,        # 文件里记录更详尽
    also_timestamp_file: bool = True        # 额外再写一份带时间戳的文件
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)          # 总开关给足，靠 handler 控制实际输出
    logger.propagate = False                # 关键：不把日志继续往 root 传

    # 清掉自己已有的 handlers，防止重复添加
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 文件 handler（固定文件）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)

    # 可选：时间戳文件
    if also_timestamp_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_file = str(Path(log_file).with_name(
            f"{Path(log_file).stem}_{ts}{Path(log_file).suffix}"
        ))
        fh2 = logging.FileHandler(ts_file, encoding="utf-8")
        fh2.setLevel(file_level)
        fh2.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(fh2)

    # 控制台 handler（把级别提高到 WARNING/ERROR 就几乎不出东西了）
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # 彻底移除 root 的默认 handler，避免别人用 logging.info() 打进控制台
    root = logging.getLogger()
    root.propagate = False
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.WARNING)  # 根日志提高阈值，兜底

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

    def __init__(self, supervised_criterion=None, sup_weight=0.5, entropy_coef=0.01, use_baseline=True):
        super().__init__()
        self.supervised_criterion = supervised_criterion  # e.g. SafeSmoothL1Loss()
        self.sup_weight = sup_weight                      # 监督项权重 λ_sup
        self.entropy_coef = entropy_coef                  # 熵系数 β
        self.use_baseline = use_baseline

        # 简单的移动平均 baseline（也可以在外面管理）
        self.register_buffer("baseline", torch.tensor(0.0))

    @torch.no_grad()
    def _update_baseline(self, reward, momentum=0.99):
        # 标量 baseline，用批次均值做 EMA
        batch_mean = reward.mean()
        self.baseline = momentum * self.baseline + (1 - momentum) * batch_mean

    def forward(self, model_outputs, targets, reward):
        """
        返回一个 dict:
          total_loss, policy_loss, supervised_loss, entropy, weight, mean_reward
        """
        logits = model_outputs.get("logits", None)
        if logits is None:
            raise ValueError("model_outputs 需要包含 'logits'（策略头输出 [B,K]）。")

        # ========= 策略：离散动作 =========
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # 兼容 [K]
        probs = F.softmax(logits, dim=-1)                 # [B,K]
        dist  = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()                            # [B]
        logp = dist.log_prob(actions)                      # [B]
        entropy = dist.entropy().mean()                    # 标量

        # ========= baseline & advantage =========
        if self.use_baseline:
            # 训练初期 baseline 是 0，逐步更新
            self._update_baseline(reward)
            advantage = reward - self.baseline             # [B]
        else:
            # 退一步，用 batch 内均值做中心化
            advantage = reward - reward.mean()

        # 注意: advantage 不需要梯度
        advantage = advantage.detach()

        # ========= 策略损失 =========
        policy_loss = -(advantage * logp).mean() - self.entropy_coef * entropy

        # ========= 监督项（可选）=========
        supervised_loss = torch.tensor(0.0, device=logits.device)
        if (self.supervised_criterion is not None) and ("regression" in model_outputs):
            y1_pred = model_outputs["regression"].view(-1)   # [B]
            supervised_loss = self.supervised_criterion(y1_pred.float(), targets.float())

        # ========= 动态权重（如果你喜欢用你的波动率逻辑）=========
        # 也可以固定 self.sup_weight 不变
        try:
            volatility = torch.abs(targets).mean()
            current_weight = float(torch.clamp(0.65 - volatility * 25, 0.2, 0.7).item())
        except Exception:
            current_weight = self.sup_weight

        total_loss = current_weight * supervised_loss + (1 - current_weight) * policy_loss

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "supervised_loss": supervised_loss,
            "entropy": entropy,
            "weight": current_weight,
            "mean_reward": reward.mean()
        }
class SafeSmoothL1Loss(nn.Module):
    """ y_pred, y_true -> 标量 loss """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        # 都是 [B] 或 [B,1]
        y_pred = torch.clamp(y_pred, -50, 50)
        y_true = torch.clamp(y_true, -50, 50)

        diff = y_pred - y_true
        diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))

        abs_diff = torch.abs(diff)
        mask = abs_diff < self.beta

        l2 = 0.5 * (diff ** 2) / self.beta
        l1 = abs_diff - 0.5 * self.beta
        loss = torch.where(mask, l2, l1).mean()

        if not torch.isfinite(loss):
            return torch.zeros((), device=y_pred.device)
        return loss
class CSVDataset(Dataset):
    """
    文本管线版：读取预处理后的 CSV（含 text/target/date），
    在 __getitem__ 中用 tokenizer 把 text -> enc（input_ids 等），返回 (enc, y, date)
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str = "text",
        target_col: str = "target",
        max_length: int = 128,
        drop_na_target: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # 避免重复 handler
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.target_col = target_col
        self.max_length = max_length

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # === 读取 CSV ===
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"读取 CSV 失败: {data_path} | {e}")

        # 有些导出会出现“只有一列 0”的情况，这里做一次兜底
        if df.shape[1] == 1 and df.columns[0] in ("0", "Unnamed: 0"):
            # 如果是误把 index 或整行串进了一列，尽力恢复；否则给出清晰报错
            self.logger.warning(f"检测到单列 CSV（列名={df.columns.tolist()}），"
                                f"请确认 {data_path} 是否为带表头的正规 CSV（text/target/date）")
            # 继续按一列处理，但没有 text/target 仍会在后续抛错

        # === 文本列：优先使用 text，其次自动寻找/拼接常见字段 ===
        self.text_col = self._pick_or_build_text_column(df, prefer=self.text_col)

        # === 目标列：必须转成数值 ===
        if self.target_col not in df.columns:
            raise ValueError(f"数据中找不到目标列：{self.target_col}，现有列={df.columns.tolist()}")

        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        if drop_na_target:
            before = len(df)
            df = df.dropna(subset=[self.target_col]).reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                self.logger.info(f"丢弃无效目标行: {dropped}")

        # 文本转字符串（缺失置空）
        df[self.text_col] = df[self.text_col].fillna("").astype(str)

        # 日期可选
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        self.df = df.reset_index(drop=True)
        self.logger.info(f"✅ 加载完成: 样本数={len(self.df)} | 文本列={self.text_col} | 目标列={self.target_col}")

    # ---------- 内部工具 ----------

    def _pick_or_build_text_column(self, df: pd.DataFrame, prefer: str) -> str:
        """
        选出文本列。如果没有 prefer，就尝试常见命名；
        若仍没有，尝试从多个字段拼接一个 'text' 列；都没有则报错（提示现有列）
        """
        if prefer in df.columns:
            return prefer

        # 常见的文本字段
        common_text = ["text", "tweet", "content", "message", "body"]
        for c in common_text:
            if c in df.columns:
                if c != "text":
                    # 统一叫 text，避免后续混乱
                    df.rename(columns={c: "text"}, inplace=True)
                return "text"

        # 尝试从多个字段拼接一个 text（适配你之前的数据可能带有 mentions/hashtags/url 等）
        candidates = [c for c in df.columns if c.lower().startswith("text") or c.lower().startswith("tweet")]
        parts = []
        for c in candidates:
            if df[c].dtype == object:
                parts.append(df[c].astype(str))
        # 一些可能的补充字段
        for c in ["screen_name", "user_name", "symbols", "hashtags"]:
            if c in df.columns and df[c].dtype == object:
                parts.append(df[c].astype(str))

        if parts:
            df["text"] = ""
            for p in parts:
                df["text"] = (df["text"] + " " + p).str.strip()
            # 万一全空，仍然报错
            if (df["text"].fillna("").str.len() > 0).any():
                self.logger.info(f"未找到 '{prefer}'，已从 {len(parts)} 个字段拼接生成 'text'")
                return "text"

        # 走到这里说明真的没有任何文本可用
        raise ValueError(
            f"数据中找不到文本列（尝试了 '{prefer}', {['text','tweet','content','message','body']}，以及前缀 text*/tweet* 拼接）。"
            f"现有列={df.columns.tolist()}。请检查预处理脚本是否按约定导出 text/target 列。"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text   = str(row[self.text_col])       # 文本
        target = float(row[self.target_col])   # 真实标签（你是股价/收益）
        date   = str(row["date"]) if "date" in row and pd.notna(row["date"]) else ""
        return text, target, date
def make_collate_fn(tokenizer, max_length=128):
    def collate(batch):
        texts, ys, dates = zip(*batch)  # 来自 CSVDataset 的 (text, y, date)
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding="longest",       # 或 "max_length"
            truncation=True,
            max_length=max_length,
        )
        y = torch.tensor(ys, dtype=torch.float32)
        return enc, y, list(dates)
    return collate

# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def compute_reward(model2_result, targets,lookback=5):
    """模拟实际交易效果 奖励"""
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
class SafetyModule:
    """
    训练稳定性保护：
    - create_backup: 备份 LoRA/模型权重
    - check_inputs: 兼容 dict(tensor) / tensor，检查 NaN/Inf/None
    - protect_outputs: 简单 logits 裁剪，避免极端数值
    - safe_forward_context: 前向安全上下文（可选择是否抛出异常）
    - check_gradients: 清洗 NaN/Inf 梯度 +（可选）逐元素裁剪 + 全局范数裁剪
    """
    def __init__(self, backup_dir="./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # --------- 备份 ----------
    def create_backup(self, model, epoch, save_lora_only=True):
        """
        若是 PEFT/LoRA 包装的模型，model.save_pretrained() 会只保存 LoRA 适配器权重（取决于库版本）。
        如果你想保存完整模型，可改为 torch.save(model.state_dict(), path/xxx.pt)
        """
        try:
            backup_path = self.backup_dir / f"lora_backup_epoch_{epoch}"
            model.save_pretrained(backup_path)
            print(f"💾 LoRA 权重已备份到: {backup_path}")
        except Exception as e:
            print(f"⚠️ 备份失败: {e}")

    # --------- 输入检查 ----------
    def _is_tensor_bad(self, t: torch.Tensor) -> bool:
        try:
            if not torch.is_tensor(t):
                return True
            if not torch.isfinite(t).all():
                return True
            return False
        except Exception:
            return True

    def _sanitize_tensor_(self, t: torch.Tensor) -> torch.Tensor:
        """
        原地清理 NaN/Inf -> 0，并限制极端值（防止梯度/数值爆炸）。
        """
        if not torch.is_tensor(t):
            return t
        # 将非有限值置零
        mask = ~torch.isfinite(t)
        if mask.any():
            t[mask] = 0.0
        # 可选：对异常大值进行硬裁剪（数值可按需调整）
        t.clamp_(min=-1e6, max=1e6)
        return t

    def check_inputs(self, inputs, targets):
        """
        检查输入是否安全，支持：
          - inputs: Tensor 或 dict[str, Tensor]（例如 tokenizer 的输出）
          - targets: Tensor
        返回 True 表示安全，False 表示不安全；若发现问题会尝试原地修复。
        """
        try:
            if inputs is None or targets is None:
                print("⚠️ inputs 或 targets 为 None")
                return False

            bad = False
            # 处理 inputs
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    if not torch.is_tensor(v):
                        print(f"⚠️ inputs['{k}'] 不是张量")
                        bad = True
                        continue
                    if self._is_tensor_bad(v):
                        print(f"⚠️ inputs['{k}'] 含 NaN/Inf，已修复")
                        self._sanitize_tensor_(v)
                        bad = True  # 标记发现过问题，但已修复
            elif torch.is_tensor(inputs):
                if self._is_tensor_bad(inputs):
                    print("⚠️ inputs 含 NaN/Inf，已修复")
                    self._sanitize_tensor_(inputs)
                    bad = True
            else:
                print("⚠️ inputs 类型异常，既不是 dict 也不是 tensor")
                return False

            # 处理 targets
            if not torch.is_tensor(targets):
                print("⚠️ targets 不是张量")
                return False
            if self._is_tensor_bad(targets):
                print("⚠️ targets 含 NaN/Inf，已修复")
                self._sanitize_tensor_(targets)
                bad = True

            # 发现问题但已修复，仍然允许继续训练
            return True

        except Exception as e:
            print(f"⚠️ check_inputs 异常: {e}")
            return False

    # --------- 输出保护 ----------
    def protect_outputs(self, outputs):
        """
        对模型输出（logits/连续值）做保底裁剪，避免极端数值。
        支持 tensor 或 dict[str, tensor]
        """
        try:
            if isinstance(outputs, dict):
                new_out = {}
                for k, v in outputs.items():
                    if torch.is_tensor(v):
                        v = torch.clamp(v, -1e4, 1e4)
                        v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
                    new_out[k] = v
                return new_out
            elif torch.is_tensor(outputs):
                v = torch.clamp(outputs, -1e4, 1e4)
                v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
                return v
            else:
                return outputs
        except Exception as e:
            print(f"⚠️ protect_outputs 异常: {e}")
            return outputs

    # --------- 前向安全上下文 ----------
    @contextmanager
    def safe_forward_context(self, rethrow: bool = True):
        """
        用法：
            with safety.safe_forward_context():
                outputs = model(**enc)

        rethrow=True：记录错误后重新抛出，便于上层逻辑中断并进入 except 分支。
        rethrow=False：仅打印错误，继续执行（不推荐，容易产生未定义变量）。
        """
        try:
            yield
        except RuntimeError as e:
            print(f"⚠️ 前向传播出错: {e}")
            if rethrow:
                raise
        except Exception as e:
            print(f"⚠️ 前向传播未知异常: {e}")
            if rethrow:
                raise

    # --------- 梯度检查（关键补充） ----------
    def check_gradients(self, model: nn.Module, max_grad_norm: float = 1.0, clip_value: float | None = None):
        """
        在 loss.backward() 之后、optimizer.step() 之前调用：
            safety.check_gradients(model, max_grad_norm, clip_value)

        功能：
        - 将参数梯度中的 NaN/Inf 置零
        - 可选：逐元素裁剪到 [-clip_value, clip_value]
        - 全局范数裁剪到 max_grad_norm
        """
        try:
            # 1) 清理 NaN/Inf
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad
                # 非有限值 → 0
                mask = ~torch.isfinite(g)
                if mask.any():
                    g[mask] = 0.0
                # 可选：逐元素裁剪
                if clip_value is not None:
                    g.clamp_(min=-float(clip_value), max=float(clip_value))

            # 2) 全局范数裁剪
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            return True

        except Exception as e:
            print(f"⚠️ check_gradients 异常: {e}")
            return False
def _tofloat(x):
    import torch
    return x.item() if isinstance(x, torch.Tensor) else float(x)
def _to_model_inputs(enc, device):
    # 允许 enc 是 BatchEncoding 或 dict
    if isinstance(enc, BatchEncoding):
        enc = dict(enc)
    elif not isinstance(enc, dict):
        raise TypeError(f"enc 应是 dict/BatchEncoding，实际是 {type(enc)}")

    if "input_ids" not in enc:
        raise KeyError(f"enc 缺少 'input_ids'，现有键: {list(enc.keys())}")

    # embedding 需要 long/int
    if enc["input_ids"].dtype != torch.long:
        enc["input_ids"] = enc["input_ids"].long()
    if "attention_mask" in enc and enc["attention_mask"].dtype != torch.long:
        enc["attention_mask"] = enc["attention_mask"].long()

    # 搬到设备
    return {k: v.to(device, non_blocking=True) for k, v in enc.items()}
class TextGenerator:
    """
    仅用于“文本生成拿奖励”，不反传梯度。
    从 model1_dir 加载 Causal LM（本地），常驻内存，避免频繁加载。
    """
    def __init__(self, model_dir: str, device: torch.device, max_length: int = 128):
        self.device = device
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True, trust_remote_code=True, local_files_only=True
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|pad|>"

        # 只做推理，使用 Causal LM
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=None,              # FP32 稳定
            attn_implementation="eager",   # 避免 sdpa 的滑窗 warning/断言
        ).to(device).eval()

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[str]:
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        gen = self.lm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        # 这里直接解码全部（包含 prompt），如果你只想要新增部分，可以额外截取
        return self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def main(config_path: str):
    # ===== 0. 基本准备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_loader = ConfigLoader(config_path)
    train_cfg = config_loader.config
    train_cfg['training']['stability'] = train_cfg.get('stability', {
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

    # 日志 & 目录
    log_file_from_cfg = train_cfg['paths']['log_file'] 
    logger = setup_logger(
        name="training",
        log_file=log_file_from_cfg,
        console_level=logging.WARNING,   # 终端干净
        file_level=logging.DEBUG,        # 文件里全量
        also_timestamp_file=True
    )
    # 加载 YAML 配置
    logger.info("🚀 开始运行训练脚本")
    logger.info(f"📄 使用配置文件: {config_path}")
    create_output_directories(train_cfg)

    # --- 路径改为 Path.exists()（必要修正） ---
    train_dir = Path(train_cfg["paths"]["train_data_path"]).resolve()
    val_dir   = Path(train_cfg["paths"]["val_data_path"]).resolve()
    test_dir  = Path(train_cfg["paths"]["test_data_path"]).resolve()
    if not train_dir.exists(): raise RuntimeError(f"训练数据不存在：{train_dir}")
    if not test_dir.exists():  raise RuntimeError(f"测试数据不存在：{test_dir}")

    # ===== 1. tokenizer（在 DataLoader 前） =====
    model1_dir = Path(train_cfg["paths"]["model1_dir"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model1_dir,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True    # ⭐ 强制只加载本地
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ===== 2. 构建 base 模型 + LoRA（只做一次）=====
    hf_config = AutoConfig.from_pretrained(model1_dir / "config.json")
    base_model = AutoModelForSequenceClassification.from_config(hf_config)

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(base_model, peft_config).to(device)

    # 显存友好设置（必要且只影响训练期缓存/激活）
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # 同步 embed 词表大小 + pad/eos
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # （可选）加载已有权重（LoRA 或全量），按需保留：
    weights_path = Path(train_cfg["paths"]["output_model"]).resolve()
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("✅ model1权重加载完成")
    else:
        logger.warning(f"⚠️ 未找到已有权重：{weights_path}，将从初始化参数开始训练")

    # ===== 3. 构建数据集 & DataLoader（使用 CSVDataset + collate） =====
    logger.info(f"🔍 加载训练数据集: {train_dir}")
    train_dataset = CSVDataset(
        data_path=train_dir,
        tokenizer=tokenizer,
        text_col="text",
        target_col="target",
        max_length=train_cfg['training'].get('max_length', 128),
    )
    logger.info(f"📊 训练样本数: {len(train_dataset)}")

    val_dataset = None
    if val_dir.exists():
        logger.info(f"🔍 加载验证数据集: {val_dir}")
        val_dataset = CSVDataset(
            data_path=val_dir,
            tokenizer=tokenizer,
            text_col="text",
            target_col="target",
            max_length=train_cfg['training'].get('max_length', 128),
        )
        logger.info(f"📊 验证集样本数: {len(val_dataset)}")

    # collate_fn
    collate = make_collate_fn(tokenizer, max_length=train_cfg['training'].get('max_length', 128))
    # ------- Debug 开关（不改配置，用代码控制）-------
    DEBUG_SMALL_TRAIN = True      # 想全量训练就改成 False
    DEBUG_TRAIN_SAMPLES = 100     # 训练集只取前 100 条
    DEBUG_MAX_BATCHES = 50        # 每个 epoch 最多跑 50 个 batch；不限制就设 None

    # ------- 训练 DataLoader -------
    train_ds_for_loader = train_dataset
    if DEBUG_SMALL_TRAIN:
        n = min(DEBUG_TRAIN_SAMPLES, len(train_dataset))
        train_ds_for_loader = Subset(train_dataset, range(n))
        print(f"⚠️ Debug 模式：仅使用前 {n} 条训练样本")

    train_loader = DataLoader(
        train_ds_for_loader,
        batch_size=train_cfg['training'].get('batch_size', 16),
        shuffle=True,
        num_workers=0,           # 先设 0，稳定后再调大
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg['training'].get('batch_size', 16),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate
        )

    hidden_size = model.config.hidden_size
    model1_dtype = next(model.parameters()).dtype
    reg_head = nn.Linear(hidden_size, 1, device=device, dtype=model1_dtype)

    # ===== 4. 模型2（LoRA 推理器） =====
    safety = SafetyModule()   # 你自己实现的带 check_inputs/protect 的版本
    try:
        model2 = DeepSeekInfer()     # 你自己的类
        model2.model.to(device)
        print("加载的 model2 类型:", type(model2))
    except Exception as e:
        raise RuntimeError(f"无法加载 model2: {e}")

    # ===== 4.1 常驻“生成用”模型一（Causal LM）—— 仅用于奖励 =====
    text_gen = TextGenerator(
        model_dir=model1_dir,
        device=device,
        max_length=train_cfg['training'].get('max_length', 128)
    )

    # ===== 5. 优化器 & 损失 =====
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(train_cfg['training']['learning_rate']),
                                 eps=1e-6)
    # AMP: 仅在 FP16 时启用 GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)

    max_grad_norm = train_cfg['training'].get('max_grad_norm', 1.0)
    loss_fn = RLoss(
        supervised_criterion=SafeSmoothL1Loss(beta=1.0).to(device),
        sup_weight=float(train_cfg['training'].get('base_loss_weight', 0.5)),
        entropy_coef=0.01,
        use_baseline=True
    ).to(device)

    grad_monitor = GradientMonitor(model)
    grad_monitor.attach()
    torch.autograd.set_detect_anomaly(True)
    
    # ===== 6. 训练循环（只用 enc，不要用未定义的 inputs）=====
    best_val_loss = float("inf")
    for epoch in range(train_cfg['training']['epochs']): 
        model.train()
        epoch_loss = 0.0
        nan_batch_count = 0
        valid_batch_count = 0
        print(f"\n【调试】开始第 {epoch+1} 轮训练")
        model.train()
        reg_head.train()

        for batch_idx, (enc, targets, date) in enumerate(train_loader):
            enc = _to_model_inputs(enc, device)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            # 1) 模型一前向（AMP）
            with torch.cuda.amp.autocast(enabled=USE_AUTOCAST, dtype=AMP_DTYPE):
                outputs = model(**enc, output_hidden_states=True)        # AutoModelForSequenceClassification
                policy_logits = outputs.logits                            # [B, K]
                last_hidden = outputs.hidden_states[-1]                   # [B, L, H]
                cls_vec = last_hidden[:, 0, :].to(reg_head.weight.dtype) # [B, H]
                reg_pred = reg_head(cls_vec).squeeze(-1)                  # [B]

            # 3) 模型二：只用于产生 reward，不回传梯度（文本→文本链路）
            with torch.no_grad():
                batch_texts = tokenizer.batch_decode(
                    enc["input_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                gen_texts = text_gen.generate(
                    prompts=batch_texts,
                    max_new_tokens=32,     # Debug 可适当减小以加速
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )
                model2_out = model2.predict(gen_texts)                # Tensor [B] 或 [B,1]
                model2_out = model2_out.view(-1).to(device)

                direction_match = (torch.sign(model2_out) == torch.sign(targets)).float()
                reward = compute_reward(model2_out, targets)
                match_rate = direction_match.mean().item()

            # 4) 计算损失（AMP）
            with torch.cuda.amp.autocast(enabled=USE_AUTOCAST, dtype=AMP_DTYPE):
                loss_dict = loss_fn(
                    model_outputs={
                        "logits": policy_logits,
                        "regression": reg_pred
                    },
                    targets=targets,
                    reward=reward
                )
                loss = loss_dict["total_loss"]

            # 5) 反向传播 + 优化（区分 FP16/BF16）
            if USE_SCALER:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            epoch_loss += loss.item()
            valid_batch_count += 1

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"[Epoch {epoch+1} | Batch {batch_idx+1}] "
                    f"loss={_tofloat(loss):.4f} | "
                    f"sup={_tofloat(loss_dict['supervised_loss']):.4f} | "
                    f"rl={_tofloat(loss_dict['policy_loss']):.4f} | "
                    f"match={match_rate:.4f}"
                )

            avg_loss = epoch_loss / valid_batch_count if valid_batch_count > 0 else float('nan')
            logger.info(f"Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.4f} | 跳过批次: {nan_batch_count}")
            
            # 输出每层梯度最大值
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_max = param.grad.abs().max().item()
                    logger.info(f"层 {name} 梯度最大值: {grad_max}")
            
            # ✅ 删除了多余的第二次 step（必要修正）
            safety.check_gradients(
                model,
                max_grad_norm=train_cfg['training'].get('max_grad_norm', 1.0),
                clip_value=None,
            )
            
            batch_reward = reward.mean().item()

            # 每10批次报告梯度状态
            if batch_idx % 10 == 0:
                grad_monitor.report(batch_idx, epoch)

            # 若设置了 DEBUG_MAX_BATCHES，可提前跳出
            if DEBUG_MAX_BATCHES is not None and (batch_idx + 1) >= DEBUG_MAX_BATCHES:
                logger.info(f"⚠️ Debug：本轮提前在 {DEBUG_MAX_BATCHES} 个 batch 处停止")
                break

        logger.info(f"🏁 Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.4f} | 跳过批次: {nan_batch_count}")

        if val_loader:
            model.eval()
            reg_head.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_enc, val_targets, _ in val_loader:
                    val_enc = _to_model_inputs(val_enc, device)
                    val_targets = val_targets.to(device, dtype=torch.float32, non_blocking=True).view(-1)

                    val_outputs = model(**val_enc, output_hidden_states=True)
                    last_hidden = val_outputs.hidden_states[-1]   # [B, L, H]
                    cls_vec = last_hidden[:, 0, :]                # [B, H]
                    reg_pred = reg_head(cls_vec).squeeze(-1)      # [B]

                    sup_loss = loss_fn.supervised_criterion(
                        reg_pred.to(torch.float32),
                        val_targets.to(torch.float32)
                    )

                    val_loss_sum += sup_loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_sum / max(1, val_steps)
            logger.info(f"Epoch {epoch+1} 验证损失: {avg_val_loss:.4f}")

            # ===== 保存最优 =====
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = Path(train_cfg['paths']['output_model'])
                torch.save(model.state_dict(), save_path)
                logger.info(f"💾 保存最佳模型到: {save_path} (val_loss: {avg_val_loss:.4f})")
            else:
                logger.info(f"当前验证损失 {avg_val_loss:.4f} ≥ 最佳 {best_val_loss:.4f}，不保存")

            model.train()
            reg_head.train()

    # 移除梯度监控钩子
    grad_monitor.detach()

    # 报告当前epoch状态
    if valid_batch_count > 0:
        avg_loss = epoch_loss / valid_batch_count
        logger.info(f"🏁 Epoch 完成 | 平均损失: {avg_loss:.6f} | 跳过批次: {nan_batch_count}")
    else:
        logger.error(f"⛔ 本轮没有有效训练批次，尝试恢复...")
        recovered_epoch = safety.recover(epoch)
        logger.warning(f"恢复至epoch {recovered_epoch}")

    model_save_path = Path(train_cfg['paths']['output_model']) 
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 模型已保存到: {model_save_path}")

    # ===== 测试和评估阶段（保持原逻辑） =====
    logger.info("🧪 开始测试阶段...")
    print("🧪 开始测试阶段...")
    test_loader = None
    test_src = train_cfg['paths'].get('test_data_path')
    if not test_src:
        logger.info("⚠️ 未配置 test_dir，跳过测试阶段")
        print("⚠️ 未配置 test_dir，跳过测试阶段")
    else:
        if isinstance(test_src, dict) and 'test' in test_src:
            test_files = test_src['test']
        elif isinstance(test_src, (list, tuple)):
            test_files = list(test_src)
        else:
            test_files = [test_src]

        test_files = [p for p in test_files if os.path.exists(p)]
        if len(test_files) == 0:
            logger.warning("⚠️ test_dir 中没有可用的测试文件，跳过测试阶段")
            test_loader = None
        else:
            logger.info(f"🧪 测试文件数: {len(test_files)}")
            test_datasets = []
            for tf in test_files:
                try:
                    ds = CSVDataset(
                        data_path=tf,
                        tokenizer=tokenizer,
                        text_col="text",
                        target_col="target",
                        max_length=train_cfg['training'].get('max_length', 128),
                        logger=logging.getLogger("CSVDataset")
                    )
                    logger.info(f"  ✔ 载入测试集: {tf} | 样本={len(ds)}")
                    test_datasets.append(ds)
                except Exception as e:
                    logger.error(f"  ❌ 加载测试文件失败: {tf} | {e}", exc_info=True)

            if len(test_datasets) == 0:
                logger.warning("⚠️ 没有任何有效的测试数据集，跳过测试阶段")
                test_loader = None
            else:
                if len(test_datasets) > 1:
                    final_test_dataset = torch.utils.data.ConcatDataset(test_datasets)
                    logger.info(f"✅ 合并测试集完成，总样本数: {len(final_test_dataset)}")
                else:
                    final_test_dataset = test_datasets[0]

                test_loader = DataLoader(
                    final_test_dataset,
                    batch_size=train_cfg['training'].get('batch_size', 16),
                    shuffle=False,
                    num_workers=train_cfg['training'].get('num_workers', 0),
                    pin_memory=True,
                    collate_fn=collate
                )

    if test_loader is not None:
        model.eval()
        reg_head.eval()
        all_preds, all_targets, all_dates = [], [], []

        with torch.no_grad():
            for bidx, (enc, targets, dates) in enumerate(test_loader):
                enc = _to_model_inputs(enc, device)
                targets = targets.to(device, dtype=torch.float32, non_blocking=True)

                outputs = model(**enc, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]      # [B, L, H]
                cls_vec = last_hidden[:, 0, :]
                cls_vec = cls_vec.to(reg_head.weight.dtype)
                reg_pred = reg_head(cls_vec).squeeze(-1)     # [B]

                all_preds.extend(reg_pred.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())
                if isinstance(dates, (list, tuple)):
                    all_dates.extend([str(d) if d is not None else "" for d in dates])
                else:
                    try:
                        all_dates.extend(list(dates))
                    except Exception:
                        all_dates.extend([""] * len(reg_pred))

                if (bidx + 1) % 100 == 0:
                    logger.info(f"🔄 测试进度: {bidx+1}/{len(test_loader)} 批次")

        if len(all_targets) > 0:
            preds_np = np.asarray(all_preds, dtype=np.float32)
            tgs_np   = np.asarray(all_targets, dtype=np.float32)

            mae  = float(np.mean(np.abs(preds_np - tgs_np)))
            rmse = float(np.sqrt(np.mean((preds_np - tgs_np) ** 2)))

            pred_sign = np.sign(preds_np)
            tg_sign   = np.sign(tgs_np)
            valid_idx = (pred_sign != 0) & (tg_sign != 0)
            dir_acc   = float(np.mean(pred_sign[valid_idx] == tg_sign[valid_idx])) if valid_idx.any() else float('nan')

            logger.info("📊 测试指标：")
            logger.info(f"  • MAE  = {mae:.6f}")
            logger.info(f"  • RMSE = {rmse:.6f}")
            logger.info(f"  • 方向准确率 = {dir_acc*100 if np.isfinite(dir_acc) else float('nan'):.2f}%")

            out_df = pd.DataFrame({
                "date": all_dates[:len(all_preds)],
                "prediction": all_preds,
                "target": all_targets
            })
            csv_path = save_structured_data(
                out_df,
                train_cfg,
                filename=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            logger.info(f"💾 已保存测试预测到: {csv_path}")

            eval_dir = Path(train_cfg['paths'].get('eval_output_dir', 'results/model1_eval'))
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_path = eval_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(eval_path, "w") as f:
                json.dump(convert_to_serializable({
                    "MAE": mae,
                    "RMSE": rmse,
                    "Direction_Accuracy": dir_acc,
                    "num_samples": len(all_targets),
                    "test_files": test_files
                }), f, indent=4)
            logger.info(f"📝 已保存评估摘要到: {eval_path}")
        else:
            logger.warning("⚠️ 测试阶段未产生有效样本，跳过指标与文件保存")

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
