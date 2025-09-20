from torch.utils.data import DataLoader
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# 解决 import 路径
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------ 默认配置路径（写死，便于直接运行） ------------------
DEFAULT_CONFIG_PATH = Path("/home/liyakun/twitter-stock-prediction/configs/model2.yaml")


# ------------------ 日志 ------------------
def setup_logger(name: str,
                 log_file: Path,
                 console_level=logging.INFO,
                 file_level=logging.DEBUG,
                 also_timestamp_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # 固定日志文件
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # 时间戳日志文件（额外）
    if also_timestamp_file:
        ts_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{log_file.suffix}"
        ts_path = log_file.parent / ts_name
        fh2 = logging.FileHandler(str(ts_path), encoding="utf-8")
        fh2.setLevel(file_level)
        fh2.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh2)

    logger.debug(f"日志写入: {log_file}")
    return logger


# ------------------ 配置 ------------------
def load_config(path: Path):
    if not Path(path).exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------ 数据集 ------------------
class TextTargetDataset(Dataset):
    """
    读取 CSV（必须包含 text, target；可选 date）
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        cols = [c.lower() for c in df.columns]
        df.columns = cols

        if "text" not in df.columns:
            raise ValueError(f"数据中找不到 'text' 列，现有列：{list(df.columns)}")
        if "target" not in df.columns:
            raise ValueError(f"数据中找不到 'target' 列，现有列：{list(df.columns)}")

        self.texts = df["text"].astype(str).tolist()
        self.targets = torch.tensor(df["target"].astype(float).values, dtype=torch.float32)
        self.dates = df["date"].astype(str).tolist() if "date" in df.columns else [""] * len(df)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.targets[i], self.dates[i]


# ------------------ 特征抽取 ------------------
class EmbeddingExtractor:
    """
    优先调用你自己的 src.model1.predict.infer_embeddings(texts, tokenizer_dir, device) -> torch.FloatTensor [N,D]
    回退：用 HuggingFace 的 tokenizer+AutoModel，从最后隐层取 CLS 向量（或平均池化）
    """
    def __init__(self, tokenizer_dir: str, device: torch.device, logger: logging.Logger):
        self.tokenizer_dir = tokenizer_dir
        self.device = device
        self.logger = logger
        self.logger.info(f"🧩 抽特征：回退 HuggingFace @ {tokenizer_dir}")
        self._build_hf_stack()

    def _build_hf_stack(self):
        from transformers import AutoTokenizer, AutoModel
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir, use_fast=True)
        self.hf_model = AutoModel.from_pretrained(self.tokenizer_dir).to(self.device)
        self.hf_model.eval()

    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 128, batch_size: int = 32) -> torch.Tensor:
        # 1) 你的自定义函数
        if self._user_infer is not None:
            try:
                embs = self._user_infer(texts=texts,
                                        tokenizer_dir=self.tokenizer_dir,
                                        device=str(self.device),
                                        max_length=max_length,
                                        batch_size=batch_size)
                if isinstance(embs, np.ndarray):
                    embs = torch.tensor(embs, dtype=torch.float32)
                return embs.to(self.device)
            except Exception as e:
                self.logger.warning(f"⚠️ 自定义 infer_embeddings 失败，改用 HF：{e}")
                self._user_infer = None  # 后续直接走 HF

        # 2) HuggingFace 回退：CLS
        from transformers import BatchEncoding
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.hf_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.hf_model(**enc, output_hidden_states=True)
            # 取最后一层的第 1 个 token (CLS) 向量
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            cls_vec = last_hidden[:, 0, :]           # [B, H]
            embs.append(cls_vec.float().detach().cpu())

        embs = torch.cat(embs, dim=0)  # [N, H]
        return embs.to(self.device)


# ------------------ 模型构建（从目录读取结构）------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 512, output_dim: int = 1, num_layers: int = 2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(d, hidden_size), nn.ReLU()]
            d = hidden_size
        layers += [nn.Linear(d, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model_from_dir(model_dir: str, logger: logging.Logger, device: torch.device):
    """
    优先尝试 HuggingFace AutoConfig/AutoModel；
    否则按简单 config.json 里 input_dim/hidden_size/output_dim 造一个 MLP。
    返回：(model, model_kind, output_dim)
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"模型目录缺少 config.json: {model_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    # 1) HuggingFace 风格
    try:
        from transformers import AutoConfig, AutoModel
        hf_config = AutoConfig.from_pretrained(str(model_dir))
        model = AutoModel.from_pretrained(str(model_dir), config=hf_config).to(device)
        logger.info(f"🧠 使用 HF 模型: {getattr(hf_config, 'architectures', ['AutoModel'])[0]}")
        # 我们做的是回归头训练，所以这里需要一个线性头将隐藏向量 -> 标量
        head = nn.Linear(hf_config.hidden_size, 1).to(device)
        return (model, "hf_backbone+linear", 1, head)
    except Exception:
        logger.info("ℹ️ 非 HF 架构，按简单 MLP 配置构建")

    # 2) 简单 MLP
    input_dim = int(cfg_json.get("input_dim", 768))
    hidden_size = int(cfg_json.get("hidden_size", 512))
    output_dim = int(cfg_json.get("output_dim", 1))
    num_layers = int(cfg_json.get("num_layers", 2))

    model = MLP(input_dim=input_dim,
                hidden_size=hidden_size,
                output_dim=output_dim,
                num_layers=num_layers).to(device)
    logger.info(f"🧠 使用 MLP: in={input_dim}, hidden={hidden_size}, out={output_dim}, layers={num_layers}")
    return (model, "mlp", output_dim, None)


# ------------------ 损失（单独写一个类）------------------
class SafeSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self._crit = nn.SmoothL1Loss()

    def forward(self, pred, target):
        if not torch.isfinite(pred).all():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        if not torch.isfinite(target).all():
            target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        return self._crit(pred, target)


# ------------------ 训练主流程 ------------------
def main(config_path: Path = DEFAULT_CONFIG_PATH):
    cfg = load_config(config_path)

    # 路径配置
    train_csv = cfg["paths"].get("train_data_path", "/home/liyakun/twitter-stock-prediction/data/processed/train_dataset.csv")
    val_csv   = cfg["paths"].get("val_data_path",   "/home/liyakun/twitter-stock-prediction/data/processed/val_dataset.csv")
    out_model = Path(cfg["paths"]["output_model"])
    log_file  = Path(cfg["paths"]["log_file"])
    model2_dir = cfg["paths"].get("model2_dir", "/home/liyakun/twitter-stock-prediction/models/model2")
    tokenizer_dir = cfg["env"].get("tokenizer_dir", "/home/liyakun/LLaMA-Factory-main/deepseek1.5B")

    # 超参
    bs      = int(cfg["training"].get("batch_size", 16))
    lr      = float(cfg["training"].get("learning_rate", 1e-4))
    epochs  = int(cfg["training"].get("epochs", 10))
    max_len = int(cfg["training"].get("max_length", 128))

    # Debug（不改配置，用开关）
    DEBUG_SMALL_TRAIN = cfg["training"].get("debug_small_train", True)
    DEBUG_TRAIN_SAMPLES = int(cfg["training"].get("debug_train_samples", 100))
    DEBUG_MAX_BATCHES = cfg["training"].get("debug_max_batches", None)  # None 或 正整数

    # 日志
    logger = setup_logger("model2_train", log_file,
                          console_level=logging.INFO,
                          file_level=logging.DEBUG,
                          also_timestamp_file=True)
    logger.info(f"📄 使用配置: {config_path}")
    logger.info(f"📁 训练集: {train_csv}")
    logger.info(f"📁 验证集: {val_csv}")
    logger.info(f"🧭 模型目录（读取结构）: {model2_dir}")
    logger.info(f"🧩 特征抽取: 优先 src.model1.predict.infer_embeddings；回退 HF @ {tokenizer_dir}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"⚙️ 设备: {device}")

    # 数据集
    train_ds = TextTargetDataset(train_csv)
    val_ds   = TextTargetDataset(val_csv)

    if DEBUG_SMALL_TRAIN:
        n = min(DEBUG_TRAIN_SAMPLES, len(train_ds))
        train_ds_loader = Subset(train_ds, range(n))
        logger.warning(f"⚠️ Debug 模式：仅使用前 {n} 条训练样本")
    else:
        train_ds_loader = train_ds

    # 特征抽取器
    extractor = EmbeddingExtractor(tokenizer_dir=tokenizer_dir, device=device, logger=logger)

    # 先抽一小批，得到特征维度
    probe_texts = [train_ds[0][0]]
    probe_emb = extractor.encode(probe_texts, max_length=max_len, batch_size=1)  # [1, D]
    feat_dim = probe_emb.shape[-1]
    logger.info(f"🧷 训练特征维度: {feat_dim}")

    # 构建模型二
    model2, model_kind, out_dim, hf_head = build_model_from_dir(model2_dir, logger, device)

    # 如果是 HF backbone，就加一个线性头把隐藏向量 -> 标量；训练时输入必须是“特征向量”，
    # 所以这里我们简单包装一下：把特征送到线性头；HF backbone只在“加载结构/保持目录一致性”层面占位
    use_linear_head_only = (model_kind == "hf_backbone+linear") and (hf_head is not None)
    if use_linear_head_only:
        # 冻结 HF 主体（只训练线性头）
        for p in model2.parameters():
            p.requires_grad = False
        reg_head = hf_head  # nn.Linear(hidden_size, 1)
        params = list(reg_head.parameters())
        logger.info("🧱 训练策略：冻结 HF 主体，仅训练线性回归头")
    else:
        reg_head = None
        params = list(model2.parameters())
        # 校验输入维度匹配（仅对 MLP 有意义）
        if model_kind == "mlp":
            try:
                # 简单前传一次检查
                _ = model2(torch.zeros(1, feat_dim, device=device))
            except Exception as e:
                logger.error(f"❌ MLP 前向检查失败，可能是 input_dim 不匹配（期望与特征维度 {feat_dim} 一致）: {e}")
                raise

    # 优化器 & 损失
    opt = torch.optim.AdamW(params, lr=lr)
    crit = SafeSmoothL1Loss()

    # DataLoader（按批抽特征，节省内存）
    # ========= DataLoader =========
    def collate(batch):
        texts, targets, dates = zip(*batch)
        return list(texts), torch.stack(targets), list(dates)

    train_loader = DataLoader(train_ds_loader, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate)

    # --- 统计 y 分布（兼容 (text, target, date) 结构）---
    target_only_loader = DataLoader(
        train_ds_loader,  # 用与你训练一致的数据源（可能是 Subset）
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: torch.stack([b[1] for b in batch])  # 只堆叠 target
    )
    with torch.no_grad():
        y_chunks = [yb.float().view(-1).cpu() for yb in target_only_loader]
        y_all = torch.cat(y_chunks, dim=0)
    logger.info(
        f"[y 统计] mean={y_all.mean().item():.4f}, "
        f"std={y_all.std().item():.4f}, "
        f"min={y_all.min().item():.4f}, "
        f"max={y_all.max().item():.4f}, "
        f"n={y_all.numel()}"
    )

    # ========= 训练配置 =========
    out_model.parent.mkdir(parents=True, exist_ok=True)
    SAVE_ONLY_LAST = True       # 只在训练结束时把“最佳”一次性落盘；想每次刷最佳就保存，改 False
    best_val = float("inf")
    best_state = None           # 缓存最佳权重（按策略决定缓存 reg_head 还是 model2）

    # ========= 训练循环 =========
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model2.train()
        if reg_head is not None:
            reg_head.train()

        run_loss = 0.0
        seen = 0

        logger.info(f"\n【训练】开始第 {epoch} 轮")
        for b, (texts, y, _) in enumerate(train_loader, start=1):
            if DEBUG_MAX_BATCHES is not None and b > int(DEBUG_MAX_BATCHES):
                logger.info(f"🔧 Debug: 仅跑前 {DEBUG_MAX_BATCHES} 个 batch")
                break

            with torch.no_grad():
                X = extractor.encode(texts, max_length=max_len, batch_size=bs)  # [B, D]
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            if use_linear_head_only:
                pred = reg_head(X).squeeze(-1)    # [B,1] -> [B]
            else:
                pred = model2(X).squeeze(-1)      # [B,1] -> [B]

            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            run_loss += loss.item() * y.size(0)
            seen += y.size(0)

            if b % 20 == 0:
                logger.info(f"[Epoch {epoch}] batch {b} | loss={loss.item():.6f}")

        # —— 本轮训练均值 & 时间
        avg_train = run_loss / max(1, seen)
        dt = time.time() - t0
        logger.info(f"✅ Epoch {epoch} 完成 | 训练损失: {avg_train:.6f} | 用时 {dt:.1f}s")

        # ========= 验证 =========
        model2.eval()
        if reg_head is not None:
            reg_head.eval()

        val_loss = 0.0
        vseen = 0
        with torch.no_grad():
            for texts, y, _ in val_loader:
                X = extractor.encode(texts, max_length=max_len, batch_size=bs)
                y = y.to(device)
                if use_linear_head_only:
                    pred = reg_head(X).squeeze(-1)
                else:
                    pred = model2(X).squeeze(-1)
                l = crit(pred, y)
                val_loss += l.item() * y.size(0)
                vseen += y.size(0)
        avg_val = val_loss / max(1, vseen)
        logger.info(f"🧪 验证损失: {avg_val:.6f}")

        # ========= 当前 vs 历史最佳 + 保存策略 =========
        new_best = avg_val < best_val
        logger.info(
            f"Epoch {epoch}/{epochs} | 训练损失={avg_train:.6f} | 验证损失={avg_val:.6f} "
            f"| 最佳={ (best_val if best_val < float('inf') else float('nan')) :.6f}"
            f"{' ✅ 新最佳' if new_best else ''}"
        )

        if new_best:
            best_val = avg_val
            if SAVE_ONLY_LAST:
                # 只在最后保存：这里缓存权重
                best_state = {"head": reg_head.state_dict()} if use_linear_head_only \
                            else {"model": model2.state_dict()}
            else:
                # 立刻落盘
                if use_linear_head_only:
                    torch.save(reg_head.state_dict(), str(out_model))
                else:
                    torch.save(model2.state_dict(), str(out_model))
                logger.info(f"💾 已保存最佳模型 -> {out_model} (val={avg_val:.6f})")

    # ========= 训练结束：一次性保存全程最佳（若启用）=========
    if SAVE_ONLY_LAST and best_state is not None:
        if use_linear_head_only and "head" in best_state:
            torch.save(best_state["head"], str(out_model))
        elif not use_linear_head_only and "model" in best_state:
            torch.save(best_state["model"], str(out_model))
        logger.info(f"🎉 训练完成，已保存全程最佳模型 -> {out_model} (best val={best_val:.6f})")
    else:
        logger.info("🎉 训练完成")


if __name__ == "__main__":
    main(DEFAULT_CONFIG_PATH)
