#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datetime import datetime
import logging
import json

import torch
import pandas as pd
import yaml

# =========================
# 可自定义的默认配置
# =========================
DEFAULT_CONFIG = "/home/liyakun/twitter-stock-prediction/configs/model1.yaml"
# 如果你希望直接在脚本里给几条示例文本，就填到这里；否则设为 None，会自动从 test_data_path 读取
SAMPLE_TEXTS = None
# 从测试集取多少条做演示（当 SAMPLE_TEXTS=None 时生效）
TEST_TAKE_N = 5
# 输出结果保存路径（可选）
DEFAULT_OUTPUT_JSON = "/home/liyakun/twitter-stock-prediction/results/model1_predict_preview.json"

# =========================
# 简单日志
# =========================
logger = logging.getLogger("predict")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# =========================
# 你之前给的模型一占位类
# =========================
class DeepSeekTextModel(torch.nn.Module):
    def __init__(self, model_weights_path: str, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # 示例：简单线性层替代真实模型
        self.model = torch.nn.Linear(100, 50).to(self.device)
        # 加载权重（如果存在）
        if model_weights_path and Path(model_weights_path).exists():
            try:
                state_dict = torch.load(model_weights_path, map_location=self.device)
                # 兼容只存了 state_dict 或整个 checkpoint 的情况
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ 模型一权重已加载: {model_weights_path}")
            except Exception as e:
                logger.warning(f"⚠️ 加载权重失败（将使用随机初始化）: {model_weights_path} | {e}")
        else:
            logger.warning(f"⚠️ 模型权重不存在（将使用随机初始化）: {model_weights_path}")

    def forward(self, x):
        return self.model(x)

    def generate_text(self, input_tensor):
        """
        生成可读文本输出（演示用）
        """
        summaries = []
        for row in input_tensor:
            text = f"矩阵行和: {row.sum().item():.4f}"
            summaries.append(text)
        return summaries


# =========================
# 工具函数
# =========================
def load_config(config_path: str) -> dict:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def pick_model1_weights_from_config(cfg: dict) -> str:
    # 兼容两种常见写法：paths.output_model 或 models.model1_path
    paths = cfg.get("paths", {})
    if "output_model" in paths:
        return paths["output_model"]
    models = cfg.get("models", {})
    if "model1_path" in models:
        return models["model1_path"]
    raise KeyError("未在配置中找到模型一权重路径（期望 paths.output_model 或 models.model1_path）。")

def pick_test_csv_from_config(cfg: dict) -> str:
    paths = cfg.get("paths", {})
    # 兼容 test_data_path / test_dir / {"test":[...]}
    if "test_data_path" in paths:
        return paths["test_data_path"]
    if "test_dir" in paths:
        test_src = paths["test_dir"]
        if isinstance(test_src, str):
            return test_src
        if isinstance(test_src, dict) and "test" in test_src and test_src["test"]:
            return test_src["test"][0]
        if isinstance(test_src, (list, tuple)) and len(test_src) > 0:
            return test_src[0]
    raise KeyError("未在配置中找到测试文件路径（期望 paths.test_data_path 或 paths.test_dir）。")

def take_texts_from_csv(csv_path: str, n: int = 5) -> list:
    df = pd.read_csv(csv_path)
    # 兼容列名
    cand_cols = ["text", "tweet", "content", "message", "body"]
    text_col = None
    for c in cand_cols:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"测试CSV中找不到文本列（尝试 {cand_cols}）。现有列={list(df.columns)}")
    texts = df[text_col].astype(str).fillna("").tolist()
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise ValueError("测试CSV中没有有效文本。")
    return texts[:n]

def texts_to_embeddings(texts: list, device: torch.device, dim: int = 100) -> torch.Tensor:
    """
    ⚠️ 演示用：把文本“伪装”为向量。
    真实情况请用 tokenizer + model.encode() 得到向量。
    """
    torch.manual_seed(42)
    x = torch.randn(len(texts), dim, device=device)
    return x

def save_preview(results: list, out_path: str):
    os.makedirs(Path(out_path).parent, exist_ok=True)
    payload = []
    for r in results:
        row = {
            "input_text": r["input_text"],
            "output_text": r["output_text"],
            # 矩阵大对象不直接全量写入，给个摘要
            "output_matrix_shape": list(r["output_matrix"].shape),
            "output_matrix_sum": float(r["output_matrix"].sum().item()),
        }
        payload.append(row)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"📝 结果预览已保存: {out_path}")


# =========================
# 主流程（无需命令行参数）
# =========================
def main():
    config_path = DEFAULT_CONFIG
    logger.info(f"使用配置: {config_path}")

    cfg = load_config(config_path)
    weights_path = pick_model1_weights_from_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 1) 构造输入文本
    if SAMPLE_TEXTS and len(SAMPLE_TEXTS) > 0:
        input_texts = SAMPLE_TEXTS
        logger.info(f"使用脚本内置 SAMPLE_TEXTS，共 {len(input_texts)} 条")
    else:
        test_csv = pick_test_csv_from_config(cfg)
        logger.info(f"从测试集读取文本: {test_csv}（取前 {TEST_TAKE_N} 条）")
        input_texts = take_texts_from_csv(test_csv, n=TEST_TAKE_N)

    # 2) 初始化模型一
    model1 = DeepSeekTextModel(weights_path, device=device)
    model1.eval()

    # 3) 文本 -> 向量（演示）
    inputs = texts_to_embeddings(input_texts, device=device, dim=100)

    # 4) 前向得到“矩阵表示” & 生成可读文本
    results = []
    with torch.no_grad():
        output_matrix = model1(inputs)                      # [B, 50]
        output_texts = model1.generate_text(output_matrix)  # list[str]

        for text, mat, txt in zip(input_texts, output_matrix, output_texts):
            results.append({
                "input_text": text,
                "output_matrix": mat.detach().cpu(),
                "output_text": txt
            })
            # 简要打印
            print("———")
            print(f"输入: {text[:120]}{'...' if len(text) > 120 else ''}")
            print(f"输出矩阵形状: {tuple(mat.shape)} | 行和摘要文本: {txt}")

    # 5) 可选保存一个预览json（不保存大矩阵本体，避免文件过大）
    try:
        save_preview(results, DEFAULT_OUTPUT_JSON)
    except Exception as e:
        logger.warning(f"结果预览保存失败（忽略）: {e}")

    logger.info("✅ 预测完成")
    return results


if __name__ == "__main__":
    main()
