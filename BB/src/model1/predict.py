# predict.py  —— BB版：在不改动原有函数/类/常量的前提下，新增文本生成接口并让 main 不落盘
import os
from pathlib import Path
from datetime import datetime
import logging
import json

import torch
import pandas as pd
import yaml

# === 新增：仅添加必要的类型与模型依赖；不移除原有 import ===
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_CONFIG = "/home/liyakun/twitter-stock-prediction/configs/model1.yaml"
# 如果你希望直接在脚本里给几条示例文本，就填到这里；否则设为 None，会自动从 test_data_path 读取
SAMPLE_TEXTS = None
# 从测试集取多少条做演示（当 SAMPLE_TEXTS=None 时生效）
TEST_TAKE_N = 5
# 输出结果保存路径（可选）
DEFAULT_OUTPUT_JSON = "/home/liyakun/twitter-stock-prediction/BB/results/model1_predict_preview.json"

logger = logging.getLogger("predict")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# 你之前给的模型一占位类（保留，未改动）
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

# === 新增：供模型二直接调用的文本生成接口（不落盘、直接返回 List[str]）===
@torch.inference_mode()
def predict_texts(
    input_texts: List[str],
    config_path: str = DEFAULT_CONFIG,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    eos_token: Optional[str] = None,
) -> List[str]:
    # 0) 基本校验
    if not isinstance(input_texts, list) or not all(isinstance(x, str) for x in input_texts):
        raise TypeError("input_texts 必须是 List[str]")

    # 1) 只从配置读取目录（不再使用写死路径）
    cfg = load_config(config_path)
    paths = cfg.get("paths", {}) or {}
    env   = cfg.get("env", {}) or {}
    model1_dir = paths.get("model1_dir") or env.get("tokenizer_dir")
    if not model1_dir:
        raise KeyError("配置缺少模型一目录：请在 model1.yaml 设置 paths.model1_dir 或 env.tokenizer_dir。")

    raw = Path(model1_dir)
    if raw.is_absolute():
        candidates = [raw]
    else:
        cfg_path = Path(config_path).resolve()
        cfg_dir  = cfg_path.parent
        cwd      = Path.cwd().resolve()
        # 依次尝试：cwd、cwd/..、cwd/../..、配置目录、配置目录的上一级
        candidates = [
            (cwd / raw).resolve(),
            (cwd.parent / raw).resolve(),
            (cwd.parent.parent / raw).resolve(),
            (cfg_dir / raw).resolve(),
            (cfg_dir.parent / raw).resolve(),
        ]

    p = None
    for cand in candidates:
        if cand.exists() and ((cand / "tokenizer.json").exists() or (cand / "tokenizer_config.json").exists()):
            p = cand
            break

    if p is None:
        tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "未找到模型一目录（需包含 tokenizer.json 或 tokenizer_config.json）。已尝试：" + tried
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 缓存 tokenizer / model（首次构建，后续复用；目录变更则重建）
    if not hasattr(predict_texts, "_cache"):
        predict_texts._cache = {}
    cache = predict_texts._cache
    need_reload = (cache.get("dir") != str(p)) or ("tok" not in cache) or ("model" not in cache)

    if need_reload:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(
            str(p), use_fast=True, trust_remote_code=True, local_files_only=True
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or "<|pad|>"

        # 关键：FP32 + eager 注意力（更稳，避免 sdpa 滑窗内核断言）
        model = AutoModelForCausalLM.from_pretrained(
            str(p),
            trust_remote_code=True,
            torch_dtype=None,                 # ← FP32（比 fp16 更不易 NaN/Inf）
            local_files_only=True,
            attn_implementation="eager",
        ).to(device).eval()

        cache["tok"], cache["model"], cache["dir"] = tok, model, str(p)

        if not hasattr(predict_texts, "_printed_once"):
            logger.info(f"设备: {device}")
            logger.info(f"模型一目录: {p}")
            predict_texts._printed_once = True

    tok, model = cache["tok"], cache["model"]

    # 3) 编码输入
    enc = tok(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(cfg.get("training", {}).get("max_length", 128)),
    ).to(device)

    # 4) 组装 logits_processor（清洗 NaN/Inf，避免 multinomial 触发断言）
    from transformers import LogitsProcessorList
    logits_processors = LogitsProcessorList()
    try:
        from transformers import InfNanRemoveLogitsProcessor
        logits_processors.append(InfNanRemoveLogitsProcessor())
    except Exception:
        # 兼容老版本 transformers：自定义一个简单清洗器（函数内定义，不新增全局符号）
        class _SafeLogitsProcessor:
            def __call__(self, input_ids, scores):
                # 将 NaN/Inf 替换为有限值，并把极端值裁剪到 [-1e4, 1e4]
                scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                scores.clamp_(min=-1e4, max=1e4)
                return scores
        logits_processors.append(_SafeLogitsProcessor())

    # 5) 生成（关闭 AMP 混精；若仍担心可将 do_sample=False 临时验证）
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=tok.pad_token_id,
        eos_token_id=(tok.convert_tokens_to_ids(eos_token) if eos_token else tok.eos_token_id),
        logits_processor=logits_processors,
    )

    # 禁用自动混精，进一步稳住数值
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            out = model.generate(**enc, **gen_kwargs)
    else:
        out = model.generate(**enc, **gen_kwargs)

    # 6) 仅返回新增 token（不回传原输入）
    input_lengths = (enc["input_ids"] != tok.pad_token_id).sum(dim=1).tolist()
    generated_texts: List[str] = []
    for i, seq in enumerate(out):
        start = input_lengths[i]
        new_tokens = seq[start:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        generated_texts.append(text.strip())

    return generated_texts


def main():
    config_path = DEFAULT_CONFIG
    logger.info(f"使用配置: {config_path}")

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 1) 构造输入文本（沿用原脚本 SAMPLE_TEXTS / 测试CSV 逻辑）
    if SAMPLE_TEXTS and len(SAMPLE_TEXTS) > 0:
        input_texts = SAMPLE_TEXTS
        logger.info(f"使用脚本内置 SAMPLE_TEXTS，共 {len(input_texts)} 条")
    else:
        test_csv = pick_test_csv_from_config(cfg)
        logger.info(f"从测试集读取文本: {test_csv}（取前 {TEST_TAKE_N} 条）")
        input_texts = take_texts_from_csv(test_csv, n=TEST_TAKE_N)

    # 2) 调用新增的 predict_texts：直接返回“生成文本”（不写任何文件）
    outputs = predict_texts(
        input_texts=input_texts,
        config_path=config_path,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # 3) 打印结果（仅控制台查看），不保存文件
    for i, (inp, out) in enumerate(zip(input_texts, outputs), 1):
        print("———")
        print(f"[{i}] 输入: {inp[:120]}{'...' if len(inp) > 120 else ''}")
        print(f"    输出: {out}")

    logger.info("✅ 预测完成（未生成任何文件）")
    return outputs

if __name__ == "__main__":
    main()
