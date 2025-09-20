# inference.py
import os
from pathlib import Path
from typing import List, Sequence, Union, Optional
import logging
import torch

# 文本路径才需要；仅本文件内部用
try:
    from transformers import AutoTokenizer, AutoModel
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# ========== 配置读取 ==========
def _load_config(cfg_path: Union[str, Path]) -> dict:
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

DEFAULT_CONFIG_PATH = Path("/home/liyakun/twitter-stock-prediction/configs/model2.yaml")

# ========== 模型二：自定义 MLP ==========
class DeepSeekMatrixModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 1536, output_dim: int = 1, num_layers: int = 30):
        super().__init__()
        layers: List[torch.nn.Module] = [torch.nn.Linear(input_dim, hidden_size), torch.nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(hidden_size, output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# ========== 轻量文本编码器（本文件自包含）==========
class _TextEncoder:
    def __init__(self, model_dir: Path, device: torch.device, max_length: int = 128, logger: Optional[logging.Logger] = None):
        if not _HF_AVAILABLE:
            raise RuntimeError("transformers 未安装，无法进行文本编码。")
        self.logger = logger or logging.getLogger("inference")
        self.device = device
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(
            str(model_dir), use_fast=True, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True,
            torch_dtype=None, attn_implementation="eager"
        ).to(device).eval()

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|pad|>"

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        vecs: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i+batch_size])
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            out = self.model(**enc)
            last = out.last_hidden_state                  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1)    # [B, T, 1]
            summed = (last * mask).sum(dim=1)             # [B, H]
            counts = mask.sum(dim=1).clamp_min(1)         # [B, 1]
            vec = summed / counts
            vecs.append(vec)
        return torch.cat(vecs, dim=0)                     # [B, H]

# ========== 推理封装 ==========
class DeepSeekInfer:
    """
    predict(x):
      - x 为 torch.Tensor（cls_vec）：直接送入模型二（保持旧用法）
      - x 为 List[str]（文本）：本文件内文本编码后送入模型二（自包含，不依赖其它脚本）
    """
    def __init__(self, device: str = "cuda", config_path: Optional[Union[str, Path]] = None, **kwargs):
        self.logger = logging.getLogger("inference")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 读取配置
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        cfg = _load_config(self.config_path)
        paths = cfg.get("paths", {})
        env   = cfg.get("env", {})
        train_sec = cfg.get("training", {})
        model_cfg = cfg.get("model", {}) or {}
        params    = model_cfg.get("params", {}) or {}
        model2_sec= cfg.get("model2", {}) or {}

        cfg_dir = self.config_path.parent
        project_root = cfg_dir.parent

        # 解析 model2_dir（相对路径支持）
        def _resolve_rel(p: Optional[str]) -> Optional[Path]:
            if not p:
                return None
            raw = Path(p)
            if raw.is_absolute():
                return raw
            # 依次尝试：config_dir, project_root, project_root/BB, cwd
            candidates = [
                (cfg_dir / raw).resolve(),
                (project_root / raw).resolve(),
                (project_root / "BB" / raw).resolve(),
                (Path.cwd() / raw).resolve(),
            ]
            for c in candidates:
                if c.exists():
                    return c
            # 如都不存在，返回第一候选以便错误提示
            return candidates[0]

        # 解析文件路径（允许把文件名映射到 model2_dir 下）
        def _resolve_file(p: Optional[str], model2_dir: Path) -> Optional[Path]:
            if not p:
                return None
            raw = Path(p)
            if raw.is_absolute():
                return raw
            # 多基准尝试
            cands = [
                (cfg_dir / raw).resolve(),
                (project_root / raw).resolve(),
                (project_root / "BB" / raw).resolve(),
                (model2_dir / raw.name).resolve(),
                (Path.cwd() / raw).resolve(),
            ]
            for c in cands:
                if c.exists():
                    return c
            return cands[0]

        # model2_dir（文本编码优先用这里的 tokenizer）
        self.model2_dir: Path = _resolve_rel(paths.get("model2_dir") or "models/model2")
        if self.model2_dir is None:
            self.model2_dir = Path("/home/liyakun/twitter-stock-prediction/BB/models/model2")

        # output_model：优先 env.prediction_model_path，其次 paths.output_model
        output_cfg = env.get("prediction_model_path") or paths.get("output_model") or "models/model2/best_model.pt"
        self.output_model: Path = _resolve_file(output_cfg, self.model2_dir)
        if not self.output_model or not self.output_model.exists():
            raise FileNotFoundError(f"基础模型权重不存在: {self.output_model}")

        # lora：可为目录或文件；目录下尝试常见文件名
        lora_cfg = paths.get("lora_output_dir")
        self.lora_path: Optional[Path] = None
        if lora_cfg:
            lp = _resolve_file(lora_cfg, self.model2_dir)
            if lp:
                if lp.is_dir():
                    for name in ("lora_weights.pt", "adapter_model.bin", "adapter.safetensors"):
                        cand = (lp / name)
                        if cand.exists():
                            lp = cand
                            break
                self.lora_path = lp if lp.exists() else None

        # 文本编码目录：优先 model2_dir；如无 tokenizer，再考虑 env.tokenizer_dir
        tok_dir = self.model2_dir
        if not ((tok_dir / "tokenizer.json").exists() or (tok_dir / "tokenizer_config.json").exists()):
            tk_cfg = env.get("tokenizer_dir")
            if tk_cfg:
                maybe = _resolve_rel(tk_cfg)
                if maybe and maybe.exists():
                    tok_dir = maybe
        self.text_encoder_dir: Path = tok_dir

        self.max_length: int = int(train_sec.get("max_length", 128))

        # —— 构建模型二（自定义 MLP）——
        cfg_input_dim = params.get("input_dim")
        hidden_size   = int(params.get("hidden_size", model2_sec.get("hidden_size", 1536)))
        num_layers    = int(params.get("num_layers",  model2_sec.get("num_layers", 30)))
        output_dim    = int(params.get("output_dim",  model2_sec.get("output_dim", 1)))

        tmp_input_dim = int(cfg_input_dim) if cfg_input_dim is not None else 1536
        self.model = DeepSeekMatrixModel(
            input_dim=tmp_input_dim, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers
        ).to(self.device)

        # 加载基础权重（并自动推断 input_dim）
        ckpt = torch.load(str(self.output_model), map_location=self.device)
        state_dict = self._extract_state_dict(ckpt)
        # 若配置未给 input_dim，尝试从首层权重推断
        first_linear_key = None
        for k, v in state_dict.items():
            if k.endswith(".layers.0.weight") or k.endswith(".0.weight"):
                first_linear_key = k
                break
        if first_linear_key and cfg_input_dim is None:
            inferred = state_dict[first_linear_key].shape[1]
            if inferred != tmp_input_dim:
                self.model = DeepSeekMatrixModel(
                    input_dim=inferred, hidden_size=hidden_size, output_dim=output_dim, num_layers=num_layers
                ).to(self.device)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            self.logger.info(f"ℹ️ 加载基础权重（missing={len(missing)}, unexpected={len(unexpected)})")

        # LoRA（可选）
        if self.lora_path and self.lora_path.exists() and self.lora_path.is_file():
            lora_ckpt = torch.load(str(self.lora_path), map_location=self.device)
            lora_sd = self._extract_state_dict(lora_ckpt)
            self.model.load_state_dict(lora_sd, strict=False)
            self.logger.info(f"✅ LoRA 权重已叠加: {self.lora_path}")
        else:
            self.logger.info("ℹ️ 未找到 LoRA 权重，跳过叠加")

        self.model.eval()
        self.logger.info(f"✅ 模型已加载到 {self.device}")

        # 文本编码器（只有在传文本预测时才会用到）
        self._encoder: Optional[_TextEncoder] = None
        if _HF_AVAILABLE and ((self.text_encoder_dir / "tokenizer.json").exists() or (self.text_encoder_dir / "tokenizer_config.json").exists()):
            try:
                self._encoder = _TextEncoder(self.text_encoder_dir, self.device, self.max_length, self.logger)
            except Exception as e:
                self.logger.warning(f"⚠️ 初始化文本编码器失败，将仅支持矩阵输入。原因: {e}")

    @staticmethod
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt
        return {k.replace("module.", ""): v for k, v in sd.items()}

    @torch.no_grad()
    def _forward_vec(self, X: torch.Tensor) -> torch.Tensor:
        # 支持 [B, D] 或 [B, T, D]：统一到 [B, D]
        if X.dim() == 3:
            B, T, D = X.shape
            if T != 1:
                X = X.mean(dim=1)
            else:
                X = X.view(B, D)
        elif X.dim() != 2:
            raise ValueError(f"features 张量维度不支持: {tuple(X.shape)}，期望 [B, D] 或 [B, T, D]")
        X = X.to(self.device)
        out = self.model(X).squeeze(-1)
        return out

    @torch.no_grad()
    def _forward_texts(self, texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        if self._encoder is None:
            raise RuntimeError(
                "文本编码器不可用：请确保 model2_dir（或 env.tokenizer_dir）下存在 tokenizer.json 或 tokenizer_config.json。"
            )
        X = self._encoder.encode(texts, batch_size=batch_size)  # [B, H]
        need_dim = self.model.layers[0].in_features
        if X.shape[-1] != need_dim:
            raise RuntimeError(f"编码后特征维 {X.shape[-1]} 与模型输入维 {need_dim} 不一致；"
                               f"可在 model2.yaml 的 model.params.input_dim 中指定。")
        out = self.model(X).squeeze(-1)
        return out

    @torch.no_grad()
    def predict(self, x: Union[torch.Tensor, Sequence[str]]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return self._forward_vec(x)
        if isinstance(x, (list, tuple)) and all(isinstance(t, str) for t in x):
            return self._forward_texts(list(x))
        raise TypeError("DeepSeekInfer.predict(x) 仅支持 torch.Tensor 或 List[str]。")

# ========== 自测 ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    infer = DeepSeekInfer()  # 使用默认 /configs/model2.yaml

    # A) 矩阵输入（保持旧用法）
    d = infer.model.layers[0].in_features
    dummy = torch.randn(2, 1, d)
    y = infer.predict(dummy)
    print("[矩阵] 输出 shape:", y.shape)

    # B) 文本输入（如果 model2_dir 下有 tokenizer.json）
    try:
        preds = infer.predict(["$TSLA jumps after earnings", "Summarize $NVDA guidance sentiment."])
        print("[文本] 输出 shape:", preds.shape)
    except Exception as e:
        print("文本路径不可用：", e)
