import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
import logging
from typing import Optional, Tuple, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StockPredictor")
def remap_deepseek_keys(state_dict):
    """完整的DeepSeek权重映射函数"""
    mapping = {
        # Embedding 和输出层
        'embed_tokens.weight': 'embed_tokens.weight',
        'lm_head.weight': 'lm_head.weight',
        'norm.weight': 'norm.weight',
    }
    
    # 为每一层添加映射 (假设28层)
    for i in range(28):
        # 注意层索引从0开始
        mapping.update({
            # 层归一化
            f'layers.{i}.input_layernorm.weight': f'layers.{i}.input_norm.weight',
            f'layers.{i}.post_attention_layernorm.weight': f'layers.{i}.post_attention_norm.weight',
            
            # 注意力层 - 权重
            f'layers.{i}.self_attn.q_proj.weight': f'layers.{i}.attention.q_proj.weight',
            f'layers.{i}.self_attn.k_proj.weight': f'layers.{i}.attention.k_proj.weight',
            f'layers.{i}.self_attn.v_proj.weight': f'layers.{i}.attention.v_proj.weight',
            f'layers.{i}.self_attn.o_proj.weight': f'layers.{i}.attention.o_proj.weight',
            
            # 注意力层 - 偏置 (如果存在)
            f'layers.{i}.self_attn.q_proj.bias': f'layers.{i}.attention.q_proj.bias',
            f'layers.{i}.self_attn.k_proj.bias': f'layers.{i}.attention.k_proj.bias',
            f'layers.{i}.self_attn.v_proj.bias': f'layers.{i}.attention.v_proj.bias',
            f'layers.{i}.self_attn.o_proj.bias': f'layers.{i}.attention.o_proj.bias',
            
            # MLP层 - 权重
            f'layers.{i}.mlp.gate_proj.weight': f'layers.{i}.mlp.gate_proj.weight',
            f'layers.{i}.mlp.up_proj.weight': f'layers.{i}.mlp.up_proj.weight',
            f'layers.{i}.mlp.down_proj.weight': f'layers.{i}.mlp.down_proj.weight',
            
            # MLP层 - 偏置 (如果存在)
            f'layers.{i}.mlp.gate_proj.bias': f'layers.{i}.mlp.gate_proj.bias',
            f'layers.{i}.mlp.up_proj.bias': f'layers.{i}.mlp.up_proj.bias',
            f'layers.{i}.mlp.down_proj.bias': f'layers.{i}.mlp.down_proj.bias',
        })
    
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = mapping.get(old_key, old_key)
        new_state_dict[new_key] = value
    
    # 添加调试信息
    print(f"📊 权重映射完成: {len(state_dict)} -> {len(new_state_dict)} 个参数")
    
    return new_state_dict
def remap_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    重映射状态字典键名 - 简化版本
    """
    new_state_dict = {}
    
    for old_key, param in state_dict.items():
        new_key = old_key
        
        # 只需要做简单的字符串替换
        if "self_attn" in new_key:
            new_key = new_key.replace("self_attn", "attention")
        
        if "input_layernorm" in new_key:
            new_key = new_key.replace("input_layernorm", "input_norm")
        
        if "post_attention_layernorm" in new_key:
            new_key = new_key.replace("post_attention_layernorm", "post_attention_norm")
        
        new_state_dict[new_key] = param
    
    return new_state_dict
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 预计算旋转位置编码
        self._precompute_rotary_emb()
        
    def _precompute_rotary_emb(self):
        # 确保所有张量都在同一设备上
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.device) / self.dim))
        t = torch.arange(self.max_position_embeddings, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, x, seq_len=None):
        return self.cos_cached, self.sin_cached

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转一半的维度"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
class DeepSeekMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=True)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=True)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=True) 
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """准备因果注意力掩码"""
    batch_size, seq_length = input_shape
    mask = torch.full(
        (seq_length, seq_length),
        torch.finfo(inputs_embeds.dtype).min,
        device=inputs_embeds.device,
    )
    mask_cond = torch.arange(mask.size(-1), device=mask.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(seq_length, past_key_values_length, dtype=mask.dtype, device=mask.device), mask], dim=-1)
    
    return mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length + past_key_values_length)
class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        
        # 归一化层
        self.input_norm = nn.LayerNorm(config["hidden_size"], eps=config["rms_norm_eps"],bias=False)
        self.post_attention_norm = nn.LayerNorm(config["hidden_size"], eps=config["rms_norm_eps"],bias=False)
        
        # 注意力层和MLP层
        self.attention = DeepSeekAttention(config)
        self.mlp = DeepSeekMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_value: 过去的键值对（用于KV缓存）
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用KV缓存
            
        Returns:
            输出元组: (hidden_states, attentions, present_key_value)
        """
        # 自注意力部分
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states  # 残差连接
        
        # MLP部分
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # 残差连接
        
        # 组织输出
        outputs = (hidden_states,)
        
        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        # 如果需要使用缓存
        if use_cache:
            outputs += (attention_outputs[2],)  # present_key_value
        
        return outputs

def adapt_deepseek_weights(pretrained_weights, num_heads=6):
    """
    正确适配DeepSeek权重格式
    """
    adapted_weights = {}
    
    for key, value in pretrained_weights.items():
        # 处理k_proj和v_proj的权重 (需要转置并重复)
        if ('k_proj.weight' in key or 'v_proj.weight' in key) and value.shape == torch.Size([256, 1536]):
            # 先转置: [256, 1536] -> [1536, 256]
            value = value.t()
            # 然后重复6次: [1536, 256] -> [1536, 1536]
            value = value.repeat(1, num_heads)
            logger.debug(f"🔄 适配权重: {key} {value.shape}")
        
        # 处理k_proj和v_proj的偏置 (需要重复)
        elif ('k_proj.bias' in key or 'v_proj.bias' in key) and value.shape == torch.Size([256]):
            # 重复6次: [256] -> [1536]
            value = value.repeat(num_heads)
            logger.debug(f"📏 适配偏置: {key} {value.shape}")
        
        else:
            # 其他权重保持不变
            adapted_weights[key] = value
            logger.debug(f"✅ 保持原样: {key} {value.shape}")
    
    logger.info(f"✅ 权重适配完成，共处理 {len(adapted_weights)} 个参数")
    return adapted_weights
class DeepSeekAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self._init_rope()
        assert self.hidden_size == self.num_heads * self.head_dim, "隐藏大小必须是头数×头维度的整数倍"
    def _init_rope(self):
        # 旋转位置编码初始化
        pass
    
    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_length, _ = x.shape
        
        # 查询投影 - 保持整体投影
        query_states = self.q_proj(x)  # [batch, seq_len, hidden_size]
        
        # 关键修改：键值投影 - 直接投影到多头格式
        key_states = self.k_proj(x)    # [batch, seq_len, num_heads * head_dim]
        value_states = self.v_proj(x)   # [batch, seq_len, num_heads * head_dim]
        
        # 重塑为多头格式
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用旋转位置编码（如果有）
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_ids)
        
        # 注意力计算
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.num_heads * self.head_dim)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, (key_states, value_states)
def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated
class DeepSeekTransformerBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # DeepSeek 特有的残差权重参数
        self.attention_residual_weight = nn.Parameter(torch.ones(1))
        self.mlp_residual_weight = nn.Parameter(torch.ones(1))
    
        # ⚠️ 关键修改：重命名归一化层以匹配权重文件
        self.input_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["rms_norm_eps"], bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["rms_norm_eps"], bias=False)
        
        # 注意力层
        self.attention = DeepSeekAttention(config)
        
        # MLP层
        self.mlp = DeepSeekMLP(config)

    def forward(self, x, attention_mask=None):
        # 保持原有的前向传播逻辑
        residual = x
        x = self.input_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + self.attention_residual_weight * x
        
        residual = x
        x = self.post_attention_norm(x)
        x = self.mlp(x)
        x = residual + self.mlp_residual_weight * x
        
        return x

class DeepSeekCompatibleModel(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.padding_idx = getattr(config, 'pad_token_id', 0)
        
        # 1. 词嵌入层 (您之前缺少的)
        self.embed_tokens = nn.Embedding(
            config['vocab_size'], 
            config['hidden_size'], 
            self.padding_idx
        )
        
        # 2. Transformer层 (需要完全重新实现)
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(config) for _ in range(config['num_layers'])
        ])
        
        # 3. 归一化层
        self.norm = nn.LayerNorm(config['hidden_size'], eps=config.get('rms_norm_eps', 1e-6),bias=False)
        
        # 4. 输出投影层
        self.output_proj = nn.Linear(config['hidden_size'], config['output_dim'],bias=False )
        
        
        # 初始化权重
        self.apply(self._init_weights)
        # 特殊初始化
        self._tie_weights()
    
    def _init_weights(self, module):
        """DeepSeek风格的权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _tie_weights(self):
        """权重绑定（如果有的话）"""
        pass
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 嵌入查找
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # 准备注意力掩码
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, inputs_embeds.shape[:2], inputs_embeds, 0
            )
        
        hidden_states = inputs_embeds
        presents = () if use_cache else None
        all_self_attns = () if output_attentions else None
        
        # 通过所有层
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                presents = presents + (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 股票预测输出
        logits = self.output_proj(hidden_states)  # 取最后一个token
        
        return logits

class DeepSeekPredictor:
    CONFIG_PATH = "/home/liyakun/twitter-stock-prediction/configs/model2.yaml"
    
    def __init__(self, device=None, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        logger.info(f"📊 使用设备: {self.device}")
        
        # 加载配置
        self.config = self._load_config(self.CONFIG_PATH)
        
        # 提取模型参数
        self.model_params = self.config["model"]["params"]
        self.deepseek_params = self.config["deepseek"]
       
        # 关键参数 - 需要确保与DeepSeek架构匹配
        self.input_dim = self.model_params["input_dim"]
        self.fixed_seq_len = self.model_params["seq_len"]
        self.model_params['use_bias'] = False
        # DeepSeek架构参数
        self.hidden_size = self.model_params["hidden_size"]
        self.num_layers = self.model_params["num_layers"]
        self.num_attention_heads = self.model_params["num_attention_heads"]
        self.vocab_size = self.model_params["vocab_size"]
        self.head_dim = self.model_params["head_dim"]
        
        # DeepSeek特定参数
        self.num_key_value_heads = self.deepseek_params["num_key_value_heads"]
        self.intermediate_size = self.deepseek_params["intermediate_size"]
        self.max_position_embeddings = self.deepseek_params["max_position_embeddings"]
        
        # 构建模型 - 使用新的DeepSeekCompatibleModel
        self.model = self._build_model_from_config()
        
        # 加载预训练权重
        if model_path:
            self.load_model(model_path)
        else:
            default_path = "/home/liyakun/twitter-stock-prediction/models/model2/best_model.pt"
            if os.path.exists(default_path):
                self.load_model(default_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_model_from_config(self):
        """构建模型并加载权重"""
        
        # 调试：打印传递给DeepSeekCompatibleModel的配置
        print("=== 传递给 DeepSeekCompatibleModel 的配置 ===")
        print("配置类型:", type(self.model_params))
        print("配置键:", list(self.model_params.keys()))
        
        # 检查是否包含所有必需参数
        required_params = ["num_layers", "hidden_size", "num_attention_heads", "vocab_size"]
        missing_params = [p for p in required_params if p not in self.model_params]
        
        if missing_params:
            print(f"❌ 缺少参数: {missing_params}")
            print("当前配置内容:", self.model_params)
            raise ValueError(f"配置缺少参数: {missing_params}")
        
        # 创建完整的配置
        full_config = {
            **self.model_params,  # 包含 num_layers 等参数
            "num_key_value_heads": self.deepseek_params["num_key_value_heads"],
            "intermediate_size": self.deepseek_params["intermediate_size"],
            "max_position_embeddings": self.deepseek_params["max_position_embeddings"],
            "rms_norm_eps": 1e-6,
        }
        
        print("✅ 完整配置:", full_config)
        
        # 现在创建模型
        model = DeepSeekCompatibleModel(full_config, self.device)
        return model
    def load_model(self, model_path):
        """加载预训练权重"""
        try:
            if os.path.exists(model_path):
                logger.info(f"📦 加载预训练权重: {model_path}")
                
                # 加载权重文件
                if model_path.endswith('.safetensors'):
                    from safetensors import safe_open
                    state_dict = {}
                    with safe_open(model_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                else:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                    
                    # 提取状态字典
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                
                logger.info(f"📊 权重文件包含 {len(state_dict)} 个参数")
                
                # 重新映射权重
                remapped_state_dict = remap_deepseek_keys(state_dict)
                
                # 加载权重 (使用 strict=False 允许部分参数不匹配)
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    remapped_state_dict, strict=False
                )
                
                # 记录不匹配的参数
                if missing_keys:
                    logger.warning(f"⚠️  缺失 {len(missing_keys)} 个参数:")
                    for key in missing_keys[:10]:
                        logger.warning(f"  - {key}")
                    if len(missing_keys) > 10:
                        logger.warning(f"  ... 还有 {len(missing_keys) - 10} 个缺失参数")
                
                if unexpected_keys:
                    logger.warning(f"⚠️  发现 {len(unexpected_keys)} 个意外参数:")
                    for key in unexpected_keys[:10]:
                        logger.warning(f"  - {key}")
                    if len(unexpected_keys) > 10:
                        logger.warning(f"  ... 还有 {len(unexpected_keys) - 10} 个意外参数")
                
                logger.info("✅ 权重加载完成")
            else:
                logger.warning("ℹ️ 未找到预训练权重，使用随机初始化")
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def predict(self, input_data):
        """预测方法 - 需要适配新的模型输出格式"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float().to(self.device)
            
            if len(input_data.shape) == 2:
                input_data = input_data.unsqueeze(0)  # 添加batch维度
            
            if input_data.shape[1] != self.fixed_seq_len:
                raise ValueError(f"❌ 输入序列长度应为 {self.fixed_seq_len}, 但得到 {input_data.shape[1]}")
            
            # 新的模型输出是字典格式
            output = self.model(input_data)
            
            # 根据您的需求选择输出
            # 如果是分类任务，可能使用 logits
            # 如果是回归任务，可能使用 hidden_states 或 last_hidden_state
            if isinstance(output, dict):
                # 返回隐藏状态（适用于特征提取）
                return output["last_hidden_state"].cpu().numpy()
            else:
                # 直接返回输出（适用于分类头）
                return output.cpu().numpy()

    def get_embeddings(self, input_data):
        """获取输入数据的嵌入表示"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float().to(self.device)
            
            if len(input_data.shape) == 2:
                input_data = input_data.unsqueeze(0)
            
            # 获取词嵌入
            embeddings = self.model.embed_tokens(input_data.long())
            return embeddings.cpu().numpy()

    def save_model(self, save_path):
        """保存模型"""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"💾 模型已保存到: {save_path}")

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_layers": self.model.num_layers,
            "hidden_size": self.model.config["hidden_size"],
            "device": str(self.device)
        }


if __name__ == "__main__":
    logger.info("🚀 启动预测器测试...")
    predictor = DeepSeekPredictor()
    
    # 2. 加载并清理预训练权重
    pretrained_path = "/home/liyakun/twitter-stock-prediction/models/model2/best_model.pt"
    if os.path.exists(pretrained_path):
        logger.info("📦 加载预训练权重...")
        pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # 🎯 在这里调用适配函数！
        logger.info("🔄 适配DeepSeek权重格式...")
        adapted_state_dict = adapt_deepseek_weights(pretrained_state_dict, num_heads=6)
        
        # 获取当前模型的state_dict
        model_state_dict = predictor.model.state_dict()
        
        # 只加载匹配的参数（过滤掉bias参数）
        filtered_state_dict = {}
        for key in model_state_dict.keys():
            if key in adapted_state_dict:  # 使用适配后的权重字典
                # 只加载weight参数，跳过bias参数
                if 'bias' not in key:
                    filtered_state_dict[key] = adapted_state_dict[key]
                    logger.debug(f"✅ 加载参数: {key}")
                else:
                    logger.debug(f"⏭️  跳过bias参数: {key}")
            else:
                logger.warning(f"⚠️  缺失参数: {key}")
        
        # 加载过滤后的权重
        predictor.model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"✅ 权重加载完成，跳过了所有bias参数")
    
    # 准备模拟输入数据
    input_dim = predictor.model_params["input_dim"]
    seq_len = predictor.model_params["seq_len"]
    mock_data = np.random.randn(seq_len, input_dim).astype(np.float32)
    
    logger.info(f"🧪 测试输入: shape={mock_data.shape}")

    # 执行预测
    try:
        prediction = predictor.predict(mock_data)
        logger.info(f"📈 预测结果: shape={prediction.shape}, 值范围: [{prediction.min():.4f}, {prediction.max():.4f}]")
        
        # 测试批量预测
        batch_size = 3
        batch_data = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        batch_prediction = predictor.predict(batch_data)
        logger.info(f"📊 批量预测结果: shape={batch_prediction.shape}")
        
    except Exception as e:
        logger.error(f"❌ 预测失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("✅ 测试完成")