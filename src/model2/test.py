#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

class DeepSeekInfer:
    """
    DeepSeek 推理/微调类，支持：
    - 固定本地路径加载基座模型
    - LoRA 微调（第一次初始化或加载已有 LoRA）
    - BF16/FP16/4bit/8bit 量化
    """
    def __init__(
        self,
        model_path: str,
        adapter_path: str = None,
        bf16: bool = True,
        fp16: bool = False,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device_map: str = "auto",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device_map = device_map

        # dtype 和量化
        self.torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
        self.quant_cfg = None
        if use_4bit:
            self.quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif use_8bit:
            self.quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 加载基座模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=self.torch_dtype,
            quantization_config=self.quant_cfg,
            trust_remote_code=True,
            local_files_only=True
        )

        # 判断 LoRA 是否存在
        if adapter_path and os.path.exists(adapter_path):
            # 加载已有 LoRA
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path, local_files_only=True)
            print(f"[INFO] 已加载 LoRA 权重：{adapter_path}")
        else:
            # 第一次训练，初始化 LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],  # 针对注意力层
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.base_model, lora_config)
            print("[INFO] 第一次训练，已初始化 LoRA 权重")

        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        no_sample: bool = False
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = not no_sample and temperature > 0
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    def save_lora(self, save_path: str):
        """
        保存 LoRA 权重到指定目录
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_pretrained(save_path)
        print(f"[INFO] LoRA 权重已保存到 {save_path}")

# ================================
# 固定配置（本地路径，无需命令行）
# ================================
MODEL_PATH = "/home/liyakun/twitter-stock-prediction/models/model2"
ADAPTER_PATH = None  # 第一次微调没有 LoRA 权重
BF16 = True

if __name__ == "__main__":
    torch.manual_seed(42)
    model2 = DeepSeekInfer(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        bf16=BF16
    )

    # 测试推理
    prompt = "测试推理效果"
    output = model2.predict(prompt)
    print("=== Generated ===")
    print(output)

    # 保存 LoRA 权重示例
    # lora_save_dir = "/home/liyakun/twitter-stock-prediction/models/model2_lora"
    # model2.save_lora(lora_save_dir)
