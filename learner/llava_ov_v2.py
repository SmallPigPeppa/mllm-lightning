import torch
import lightning as L
from mllm.llava_onevision_qwen2_0_5b_ov_hf.custom_models.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration
from transformers.models.llava_onevision.configuration_llava_onevision import LlavaOnevisionConfig
from peft import LoraConfig, TaskType, get_peft_model

class LlavaSFTModule(L.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            torch_dtype: torch.dtype = torch.bfloat16,
            optim_args: dict | None = None,
            lora_args: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])
        self.optim_args = optim_args
        model_config = LlavaOnevisionConfig.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )

        print(type(model_config.text_config))
        print(model_config.text_config)
        print("hidden_size =", model_config.text_config.hidden_size)
        print("intermediate_size =", model_config.text_config.intermediate_size)
        print("num_hidden_layers =", model_config.text_config.num_hidden_layers)
        print("vocab_size =", model_config.text_config.vocab_size)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=model_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=True,
        )
        self.model.config.use_cache = False  # 训练时关闭 KV cache

        if lora_args and lora_args.get("enabled", False):
            target_modules = lora_args.get("target_modules") or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_args.get("r", 64),
                lora_alpha=lora_args.get("alpha", 128),
                lora_dropout=lora_args.get("dropout", 0.05),
                bias=lora_args.get("bias", "none"),
                target_modules=target_modules,
                modules_to_save=lora_args.get("modules_to_save", None),
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()




    def training_step(self, batch, batch_idx):
        loss = self.model(**{k: v for k, v in batch.items() if k != "sample_ids"}).loss
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_ids"].size(0),
            sync_dist=False,
        )
        return loss

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_args = self.optim_args or {}
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": optim_args.get("weight_decay", 1e-2)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=optim_args.get("lr", 1e-5),
            betas=optim_args.get("betas", (0.9, 0.95)),
        )
