import torch
import lightning as L
from transformers import LlavaOnevisionForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model


class LlavaSFTModule(L.LightningModule):
    def __init__(
            self,
            *,
            model_name_or_path: str,
            lr: float = 1e-5,
            weight_decay: float = 0.01,
            trust_remote_code: bool = True,
            torch_dtype: torch.dtype = torch.bfloat16,
            lora_args: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.config.use_cache = False


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
                bias=lora_args.get("bias", None),
                target_modules=target_modules,
                modules_to_save=lora_args.get("modules_to_save", None),
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def _forward_one(self, packed_batch):
        model_inputs = {k: v for k, v in packed_batch.items() if k != "sample_ids"}
        loss = self.model(**model_inputs).loss
        bs = packed_batch["input_ids"].size(0)
        return loss, bs

    def training_step(self, batch, batch_idx):
        total_loss = None
        total_bs = 0

        if batch["text_batch"] is not None:
            loss_text, bs_text = self._forward_one(batch["text_batch"])
            total_loss = loss_text * bs_text if total_loss is None else total_loss + loss_text * bs_text
            total_bs += bs_text
            self.log("train/loss_text", loss_text, prog_bar=False, on_step=True, batch_size=bs_text, sync_dist=True)

        if batch["mm_batch"] is not None:
            loss_mm, bs_mm = self._forward_one(batch["mm_batch"])
            total_loss = loss_mm * bs_mm if total_loss is None else total_loss + loss_mm * bs_mm
            total_bs += bs_mm
            self.log("train/loss_mm", loss_mm, prog_bar=False, on_step=True, batch_size=bs_mm, sync_dist=True)

        if total_bs == 0:
            raise RuntimeError("Both text_batch and mm_batch are empty.")

        loss = total_loss / total_bs
        self.log("train/loss", loss, prog_bar=True, on_step=True, batch_size=total_bs, sync_dist=True)
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

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )
