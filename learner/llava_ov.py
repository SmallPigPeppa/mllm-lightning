import json
import os

import lightning as L
import torch
from transformers import LlavaOnevisionForConditionalGeneration


class LlavaSFTModule(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        trust_remote_code: bool = True,
        use_gradient_checkpointing: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        debug_batch_dir: str | None = None,
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

        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _dump_batch_debug(self, batch, batch_idx: int):
        if not self.hparams.debug_batch_dir:
            return
        os.makedirs(self.hparams.debug_batch_dir, exist_ok=True)
        path = os.path.join(self.hparams.debug_batch_dir, f"rank{self.global_rank}.jsonl")
        record = {
            "batch_idx": batch_idx,
            "sample_ids": batch.get("sample_ids"),
            "keys": list(batch.keys()),
            "tensor_shapes": {
                k: list(v.shape)
                for k, v in batch.items()
                if torch.is_tensor(v)
            },
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def training_step(self, batch, batch_idx):
        self._dump_batch_debug(batch, batch_idx)
        model_inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        loss = self.model(**model_inputs).loss
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
            target = no_decay_params if param.ndim < 2 or "bias" in name or "norm" in name.lower() else decay_params
            target.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )
