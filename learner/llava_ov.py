import torch
import lightning as L
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class LlavaSFTModule(L.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            lr: float = 1e-5,
            weight_decay: float = 0.01,
            trust_remote_code: bool = True,
            use_gradient_checkpointing: bool = True,
            torch_dtype: torch.dtype = torch.bfloat16,
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
            (no_decay_params if param.ndim < 2 or "bias" in name or "norm" in name.lower() else decay_params).append(
                param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )


