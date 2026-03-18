import os
import re

import torch
import lightning as L
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DeepSpeedStrategy


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_datasets: list[dict],
        batch_size: int = 1,
        num_workers: int = 2,
        max_length: int = 1024,
        streaming: bool = False,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        stopping_strategy: str = "all_exhausted",
        trust_remote_code: bool = True,
        cache_dir: str = "/ppio_net0/huggingface",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = None
        self.train_dataset = None

    def setup(self, stage=None):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.hparams.model_name_or_path,
                trust_remote_code=self.hparams.trust_remote_code,
                cache_dir=self.hparams.cache_dir,
            )
            if getattr(self.processor, "tokenizer", None):
                self.processor.tokenizer.padding_side = "right"

        if self.train_dataset is not None:
            return

        datasets, weights = [], []
        for cfg in self.hparams.train_datasets:
            is_streaming = cfg.get("streaming", self.hparams.streaming)
            ds = load_dataset(
                cfg["path"],
                split=cfg.get("split", "train"),
                streaming=is_streaming,
                cache_dir=self.hparams.cache_dir,
            )

            ds = ds.shuffle(
                seed=self.hparams.seed,
                buffer_size=cfg.get("shuffle_buffer_size", self.hparams.shuffle_buffer_size),
            ) if is_streaming else ds.shuffle(seed=self.hparams.seed)

            datasets.append(ds)
            weights.append(float(cfg.get("weight", 1.0)))

        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            total = sum(weights)
            self.train_dataset = interleave_datasets(
                datasets,
                probabilities=[w / total for w in weights],
                seed=self.hparams.seed,
                stopping_strategy=self.hparams.stopping_strategy,
            )

    def collate_fn(self, batch):
        images, texts, sample_ids = [], [], []

        for sample in batch:
            if sample.get("image") is None or sample.get("conversations") is None:
                continue

            messages = []
            first_user, has_assistant = True, False

            for turn in sample["conversations"]:
                role = "user" if turn.get("from") in {"human", "user"} else "assistant"
                text = re.sub(r"<image>\s*", "", (turn.get("value") or "").strip()).strip()

                if role == "user":
                    content = []
                    if first_user:
                        content.append({"type": "image"})
                        first_user = False
                    content.append({"type": "text", "text": text or "Please describe this image."})
                    messages.append({"role": "user", "content": content})
                elif text:
                    has_assistant = True
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    })

            if not messages or not has_assistant:
                continue

            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.append(sample["image"].convert("RGB"))
            sample_ids.append(sample.get("id"))

        if not images:
            raise RuntimeError("No valid samples found in batch.")

        model_inputs = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)

        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = sample_ids
        return model_inputs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
        )


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
            "train_loss",
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
            (no_decay_params if param.ndim < 2 or "bias" in name or "norm" in name.lower() else decay_params).append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )


def main():
    L.seed_everything(42, workers=True)

    model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    output_dir = "./outputs/llava_onevision_05b_zero3_sft_1epoch"

    dm = MultiModalDataModule(
        model_name_or_path=model_name,
        train_datasets=[
            {
                "path": "lmms-lab/LLaVA-NeXT-Data",
                "split": "train",
                "weight": 1.0,
                "streaming": False,
            }
        ],
        batch_size=1,
        num_workers=2,
        max_length=8192,
        streaming=False,
    )

    use_bf16 = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    precision = "bf16-mixed" if use_bf16 else "16-mixed"

    lit_model = LlavaSFTModule(
        model_name_or_path=model_name,
        lr=1e-5,
        weight_decay=0.01,
        trust_remote_code=True,
        use_gradient_checkpointing=True,
        torch_dtype=model_dtype,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="epoch{epoch:02d}",
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        monitor=None,
    )

    logger = CSVLogger(save_dir=output_dir, name="logs")

    strategy = DeepSpeedStrategy(
        config={
            "zero_optimization": {
                "stage": 3,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8,
            },
            "gradient_clipping": 1.0,
        }
    )

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices="auto",
        max_epochs=1,
        precision=precision,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        logger=logger,
        default_root_dir=output_dir,
    )

    trainer.fit(lit_model, datamodule=dm)

    hf_save_dir = os.path.join(output_dir, "hf_model")
    os.makedirs(hf_save_dir, exist_ok=True)

    trainer.strategy.barrier()
    if trainer.is_global_zero:
        lit_model.model.save_pretrained(hf_save_dir)
        dm.processor.save_pretrained(hf_save_dir)
    trainer.strategy.barrier()

    print(f"训练完成，HF 权重已保存到: {hf_save_dir}")
    print(f"Lightning checkpoint: {ckpt_callback.last_model_path}")


if __name__ == "__main__":
    main()