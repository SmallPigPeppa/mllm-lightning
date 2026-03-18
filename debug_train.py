import os
import re
from typing import Optional

import torch
import lightning as L

from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DeepSpeedStrategy


def clean_text(text: Optional[str]) -> str:
    text = (text or "").strip()
    text = re.sub(r"<image>\s*", "", text).strip()
    return text


def normalize_role(raw_role: str) -> str:
    return "user" if raw_role in ["human", "user"] else "assistant"


def build_llava15_prompt_and_loss_mask(
    conversations,
    tokenizer,
    max_length: int,
):
    """
    按 llava-1.5 的文本模板构造训练文本，并生成 assistant-only 的 loss mask。

    llava-1.5 多轮模板官方格式大致是：
    USER: <image>\n<prompt1> ASSISTANT: <answer1></s>
    USER: <prompt2> ASSISTANT: <answer2></s>
    ...
    """
    eos_token = tokenizer.eos_token or "</s>"

    full_text = ""
    assistant_spans = []
    seen_image = False
    has_assistant = False

    for turn in conversations:
        role = normalize_role(turn.get("from", ""))
        text = clean_text(turn.get("value"))

        if role == "user":
            if not text:
                text = "Please describe this image."

            if not seen_image:
                seg = f"USER: <image>\n{text} "
                seen_image = True
            else:
                seg = f"USER: {text} "

            full_text += seg

        else:
            if not text:
                continue

            has_assistant = True

            # 先写入 assistant 前缀，但不对前缀本身算 loss
            full_text += "ASSISTANT: "
            start = len(
                tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )

            # assistant 回复内容 + eos 算 loss
            full_text += text + eos_token
            end = len(
                tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
            )
            assistant_spans.append((start, end))

    if not full_text or not has_assistant:
        return None, None

    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    seq_len = len(tokenized["input_ids"])

    loss_mask = [0] * seq_len
    for s, e in assistant_spans:
        s = max(0, min(s, seq_len))
        e = max(0, min(e, seq_len))
        for i in range(s, e):
            loss_mask[i] = 1

    if sum(loss_mask) == 0:
        return None, None

    return full_text, loss_mask


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_datasets: list[dict],
        batch_size: int = 1,
        num_workers: int = 4,
        max_length: int = 2048,
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
            # 训练一般用 right padding 更自然
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                self.processor.tokenizer.padding_side = "right"

        if self.train_dataset is not None:
            return

        datasets = []
        weights = []

        for cfg in self.hparams.train_datasets:
            ds = load_dataset(
                cfg["path"],
                split=cfg.get("split", "train"),
                streaming=cfg.get("streaming", self.hparams.streaming),
                cache_dir=self.hparams.cache_dir,
            )

            if cfg.get("streaming", self.hparams.streaming):
                ds = ds.shuffle(
                    seed=self.hparams.seed,
                    buffer_size=cfg.get(
                        "shuffle_buffer_size",
                        self.hparams.shuffle_buffer_size,
                    ),
                )
            else:
                ds = ds.shuffle(seed=self.hparams.seed)

            datasets.append(ds)
            weights.append(float(cfg.get("weight", 1.0)))

        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
            self.train_dataset = interleave_datasets(
                datasets,
                probabilities=weights,
                seed=self.hparams.seed,
                stopping_strategy=self.hparams.stopping_strategy,
            )

    def collate_fn(self, batch):
        images = []
        texts = []
        loss_masks = []
        sample_ids = []

        tokenizer = self.processor.tokenizer

        for x in batch:
            if x.get("image") is None or x.get("conversations") is None:
                continue

            prompt_text, loss_mask = build_llava15_prompt_and_loss_mask(
                x["conversations"],
                tokenizer=tokenizer,
                max_length=self.hparams.max_length,
            )
            if prompt_text is None:
                continue

            images.append(x["image"].convert("RGB"))
            texts.append(prompt_text)
            loss_masks.append(loss_mask)
            sample_ids.append(x.get("id"))

        if not images:
            raise RuntimeError("No valid samples found in batch.")

        # 这里仍然让 processor 统一处理文本+图像，避免自己拼 image token 出现不一致
        model_inputs = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # assistant-only loss mask
        bs, seq_len = labels.shape
        batch_loss_mask = torch.zeros((bs, seq_len), dtype=torch.bool)

        for i, mask in enumerate(loss_masks):
            valid_len = min(len(mask), seq_len)
            batch_loss_mask[i, :valid_len] = torch.tensor(mask[:valid_len], dtype=torch.bool)

        labels[~batch_loss_mask] = -100

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
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["torch_dtype"])

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )

        # 训练时建议关掉 cache
        self.model.config.use_cache = False

        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def training_step(self, batch, batch_idx):
        model_inputs = {
            k: v
            for k, v in batch.items()
            if k != "sample_ids"
        }

        outputs = self.model(**model_inputs)
        loss = outputs.loss

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
        decay_params = []
        no_decay_params = []

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "norm" in n.lower():
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
        )
        return optimizer


def infer_precision():
    if not torch.cuda.is_available():
        return "32-true"

    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return "bf16-mixed"

    return "16-mixed"


def main():
    L.seed_everything(42, workers=True)

    model_name = "llava-hf/llava-1.5-7b-hf"
    output_dir = "./outputs/llava15_sft_1epoch"

    dm = MultiModalDataModule(
        model_name_or_path=model_name,
        train_datasets=[
            {
                "path": "lmms-lab/LLaVA-NeXT-Data",
                "split": "train",
                "weight": 1.0,
                "streaming": False,   # 你这里要“严格一个 epoch”，建议先用非 streaming
            }
        ],
        batch_size=1,
        num_workers=4,
        max_length=2048,
        streaming=False,
    )

    model_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )

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

    ds_config = {
        "zero_optimization": {
            "stage": 3,
        }
    }

    ds_strategy = DeepSpeedStrategy(config=ds_config)

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=ds_strategy,
        devices="auto",
        max_steps=1,
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        logger=logger,
        default_root_dir=output_dir,
    )

    trainer.fit(lit_model, datamodule=dm)


    # 保存 HF 可直接加载的权重
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