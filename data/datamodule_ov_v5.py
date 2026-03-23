import re
from typing import List

import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
# from transformers import AutoProcessor
from mllm.llava_onevision_qwen2_0_5b_ov_hf.custom_models.processing_llava_onevision import LlavaOnevisionProcessor


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str,
            train_datasets: List[dict],
            batch_size: int = 1,
            num_workers: int = 2,
            max_length: int = 1024,
            seed: int = 42,
            stopping_strategy: str = "all_exhausted",
            trust_remote_code: bool = True,
            cache_dir: str = "/ppio_net0/huggingface",
            ignore_index: int = -100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = None
        self.train_dataset = None

    def setup(self, stage=None):
        if self.processor is None:
            self.processor = LlavaOnevisionProcessor.from_pretrained(
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
            is_streaming = cfg.get("streaming", False)
            ds = load_dataset(
                cfg["path"],
                split=cfg.get("split", "train"),
                streaming=is_streaming,
                cache_dir=self.hparams.cache_dir,
            )
            ds = (
                ds.shuffle(
                    seed=self.hparams.seed,
                    buffer_size=cfg.get("shuffle_buffer_size", 10000),
                )
                if is_streaming
                else ds.shuffle(seed=self.hparams.seed)
            )
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
        texts, images, sample_ids = [], [], []

        image_token = getattr(self.processor, "image_token", "<image>")
        video_token = getattr(self.processor, "video_token", "<video>")

        for sample in batch:
            conversations = sample.get("conversations") or []
            if not conversations:
                continue

            messages = []
            used_image = False
            has_assistant = False

            for turn in conversations:
                role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", }.get(
                    turn.get("from"))
                if role is None:
                    continue

                text = (turn.get("value") or "").replace(image_token, "").replace(video_token, "").strip()
                if not text:
                    continue

                if role == "user":
                    content = [{"type": "text", "text": text}]
                    if sample.get("image") is not None and not used_image:
                        content.insert(0, {"type": "image"})
                        used_image = True
                    messages.append({"role": "user", "content": content})
                else:
                    has_assistant = True
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text}],
                        }
                    )

            if not messages or not has_assistant:
                continue

            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            sample_ids.append(sample.get("id"))

            if used_image:
                images.append(sample["image"].convert("RGB"))

        if not texts:
            raise RuntimeError("No valid samples found in batch.")

        model_inputs = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        # TODO: labels 应该去掉 user 输入, 其他 code, 例如 llama factory 怎么处理的？
        labels.masked_fill_(model_inputs["attention_mask"] == 0, self.hparams.ignore_index)

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels.masked_fill_(model_inputs["input_ids"] == image_token_id, self.hparams.ignore_index)

        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is not None:
            labels.masked_fill_(model_inputs["input_ids"] == video_token_id, self.hparams.ignore_index)

        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = sample_ids
        return model_inputs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
        )
