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
        processor = self.processor
        texts, images, sample_ids = [], [], []

        image_token = getattr(processor, "image_token", "<image>")
        video_token = getattr(processor, "video_token", "<video>")
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
        }

        for sample in batch:
            image = sample.get("image")
            need_image = image is not None
            messages = []

            for turn in sample.get("conversations", []):
                role = role_map.get(turn.get("from"))
                text = (turn.get("value") or "")

                # Remove raw <image>/<video> markers from text.
                # They will be added back as structured content by the chat template.
                text = text.replace(image_token, "").replace(video_token, "").strip()

                content = [{"type": "text", "text": text}]
                # 只有第一轮 query 插入图片
                if role == "user" and need_image:
                    content.insert(0, {"type": "image"})
                    need_image = False

                messages.append({"role": role, "content": content})

            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            sample_ids.append(sample.get("id"))

            if image is not None and not need_image:
                images.append(image.convert("RGB"))

        model_inputs = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )
        # TODO: labels 应该去掉 user 输入, 其他 code, 例如 llama factory 怎么处理的？ 把label 的逻辑处理好，如果是多轮对话，那么去掉每一轮的 user输入的label
        labels = model_inputs["input_ids"].clone()
        labels.masked_fill_(model_inputs["attention_mask"] == 0, self.hparams.ignore_index)

        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is not None:
            labels.masked_fill_(model_inputs["input_ids"] == image_token_id, self.hparams.ignore_index)

        video_token_id = getattr(processor, "video_token_id", None)
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
