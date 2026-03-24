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
                trust_remote_code=True,
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

    def _get_prefix_input_ids(self, messages, image=None):
        """Return input_ids for one message prefix using the training processor path."""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            padding=False,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors=None,
        )
        return inputs["input_ids"][0]

    def _build_assistant_labels(self, messages, image, input_ids_row, attention_mask_row):
        """Keep labels only on assistant turns."""
        ignore_index = self.hparams.ignore_index
        seq_len = int(attention_mask_row.sum().item())
        labels = torch.full_like(input_ids_row, ignore_index)

        prefix_lens = [0]
        for end in range(1, len(messages)):
            prefix_ids = self._get_prefix_input_ids(messages[:end], image=image)
            prefix_lens.append(min(len(prefix_ids), seq_len))
        prefix_lens.append(seq_len)

        for turn_idx, (start, end) in enumerate(zip(prefix_lens[:-1], prefix_lens[1:])):
            if messages[turn_idx]["role"] == "assistant" and end > start:
                labels[start:end] = input_ids_row[start:end]

        return labels

    def collate_fn(self, batch):
        processor = self.processor
        texts, images, sample_ids = [], [], []
        batch_messages, batch_image_objs = [], []

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
            image_rgb = None
            messages = []

            for turn in sample.get("conversations", []):
                role = role_map.get(turn.get("from"))
                text = (turn.get("value") or "")
                # Remove raw <image>/<video> markers from text.
                # They will be added back as structured content by the chat template.
                text = text.replace(image_token, "").replace(video_token, "").strip()
                content = [{"type": "text", "text": text}]
                # Insert the image only into the first user turn.
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
            batch_messages.append(messages)

            if image is not None and not need_image:
                image_rgb = image.convert("RGB")
                images.append(image_rgb)
            batch_image_objs.append(image_rgb)

        model_inputs = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )

        labels = torch.stack([
            self._build_assistant_labels(messages, image_rgb, input_ids_row, attention_mask_row)
            for messages, image_rgb, input_ids_row, attention_mask_row in zip(
                batch_messages,
                batch_image_objs,
                model_inputs["input_ids"],
                model_inputs["attention_mask"],
            )
        ])

        labels.masked_fill_(model_inputs["attention_mask"] == 0, self.hparams.ignore_index)

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
