import re
from typing import List, Optional, Dict, Any

import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoProcessor


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
            is_streaming = cfg.get("streaming", False)
            ds = load_dataset(
                cfg["path"],
                split=cfg.get("split", "train"),
                streaming=is_streaming,
                cache_dir=self.hparams.cache_dir,
            )

            ds = (
                ds.shuffle(seed=self.hparams.seed, buffer_size=cfg.get("shuffle_buffer_size", 10000, ), )
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

    def _normalize_role(self, role: Optional[str]) -> Optional[str]:
        if role in {"human", "user"}:
            return "user"
        if role in {"gpt", "assistant"}:
            return "assistant"
        return None

    def _get_image_token(self) -> str:
        return getattr(self.processor, "image_token", "<image>")

    def _get_video_token(self) -> str:
        return getattr(self.processor, "video_token", "<video>")

    def _strip_mm_tokens(self, text: str) -> str:
        if not text:
            return ""

        image_token = re.escape(self._get_image_token())
        video_token = re.escape(self._get_video_token())

        text = re.sub(rf"{image_token}\s*", "", text)
        text = re.sub(rf"{video_token}\s*", "", text)
        return text.strip()

    def _build_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        convs = sample.get("conversations") or []
        if not convs:
            return None

        image = sample.get("image", None)
        has_image = image is not None

        messages = []
        first_user = True
        has_assistant = False

        for turn in convs:
            role = self._normalize_role(turn.get("from"))
            if role is None:
                continue

            text = self._strip_mm_tokens((turn.get("value") or "").strip())

            if role == "user":
                content = []

                # 只在第一个 user turn 注入 image，占位 token 交给 chat template 生成
                if first_user and has_image:
                    content.append({"type": "image"})
                first_user = False

                # 允许 image-only 的第一轮 user 消息
                if text:
                    content.append({"type": "text", "text": text})

                if not content:
                    continue

                messages.append({"role": "user", "content": content})

            elif role == "assistant":
                if not text:
                    continue

                has_assistant = True
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                )

        if not messages or not has_assistant:
            return None

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": prompt,
            "image": image.convert("RGB") if has_image else None,
            "sample_id": sample.get("id"),
            "has_image": has_image,
        }

    def collate_fn(self, batch):
        items = []
        for sample in batch:
            item = self._build_sample(sample)
            if item is not None:
                items.append(item)

        if not items:
            raise RuntimeError("No valid samples found in batch.")

        texts = [x["text"] for x in items]
        images = [x["image"] for x in items if x["image"] is not None]
        sample_ids = [x["sample_id"] for x in items]
        has_image_mask = torch.tensor([x["has_image"] for x in items], dtype=torch.bool)

        processor_kwargs = dict(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )

        # 只要 batch 里存在图像，就把图像按“在 texts 中出现的带图样本顺序”传进去
        if len(images) > 0:
            processor_kwargs["images"] = images

        model_inputs = self.processor(**processor_kwargs)

        labels = model_inputs["input_ids"].clone()
        labels = labels.masked_fill(model_inputs["attention_mask"] == 0, self.hparams.ignore_index)

        # 不对 mm placeholder 本身计算 loss
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels = labels.masked_fill(model_inputs["input_ids"] == image_token_id, self.hparams.ignore_index)

        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is not None:
            labels = labels.masked_fill(model_inputs["input_ids"] == video_token_id, self.hparams.ignore_index)

        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = sample_ids
        model_inputs["has_image"] = has_image_mask
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
