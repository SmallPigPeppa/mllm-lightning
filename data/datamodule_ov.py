import os
import re
import json
import lightning as L
from torch.utils.data import DataLoader, get_worker_info
from datasets import load_dataset, interleave_datasets
from transformers import AutoProcessor
from typing import List


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_datasets: List[dict],
        batch_size: int = 1,
        num_workers: int = 2,
        max_length: int = 1024,
        streaming: bool = False,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        stopping_strategy: str = "all_exhausted",
        trust_remote_code: bool = True,
        cache_dir: str = "/ppio_net0/huggingface",
        skip_text_only: bool = True,
        debug_samples: bool = False,
        debug_log_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = None
        self.train_dataset = None

    def _debug_print(self, msg: str):
        if not self.hparams.debug_samples:
            return
        rank = int(os.environ.get("RANK", "0"))
        worker = get_worker_info().id if get_worker_info() is not None else 0
        full_msg = f"[rank={rank} worker={worker}] {msg}"
        print(full_msg, flush=True)

        if self.hparams.debug_log_dir:
            os.makedirs(self.hparams.debug_log_dir, exist_ok=True)
            path = os.path.join(
                self.hparams.debug_log_dir,
                f"rank{rank}_worker{worker}.log"
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")

    def _sample_to_str(self, sample):
        convs = sample.get("conversations") or []
        has_image = sample.get("image") is not None
        image_info = None
        if has_image:
            try:
                image_info = {
                    "size": getattr(sample["image"], "size", None),
                    "mode": getattr(sample["image"], "mode", None),
                }
            except Exception:
                image_info = "unavailable"

        preview = []
        for i, turn in enumerate(convs[:10]):
            text = (turn.get("value") or "").replace("\n", "\\n")
            preview.append({
                "idx": i,
                "from": turn.get("from"),
                "value": text[:300],
            })

        return json.dumps(
            {
                "id": sample.get("id"),
                "has_image": has_image,
                "image_info": image_info,
                "num_turns": len(convs),
                "conversations_preview": preview,
            },
            ensure_ascii=False,
        )

    def _is_valid_mm_sample(self, sample):
        convs = sample.get("conversations") or []
        has_image = sample.get("image") is not None
        has_user = any(
            t.get("from") in {"human", "user"} and (t.get("value") or "").strip()
            for t in convs
        )
        has_assistant = any(
            t.get("from") in {"gpt", "assistant"} and (t.get("value") or "").strip()
            for t in convs
        )

        ok = has_image and has_user and has_assistant
        if self.hparams.debug_samples and not ok:
            self._debug_print(f"FILTER_SKIP sample={self._sample_to_str(sample)}")
        return ok

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

            # 只训练多模态时，直接过滤掉纯文本
            if self.hparams.skip_text_only:
                ds = ds.filter(self._is_valid_mm_sample)

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
            if self.hparams.debug_samples:
                self._debug_print(f"RAW sample={self._sample_to_str(sample)}")

            convs = sample.get("conversations") or []
            if not convs:
                self._debug_print(f"SKIP no_conversations sample={self._sample_to_str(sample)}")
                continue

            has_image = sample.get("image") is not None
            if self.hparams.skip_text_only and not has_image:
                self._debug_print(f"SKIP text_only sample={self._sample_to_str(sample)}")
                continue

            messages = []
            first_user = True
            has_assistant = False

            for turn in convs:
                src = turn.get("from")
                text = re.sub(r"<image>\s*", "", (turn.get("value") or "").strip()).strip()

                if src in {"human", "user"}:
                    content = []
                    if first_user and has_image:
                        content.append({"type": "image"})
                    first_user = False

                    if text:
                        content.append({"type": "text", "text": text})

                    if content:
                        messages.append({"role": "user", "content": content})

                elif src in {"gpt", "assistant"} and text:
                    has_assistant = True
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    })

            if not messages or not has_assistant:
                self._debug_print(f"SKIP invalid_messages sample={self._sample_to_str(sample)}")
                continue

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            texts.append(prompt)
            sample_ids.append(sample.get("id"))

            if has_image:
                images.append(sample["image"].convert("RGB"))

            self._debug_print(
                f"KEEP id={sample.get('id')} has_image={has_image} prompt_preview={prompt[:500]!r}"
            )

        if not texts:
            raise RuntimeError("No valid samples found in batch after filtering.")

        # 现在 skip_text_only=True 时，理论上这里一定是全图文 batch
        if len(images) != len(texts):
            raise RuntimeError(
                f"images/texts size mismatch: len(images)={len(images)}, len(texts)={len(texts)}"
            )

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
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
        )