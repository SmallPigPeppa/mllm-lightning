import json
import os
import re
from typing import Any, Dict, List, Optional

import lightning as L
from datasets import IterableDataset as HFIterableDataset
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoProcessor


def _clean_text(text: Optional[str]) -> str:
    return re.sub(r"<image>\s*", "", (text or "").strip()).strip()


def _has_assistant_answer(convs: List[dict]) -> bool:
    for turn in convs:
        if turn.get("from") in {"gpt", "assistant"} and _clean_text(turn.get("value")):
            return True
    return False


def _sample_record(sample: Dict[str, Any], action: str, reason: Optional[str] = None, chat_text: Optional[str] = None) -> Dict[str, Any]:
    convs = sample.get("conversations") or []
    return {
        "sample_id": sample.get("id"),
        "action": action,
        "reason": reason,
        "has_image": sample.get("image") is not None,
        "num_turns": len(convs),
        "conversations": [
            {
                "from": turn.get("from"),
                "value": turn.get("value"),
            }
            for turn in convs
        ],
        "chat_text": chat_text,
    }


def _append_jsonl(log_dir: str, record: Dict[str, Any]) -> None:
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    rank = os.environ.get("RANK", "0")
    path = os.path.join(log_dir, f"rank{rank}_worker{worker_id}.jsonl")

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class StreamingSampleFilter(IterableDataset):
    def __init__(self, base_dataset, skip_text_only: bool = False, debug_samples: bool = False, debug_log_dir: Optional[str] = None):
        super().__init__()
        self.base_dataset = base_dataset
        self.skip_text_only = skip_text_only
        self.debug_samples = debug_samples
        self.debug_log_dir = debug_log_dir

    def __iter__(self):
        for sample in self.base_dataset:
            convs = sample.get("conversations") or []
            has_image = sample.get("image") is not None

            reason = None
            if not convs:
                reason = "empty_conversations"
            elif self.skip_text_only and not has_image:
                reason = "skip_text_only"
            elif not _has_assistant_answer(convs):
                reason = "no_assistant_answer"

            if self.debug_samples:
                _append_jsonl(
                    self.debug_log_dir,
                    _sample_record(
                        sample,
                        action="skip" if reason else "yield",
                        reason=reason,
                    ),
                )

            if reason is not None:
                continue

            yield sample


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
        skip_text_only: bool = False,
        debug_samples: bool = False,
        debug_log_dir: Optional[str] = None,
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

        if isinstance(self.train_dataset, HFIterableDataset) and (
            self.hparams.skip_text_only or self.hparams.debug_samples
        ):
            self.train_dataset = StreamingSampleFilter(
                self.train_dataset,
                skip_text_only=self.hparams.skip_text_only,
                debug_samples=self.hparams.debug_samples,
                debug_log_dir=self.hparams.debug_log_dir,
            )

    def _build_messages(self, sample: Dict[str, Any]):
        convs = sample.get("conversations") or []
        has_image = sample.get("image") is not None
        messages = []
        first_user = True
        has_assistant = False

        for turn in convs:
            src = turn.get("from")
            text = _clean_text(turn.get("value"))

            if src in {"human", "user"}:
                content = []
                if first_user and has_image:
                    content.append({"type": "image"})
                first_user = False
                content.append({"type": "text", "text": text or ""})
                messages.append({"role": "user", "content": content})
            elif src in {"gpt", "assistant"} and text:
                has_assistant = True
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                )

        return messages, has_image, has_assistant

    def collate_fn(self, batch):
        images, texts, sample_ids = [], [], []

        for sample in batch:
            messages, has_image, has_assistant = self._build_messages(sample)

            if self.hparams.skip_text_only and not has_image:
                if self.hparams.debug_samples:
                    _append_jsonl(
                        self.hparams.debug_log_dir,
                        _sample_record(sample, action="skip_in_collate", reason="skip_text_only_guard"),
                    )
                continue

            if not messages or not has_assistant:
                if self.hparams.debug_samples:
                    _append_jsonl(
                        self.hparams.debug_log_dir,
                        _sample_record(sample, action="skip_in_collate", reason="invalid_messages"),
                    )
                continue

            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if self.hparams.debug_samples:
                _append_jsonl(
                    self.hparams.debug_log_dir,
                    _sample_record(sample, action="use", chat_text=chat_text),
                )

            texts.append(chat_text)
            sample_ids.append(sample.get("id"))
            if has_image:
                images.append(sample["image"].convert("RGB"))

        if not texts:
            raise RuntimeError(
                "All samples in the current batch were skipped. "
                "For streaming data, prefer batch_size=1 together with skip_text_only=True."
            )

        if len(images) == len(texts):
            model_inputs = self.processor(
                images=images if images else None,
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
                return_tensors="pt",
            )
        elif len(images) == 0:
            model_inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
                return_tensors="pt",
            )
        else:
            raise RuntimeError(
                f"Mixed text-only and multimodal samples in one batch are not supported: "
                f"num_texts={len(texts)}, num_images={len(images)}. "
                "Use batch_size=1 or enable skip_text_only=True."
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
