import json
import os
import re
from typing import List, Optional

import lightning as L
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from datasets import load_dataset, interleave_datasets
from datasets.distributed import split_dataset_by_node
from transformers import AutoProcessor


class LazyValidSampleDataset(IterableDataset):
    """Lazily skips invalid raw samples and yields only prepared samples.

    Works for both HF IterableDataset and map-style Dataset that has been
    converted via `.to_iterable_dataset(...)`.
    """

    def __init__(self, base_dataset, processor, debug_print_samples: bool = False, debug_max_chars: int = 300):
        super().__init__()
        self.base_dataset = base_dataset
        self.processor = processor
        self.debug_print_samples = debug_print_samples
        self.debug_max_chars = debug_max_chars

    def _prepare_sample(self, sample) -> Optional[dict]:
        convs = sample.get("conversations") or []
        if not convs:
            return None

        has_image = sample.get("image") is not None
        messages = []
        first_user = True
        has_assistant = False
        assistant_texts = []

        for turn in convs:
            src = turn.get("from")
            text = re.sub(r"<image>\s*", "", (turn.get("value") or "").strip()).strip()

            if src in {"human", "user"}:
                content = []
                if first_user and has_image:
                    content.append({"type": "image"})
                first_user = False
                content.append({"type": "text", "text": text or ""})
                messages.append({"role": "user", "content": content})

            elif src in {"gpt", "assistant"} and text:
                has_assistant = True
                assistant_texts.append(text)
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                })

        if not messages or not has_assistant:
            return None

        rendered_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        image = sample["image"].convert("RGB") if has_image else None
        item = {
            "sample_id": sample.get("id"),
            "has_image": has_image,
            "image": image,
            "text": rendered_text,
            "debug_info": {
                "sample_id": sample.get("id"),
                "has_image": has_image,
                "image_size": list(image.size) if image is not None else None,
                "num_turns": len(convs),
                "text_preview": rendered_text[: self.debug_max_chars],
                "assistant_preview": (assistant_texts[-1][: self.debug_max_chars] if assistant_texts else ""),
            },
        }

        if self.debug_print_samples:
            print(f"[sample pid={os.getpid()}] {json.dumps(item['debug_info'], ensure_ascii=False)}", flush=True)

        return item

    def __iter__(self):
        ds = self.base_dataset
        worker = get_worker_info()
        if worker is not None and hasattr(ds, "shard"):
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)

        for sample in ds:
            item = self._prepare_sample(sample)
            if item is not None:
                yield item


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
        debug_print_samples: bool = False,
        debug_max_chars: int = 300,
        iterable_num_shards: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = None
        self.train_dataset = None

    def _build_labels(self, model_inputs):
        labels = model_inputs["input_ids"].clone()
        labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)
        model_inputs["labels"] = labels
        return model_inputs

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
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer is not None else 1
        global_rank = getattr(self.trainer, "global_rank", 0) if self.trainer is not None else 0

        for cfg in self.hparams.train_datasets:
            is_streaming = cfg.get("streaming", False)
            ds = load_dataset(
                cfg["path"],
                split=cfg.get("split", "train"),
                streaming=is_streaming,
                cache_dir=self.hparams.cache_dir,
            )

            # Key point:
            # - NEVER call HF `filter` here for map-style datasets.
            # - For `streaming=False`, convert once to iterable and then do lazy skip later.
            if not is_streaming:
                ds = ds.to_iterable_dataset(
                    num_shards=cfg.get("iterable_num_shards", self.hparams.iterable_num_shards)
                )

            ds = ds.shuffle(
                seed=self.hparams.seed,
                buffer_size=cfg.get("shuffle_buffer_size", 10000),
            )

            if world_size > 1:
                ds = split_dataset_by_node(ds, rank=global_rank, world_size=world_size)

            datasets.append(ds)
            weights.append(float(cfg.get("weight", 1.0)))

        if len(datasets) == 1:
            merged = datasets[0]
        else:
            total = sum(weights)
            merged = interleave_datasets(
                datasets,
                probabilities=[w / total for w in weights],
                seed=self.hparams.seed,
                stopping_strategy=self.hparams.stopping_strategy,
            )

        self.train_dataset = LazyValidSampleDataset(
            base_dataset=merged,
            processor=self.processor,
            debug_print_samples=self.hparams.debug_print_samples,
            debug_max_chars=self.hparams.debug_max_chars,
        )

    def collate_fn(self, batch):
        if not batch:
            raise RuntimeError("Empty batch received from LazyValidSampleDataset.")

        mm_items = [x for x in batch if x["has_image"]]
        txt_items = [x for x in batch if not x["has_image"]]

        output = {
            "text_batch": None,
            "mm_batch": None,
            "debug_samples": [x["debug_info"] for x in batch],
        }

        if txt_items:
            text_batch = self.processor(
                text=[x["text"] for x in txt_items],
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
                return_tensors="pt",
            )
            text_batch = self._build_labels(text_batch)
            text_batch["sample_ids"] = [x["sample_id"] for x in txt_items]
            output["text_batch"] = text_batch

        if mm_items:
            mm_batch = self.processor(
                images=[x["image"] for x in mm_items],
                text=[x["text"] for x in mm_items],
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
                return_tensors="pt",
            )
            mm_batch = self._build_labels(mm_batch)
            mm_batch["sample_ids"] = [x["sample_id"] for x in mm_items]
            output["mm_batch"] = mm_batch

        return output

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
