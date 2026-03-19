import json
import os
import re
import lightning as L
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoProcessor
from typing import List


def is_valid_raw_sample(sample):
    convs = sample.get("conversations") or []
    if not convs:
        return False

    has_user = False
    has_assistant = False
    for turn in convs:
        src = turn.get("from")
        text = (turn.get("value") or "").strip()
        if src in {"human", "user"}:
            has_user = True
        elif src in {"gpt", "assistant"} and text:
            has_assistant = True

    return has_user and has_assistant


class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_datasets: List[dict],
        batch_size: int = 1,
        num_workers: int = 2,
        max_length: int = 1024,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        stopping_strategy: str = "all_exhausted",
        trust_remote_code: bool = True,
        cache_dir: str = "/ppio_net0/huggingface",
        debug_print_samples: bool = False,
        debug_max_chars: int = 300,
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

            # 这是惰性的，不会在开始训练前把全部样本扫完
            ds = ds.filter(is_valid_raw_sample)

            if is_streaming:
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
            self.train_dataset = interleave_datasets(
                datasets,
                probabilities=[w / total for w in weights],
                seed=self.hparams.seed,
                stopping_strategy=self.hparams.stopping_strategy,
            )

    def _build_labels(self, model_inputs):
        labels = model_inputs["input_ids"].clone()
        labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)
        model_inputs["labels"] = labels
        return model_inputs

    def _prepare_sample(self, sample):
        convs = sample.get("conversations") or []
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

        debug_info = {
            "sample_id": sample.get("id"),
            "has_image": has_image,
            "image_size": list(image.size) if image is not None else None,
            "num_turns": len(convs),
            "text_preview": rendered_text[: self.hparams.debug_max_chars],
            "assistant_preview": (
                assistant_texts[-1][: self.hparams.debug_max_chars]
                if assistant_texts else ""
            ),
        }

        return {
            "sample_id": sample.get("id"),
            "has_image": has_image,
            "image": image,
            "text": rendered_text,
            "debug_info": debug_info,
        }

    def collate_fn(self, batch):
        items = []
        for sample in batch:
            item = self._prepare_sample(sample)
            if item is not None:
                items.append(item)

        if not items:
            raise RuntimeError("All samples in this batch are invalid.")

        mm_items = [x for x in items if x["has_image"]]
        txt_items = [x for x in items if not x["has_image"]]

        output = {
            "text_batch": None,
            "mm_batch": None,
            "debug_samples": [x["debug_info"] for x in items],
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

        if self.hparams.debug_print_samples:
            for info in output["debug_samples"]:
                print(
                    f"[collate pid={os.getpid()}] {json.dumps(info, ensure_ascii=False)}",
                    flush=True,
                )

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