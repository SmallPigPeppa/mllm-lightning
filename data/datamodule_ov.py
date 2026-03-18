import lightning as L
from torch.utils.data import DataLoader
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
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
        )
