import lightning as L
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoProcessor



class MultiModalDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_datasets: list[dict],
        batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 2048,
        streaming: bool = True,
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
        self.processor = AutoProcessor.from_pretrained(
            self.hparams.model_name_or_path,
            trust_remote_code=self.hparams.trust_remote_code,
            cache_dir=self.hparams.cache_dir,
        )

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
        images, texts, sample_ids = [], [], []

        for x in batch:
            if x.get("image") is None or x.get("conversations") is None:
                continue

            images.append(x["image"].convert("RGB"))
            sample_ids.append(x.get("id"))

            text = "\n".join(
                f"{'USER' if t['from'] in ['human', 'user'] else 'ASSISTANT'}: {t['value'].strip()}"
                for t in x["conversations"]
            )
            texts.append(text)

        if not images:
            raise RuntimeError("empty batch")

        out = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
            return_tensors="pt",
        )

        out["labels"] = out["input_ids"].clone()
        out["labels"][out["attention_mask"] == 0] = -100
        out["sample_ids"] = sample_ids
        return out

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

if __name__ == "__main__":
    import os
    from huggingface_hub import constants as hf_constants

    print("HF_HOME env =", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE env =", os.environ.get("HF_HUB_CACHE"))
    print("HF_DATASETS_CACHE env =", os.environ.get("HF_DATASETS_CACHE"))
    print("TRANSFORMERS_CACHE env =", os.environ.get("TRANSFORMERS_CACHE"))

    print("huggingface_hub HF_HOME =", hf_constants.HF_HOME)
    print("huggingface_hub HF_HUB_CACHE =", hf_constants.HF_HUB_CACHE)
    import pdb;pdb.set_trace()

    from huggingface_hub import snapshot_download

    # ===== 1. 设置 Hugging Face 缓存目录 =====
    HF_ROOT = "/ppio_net0/huggingface"
    os.makedirs(HF_ROOT, exist_ok=True)

    os.environ["HF_HOME"] = HF_ROOT
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # ===== 2. 自动下载到 HF 缓存，不再手动指定 local_dir =====
    print("开始下载数据集到 Hugging Face 缓存...")
    cached_repo_path = snapshot_download(
        repo_id="lmms-lab/LLaVA-NeXT-Data",
        repo_type="dataset",
        max_workers=16,
    )
    print(f"数据集仓库缓存完成: {cached_repo_path}")

    # ===== 3. 直接按 repo 名加载；datasets 会复用缓存 =====
    dm = MultiModalDataModule(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        train_datasets=[
            {
                "path": "lmms-lab/LLaVA-NeXT-Data",
                "split": "train",
                "weight": 1.0,
                "streaming": False,
            }
        ],
        batch_size=4,
        num_workers=4,
        streaming=False,
        cache_dir=HF_ROOT,
    )

    print("开始 setup ...")
    dm.setup("fit")
    print("setup 完成")

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    print("batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)
    print("sample_ids:", batch["sample_ids"][:2])