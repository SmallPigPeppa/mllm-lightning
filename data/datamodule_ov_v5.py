from typing import List
import lightning as L
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
# from transformers import AutoProcessor
from mllm.llava_onevision_qwen2_0_5b_ov_hf.custom_models.processing_llava_onevision import LlavaOnevisionProcessor

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}

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
        texts, image_batches, loss_masks, sample_ids = [], [], [], []

        image_token = getattr(self.processor, "image_token", "<image>")
        video_token = getattr(self.processor, "video_token", "<video>")
        max_length = self.hparams.max_length
        ignore_index = self.hparams.ignore_index

        for sample in batch:
            conversations = sample.get("conversations") or []
            # sample 级图片池：当 turn 没显式给图时，按 <image> 出现顺序顺延取图
            pooled_images = sample.get("images", sample.get("image"))
            if pooled_images is None:
                pooled_images = []
            elif not isinstance(pooled_images, list):
                pooled_images = [pooled_images]
            pooled_idx = 0

            messages, sample_images = [], []
            has_assistant = False

            for turn in conversations:
                role = ROLE_MAP.get(turn.get("from"))
                if role is None:
                    continue

                # turn 级图片：支持 turn["images"] / turn["image"]
                turn_images = turn.get("images", turn.get("image"))
                if turn_images is None:
                    turn_images = []
                elif not isinstance(turn_images, list):
                    turn_images = [turn_images]

                raw_text = (turn.get("value") or "").replace(video_token, "")
                need_images = raw_text.count(image_token)
                raw_text = raw_text.replace(image_token, "").strip()

                # turn 没带图时，从 sample 级图片池按占位符数量补图
                if need_images > 0 and not turn_images and pooled_idx < len(pooled_images):
                    turn_images = pooled_images[pooled_idx: pooled_idx + need_images]
                    pooled_idx += len(turn_images)

                content = [{"type": "image"} for _ in turn_images]
                if raw_text:
                    content.append({"type": "text", "text": raw_text})
                if not content:
                    continue

                messages.append({"role": role, "content": content})
                sample_images.extend(img.convert("RGB") for img in turn_images)
                has_assistant |= (role == "assistant")

            if not messages or not has_assistant:
                continue

            # 渲染完整对话
            full_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # 先对单样本编码，用于构造 assistant-only loss mask
            one_inputs = self.processor(
                text=[full_text],
                images=[sample_images] if sample_images else None,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            one_ids = one_inputs["input_ids"][0]
            loss_mask = torch.zeros_like(one_ids, dtype=torch.bool)

            # 用“前缀长度差”定位每一轮 assistant 在 token 序列中的区间
            prefix_messages = []
            prefix_image_count = 0
            prev_len = 0

            for msg in messages:
                prefix_messages.append(msg)
                prefix_image_count += sum(x["type"] == "image" for x in msg["content"])

                prefix_text = self.processor.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prefix_ids = self.processor(
                    text=[prefix_text],
                    images=[sample_images[:prefix_image_count]] if prefix_image_count else None,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )["input_ids"][0]

                cur_len = prefix_ids.size(0)
                if msg["role"] == "assistant":
                    loss_mask[prev_len:cur_len] = True

                prev_len = cur_len
                if prev_len >= one_ids.numel():
                    break

            texts.append(full_text)
            image_batches.append(sample_images)
            loss_masks.append(loss_mask)
            sample_ids.append(sample.get("id"))

        if not texts:
            raise RuntimeError("No valid samples found in batch.")

        # 批量编码；支持每个样本对应多张图
        model_inputs = self.processor(
            text=texts,
            images=image_batches if any(len(x) > 0 for x in image_batches) else None,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # 只保留 assistant token 的 label，其余全部置为 ignore_index
        labels = torch.full_like(model_inputs["input_ids"], ignore_index)
        for i, mask in enumerate(loss_masks):
            n = min(mask.numel(), labels.size(1))
            labels[i, :n][mask[:n]] = model_inputs["input_ids"][i, :n][mask[:n]]

        # padding 不参与 loss
        labels.masked_fill_(model_inputs["attention_mask"] == 0, ignore_index)

        # 多模态占位 token 不参与 loss
        for token_name in ("image_token_id", "video_token_id"):
            token_id = getattr(self.processor, token_name, None)
            if token_id is not None:
                labels.masked_fill_(model_inputs["input_ids"] == token_id, ignore_index)

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
