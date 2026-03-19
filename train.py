import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from data.datamodule_ov import MultiModalDataModule
from learner.llava_ov import LlavaSFTModule


def main():
    L.seed_everything(42, workers=True)
    model_name = "/ppio_net0/code/mllm-lightning/mllm/llava-onevision-qwen2-0.5b-ov-hf"
    output_dir = "./outputs/llava_onevision_05b_zero3_sft_1epoch"

    dm = MultiModalDataModule(
        model_name_or_path=model_name,
        train_datasets=[
            {
                "path": "lmms-lab/LLaVA-NeXT-Data",
                "split": "train",
                "weight": 1.0,
                "streaming": False,
            }
        ],
        batch_size=1,
        num_workers=2,
        max_length=2048,
        streaming=True,
    )

    model_dtype = torch.bfloat16

    lit_model = LlavaSFTModule(
        model_name_or_path=model_name,
        lr=1e-5,
        weight_decay=0.01,
        trust_remote_code=True,
        use_gradient_checkpointing=True,
        torch_dtype=model_dtype,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="epoch{epoch:02d}",
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        monitor=None,
    )

    strategy = DeepSpeedStrategy(
        config={
            "zero_optimization": {
                "stage": 3,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8,
            },
            "gradient_clipping": 1.0,
        }
    )

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices="auto",
        max_steps=100,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        logger=WandbLogger(
            name=f"llava_onevision_05b_zero3_sft",
            project="mllm-lightning",
            log_model=False,
        ),
        default_root_dir=output_dir,
    )

    trainer.fit(lit_model, datamodule=dm)

    hf_save_dir = os.path.join(output_dir, "hf_model")
    os.makedirs(hf_save_dir, exist_ok=True)

    trainer.strategy.barrier()
    if trainer.is_global_zero:
        lit_model.model.save_pretrained(hf_save_dir)
        dm.processor.save_pretrained(hf_save_dir)
    trainer.strategy.barrier()

    print(f"训练完成，HF 权重已保存到: {hf_save_dir}")
    print(f"Lightning checkpoint: {ckpt_callback.last_model_path}")


if __name__ == "__main__":
    main()
