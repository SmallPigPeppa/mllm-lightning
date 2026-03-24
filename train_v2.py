import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy,DDPStrategy

# from data.datamodule_ov_v4 import MultiModalDataModule
from data.datamodule_llava_ov_779k import MultiModalDataModule
from learner.llava_ov_v2 import LlavaSFTModule


def main():
    L.seed_everything(42, workers=True)
    model_name = "/ppio_net0/code/mllm-lightning/mllm/llava_onevision_qwen2_0_5b_ov_hf"
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
        num_workers=0,
        max_length=2048,
    )

    lit_model = LlavaSFTModule(
        model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        optim_args={
            "lr": 2e-5,
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        lora_args = {
            "enabled": True,
            "r": 128,
            "alpha": 256,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "bias": "none",
        }

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

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        max_steps=2000,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
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

    print(f"HF weights saved to: {hf_save_dir}")
    print(f"Lightning checkpoint saved to: {ckpt_callback.last_model_path}")


if __name__ == "__main__":
    main()
