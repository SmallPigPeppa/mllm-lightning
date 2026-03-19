from huggingface_hub import snapshot_download

WORKERS = 16

if __name__ == "__main__":
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        local_dir="../download/Qwen3-VL-2B-Instruct",
        max_workers=WORKERS,
    )

    snapshot_download(
        repo_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        local_dir="../download/llava-onevision-qwen2-0.5b-ov-hf",
        max_workers=WORKERS,
    )
