from pathlib import Path
from huggingface_hub import snapshot_download

WORKERS = 16
BASE_DIR = Path("/ppio_net0/code/download")


def download_model(repo_id: str, base_dir: Path = BASE_DIR) -> str:
    # Save to: <base_dir>/<repo_name>
    local_dir = base_dir / repo_id.rsplit("/", 1)[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        max_workers=WORKERS,
    )


if __name__ == "__main__":
    for repo_id in [
        # "Qwen/Qwen3-VL-2B-Instruct",
        # "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "llava-hf/llava-1.5-7b-hf",
    ]:
        download_model(repo_id)