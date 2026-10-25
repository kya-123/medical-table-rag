import argparse
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TARGET = BASE / "model" / "models"

REPOS = [
    ("Xorbits/bge-large-zh-v1.5", "Xorbits--bge-large-zh-v1.5"),
    ("BAAI/bge-reranker-base", "BAAI--bge-reranker-base"),
]


ALLOW = ["*.json", "*.txt", "*.safetensors", "*.model", "*.py", "1_Pooling/*"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="https://hf-mirror.com",
                    help="HF endpoint（默认国内镜像 hf-mirror.com）")
    args = ap.parse_args()


    os.environ.setdefault("HF_ENDPOINT", args.mirror)
    os.environ.pop("HF_HUB_OFFLINE", None)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺 huggingface_hub，先装依赖：pip install -r requirements.txt")
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)
    for repo_id, folder in REPOS:
        dest = TARGET / folder / "snapshots" / "master"
        if (dest / "config.json").exists():
            print(f"[skip] 已存在：{folder}")
            continue
        print(f"[get ] {repo_id}  →  {dest}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            allow_patterns=ALLOW,
            endpoint=args.mirror,
        )
        print(f"[ ok ] {folder}")

    print("\n完成。启动服务：python rag_system.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
