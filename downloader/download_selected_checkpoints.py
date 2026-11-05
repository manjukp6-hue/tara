import os
import sys
from huggingface_hub import snapshot_download

def main():
    token = os.environ.get('HF_TOKEN')
    token_file = os.path.expanduser('~/.cache/huggingface/token')
    if not token and os.path.exists(token_file):
        with open(token_file) as f:
            token = f.read().strip()

    patterns = [
        'tara_checkpoints/normal_checkpoint/step_7700/*'
    ]

    print("[Checkpoints Download] Starting download for step_7700 from manjukp6/tara...", flush=True)
    snapshot_download(
        repo_id='manjukp6/tara',
        repo_type='model',
        local_dir='.',
        allow_patterns=patterns,
        token=token,
        max_workers=4
    )
    print("[Checkpoints Download] Successfully downloaded all files!", flush=True)

if __name__ == '__main__':
    main()
