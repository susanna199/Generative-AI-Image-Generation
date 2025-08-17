# your_project_name/scripts/train.py

import os
import sys
from pathlib import Path
import re

# --- Add the 'src' directory to the Python path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ----------------------------------------------------

from src import config

def download_training_script():
    # ... (This function is the same as before) ...
    script_path = project_root / "scripts" / "train_dreambooth_lora.py"
    if not script_path.exists():
        print("Downloading official training script...")
        import requests
        url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py"
        response = requests.get(url)
        response.raise_for_status()
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Training script downloaded successfully.")
    else:
        print("Official training script already exists.")

def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Finds the path to the latest checkpoint folder."""
    if not output_dir.exists():
        return None
    
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
        
    # Find the checkpoint with the highest step number
    latest_checkpoint = max(checkpoints, key=lambda d: int(re.search(r"(\d+)", d.name).group(1)))
    return latest_checkpoint

def main():
    download_training_script()
    config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- NEW: Automatically find the latest checkpoint ---
    resume_checkpoint_path = find_latest_checkpoint(config.MODEL_OUTPUT_DIR)
    resume_argument = ""
    if resume_checkpoint_path:
        resume_argument = f'--resume_from_checkpoint="{resume_checkpoint_path}"'
        print(f"✅ Found latest checkpoint: {resume_checkpoint_path}. Resuming training.")
    else:
        print("ℹ️ No checkpoint found. Starting a fresh training run.")
    # --- END NEW ---

    command = f"""
    accelerate launch scripts/train_dreambooth_lora.py \\
      --pretrained_model_name_or_path="{config.BASE_MODEL_ID}" \\
      --instance_data_dir="{config.PROCESSED_DATA_FOR_LORA}" \\
      --output_dir="{config.MODEL_OUTPUT_DIR}" \\
      --instance_prompt="a photo of {config.TRIGGER_WORD}" \\
      --resolution={config.IMAGE_RESOLUTION[0]} \\
      --train_batch_size={config.TRAIN_BATCH_SIZE} \\
      --gradient_accumulation_steps=1 \\
      --checkpointing_steps={config.CHECKPOINTING_STEPS} \\
      --learning_rate={config.LEARNING_RATE} \\
      --lr_scheduler="constant" \\
      --lr_warmup_steps=0 \\
      --max_train_steps={config.MAX_TRAIN_STEPS} \\
      --seed="42" \\
      {resume_argument}
    """

    print("\n" + "="*80)
    print("STARTING FINE-TUNING PROCESS")
    # ... (rest of the script is the same) ...
    os.system(command.strip().replace("\\\n", " "))
    print("\n✅ Fine-tuning complete!")

if __name__ == '__main__':
    main()