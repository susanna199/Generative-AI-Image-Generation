# your_project_name/scripts/prepare_dataset.py

import os
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from collections import defaultdict

# --- Add the 'src' directory to the Python path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ----------------------------------------------------

from src import config

def main():
    print("--- Starting Dataset Preparation ---")
    config.PROCESSED_DATA_FOR_LORA.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load Captioning Model ---
    print(f"Loading captioning model: {config.CAPTIONING_MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(config.CAPTIONING_MODEL_ID)
    captioning_model = BlipForConditionalGeneration.from_pretrained(
        config.CAPTIONING_MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    print(f"BLIP model loaded to '{device}'.")

    # --- Step 2: Select Files from Target Classes ---
    print(f"Selecting up to {config.IMAGES_PER_CLASS} images for {len(config.TARGET_CLASSES)} target classes...")
    train_list_path = config.RAW_FOOD101_ROOT / "meta" / "train.txt"
    with open(train_list_path, 'r') as f:
        all_train_files = [line.strip() for line in f.readlines()]

    # Group all available training files by their class
    files_by_class = defaultdict(list)
    for file_path in all_train_files:
        class_name = file_path.split('/')[0]
        if class_name in config.TARGET_CLASSES:
            files_by_class[class_name].append(file_path)

    # Create the final list by taking the desired number from each class
    final_files_to_process = []
    for class_name in config.TARGET_CLASSES:
        # Take up to IMAGES_PER_CLASS images, or fewer if not that many are available
        files_for_class = files_by_class[class_name][:config.IMAGES_PER_CLASS]
        final_files_to_process.extend(files_for_class)
        print(f"  - Found {len(files_for_class)} images for class '{class_name}'")

    print(f"Total images to process: {len(final_files_to_process)}")

    # --- Step 3: Loop through, process, and save each image ---
    print("Processing images and generating captions...")
    for item_path in tqdm(final_files_to_process, desc="Processing Images"):
        try:
            class_name = item_path.split('/')[0].replace('_', ' ')
            image_path = config.RAW_FOOD101_ROOT / "images" / f"{item_path}.jpg"
            image = Image.open(image_path).convert("RGB").resize(config.IMAGE_RESOLUTION)

            # Generate caption
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            generated_ids = captioning_model.generate(**inputs, max_new_tokens=50)
            base_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            final_caption = f"{config.TRIGGER_WORD}, a professional photo of {class_name}, {base_caption}"

            # Save files
            output_filename_base = item_path.replace('/', '_') # e.g., 'sushi_12345'
            image.save(config.PROCESSED_DATA_FOR_LORA.joinpath(f"{output_filename_base}.png"))
            with open(config.PROCESSED_DATA_FOR_LORA.joinpath(f"{output_filename_base}.txt"), 'w') as f:
                f.write(final_caption)

        except Exception as e:
            print(f"Skipping file {item_path} due to error: {e}")

    print("\n✅ Dataset preparation complete!")

if __name__ == '__main__':
    main()