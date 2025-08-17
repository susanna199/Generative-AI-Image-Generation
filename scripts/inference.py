# your_project_name/scripts/inference.py

import torch
from diffusers import StableDiffusionPipeline
import sys
from pathlib import Path

# --- Add the 'src' directory to the Python path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ----------------------------------------------------

from src import config

def generate_image(prompt: str, output_path: Path):
    """
    Generates an image based on a prompt using the fine-tuned LoRA model.
    """
    print("--- Starting Image Generation ---")
    
    # --- 1. Load the Base Model ---
    print(f"Loading base model: {config.BASE_MODEL_ID}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.BASE_MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")

    # --- 2. Load and Attach the LoRA Weights ---
    lora_model_path = config.MODEL_OUTPUT_DIR / "pytorch_lora_weights.safetensors"
    print(f"Loading LoRA weights from: {lora_model_path}...")
    pipe.load_lora_weights(lora_model_path)
    print("LoRA weights attached successfully.")

    # --- 3. Generate the Image ---
    print(f"Generating image for prompt: '{prompt}'...")
    # It's important to use the trigger word from your training
    full_prompt = f"{config.TRIGGER_WORD}, {prompt}"
    
    # For better results, a negative prompt can be useful
    negative_prompt = "blurry, low quality, cartoon, 3d, ugly, deformed"
    
    image = pipe(
        full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # --- 4. Save the Image ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"✅ Image saved successfully to: {output_path}")
    return image

if __name__ == '__main__':
    # This is an example of how to run the function.
    # The prompt you want to test with.
    test_prompt = "a top-down shot of a vibrant bowl of ramen with a soft-boiled egg, chashu pork, and nori, steam rising, detailed, appetizing"
    
    # Where to save the output image.
    output_file_path = config.IMAGE_OUTPUT_DIR / "test_ramen.png"
    
    generate_image(prompt=test_prompt, output_path=output_file_path)