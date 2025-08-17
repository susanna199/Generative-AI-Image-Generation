# your_project_name/src/config.py

from pathlib import Path

# ... (Keep PROJECT_ROOT and other paths the same) ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_FOOD101_ROOT = RAW_DATA_DIR / "food-101" # Make sure this path is correct!
PROCESSED_DATA_FOR_LORA = PROCESSED_DATA_DIR / "food_for_lora"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "generated_images"

# --- Model & Tokenizer IDs ---
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CAPTIONING_MODEL_ID = "Salesforce/blip-image-captioning-large"

# --- Data Preparation Parameters ---
TRIGGER_WORD = "pro_food_photo"
IMAGE_RESOLUTION = (512, 512)

# !! NEW !!: Define the specific classes and image counts you want to use.
# Make sure the class names match the folder names in your raw data exactly.
TARGET_CLASSES = [
    'sushi',
    'pizza',
    'waffles',
    'donuts',
    'ramen'
]
IMAGES_PER_CLASS = 1000 # The script will take UP TO this many images per class.

# We no longer need the old LIMIT variable.
# LIMIT = 100 # <-- DELETE OR COMMENT OUT THIS LINE

# --- Fine-Tuning Parameters ---
TRAIN_BATCH_SIZE = 1
# !! UPDATED !!: Adjust steps for the new dataset size (5 classes * 1000 images = 5000 images)
# A good starting point is 10-20 repetitions per image. Let's start with ~3-4.
# 5000 images * 3 reps = 15000 steps
MAX_TRAIN_STEPS = 15000
LEARNING_RATE = 1e-4
CHECKPOINTING_STEPS = 1000 # Save a checkpoint every 1000 steps