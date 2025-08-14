# AI Food Stylist: Fine-Tuning Stable Diffusion for Professional Food Photography

This repository contains the complete pipeline for fine-tuning a pre-trained Stable Diffusion model to generate high-quality, stylized images of food. This project addresses the challenge of adapting a general-purpose model for a specialized domain by leveraging an automated data preparation workflow and the efficient LoRA (Low-Rank Adaptation) fine-tuning technique.

The final model is capable of generating professional-grade food photography from simple text prompts, guided by the style learned from the Food-101 dataset.

## The Challenge

The standard **Food-101 dataset** is designed for classification tasks, providing only simple class labels (e.g., `"sushi"`). For a generative model to learn the nuances of style, texture, and composition, it requires rich, descriptive captions. Manually creating these captions for the 75,000+ images in the training set is infeasible.

This project solves that problem by creating an automated data curation pipeline.

## The Pipeline

The project is broken down into two main stages:

**1. Automated Data Curation**
The `scripts/prepare_dataset.py` script uses the powerful **Salesforce BLIP** vision-language model to analyze each image from the Food-101 dataset. It generates a detailed text caption describing the image's content. This process programmatically transforms the classification dataset into a high-quality dataset perfectly formatted for text-to-image training.

**2. LoRA Fine-Tuning**
The `scripts/train.py` script orchestrates the fine-tuning process. It uses the official, highly-optimized script from Hugging Face `diffusers` to train a small LoRA adapter on top of the frozen **`runwayml/stable-diffusion-v1-5`** base model. This efficiently teaches the model the specific "professional food photography" style (`pro_food_photo`) from our curated dataset.

## 🍔 Sample Generations

*Prompt: `pro_food_photo, a vibrant bowl of ramen with a soft-boiled egg, chashu pork, and nori, steam rising, detailed, appetizing`*


*Prompt: `pro_food_photo, a stack of fluffy pancakes with melted butter, dripping maple syrup, and fresh blueberries`*


## Key Technologies

* **Base Model:** `runwayml/stable-diffusion-v1-5`
* **Captioning Model:** `Salesforce/blip-image-captioning-large`
* **Core Frameworks:** PyTorch, Hugging Face `Diffusers`, `Transformers`, `Accelerate`
* **Dataset:** Food-101

---

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-project-name>
    ```

2.  **Create a Python Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download and Place the Dataset**
    * Download the Food-101 dataset.
    * Unzip it and place the entire `food-101` folder inside the `data/raw/` directory. The final path should be `data/raw/food-101/`.

5.  **Configure the Project**
    * Open `src/config.py`. While the paths are relative, ensure the folder structure matches.
    * Adjust parameters like `LIMIT` (for testing `prepare_dataset.py`) and `MAX_TRAIN_STEPS` as needed.

---

## How to Run the Workflow

Execute the scripts from the project's root directory.

**Step 1: Prepare the Dataset**
This script will process the raw images and generate captions. For a quick test, set the `LIMIT` in `config.py` to a small number (e.g., 100). For the full run, set `LIMIT = None`.
```bash
python scripts/prepare_dataset.py
