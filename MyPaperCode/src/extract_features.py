"""
Social Media Multimodal Knowledge Evolution Analysis System
Module: Multimodal Feature Extraction (LLaVA-1.5)
Reference: "Evolutionary Dynamics of AI Discourse", Section III.A

Description:
    Feeds image-text pairs into the LLaVA model and captures the
    hidden states from the vision-language projection layer (mm_projector).
    Output tensor shape: [1, 576, 5120] (before flattening).
"""

import argparse
import torch
import os
import pandas as pd
import requests
import logging
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# LLaVA Imports (Assumes llava package is installed)
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
except ImportError:
    raise ImportError("LLaVA package not found. Please install it following https://github.com/haotian-liu/LLaVA")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # 1. Initialize Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    logger.info(f"Loading model: {model_name}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        args.load_8bit, args.load_4bit, device=args.device
    )

    # Determine conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        logger.warning(f"Overriding conv_mode to {args.conv_mode}")
    else:
        args.conv_mode = conv_mode

    # 2. Load Dataset
    logger.info(f"Loading metadata from {args.input_file}")
    if args.input_file.endswith('.xlsx'):
        df = pd.read_excel(args.input_file)
    else:
        df = pd.read_csv(args.input_file)

    # Prepare Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Register Forward Hook to Capture Features
    fusion_features = {}

    def capture_hook(module, input, output):
        # Detach and move to CPU immediately to save VRAM
        fusion_features['hidden'] = output.detach().cpu()

    # Locate mm_projector (Projection Layer)
    if hasattr(model, 'mm_projector'):
        target_layer = model.mm_projector
    elif hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
        target_layer = model.model.mm_projector
    else:
        raise ValueError("Cannot find 'mm_projector' layer in the model.")

    hook_handle = target_layer.register_forward_hook(capture_hook)
    logger.info("Hook registered on mm_projector.")

    # 4. Processing Loop
    success_count = 0
    skip_count = 0

    # Column name configuration (Modify if your CSV headers change)
    COL_CONTENT = '微博内容' if '微博内容' in df.columns else 'content'
    COL_IMG_ID = '图片ID' if '图片ID' in df.columns else 'image_id'

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        try:
            # Data Validation
            content = str(row[COL_CONTENT]) if pd.notna(row[COL_CONTENT]) else ""
            image_id = str(row[COL_IMG_ID]) if pd.notna(row[COL_IMG_ID]) else ""

            if not content.strip() or not image_id.strip():
                skip_count += 1
                continue

            image_path = os.path.join(args.image_dir, image_id)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                skip_count += 1
                continue

            # Check if already processed
            base_name = os.path.splitext(image_id)[0]
            save_path = os.path.join(args.output_dir, f"{base_name}.pt")
            if os.path.exists(save_path) and not args.overwrite:
                continue

            # Preprocess Image
            image = load_image(image_path)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # Construct Prompt
            conv = conv_templates[args.conv_mode].copy()
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + content
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + content

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).to(model.device)

            # Inference (Forward Pass Only)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                )

            # Save Captured Features
            if 'hidden' in fusion_features:
                torch.save(fusion_features['hidden'], save_path)
                success_count += 1
                del fusion_features['hidden']  # Clear for next iter

            # Clean up VRAM
            del image_tensor
            del input_ids
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            skip_count += 1

    hook_handle.remove()
    logger.info(f"Extraction complete. Processed: {success_count}, Skipped: {skip_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Feature Extraction")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LLaVA model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input-file", type=str, required=True, help="Path to metadata CSV/XLSX")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="./output/features_raw",
                        help="Output directory for .pt files")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")

    args = parser.parse_args()
    main(args)