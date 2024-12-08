# utils/image_generation.py

import requests
from PIL import Image
from io import BytesIO
import os
import logging

def generate_image_from_prompt(prompt, height=512, width=512, num_inference_steps=50):
    """
    Generates an image from a text prompt using Hugging Face Inference API.
    """
    try:
        logging.info(f"Generating image with prompt: {prompt}")
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": 7.5
            },
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
            headers=headers,
            json=payload,
        )
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data)).convert("RGB")
            logging.info("Image generated successfully via API.")
            return image
        else:
            error_message = f"API request failed with status {response.status_code}: {response.text}"
            logging.error(error_message)
            raise Exception(error_message)
    except Exception as e:
        logging.error(f"Image Generation Error: {e}")
        raise e
