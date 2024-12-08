# app_gradio.py

import gradio as gr
from utils.text_generation import generate_text_response
from utils.summarize_youtube import summarize_youtube_video
from utils.image_generation import generate_image_from_prompt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def text_query(text):
    try:
        summary = generate_text_response(text)
        logging.info(f"Text Query Success: {text}")
        return summary
    except Exception as e:
        logging.error(f"Text Query Error: {e}")
        return f"Error: {e}"

def youtube_summarize(url):
    try:
        summary = summarize_youtube_video(url)
        logging.info(f"YouTube Summarization Success for URL: {url}")
        return summary
    except Exception as e:
        logging.error(f"YouTube Summarization Error for URL {url}: {e}")
        return f"Error: {e}"

def generate_image(prompt):
    try:
        image = generate_image_from_prompt(prompt)
        logging.info(f"Image Generation Success for prompt: {prompt}")
        return image
    except Exception as e:
        logging.error(f"Image Generation Error for prompt '{prompt}': {e}")
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Multimodal AI Application")
    
    with gr.Tab("Text Query"):
        text_input = gr.Textbox(label="Enter Text", lines=4)
        text_output = gr.Textbox(label="Summary")
        summarize_btn = gr.Button("Summarize")
        summarize_btn.click(text_query, inputs=text_input, outputs=text_output)
    
    with gr.Tab("YouTube Summarization"):
        youtube_input = gr.Textbox(label="Enter YouTube URL")
        youtube_output = gr.Textbox(label="Summary")
        youtube_btn = gr.Button("Summarize Video")
        youtube_btn.click(youtube_summarize, inputs=youtube_input, outputs=youtube_output)
    
    with gr.Tab("Image Generation"):
        prompt_input = gr.Textbox(label="Enter Image Prompt")
        image_output = gr.Image(label="Generated Image")
        generate_btn = gr.Button("Generate Image")
        generate_btn.click(generate_image, inputs=prompt_input, outputs=image_output)

demo.launch()
