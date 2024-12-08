# utils/text_generation.py

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import logging

print("Initializing text generation model...")  # Debugging

try:
    model_name = "EleutherAI/gpt-neo-125M"  # Consider upgrading to "EleutherAI/gpt-neo-1.3B" for better performance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer eos_token_id before: {tokenizer.eos_token_id}")  # Debugging

    # Ensure eos_token_id is set
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 50256
        print(f"Set eos_token_id to 50256")  # Debugging
    else:
        print(f"Tokenizer eos_token_id after: {tokenizer.eos_token_id}")  # Debugging

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")  # Debugging
    else:
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")  # Debugging

    # Set model_max_length to guide the tokenizer
    tokenizer.model_max_length = 512

    model = AutoModelForCausalLM.from_pretrained(model_name)
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # Use -1 for CPU; change to 0 if using GPU
    )
    print("Text generation model initialized successfully.")  # Debugging

except Exception as e:
    print(f"Error initializing text generation model: {e}")  # Debugging
    logging.error(f"Text Generation Initialization Error: {e}")
    raise e

def generate_text_response(prompt, max_new_tokens=150):
    """
    Generates a text response using the text generation model.
    """
    try:
        print("Generating text response...")  # Debugging
        generated = text_generator(
            prompt,
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,
            do_sample=True,             # Enable sampling
            temperature=0.7,            # Balance between determinism and creativity
            top_p=0.9,                  # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256,
            repetition_penalty=1.3,     # Discourages repetition
            no_repeat_ngram_size=3      # Prevents repeating trigrams
            # early_stopping=True        # Removed to align with num_beams=1
        )
        response = generated[0]['generated_text'].strip()
        print("Text response generated successfully.")  # Debugging
        return response
    except Exception as e:
        print(f"Error generating text response: {e}")  # Debugging
        logging.error(f"Text Generation Error: {e}")
        raise e

def summarize_text(text, max_new_tokens=150):
    """
    Summarizes the given text using the text generation model.
    """
    try:
        print("Generating summary for the text...")  # Debugging
        prompt = f"Summarize the following text in a concise paragraph:\n\n{text}\n\nSummary:"
        summary = generate_text_response(prompt, max_new_tokens=max_new_tokens)
        print("Summary generated successfully.")  # Debugging
        # Remove the prompt from the generated text
        summary = summary.split("Summary:")[-1].strip()
        return summary
    except Exception as e:
        print(f"Error summarizing text: {e}")  # Debugging
        logging.error(f"Text Summarization Error: {e}")
        raise e
