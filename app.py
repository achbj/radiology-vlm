import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch
import warnings
from unittest.mock import patch

# Ignore warning
warnings.filterwarnings("ignore")

base_model_id = "microsoft/Florence-2-base"
lora_path = "./florence-vqa-lora"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading Base Model on {device.upper()}...")

# Bypass transformers import checker using patch context manager
with patch("transformers.dynamic_module_utils.check_imports", return_value=[]):
    # Fix for transformers 5.x compatibility with Florence-2
    from transformers import PretrainedConfig
    if not hasattr(PretrainedConfig, 'forced_bos_token_id'):
        PretrainedConfig.forced_bos_token_id = None
        
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)

print("Merging LoRA Adapters...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.to(device)
model.eval()
print("Model loaded successfully!")

def predict(image, question):
    if image is None or not question.strip():
        return "⚠️ Please provide both an image and a question."
        
    prompt = f"<QA> {question}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            num_beams=3
        )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<QA>", image_size=(image.width, image.height))
    
    return parsed_answer.get('<QA>', generated_text)

# gradio
with gr.Blocks(theme=gr.themes.Soft(), title="Radiology VLM") as demo:
    gr.Markdown("# 🏥 Generative AI Radiology VLM")
    gr.Markdown("Ask free-form questions about Medical X-Rays using Microsoft's Florence-2 Vision-Language Model fine-tuned via LoRA.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload X-Ray Image")
            question_input = gr.Textbox(label="Question", placeholder="e.g. What abnormalities are seen in this image?")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="VLM Diagnosis / Answer", lines=5)
            
    submit_btn.click(fn=predict, inputs=[image_input, question_input], outputs=output_text)
    question_input.submit(fn=predict, inputs=[image_input, question_input], outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=10987)
