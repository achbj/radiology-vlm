import modal
from fastapi import Request

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "transformers==4.42.4",
    "peft==0.11.1",
    "einops",
    "timm",
    "Pillow",
    "fastapi",
    "python-multipart"
)

volume = modal.Volume.from_name("vlm-model-vol")
app = modal.App("radiology-vlm-api")

@app.cls(image=image, volumes={"/vol/model": volume})
class VLMClassifier:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel
        import os
        import torch
        
        base_model_id = "microsoft/Florence-2-base"
        lora_path = "/vol/model/florence-vqa-lora"
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"Model not found at {lora_path}. Did you run train_vlm.py?")
            
        print("Loading base model and merging LoRA weights...")
        
        # Bypass the transformers import checker using unittest.mock.patch to guarantee it intercepts the call
        from unittest.mock import patch
        with patch("transformers.dynamic_module_utils.check_imports", return_value=[]):
            self.processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
        
        # Load the fine-tuned LoRA weights on top of the base model
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()

    @modal.web_endpoint(method="POST")
    async def predict(self, request: Request):
        from PIL import Image
        import torch
        import io
        
        form = await request.form()
        file = form.get("file")
        question = form.get("question")
        
        if not file or not question:
            return {"error": "Both 'file' and 'question' are required in form data"}
            
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Format the prompt exactly as we trained it
        prompt = f"<QA> {question}"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Generate the answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=3
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Florence-2 post-processor parses the special tags automatically
        parsed_answer = self.processor.post_process_generation(generated_text, task="<QA>", image_size=(image.width, image.height))
        
        return {
            "question": question,
            "answer": parsed_answer.get('<QA>', generated_text),
            "status": "success"
        }
