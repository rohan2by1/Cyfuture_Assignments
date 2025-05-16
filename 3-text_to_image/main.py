import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Load the Stable Diffusion model (may require logging in to Hugging Face)
model_id = "runwayml/stable-diffusion-v1-5"  # You can pick other models as well

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,  # float32 is safer for CPU
).to("cpu")                     # Use .to("cpu") if not on GPU

def generate_image(
    prompt, 
    num_inference_steps=30, 
    guidance_scale=7.5, 
    seed=42, 
    height=512, 
    width=512
):
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return image

# Define the Gradio UI
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", value="A fantasy landscape with castles"),
        gr.Slider(10, 100, step=1, value=30, label="Inference Steps"),
        gr.Slider(1, 20, step=0.1, value=7.5, label="Guidance Scale"),
        gr.Number(value=42, label="Random Seed"),
        gr.Slider(256, 1024, step=64, value=512, label="Height (px)"),
        gr.Slider(256, 1024, step=64, value=512, label="Width (px)"),
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Playground",
    description="Generate images from text using Stable Diffusion."
)

iface.launch()