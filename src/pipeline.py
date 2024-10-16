import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-20",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")

    pipeline(prompt="")

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    if request.seed is None:
        generator = None
    else:
        generator = Generator(pipeline.device).manual_seed(request.seed)

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=20,
    ).images[0]
