from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        revision="4bdd502bca7abd1ea57ee12fba0b0f23052958cc",
        cache_dir="./models",
        local_files_only=True,
    ).to("cuda")

    pipeline(prompt="")

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]
