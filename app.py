import os
import io
import gc
import uuid
import json
import base64
import random
import subprocess
from pathlib import Path
from typing import List, Optional

import spaces
import numpy as np
import torch
from PIL import Image

from gradio import Server
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

HF_TOKEN = os.environ.get("HF_TOKEN")

app = Server()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
EXAMPLES_DIR = BASE_DIR / "examples"

STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

ADAPTER = {
    "title": "Klein-Consistency",
    "adapter_name": "klein-consistency",
    "repo": "dx8152/Flux2-Klein-9B-Consistency",
    "weights": "Klein-consistency.safetensors",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("Using device:", device)


def apply_patch():
    import diffusers

    site_packages = os.path.dirname(diffusers.__file__)
    patch_file = os.path.join(os.path.dirname(__file__), "flux2_klein_kv.patch")
    if os.path.exists(patch_file):
        result = subprocess.run(
            ["patch", "-p2", "--forward", "--batch"],
            cwd=os.path.dirname(site_packages),
            stdin=open(patch_file),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Patch applied successfully")
        else:
            print(f"Patch output: {result.stdout}\n{result.stderr}")


apply_patch()

from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

print("Loading FLUX.2 Klein 9B KV model...")
pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=dtype,
    token=HF_TOKEN,
).to(device)
print("Base KV Model loaded successfully.")

print(f"Loading adapter: {ADAPTER['title']}")
pipe.load_lora_weights(
    ADAPTER["repo"],
    weight_name=ADAPTER["weights"],
    adapter_name=ADAPTER["adapter_name"],
)
pipe.set_adapters([ADAPTER["adapter_name"]], adapter_weights=[1.0])
print(f"Adapter loaded successfully: {ADAPTER['adapter_name']}")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(img: Image.Image, prefix: str = "output") -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    path = OUTPUT_DIR / filename
    img.save(path, format="PNG")
    return filename


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024

    try:
        if isinstance(image, list) and len(image) > 0:
            first = image[0]
        else:
            first = image

        if isinstance(first, (tuple, list)):
            path_or_img = first[0]
        else:
            path_or_img = first

        if isinstance(path_or_img, str):
            img = Image.open(path_or_img).convert("RGB")
        elif isinstance(path_or_img, Image.Image):
            img = path_or_img.convert("RGB")
        else:
            img = Image.open(path_or_img.name).convert("RGB")

        original_width, original_height = img.size

        if original_width > original_height:
            new_width = 1024
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
        else:
            new_height = 1024
            aspect_ratio = original_width / original_height
            new_width = int(new_height * aspect_ratio)

        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        new_width = max(256, min(1024, new_width))
        new_height = max(256, min(1024, new_height))

        return new_width, new_height
    except Exception:
        return 1024, 1024


def process_gallery_images(images):
    if not images:
        return []

    pil_images = []
    for item in images:
        try:
            if isinstance(item, (tuple, list)):
                path_or_img = item[0]
            else:
                path_or_img = item

            if isinstance(path_or_img, str):
                pil_images.append(Image.open(path_or_img).convert("RGB"))
            elif isinstance(path_or_img, Image.Image):
                pil_images.append(path_or_img.convert("RGB"))
            else:
                pil_images.append(Image.open(path_or_img.name).convert("RGB"))
        except Exception as e:
            print(f"Skipping invalid image item: {e}")
            continue

    return pil_images


@spaces.GPU
def infer(
    images,
    prompt,
    seed,
    randomize_seed,
    width,
    height,
    steps,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not prompt or not str(prompt).strip():
        raise ValueError("Please enter a prompt.")

    if isinstance(seed, str):
        seed = int(seed)
    if isinstance(randomize_seed, str):
        randomize_seed = randomize_seed.lower() == "true"
    if isinstance(width, str):
        width = int(width)
    if isinstance(height, str):
        height = int(height)
    if isinstance(steps, str):
        steps = int(steps)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    pil_images = process_gallery_images(images) if images else []

    if pil_images:
        width, height = update_dimensions_on_upload(pil_images[0])
        image_input = [
            img.resize((width, height), Image.LANCZOS).convert("RGB")
            for img in pil_images
        ]
    else:
        image_input = None
        width = max(256, min(MAX_IMAGE_SIZE, (int(width) // 8) * 8))
        height = max(256, min(MAX_IMAGE_SIZE, (int(height) // 8) * 8))

    try:
        generator = torch.Generator(device=device).manual_seed(seed)

        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "generator": generator,
        }

        if image_input is not None:
            pipe_kwargs["image"] = image_input

        result_image = pipe(**pipe_kwargs).images[0]
        return result_image, seed

    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_example_items():
    example_prompts = {
        "1.jpg": "Change the weather to stormy.",
        "2.jpg": "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition.",
        "3.jpg": "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent.",
        "4.jpg": "Make the texture high-resolution.",
    }

    items = []
    if EXAMPLES_DIR.exists():
        for name in sorted(os.listdir(EXAMPLES_DIR)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                items.append(
                    {
                        "file": name,
                        "url": f"/example-file/{name}",
                        "prompt": example_prompts.get(name, "Edit this image while preserving composition."),
                    }
                )
    return items


@app.api(name="hello")
def hello(name: str) -> str:
    return f"Hello, {name}!"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/example-file/{filename}")
async def example_file(filename: str):
    path = EXAMPLES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Example not found"}, status_code=404)
    return FileResponse(path)


@app.get("/download/{filename}")
async def download_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, filename=filename, media_type="image/png")


@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    seed: str = Form("0"),
    randomize_seed: str = Form("true"),
    width: str = Form("1024"),
    height: str = Form("1024"),
    steps: str = Form("4"),
    images: Optional[List[UploadFile]] = File(None),
):
    temp_paths = []
    try:
        image_paths = []

        if images:
            for upload in images:
                suffix = Path(upload.filename).suffix or ".png"
                temp_name = f"upload_{uuid.uuid4().hex}{suffix}"
                temp_path = OUTPUT_DIR / temp_name
                content = await upload.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_paths.append(str(temp_path))
                image_paths.append(str(temp_path))

        result_image, used_seed = infer(
            images=image_paths,
            prompt=prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            width=width,
            height=height,
            steps=steps,
        )

        output_filename = save_image(result_image, prefix="kv_edit")
        return JSONResponse(
            {
                "success": True,
                "seed": used_seed,
                "image_url": f"/download/{output_filename}",
                "download_url": f"/download/{output_filename}",
                "image_base64": image_to_base64(result_image),
                "adapter": ADAPTER["title"],
            }
        )

    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500,
        )
    finally:
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    examples = get_example_items()
    examples_json = json.dumps(examples)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KV-Edit-Consistency</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;900&display=swap');

    :root {{
      --bg-main: #ffd6e7;
      --bg-card: #ff66a3;
      --bg-header: #ffffff;
      --bg-result: #ffffff;
      --bg-dropdown-active: #ff66a3;
      --bg-advanced: rgba(255, 255, 255, 0.3);
      --bg-uploader: rgba(255, 255, 255, 0.4);
      --bg-button-primary: #4ade80;
      --bg-button-primary-hover: #1ac2ff;
      --bg-button-secondary: #fde047;
      --bg-button-secondary-hover: #f97316;

      --color-border: #000000;
      --color-text: #000000;
      --color-text-button: #000000;
    }}

    [data-theme="dark"] {{
      --bg-main: #2c132c;
      --bg-card: #592659;
      --bg-header: #1a1a1a;
      --bg-result: #2a2a2a;
      --bg-dropdown-active: #7f397f;
      --bg-advanced: rgba(0, 0, 0, 0.3);
      --bg-uploader: rgba(0, 0, 0, 0.4);
      --color-text: #f0f0f0;
      --color-text-button: #000000;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      font-family: 'Montserrat', sans-serif;
      margin: 0;
      padding: 0;
      color: var(--color-text);
      background-color: var(--bg-main);
      overflow-x: hidden;
      transition: background-color 0.3s ease;
    }}

    .app-container {{
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    .app-header {{
      background: var(--bg-header);
      padding: 2rem 4rem;
      border-bottom: 3px solid var(--color-border);
      color: var(--color-text);
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: background-color 0.3s ease, color 0.3s ease;
      gap: 1rem;
    }}

    .app-header h1 {{
      font-size: 2.5rem;
      font-weight: 900;
      margin: 0 0 0.5rem 0;
      line-height: 1;
    }}

    .app-header p {{
      font-size: 1rem;
      font-weight: 600;
      margin: 0;
    }}

    .app-main {{
      flex-grow: 1;
      background: var(--bg-main);
      padding: 2rem;
      transition: background-color 0.3s ease;
    }}

    .main-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 4rem 2rem;
    }}

    @media (min-width: 1024px) {{
      .main-grid {{
        grid-template-columns: 1fr 1fr;
        gap: 2rem 4rem;
      }}
    }}

    .card {{
      font-family: 'Montserrat', sans-serif;
      translate: -6px -6px;
      background: var(--bg-card);
      border: 3px solid var(--color-border);
      box-shadow: 12px 12px 0 var(--color-border);
      transition: all 0.2s ease;
      width: 100%;
    }}

    .card:hover {{
      translate: -3px -3px;
      box-shadow: 9px 9px 0 var(--color-border);
    }}

    .head {{
      font-size: 14px;
      font-weight: 900;
      width: 100%;
      background: var(--bg-header);
      padding: 12px 16px;
      color: var(--color-text);
      border-bottom: 3px solid var(--color-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .content {{
      padding: 1.5rem;
      font-size: 14px;
      font-weight: 600;
      color: var(--color-text);
    }}

    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      min-height: 52px;
      padding: 12px 20px;
      border: 3px solid var(--color-border);
      box-shadow: 4px 4px 0 var(--color-border);
      font-weight: 900;
      transition: all 0.15s ease;
      cursor: pointer;
      font-size: 14px;
      color: var(--color-text-button);
      font-family: 'Montserrat', sans-serif;
      text-decoration: none;
      background: transparent;
      line-height: 1;
    }}

    .button.primary {{
      background: var(--bg-button-primary);
    }}

    .button.primary:hover {{
      background: var(--bg-button-primary-hover);
    }}

    .button.secondary {{
      background: var(--bg-button-secondary);
    }}

    .button.secondary:hover {{
      background: var(--bg-button-secondary-hover);
    }}

    .theme-toggle {{
      padding: 8px;
      background: var(--bg-button-secondary);
      width: 52px;
      height: 52px;
      flex: 0 0 52px;
    }}

    .theme-toggle:hover {{
      background: var(--bg-button-secondary-hover);
    }}

    .button:hover {{
      translate: 2px 2px;
      box-shadow: 2px 2px 0 var(--color-border);
    }}

    .button:active {{
      translate: 4px 4px;
      box-shadow: 0 0 0 var(--color-border);
    }}

    .button:disabled {{
      background: #d1d5db;
      color: #6b7280;
      cursor: not-allowed;
      translate: 0 0;
      box-shadow: 4px 4px 0 var(--color-border);
    }}

    .form-content {{
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }}

    .form-group {{
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}

    .form-group label {{
      font-weight: 900;
    }}

    .form-actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      align-items: stretch;
      margin-top: 0.5rem;
    }}

    .form-actions .button {{
      width: 100%;
      min-width: 0;
    }}

    .input, textarea.input {{
      width: 100%;
      padding: 10px;
      border: 3px solid var(--color-border);
      background: var(--bg-result);
      color: var(--color-text);
      font-weight: 600;
      font-family: 'Montserrat', sans-serif;
      font-size: 14px;
      box-sizing: border-box;
      border-radius: 0;
      outline: none;
    }}

    textarea.input {{
      resize: vertical;
      min-height: 120px;
    }}

    .checkbox-row {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-weight: 700;
    }}

    .checkbox-row input {{
      width: 18px;
      height: 18px;
      accent-color: var(--bg-button-primary);
    }}

    .advanced-toggle {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      background: var(--bg-header);
      border: 3px solid var(--color-border);
      box-shadow: 4px 4px 0 var(--color-border);
      font-weight: 900;
      cursor: pointer;
      padding: 12px 14px;
      color: var(--color-text);
      font-family: 'Montserrat', sans-serif;
      font-size: 14px;
      text-transform: uppercase;
    }}

    .advanced-toggle:hover {{
      transform: translate(2px, 2px);
      box-shadow: 2px 2px 0 var(--color-border);
    }}

    .advanced-toggle .chevron {{
      transition: transform 0.2s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}

    .advanced-toggle.open .chevron {{
      transform: rotate(180deg);
    }}

    .advanced-panel {{
      background: var(--bg-advanced);
      padding: 1rem;
      border: 3px solid var(--color-border);
      display: none;
      flex-direction: column;
      gap: 1rem;
      margin-top: 1rem;
    }}

    .advanced-panel.open {{
      display: flex;
    }}

    .advanced-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}

    .uploader {{
      border: 3px dashed var(--color-border);
      background: var(--bg-uploader);
      padding: 1rem;
      cursor: pointer;
      min-height: 220px;
      transition: background-color 0.2s;
      color: var(--color-text);
    }}

    .uploader.dragover {{
      background: rgba(74, 222, 128, 0.5);
    }}

    .uploader input[type="file"] {{
      display: none;
    }}

    .uploader-placeholder {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 100%;
      min-height: 180px;
      gap: 0.85rem;
      font-weight: 900;
      background: none;
      border: none;
      cursor: pointer;
      color: inherit;
      text-align: center;
      padding: 1rem;
    }}

    .upload-badge {{
      width: 78px;
      height: 78px;
      border: 3px solid var(--color-border);
      background: var(--bg-button-secondary);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 6px 6px 0 var(--color-border);
    }}

    .uploader-placeholder svg {{
      width: 40px;
      height: 40px;
      display: block;
    }}

    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
      gap: 1rem;
    }}

    .image-grid .aspect-square {{
      aspect-ratio: 1/1;
      position: relative;
    }}

    .image-grid img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      border: 2px solid var(--color-border);
      display: block;
    }}

    .group {{
      position: relative;
    }}

    .image-remove-button {{
      position: absolute;
      top: 4px;
      right: 4px;
      background: rgba(0,0,0,0.7);
      color: white;
      border: none;
      border-radius: 99px;
      width: 28px;
      height: 28px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 900;
    }}

    .image-grid .group:hover .image-remove-button {{
      opacity: 1;
    }}

    .result-area {{
      width: 100%;
      height: 60vh;
      min-height: 400px;
      background: var(--bg-result);
      border: 3px solid var(--color-border);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
      position: relative;
      overflow: hidden;
    }}

    .result-area img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: none;
    }}

    .result-placeholder {{
      text-align: center;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      align-items: center;
      color: var(--color-text);
      font-weight: 700;
      padding: 1rem;
    }}

    .result-placeholder h3 {{
      font-weight: 900;
      margin: 0;
    }}

    .result-placeholder p {{
      margin: 0;
    }}

    .result-placeholder svg {{
      width: 72px;
      height: 72px;
    }}

    .result-icon-shell {{
      width: 88px;
      height: 88px;
      border: 3px solid var(--color-border);
      background: var(--bg-header);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 6px 6px 0 var(--color-border);
    }}

    .download-button {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(0,0,0,0.7);
      color: white;
      border-radius: 99px;
      padding: 8px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s;
      border: none;
      display: none;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      z-index: 2;
    }}

    .main-image-wrapper.group:hover .download-button {{
      opacity: 1;
    }}

    .loader-overlay {{
      position: absolute;
      inset: 0;
      background: rgba(0,0,0,0.45);
      display: none;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 1rem;
      z-index: 3;
    }}

    .spinner {{
      width: 52px;
      height: 52px;
      border: 4px solid rgba(255,255,255,0.3);
      border-top-color: #ffffff;
      border-radius: 999px;
      animation: spin 1s linear infinite;
    }}

    .loader-text {{
      color: #ffffff;
      font-weight: 900;
      letter-spacing: 0.03em;
    }}

    .meta-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin-top: 1rem;
      flex-wrap: wrap;
    }}

    .seed-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 8px 12px;
      background: var(--bg-header);
      border: 3px solid var(--color-border);
      box-shadow: 4px 4px 0 var(--color-border);
      font-weight: 900;
    }}

    .examples-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1rem;
    }}

    .example-card {{
      border: 3px solid var(--color-border);
      background: var(--bg-result);
      box-shadow: 4px 4px 0 var(--color-border);
      cursor: pointer;
      transition: all 0.15s ease;
      overflow: hidden;
    }}

    .example-card:hover {{
      transform: translate(-2px, -2px);
      box-shadow: 6px 6px 0 var(--color-border);
    }}

    .example-card img {{
      width: 100%;
      aspect-ratio: 1/1;
      object-fit: cover;
      display: block;
      border-bottom: 3px solid var(--color-border);
    }}

    .example-card .example-body {{
      padding: 0.75rem;
    }}

    .example-card .example-body p {{
      margin: 0;
      font-size: 12px;
      line-height: 1.5;
      font-weight: 700;
      color: var(--color-text);
    }}

    .toast-wrap {{
      position: fixed;
      top: 20px;
      right: 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      z-index: 9999;
    }}

    .toast {{
      min-width: 260px;
      max-width: 360px;
      background: var(--bg-header);
      color: var(--color-text);
      padding: 12px 14px;
      border: 3px solid var(--color-border);
      box-shadow: 6px 6px 0 var(--color-border);
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
    }}

    .toast button {{
      border: none;
      background: transparent;
      color: inherit;
      font-size: 18px;
      cursor: pointer;
      line-height: 1;
      padding: 0;
      font-weight: 900;
    }}

    .helper-text {{
      font-size: 13px;
      font-weight: 700;
      opacity: 0.9;
    }}

    .brand-block {{
      display: flex;
      flex-direction: column;
    }}

    @keyframes spin {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}

    @media (max-width: 1024px) {{
      .examples-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .app-header {{
        padding: 1.5rem 2rem;
      }}
    }}

    @media (max-width: 768px) {{
      .advanced-grid {{
        grid-template-columns: 1fr;
      }}

      .form-actions {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 640px) {{
      .app-header {{
        padding: 1.25rem 1rem;
        flex-direction: column;
        align-items: flex-start;
      }}

      .app-header h1 {{
        font-size: 2rem;
      }}

      .app-main {{
        padding: 1rem;
      }}

      .examples-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body data-theme="light">
  <div class="toast-wrap" id="toastWrap"></div>

  <div class="app-container">
    <header class="app-header">
      <div class="brand-block">
        <h1>KV-Edit-Consistency</h1>
        <p>Perform image edits with Consistency LoRA using a minimalist headless Gradio app.</p>
      </div>

      <button class="button secondary theme-toggle" id="themeToggle" aria-label="Toggle theme" title="Toggle dark mode / light mode">
        <span id="themeIcon"></span>
      </button>
    </header>

    <main class="app-main">
      <div class="main-grid">
        <section class="card">
          <div class="head">
            <span>Input</span>
          </div>

          <div class="content">
            <div class="form-content">
              <div class="form-group">
                <label>Upload Images</label>
                <div class="uploader" id="uploadZone">
                  <input id="fileInput" type="file" accept="image/*" multiple />
                  <button class="uploader-placeholder" id="uploadPlaceholder" type="button">
                    <div class="upload-badge">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square" stroke-linejoin="miter" aria-hidden="true">
                        <path d="M12 4V15"></path>
                        <path d="M8.5 7.5L12 4L15.5 7.5"></path>
                        <rect x="5" y="15.5" width="14" height="4.5"></rect>
                      </svg>
                    </div>
                    <span>Upload one or more images.</span>
                  </button>
                  <div class="image-grid" id="previewGrid" style="display:none;"></div>
                </div>
              </div>

              <div class="form-group">
                <label for="prompt">Edit Prompt</label>
                <textarea id="prompt" class="input" placeholder="e.g., Change the weather to stormy"></textarea>
              </div>

              <div class="form-group">
                <button class="advanced-toggle" id="advancedToggle" type="button">
                  <span>Advanced Options</span>
                  <span class="chevron" id="advancedChevron">
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="3">
                      <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                  </span>
                </button>

                <div class="advanced-panel" id="advancedPanel">
                  <div class="advanced-grid">
                    <div class="form-group">
                      <label for="seed">Seed</label>
                      <input id="seed" class="input" type="number" min="0" max="{MAX_SEED}" value="0" />
                    </div>

                    <div class="form-group">
                      <label for="steps">Inference Steps</label>
                      <input id="steps" class="input" type="number" min="1" max="20" value="4" />
                    </div>

                    <div class="form-group">
                      <label for="width">Width</label>
                      <input id="width" class="input" type="number" min="256" max="{MAX_IMAGE_SIZE}" step="8" value="1024" />
                    </div>

                    <div class="form-group">
                      <label for="height">Height</label>
                      <input id="height" class="input" type="number" min="256" max="{MAX_IMAGE_SIZE}" step="8" value="1024" />
                    </div>
                  </div>

                  <div class="checkbox-row">
                    <input id="randomizeSeed" type="checkbox" checked />
                    <label for="randomizeSeed" style="margin:0;">Randomize seed</label>
                  </div>
                </div>
              </div>

              <div class="form-actions">
                <button class="button primary" id="runBtn" type="button">Edit Image</button>
                <button class="button secondary" id="clearBtn" type="button">Clear</button>
              </div>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="head">
            <span>Output</span>
          </div>

          <div class="content">
            <div class="main-image-wrapper group">
              <div class="result-area" id="outputBox">
                <div class="result-placeholder" id="outputEmpty">
                  <div class="result-icon-shell">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square" stroke-linejoin="miter" aria-hidden="true">
                      <rect x="4.5" y="5" width="15" height="11"></rect>
                      <path d="M8 13l2.5-2.5L13 13"></path>
                      <path d="M13 13l2-2 2 2"></path>
                      <path d="M12 16V20"></path>
                      <path d="M8.5 16.5L12 20L15.5 16.5"></path>
                    </svg>
                  </div>
                  <span>Your edited image will appear here.</span>
                </div>

                <img id="outputImage" alt="Generated output" />

                <a id="downloadLink" class="download-button" download title="Download image">
                  <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="3">
                    <path d="M12 3v12"></path>
                    <path d="m7 10 5 5 5-5"></path>
                    <path d="M4 21h16"></path>
                  </svg>
                </a>

                <div class="loader-overlay" id="loaderOverlay">
                  <div class="spinner"></div>
                  <div class="loader-text">Processing Image</div>
                </div>
              </div>
            </div>

            <div class="meta-row">
              <div class="seed-pill">
                <span>SEED:</span>
                <strong id="usedSeed">-</strong>
              </div>
            </div>
          </div>
        </section>
      </div>

      <section class="card" style="margin-top:2rem;">
        <div class="head">
          <span>Examples</span>
        </div>
        <div class="content">
          <div class="examples-grid" id="examplesGrid"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const examples = {examples_json};

    const state = {{
      files: [],
      theme: localStorage.getItem("kv_theme") || "light",
      advancedOpen: false
    }};

    const body = document.body;
    const themeToggle = document.getElementById("themeToggle");
    const themeIcon = document.getElementById("themeIcon");

    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");
    const previewGrid = document.getElementById("previewGrid");

    const promptEl = document.getElementById("prompt");
    const seedEl = document.getElementById("seed");
    const stepsEl = document.getElementById("steps");
    const widthEl = document.getElementById("width");
    const heightEl = document.getElementById("height");
    const randomizeSeedEl = document.getElementById("randomizeSeed");

    const advancedToggle = document.getElementById("advancedToggle");
    const advancedPanel = document.getElementById("advancedPanel");

    const runBtn = document.getElementById("runBtn");
    const clearBtn = document.getElementById("clearBtn");

    const outputImage = document.getElementById("outputImage");
    const outputEmpty = document.getElementById("outputEmpty");
    const loaderOverlay = document.getElementById("loaderOverlay");
    const usedSeed = document.getElementById("usedSeed");
    const downloadLink = document.getElementById("downloadLink");

    const examplesGrid = document.getElementById("examplesGrid");
    const toastWrap = document.getElementById("toastWrap");

    function moonSvg() {{
      return `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="3">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"></path>
        </svg>
      `;
    }}

    function sunSvg() {{
      return `
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="3">
          <circle cx="12" cy="12" r="4"></circle>
          <path d="M12 2v2.2M12 19.8V22M4.22 4.22l1.56 1.56M18.22 18.22l1.56 1.56M2 12h2.2M19.8 12H22M4.22 19.78l1.56-1.56M18.22 5.78l1.56-1.56"></path>
        </svg>
      `;
    }}

    function applyTheme() {{
      body.setAttribute("data-theme", state.theme);
      themeIcon.innerHTML = state.theme === "dark" ? sunSvg() : moonSvg();
      localStorage.setItem("kv_theme", state.theme);
    }}

    function setAdvanced(open) {{
      state.advancedOpen = open;
      advancedPanel.classList.toggle("open", open);
      advancedToggle.classList.toggle("open", open);
    }}

    themeToggle.addEventListener("click", () => {{
      state.theme = state.theme === "dark" ? "light" : "dark";
      applyTheme();
    }});

    advancedToggle.addEventListener("click", () => {{
      setAdvanced(!state.advancedOpen);
    }});

    applyTheme();
    setAdvanced(false);

    function showToast(message) {{
      const toast = document.createElement("div");
      toast.className = "toast";

      const text = document.createElement("div");
      text.textContent = message;

      const btn = document.createElement("button");
      btn.type = "button";
      btn.innerHTML = "&times;";
      btn.addEventListener("click", () => toast.remove());

      toast.appendChild(text);
      toast.appendChild(btn);
      toastWrap.appendChild(toast);

      setTimeout(() => {{
        toast.remove();
      }}, 4200);
    }}

    function setLoading(loading) {{
      loaderOverlay.style.display = loading ? "flex" : "none";
      runBtn.disabled = loading;
      clearBtn.disabled = loading;
    }}

    function createThumb(file, index) {{
      const wrapper = document.createElement("div");
      wrapper.className = "aspect-square group";

      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.alt = file.name || `upload-${{index}}`;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "image-remove-button";
      removeBtn.innerHTML = "&times;";
      removeBtn.title = "Remove image";

      removeBtn.addEventListener("click", (e) => {{
        e.stopPropagation();
        state.files.splice(index, 1);
        renderPreviews();
      }});

      wrapper.appendChild(img);
      wrapper.appendChild(removeBtn);
      return wrapper;
    }}

    function renderPreviews() {{
      previewGrid.innerHTML = "";

      if (!state.files.length) {{
        uploadPlaceholder.style.display = "flex";
        previewGrid.style.display = "none";
        return;
      }}

      uploadPlaceholder.style.display = "none";
      previewGrid.style.display = "grid";

      state.files.forEach((file, index) => {{
        previewGrid.appendChild(createThumb(file, index));
      }});
    }}

    function addFiles(fileList) {{
      const valid = Array.from(fileList).filter(file => file.type.startsWith("image/"));
      if (!valid.length) {{
        showToast("Please upload valid image files.");
        return;
      }}
      state.files = [...state.files, ...valid];
      renderPreviews();
    }}

    uploadPlaceholder.addEventListener("click", () => fileInput.click());
    uploadZone.addEventListener("click", (e) => {{
      if (e.target === uploadZone) fileInput.click();
    }});

    fileInput.addEventListener("change", (e) => {{
      addFiles(e.target.files);
      fileInput.value = "";
    }});

    uploadZone.addEventListener("dragover", (e) => {{
      e.preventDefault();
      uploadZone.classList.add("dragover");
    }});

    uploadZone.addEventListener("dragleave", () => {{
      uploadZone.classList.remove("dragover");
    }});

    uploadZone.addEventListener("drop", (e) => {{
      e.preventDefault();
      uploadZone.classList.remove("dragover");
      if (e.dataTransfer.files?.length) {{
        addFiles(e.dataTransfer.files);
      }}
    }});

    function clearAll() {{
      state.files = [];
      renderPreviews();
      promptEl.value = "";
      seedEl.value = "0";
      stepsEl.value = "4";
      widthEl.value = "1024";
      heightEl.value = "1024";
      randomizeSeedEl.checked = true;
      outputImage.style.display = "none";
      outputImage.removeAttribute("src");
      outputEmpty.style.display = "flex";
      usedSeed.textContent = "-";
      downloadLink.style.display = "none";
      downloadLink.removeAttribute("href");
      setAdvanced(false);
    }}

    clearBtn.addEventListener("click", clearAll);

    async function fileFromUrl(url, filename = "example.jpg") {{
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch example image.");
      const blob = await res.blob();
      return new File([blob], filename, {{ type: blob.type || "image/jpeg" }});
    }}

    function renderExamples() {{
      examplesGrid.innerHTML = "";

      examples.forEach((item) => {{
        const card = document.createElement("div");
        card.className = "example-card";

        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.file;

        const body = document.createElement("div");
        body.className = "example-body";

        const text = document.createElement("p");
        text.textContent = item.prompt;

        body.appendChild(text);
        card.appendChild(img);
        card.appendChild(body);

        card.addEventListener("click", async () => {{
          try {{
            const file = await fileFromUrl(item.url, item.file);
            state.files = [file];
            renderPreviews();
            promptEl.value = item.prompt;
            showToast("Example loaded.");
          }} catch (err) {{
            showToast(err.message || "Failed to load example.");
          }}
        }});

        examplesGrid.appendChild(card);
      }});
    }}

    renderExamples();
    renderPreviews();

    async function submitEdit() {{
      try {{
        const prompt = promptEl.value.trim();
        if (!prompt) {{
          showToast("Please enter a prompt.");
          return;
        }}

        const formData = new FormData();
        formData.append("prompt", prompt);
        formData.append("seed", seedEl.value || "0");
        formData.append("randomize_seed", String(randomizeSeedEl.checked));
        formData.append("width", widthEl.value || "1024");
        formData.append("height", heightEl.value || "1024");
        formData.append("steps", stepsEl.value || "4");

        state.files.forEach(file => formData.append("images", file));

        setLoading(true);

        const res = await fetch("/api/edit", {{
          method: "POST",
          body: formData
        }});

        const data = await res.json();

        if (!res.ok || !data.success) {{
          throw new Error(data.error || "Processing failed.");
        }}

        outputImage.src = data.image_url + "?t=" + Date.now();
        outputImage.style.display = "block";
        outputEmpty.style.display = "none";

        usedSeed.textContent = String(data.seed);

        downloadLink.href = data.download_url;
        downloadLink.style.display = "flex";

      }} catch (err) {{
        showToast(err.message || "An unexpected error occurred.");
      }} finally {{
        setLoading(false);
      }}
    }}

    runBtn.addEventListener("click", submitEdit);
  </script>
</body>
</html>
"""
    
app.launch()