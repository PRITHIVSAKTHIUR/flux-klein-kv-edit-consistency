# **flux-klein-kv-edit-consistency**

flux-klein-kv-edit-consistency is an experimental image editing application powered by the FLUX.2-klein-9b-kv model and a custom Consistency LoRA adapter. This tool is designed to apply precise, highly consistent edits to uploaded images based on natural language prompts. It allows users to execute complex transformations—such as changing weather conditions, adjusting lighting, or upscaling textures—while strictly preserving the original subject's identity, composition, and framing. The application features a lightweight, custom-built HTML/CSS/JavaScript frontend served directly via FastAPI, completely bypassing standard Gradio UI elements for a bespoke, fast, and responsive user experience. Operating entirely on CUDA-enabled GPUs, this project demonstrates advanced, consistent image-to-image manipulation using state-of-the-art diffusion models.

---

<img width="1920" height="1080" alt="Screen Shot 2026-03-25 at 13 58 34" src="https://github.com/user-attachments/assets/483775ae-9ee9-4e3f-9d1d-87fd3e0ebc1f" />
<img width="2880" height="1800" alt="Screen Shot 2026-03-25 at 14 00 46" src="https://github.com/user-attachments/assets/43cc2b24-547e-44df-88dd-c3760d322950" />

---

<img width="2880" height="1800" alt="Screen Shot 2026-03-25 at 13 59 58" src="https://github.com/user-attachments/assets/1ca22af7-72f7-4d4e-b79f-e3557d73128a" />
<img width="2880" height="1800" alt="Screen Shot 2026-03-25 at 14 00 10" src="https://github.com/user-attachments/assets/57df70a8-ed17-4740-8349-d8968bfe4c98" />

---

### **Key Features**

* **Consistent Image Editing:** Utilizes `black-forest-labs/FLUX.2-klein-9b-kv` paired with the `dx8152/Flux2-Klein-9B-Consistency` LoRA to perform edits that maintain structural and compositional fidelity to the source image.
* **Custom Headless UI:** Features a completely custom, responsive web interface built with vanilla HTML/CSS/JS, served via FastAPI. It includes dynamic theme switching (Dark/Light mode), custom toast notifications, and a styled drag-and-drop uploader.
* **Dynamic Sizing:** Automatically calculates and snaps image dimensions to optimal sizes (multiples of 8, up to 1024x1024) based on the uploaded image's aspect ratio.
* **Advanced Inference Controls:** Provides a hidden 'Advanced Options' panel to manually adjust the Generation Seed, Inference Steps, and output Dimensions.
* **Automated Patching:** Includes a mechanism to automatically apply necessary code patches (`flux2_klein_kv.patch`) to the `diffusers` library at runtime.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── 4.jpg
├── app.py
├── flux2_klein_kv.patch
├── LICENSE.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run flux-klein-kv-edit-consistency locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU and a Hugging Face access token configured as an environment variable (`HF_TOKEN`).

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=26.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and web server libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
accelerate==1.13.0
diffusers==0.37.0
huggingface_hub==1.5.0
sentencepiece==0.2.1
protobuf==3.20.3
transformers==5.3.0
torch==2.9.1
fastapi
pillow
gradio
peft
```

### **Usage**

Once your environment is set up, the dependencies are installed, and your `HF_TOKEN` is exported, you can launch the application by running the main Python script:

```bash
export HF_TOKEN="your_huggingface_token"
python app.py
```

The script will attempt to apply the required patch to the `diffusers` library, load the heavy 9B parameter model into VRAM, and then start the web server. Access the custom UI by navigating to the local address provided in your terminal (usually `http://127.0.0.1:7860/`). Upload an image, provide an editing prompt (e.g., "Relight the image with soft golden sunset lighting"), and click "Edit Image".

### **License and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency.git](https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency.git)
