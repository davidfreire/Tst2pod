import tempfile
import requests
import torch
import runpod

from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """
Eres un juez profesional de salto de trampolín.

Estas imágenes son frames consecutivos del mismo salto.

Evalúa:
1. aproximación
2. batida
3. vuelo
4. entrada al agua

Devuelve SOLO JSON con este formato:

{
  "approach_score": 0,
  "takeoff_score": 0,
  "flight_score": 0,
  "entry_score": 0,
  "overall_score": 0,
  "faults": [],
  "summary": ""
}
"""

model = None
processor = None

def get_model():
    global model, processor

    if model is None:
        print("Cargando processor...", flush=True)
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        print("Cargando modelo...", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        print("Modelo cargado.", flush=True)

    return model, processor

def download_video(url: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(r.content)
    tmp.close()
    return tmp.name

def sample_frames(video_path: str, n_frames: int = 12):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    if total == 0:
        raise ValueError("No se pudieron leer frames del vídeo")

    if n_frames >= total:
        idx = list(range(total))
    else:
        step = total / n_frames
        idx = [min(int(i * step), total - 1) for i in range(n_frames)]

    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in frames]

def handler(job):
    print("Job recibido", flush=True)

    video_url = job["input"]["video_url"]
    video_path = download_video(video_url)
    frames = sample_frames(video_path, n_frames=12)

    model, processor = get_model()

    prompt = PROMPT + "\nEstas 12 imágenes son frames del mismo vídeo en orden temporal."

    inputs = processor(
        text=[prompt],
        images=frames,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    output_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    return {"result": output_text}

runpod.serverless.start({"handler": handler})
