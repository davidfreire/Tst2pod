import io
import base64
import tempfile
import requests

import runpod
from PIL import Image
from decord import VideoReader, cpu
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Arrancando worker...", flush=True)
print("Cargando modelo...", flush=True)

llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 32},
)

print("Modelo cargado.", flush=True)

sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=400,
)

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

def download_video(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(r.content)
    tmp.close()
    return tmp.name

def sample_frames(video_path, n_frames=24):
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
    return [Image.fromarray(f) for f in frames]

def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def handler(job):
    print("Job recibido", flush=True)

    video_url = job["input"]["video_url"]
    video = download_video(video_url)
    frames = sample_frames(video)

    content = [{"type": "text", "text": PROMPT}]
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_b64(f)}
        })

    messages = [{
        "role": "user",
        "content": content
    }]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    return {"result": outputs[0].outputs[0].text}

runpod.serverless.start({"handler": handler})
