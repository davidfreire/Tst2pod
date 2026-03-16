import runpod

def handler(job):
    print("JOB RECIBIDO:", job, flush=True)
    return {
        "ok": True,
        "input": job["input"]
    }

runpod.serverless.start({"handler": handler})
