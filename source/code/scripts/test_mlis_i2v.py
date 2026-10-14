import base64, json, sys, uuid
import urllib.request

SCRATCH = "C:/Users/ml/AppData/Local/hermes/cache/scratch"
png = open(f"{SCRATCH}/test_car.png", "rb").read()
b64 = base64.b64encode(png).decode()
image_reference = json.dumps({"image_url": f"data:image/png;base64,{b64}"})

# Build multipart/form-data by hand (stdlib only)
fields = {
    "prompt": "A sleek red sports car gliding along an empty coastal highway at golden hour, cinematic tracking shot, soft warm light, smooth 24fps",
    "model": "/models/wan22-ti2v-5b",
    "image_reference": image_reference,
    "num_frames": "121",
    "fps": "24",
    "num_inference_steps": "50",
    "guidance_scale": "5.0",
    "flow_shift": "5.0",
    "size": "1280x704",
}
boundary = uuid.uuid4().hex
body = b""
for k, v in fields.items():
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
             f"{v}\r\n").encode()
body += f"--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:18085/v1/videos",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
    method="POST",
)
try:
    r = urllib.request.urlopen(req, timeout=60)
    d = json.loads(r.read().decode())
    vid = d.get("id") or d.get("video_id")
    print("video id:", vid)
    print("status:", d.get("status"))
    print("model:", d.get("model"))
    open(f"{SCRATCH}/mlis_vid.txt", "w").write(vid or "")
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:1200])
    sys.exit(1)
