import base64, json, sys
import urllib.request

SCRATCH = "C:/Users/ml/AppData/Local/hermes/cache/scratch"
png = open(f"{SCRATCH}/test_car.png", "rb").read()
b64 = base64.b64encode(png).decode()

body = {
    "subject_image": f"data:image/png;base64,{b64}",
    "style_reference": "",
    "parameters": {"num_frames": 121, "fps": 24, "style_strength": 0.7, "motion_strength": 0.6},
}
data = json.dumps(body).encode()
req = urllib.request.Request(
    "http://127.0.0.1:18091/api/generate",
    data=data,
    headers={"Content-Type": "application/json", "Accept": "application/json"},
    method="POST",
)
try:
    r = urllib.request.urlopen(req, timeout=1200)
    d = json.loads(r.read().decode())
    print("=== /api/generate RESULT ===")
    print(json.dumps(d, indent=2)[:1500])
    open(f"{SCRATCH}/gen_result.json", "w").write(json.dumps(d))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:1200])
    sys.exit(1)
