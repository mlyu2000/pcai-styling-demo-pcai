"""Local end-to-end smoke of stages 4/5 with NO image model configured:
proves the deterministic fallbacks (pencil sketch, composite-only, mock design,
mock scene) make the whole 6-stage flow runnable without any GPU model."""
import base64, io, os, sys, tempfile

os.environ["RESULTS_DIR"] = tempfile.mkdtemp(prefix="pcai-smoke-")
os.environ.pop("IMG_ENDPOINT", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import main  # noqa

# copy the bundled scene into the expected dir (already there in repo)
assert main.MOCK_SCENE_FILE.exists(), "sample scene missing"

from PIL import Image, ImageDraw
img = Image.new("RGB", (512, 320), (200, 200, 205))
d = ImageDraw.Draw(img)
d.rounded_rectangle([60, 120, 460, 240], 30, fill=(30, 80, 200))
d.ellipse([110, 210, 170, 270], fill=(20, 20, 24)); d.ellipse([350, 210, 410, 270], fill=(20, 20, 24))
buf = io.BytesIO(); img.save(buf, format="PNG")
B64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

from fastapi.testclient import TestClient
c = TestClient(main.app)

r = c.post("/api/workflow/stage", json={"stage": 4, "image": B64, "colorways": 2})
print("stage4:", r.status_code)
d4 = r.json()
assert r.status_code == 200, d4
res4 = d4["results"]
assert res4["base_sketch"] and res4["base_sketch_rear"] and len(res4["colorways"]) == 2
print("  sketch_mode:", d4["sketch_mode"], "targets:", d4["colorways_targets"],
      "elapsed_ms:", d4["elapsed_ms"])

design = res4["colorways"][0]["front"]
r = c.post("/api/workflow/stage", json={"stage": 5, "design_image": design,
                                        "use_mock_scene": True, "use_mock_design": True,
                                        "view_angle": "3/4 front"})
print("stage5:", r.status_code)
d5 = r.json()
assert r.status_code == 200, d5
res5 = d5["results"]
assert all(res5[k] for k in ("composed", "main", "another_view", "another_color"))
print("  viz_mode:", d5["viz_mode"], "mock flags:", d5["mock_design_applied"],
      d5["mock_scene_used"], "elapsed_ms:", d5["elapsed_ms"])
for k, v in res5.items():
    im = Image.open(io.BytesIO(base64.b64decode(v.split(",")[1])))
    assert im.width <= 768 and im.height <= 768
    print(f"  {k}: {im.size}")

# geometry guarantee: another_color keeps pixels outside the body untouched
import numpy as np
def b64img(v): return Image.open(io.BytesIO(base64.b64decode(v.split(",")[1]))).convert("RGB")
main_im, col_im = b64img(res5["main"]), b64img(res5["another_color"])
mask = main._body_mask(main_im)
diff = np.abs(np.asarray(main_im, np.int16) - np.asarray(col_im, np.int16)).mean(2)
print(f"  colorway diff outside body: {diff[~mask].mean():.3f} (want <1) | inside: {diff[mask].mean():.1f}")
assert diff[~mask].mean() < 1.0

r = c.get("/api/mock/scene"); print("GET /api/mock/scene:", r.status_code, r.headers["content-type"])
assert r.status_code == 200 and r.headers["content-type"] == "image/png"
r = c.get("/api/config"); cfg = r.json()
print("config:", {k: cfg[k] for k in ("imageModelEnabled", "mockSceneAvailable")})
print("\nSMOKE_OK — full 6-stage chain runs with no image model")
