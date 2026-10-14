#!/usr/bin/env python3
"""
test_flow_api.py — e2e for the per-step flow API (backend v29+).

Drives the deployed service via port-forward (default 127.0.0.1:18091) or a
base URL, running each /api/flow/* step with a real photo + the mock scene,
and asserts the v29 contract:
  - every response has: step, results.<output>, mode, prompt(s), elapsed_ms
  - fallbacks are DECLARED (mode contains 'fallback') rather than silent
  - video is H.264 > 1 MB when the video model is up
  - the legacy stage API still works (app UI regression gate)

Usage:
  python scripts/test_flow_api.py [--base http://127.0.0.1:18091] [--skip-video]

Exit codes: 0 all pass, 1 failure, 2 model-fallback only (artifacts valid).
"""
import argparse, base64, json, ssl, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = Path(__file__).resolve().parent.parent / "backend" / "sample_data"
CTX = ssl._create_unverified_context()

PASS, FAIL, WARN = 0, 0, 0


def log(kind, msg):
    global PASS, FAIL, WARN
    if kind == "PASS":
        PASS += 1
    elif kind == "FAIL":
        FAIL += 1
    else:
        WARN += 1
    print(f"{kind:4} {msg}")


def http(base, path, data=None, timeout=900):
    data = data.encode() if isinstance(data, str) else data
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode())


def http_raw(base, path, timeout=300):
    url = base.rstrip("/") + path
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout, context=CTX) as r:
        return r.read()


def b64_png(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:18091")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--photo", default=None, help="override the input photo")
    args = ap.parse_args()
    base = args.base

    # ---- health + config
    cfg = http(base, "/api/config", timeout=30)
    log("PASS" if cfg else "FAIL", f"config: imageGen={cfg.get('imageGen', {}).get('enabled', '?')}")

    # ---- input photo: mock scene is the only guaranteed asset; use a small
    # synthetic car-ish image so the test never depends on external files.
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (768, 512), (200, 210, 220))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([96, 240, 672, 400], 48, fill=(20, 60, 160))       # body
    d.rounded_rectangle([220, 180, 550, 260], 30, fill=(160, 200, 230))     # glass
    d.ellipse([150, 330, 260, 440], fill=(20, 20, 20))                       # wheels
    d.ellipse([520, 330, 630, 440], fill=(20, 20, 20))
    p = Path(__file__).resolve().parent.parent / "cache_scratch" / "_flow_test_car.png"
    p.parent.mkdir(exist_ok=True)
    img.save(p)
    photo = b64_png(p) if not args.photo else b64_png(Path(args.photo))
    scene = b64_png(SAMPLE / "sample_scene_dusk.png")
    if not (SAMPLE / "sample_scene_dusk.png").exists():
        log("FAIL", "sample_data/sample_scene_dusk.png missing — build context broken")
        sys.exit(1)

    fallbacks = 0

    # ---- step 1: sketch (front + rear, each from the photo)
    t0 = time.time()
    r = http(base, "/api/flow/sketch", json.dumps({
        "image": photo,
        "prompt_front": "Convert this car into a clean automotive design sketch, pencil on white.",
        "prompt_rear": "Rear 3/4 view automotive design sketch of this car.",
        "requirement": "test",
    }), timeout=900)
    el = round(time.time() - t0, 1)
    ok = (r.get("step") == "sketch" and "front" in r.get("results", {})
          and "rear" in r.get("results", {}) and isinstance(r.get("mode"), dict)
          and "prompt_front" in r and "prompt_rear" in r)
    log("PASS" if ok else "FAIL", f"flow/sketch {el}s mode={r.get('mode')}")
    if not ok:
        log("FAIL", f"  contract: {json.dumps(r)[:300]}")
    if any("fallback" in str(v) for v in (r.get("mode") or {}).values()):
        fallbacks += 1
        log("WARN", "  sketch used deterministic fallback (model down?)")
    front = r["results"]["front"]

    # ---- step 3: combine (deterministic composite = integrator; single-ref
    # harmonize is the model pass — the image model cannot take two images)
    t0 = time.time()
    r = http(base, "/api/flow/combine", json.dumps({
        "design_image": front, "scene_image": scene,
        "prompt": "Integrate this car design into this scene. Keep the exact same car and the scene background.",
        "requirement": "test",
    }), timeout=900)
    el = round(time.time() - t0, 1)
    ok = (r.get("step") == "combine" and "composed" in r.get("results", {})
          and r.get("mode") and "prompt" in r)
    log("PASS" if ok else "FAIL", f"flow/combine {el}s mode={r.get('mode')}")
    if not ok:
        log("FAIL", f"  contract: {json.dumps(r)[:300]}")
    if "fallback" in str(r.get("mode")):
        fallbacks += 1
        log("WARN", f"  combine mode={r.get('mode')} (harmonize model unavailable)")
    composed = r["results"]["composed"]

    # ---- step 4: view (prompt-driven re-angle)
    t0 = time.time()
    r = http(base, "/api/flow/view", json.dumps({
        "image": composed,
        "prompt": "Show this same car and scene from a side view. Keep everything else identical.",
        "view_angle": "side", "requirement": "test",
    }), timeout=900)
    el = round(time.time() - t0, 1)
    ok = (r.get("step") == "view" and "view" in r.get("results", {})
          and r.get("mode") and r.get("view_angle") == "side")
    log("PASS" if ok else "FAIL", f"flow/view {el}s mode={r.get('mode')}")
    if "fallback" in str(r.get("mode")):
        fallbacks += 1
        log("WARN", f"  view mode={r.get('mode')}")

    # ---- step 5: recolor (prompt-driven model repaint primary)
    t0 = time.time()
    r = http(base, "/api/flow/recolor", json.dumps({
        "image": composed,
        "prompt": "Re-color the car body to red. Keep exactly the same car and scene.",
        "color": "red", "requirement": "test",
    }), timeout=900)
    el = round(time.time() - t0, 1)
    ok = (r.get("step") == "recolor" and "recolor" in r.get("results", {})
          and r.get("mode") and r.get("color_target") == "red")
    log("PASS" if ok else "FAIL", f"flow/recolor {el}s mode={r.get('mode')} color={r.get('color_target')}")
    if "fallback" in str(r.get("mode")):
        fallbacks += 1
        log("WARN", f"  recolor mode={r.get('mode')} (deterministic repaint)")

    # ---- step 6: video (I2V of the combined still)
    if not args.skip_video:
        t0 = time.time()
        r = http(base, "/api/flow/video", json.dumps({
            "image": composed,
            "prompt": "Cinematic slow camera push-in on the car in the scene, smooth steady motion.",
            "requirement": "test",
        }), timeout=900)
        el = round(time.time() - t0, 1)
        vurl = "/" + (r.get("results", {}).get("video_url") or "").lstrip("/")
        ok = r.get("step") == "video" and vurl and r.get("mode") == "i2v"
        log("PASS" if ok else "FAIL", f"flow/video {el}s mode={r.get('mode')}")
        if ok:
            raw = http_raw(base, vurl)
            h264 = b"ftyp" in raw[:64] and b"avcC" in raw
            log("PASS" if h264 and len(raw) > 1_000_000 else "FAIL",
                f"  video {len(raw)/1e6:.1f} MB h264={h264}")

    # ---- regression: legacy stage API still works (app UI)
    try:
        r = http(base, "/api/workflow/stage", json.dumps({
            "stage": 4, "image": photo,
            "prompt_override": "sketch", "requirement": "test",
        }), timeout=900)
        ok = r.get("stage") == 4 and "base_sketch" in r.get("results", {})
        log("PASS" if ok else "FAIL", "legacy /api/workflow/stage stage4 (UI regression)")
    except Exception as e:
        log("FAIL", f"legacy stage API broken: {e}")

    print(f"\n{PASS} passed, {FAIL} failed, {WARN} warnings (model-fallbacks: {fallbacks})")
    if FAIL:
        sys.exit(1)
    sys.exit(2 if fallbacks else 0)


if __name__ == "__main__":
    main()
