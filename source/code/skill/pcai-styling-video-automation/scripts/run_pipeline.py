#!/usr/bin/env python3
"""
run_pipeline.py — drives the PCAI styling video pipeline on two engines.

RECONCILED WORKFLOW (identical step sequence + identical rendered prompts on
both engines; reference semantics = gemini's "user prompt + declared inputs
only"). Since backend v32 the pcai provider drives the NATIVE per-step flow
API (/api/flow/*), which implements these exact semantics — prompt-driven,
declared inputs, deterministic machinery only as declared fallbacks:

  1. sketch_front   inputs [photo]
  2. sketch_rear    inputs [photo]        (independent call, no chaining)
  3. combine        inputs [front, scene] (deterministic composite = integrator
                    on PCAI — image model is single-reference — + model harmonize)
  4. alt_view       inputs [combine]
  5. recolor        inputs [combine]       paint-only color change ({color})
  6. video          engine-specific source (see notes below)

Providers:
  pcai    : deployed styling-demo service (lab ingress), /api/flow/* steps
  gemini  : Google AI Studio — one generateContent call per image step,
            Veo first/last-frame for the video (customer's P6 pattern)
  both    : same prompts, both engines, side-by-side output folders

ENGINE DIFFERENCES (documented in run_summary.json; each run also records
the per-step `mode` the backend actually used — model vs declared fallback):
  - video: pcai = I2V of the combined still (scene video).
           gemini = first/last-frame morph sketch_front -> sketch_rear
           (matches P6 "camera movement from first image until to second image").
    (Wan2.2 has no first/last-frame mode — model-inherent.)
  - fault tolerance: pcai steps fall back deterministically and declare it
    in `mode`; gemini is fail-fast.

Stdlib only (no pip installs). Long-running: ~3-15 min (pcai) / ~2-6 min
(gemini, fast models). Run as a background task with notify.

Exit codes: 0 = success, 1 = hard failure, 2 = model fallback (artifacts saved)
"""
import argparse, base64, json, os, ssl, sys, time, urllib.request
from pathlib import Path

LAB = "https://styling-demo.aie.cs1.ctc.sg.lab"
GEM = "https://generativelanguage.googleapis.com/v1beta"
CTX = ssl._create_unverified_context()  # lab ingress self-signed cert

STEP_ORDER = ["sketch_front", "sketch_rear", "combine", "alt_view", "recolor", "video"]
STEP_INPUTS = {  # declared inputs per step (prompt-driven only)
    "sketch_front": ["photo"],
    "sketch_rear": ["photo"],
    "combine": ["front", "scene"],
    "alt_view": ["combine"],
    "recolor": ["combine"],
    "video": ["engine: pcai=combine | gemini=front,rear(first/last frame)"],
}

# Engine-inherent differences surfaced in run_summary.json (post v32 rebuild:
# everything is prompt-driven parity EXCEPT combine's integrator + video mode.
# The image model is single-reference, so combine's deterministic composite is
# the engine's PRIMARY integrator, not a fallback — declared, not hidden.)
ENGINE_DIFFS = {
    "combine": "pcai: Qwen-Image-Edit GGUF is single-reference (a 2nd image is "
               "parsed but silently discarded, verified in-pod) -> the design-into-"
               "scene integration is the geometry-locked deterministic composite, "
               "then a model harmonize pass. gemini: native two-image model "
               "integration of [design, scene].",
    "video": "pcai: I2V of the combined still (scene video; Wan2.2 has no "
             "first/last-frame mode). gemini: first/last-frame morph "
             "sketch_front -> sketch_rear (P6 pattern).",
    "fallbacks": "pcai: each /api/flow/* step reports `mode` (model vs declared "
                 "deterministic path). gemini: fail-fast.",
}


def http(url, data=None, key=None, timeout=900):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["x-goog-api-key"] = key
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode())


def http_raw(url, key=None, timeout=900):
    req = urllib.request.Request(url, headers={"x-goog-api-key": key} if key else {})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def b64(path):
    """Read a local image -> data: URI (what the backend expects)."""
    raw = Path(path).read_bytes()
    mime = "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def raw_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def save_b64(b64val, path):
    """Strip a data: URI prefix, write raw bytes, return str(path) or None."""
    if not b64val:
        return None
    if b64val.startswith("data:") and "," in b64val:
        b64val = b64val.split(",", 1)[1]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(base64.b64decode(b64val))
    return str(path)


def load_simple_yaml(path):
    """Minimal YAML for flat key: "value" packs (no external deps)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        v = v.strip()
        if v.startswith('"') and v.endswith('"') and len(v) > 1:
            v = v[1:-1]
        out[k.strip()] = v
    return out


def render(tpl, slots):
    for k, v in slots.items():
        tpl = tpl.replace("{" + k + "}", v or "")
    return tpl


# ------------------------------------------------------------------ gemini
def gemini_generate(key, model, prompt, images_b64, timeout=300):
    parts = [{"text": prompt}]
    for b in images_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b}})
    body = json.dumps({"contents": [{"role": "user", "parts": parts}]}).encode()
    out = http(f"{GEM}/models/{model}:generateContent", data=body, key=key, timeout=timeout)
    parts_out = (out.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    for p in parts_out:
        if p.get("inline_data"):
            return p["inline_data"]["data"]
    raise RuntimeError(f"gemini returned no image: {json.dumps(out)[:400]}")


def gemini_video(key, model, prompt, first_b64, last_b64, timeout=900):
    body = json.dumps({"instances": [{
        "first_frame": {"image_bytes_b64": first_b64, "mime_type": "image/jpeg"},
        "last_frame": {"image_bytes_b64": last_b64, "mime_type": "image/jpeg"},
        "prompt": prompt}], "parameters": {}}).encode()
    op = http(f"{GEM}/models/{model}:predictLongRunning", data=body, key=key, timeout=120)
    name = op.get("name")
    if not name:
        raise RuntimeError(f"veo op missing: {json.dumps(op)[:400]}")
    url = f"{GEM}/{name}"
    t0 = time.time()
    while True:
        st = http(url, key=key, timeout=120)
        if st.get("done"):
            resp = st.get("response", {})
            vids = resp.get("generateVideoResponse", {}).get("generatedVideos", [])
            uri = (vids[0].get("video", {}) or {}).get("uri")
            if not uri:
                raise RuntimeError(f"veo done but no video uri: {json.dumps(st)[:400]}")
            return uri, time.time() - t0
        if time.time() - t0 > timeout:
            raise TimeoutError("veo did not finish in time")
        time.sleep(15)


def run_gemini(key, pack, prompts, args, out, base):
    """Reference semantics: every step = ONE user prompt + declared inputs -> model."""
    img_model = args.image_model or pack.get("image_model", "gemini-3.1-flash-image")
    vid_model = args.video_model or pack.get("video_model", "veo-3.1-fast-generate-preview")
    timings, artifacts, notes, steps = {}, {}, {}, []
    photo = raw_b64(args.photo)

    def step(name, prompt, fn, *a):
        t0 = time.time()
        val = fn(*a)
        timings[name] = round(time.time() - t0, 1)
        steps.append({"step": name, "prompt": prompt, "inputs": STEP_INPUTS[name],
                      "engine": "gemini", "model": img_model, "elapsed_s": timings[name]})
        return val

    front = step("sketch_front", prompts["sketch_front"],
                 gemini_generate, key, img_model, prompts["sketch_front"], [photo])
    artifacts["sketch_front"] = save_b64(front, out / "sketch_front.jpg")

    rear = step("sketch_rear", prompts["sketch_rear"],
                gemini_generate, key, img_model, prompts["sketch_rear"], [photo])
    artifacts["sketch_rear"] = save_b64(rear, out / "sketch_rear.jpg")

    # scene (REF 2): user file or the lab's bundled mock scene
    if args.scene:
        scene = raw_b64(args.scene)
        (out / "scene_input.jpg").write_bytes(Path(args.scene).read_bytes())
    else:
        raw = http_raw(base + "/api/mock/scene", timeout=60)
        (out / "scene_input.png").write_bytes(raw)
        scene = base64.b64encode(raw).decode()

    combine = step("combine", prompts["combine"],
                   gemini_generate, key, img_model, prompts["combine"], [front, scene])
    artifacts["combine"] = save_b64(combine, out / "combine.jpg")

    alt = step("alt_view", prompts["alt_view"],
               gemini_generate, key, img_model, prompts["alt_view"], [combine])
    artifacts["alt_view"] = save_b64(alt, out / "alt_view.jpg")

    rec = step("recolor", prompts["recolor"],
               gemini_generate, key, img_model, prompts["recolor"], [combine])
    artifacts["recolor"] = save_b64(rec, out / "recolor.jpg")

    t0 = time.time()
    uri, secs = gemini_video(key, vid_model, prompts["video"], front, rear, timeout=900)
    raw = http_raw(uri if uri.startswith("http") else "https://" + uri, key=key, timeout=600)
    (out / "final.mp4").write_bytes(raw)
    artifacts["video"] = str(out / "final.mp4")
    timings["video"] = round(secs, 1)
    steps.append({"step": "video", "prompt": prompts["video"],
                  "inputs": STEP_INPUTS["video"], "engine": "gemini", "model": vid_model,
                  "elapsed_s": timings["video"]})
    notes.append(f"image_model={img_model} video_model={vid_model} (first=sketch_front, last=sketch_rear)")
    return timings, artifacts, notes, steps


# ------------------------------------------------------------------ pcai
def _flow(base, step, payload, timeout):
    """POST one /api/flow/<step> and return the JSON response."""
    return http(base + f"/api/flow/{step}",
                data=json.dumps(payload).encode(), timeout=timeout)


def run_pcai(base, pack, prompts, args, out):
    """Same reconciled workflow, driven through the deployed service's NATIVE
    per-step flow API (backend v29+): /api/flow/sketch, /combine, /view,
    /recolor, /video. Prompt-driven with exactly the declared inputs; every
    response's `mode` is recorded so run_summary.json shows model-vs-fallback
    per step. Engine differences remaining: video mode + declared fallbacks
    (see ENGINE_DIFFS)."""
    photo = b64(args.photo)
    timings, artifacts, notes, steps = {}, {}, {}, []

    # steps 1+2: one /api/flow/sketch call; front AND rear are each generated
    # directly from the photo (independent — no chaining, Gemini parity).
    t0 = time.time()
    rsk = _flow(base, "sketch", {
        "image": photo,
        "prompt_front": prompts["sketch_front"],
        "prompt_rear": prompts["sketch_rear"],
        "requirement": args.requirement,
    }, timeout=900)
    el = round(time.time() - t0, 1)
    res = rsk.get("results", {})
    front = res.get("front", "")
    rear = res.get("rear", "")
    artifacts["sketch_front"] = save_b64(front, out / "sketch_front.png")
    artifacts["sketch_rear"] = save_b64(rear, out / "sketch_rear.png")
    mode = rsk.get("mode", {})
    steps.append({"step": "sketch_front", "prompt": prompts["sketch_front"],
                  "inputs": STEP_INPUTS["sketch_front"], "engine": "pcai",
                  "model": "qwen-image-edit", "mode": mode.get("front", "model"),
                  "elapsed_s": el})
    steps.append({"step": "sketch_rear", "prompt": prompts["sketch_rear"],
                  "inputs": STEP_INPUTS["sketch_rear"], "engine": "pcai",
                  "model": "qwen-image-edit", "mode": mode.get("rear", "model"),
                  "elapsed_s": el})
    for m, k in mode.items():
        if "fallback" in str(k):
            notes.append(f"sketch[{m}] used deterministic fallback (exit 2)")

    # step 3: /api/flow/combine — PRIMARY integrator = single-ref i2i SCENE
    # GENERATION (mode=scene-gen): the car photo is the only reference and the
    # model generates the car already integrated into the described scene
    # (grounded, lighting-matched — a deterministic paste of a studio product
    # photo reads as a floating rectangle even with the v33/v35 geometry fix,
    # verified 2026-10-01). scene_image is still sent (fallback path needs it).
    # The CAR is the user's uploaded base photo (the "screenshot base"), not the
    # step-1 design sketch.
    if args.scene:
        scene = b64(args.scene)
    else:
        raw = http_raw(base + "/api/mock/scene", timeout=60)
        (out / "scene_input.png").write_bytes(raw)
        scene = "data:image/png;base64," + base64.b64encode(raw).decode()
    t0 = time.time()
    rcb = _flow(base, "combine", {
        "design_image": photo, "scene_image": scene,
        "mode": "scene-gen",
        "scene_prompt": args.requirement or "a realistic road environment",
        "requirement": args.requirement,
    }, timeout=900)
    el = round(time.time() - t0, 1)
    combine_still = rcb.get("results", {}).get("composed", "")
    artifacts["combine"] = save_b64(combine_still, out / "combine.png")
    cmb_mode = rcb.get("mode", "scene-gen")
    steps.append({"step": "combine", "prompt": rcb.get("prompt", "scene-gen"),
                  "inputs": STEP_INPUTS["combine"], "engine": "pcai",
                  "model": "qwen-image-edit (single-ref i2i scene generation; "
                           "deterministic composite = fallback)", "mode": cmb_mode,
                  "elapsed_s": el})
    if "fallback" in str(cmb_mode):
        notes.append(f"combine used {cmb_mode} (image model unavailable; "
                     "deterministic composite fallback; exit 2)")

    # step 4: /api/flow/view — prompt-driven re-angle of the combined still
    t0 = time.time()
    rvw = _flow(base, "view", {
        "image": combine_still,
        "prompt": prompts["alt_view"],
        "view_angle": args.view,
        "requirement": args.requirement,
    }, timeout=900)
    el = round(time.time() - t0, 1)
    artifacts["alt_view"] = save_b64(rvw.get("results", {}).get("view"), out / "alt_view.png")
    vw_mode = rvw.get("mode", "model")
    steps.append({"step": "alt_view", "prompt": prompts["alt_view"],
                  "inputs": STEP_INPUTS["alt_view"], "engine": "pcai",
                  "model": "qwen-image-edit", "mode": vw_mode,
                  "elapsed_s": el})
    if "fallback" in str(vw_mode):
        notes.append(f"alt_view used {vw_mode} (exit 2)")

    # step 5: /api/flow/recolor — prompt-driven repaint of the combined still
    t0 = time.time()
    rrc = _flow(base, "recolor", {
        "image": combine_still,
        "prompt": prompts["recolor"],
        "color": args.color or "",
        "requirement": args.requirement,
    }, timeout=900)
    el = round(time.time() - t0, 1)
    artifacts["recolor"] = save_b64(rrc.get("results", {}).get("recolor"), out / "recolor.png")
    rc_mode = rrc.get("mode", "model")
    steps.append({"step": "recolor", "prompt": prompts["recolor"],
                  "inputs": STEP_INPUTS["recolor"], "engine": "pcai",
                  "model": "qwen-image-edit (prompt-driven)", "mode": rc_mode,
                  "elapsed_s": el})
    if "fallback" in str(rc_mode):
        notes.append(f"recolor used {rc_mode} (exit 2)")

    # step 6: /api/flow/video — I2V of the combined still (engine difference,
    # documented: Wan2.2 has no first/last-frame mode). The base prompt is a
    # DRIVING animation (the scene-gen combine is an in-scene car, not a
    # studio shot, so the old "studio orbit" base would conflict); the
    # requirement adds the user's style context on top.
    video_prompt = (
        "Cinematic video of the car DRIVING forward on the road in this scene. "
        "Slow camera tracking alongside the car so the road and background move "
        "past. Realistic subtle motion, matching the scene's lighting. "
        "Strictly keep all the car design elements; do not improvise with the "
        "car design."
    )
    t0 = time.time()
    rv = _flow(base, "video", {
        "image": combine_still,
        "prompt": video_prompt,
        "requirement": args.requirement,
    }, timeout=900)
    el = round(time.time() - t0, 1)
    vurl = "/" + (rv.get("results", {}).get("video_url") or "").lstrip("/")
    if not vurl:
        raise RuntimeError(f"no video_url: {json.dumps(rv)[:300]}")
    raw = http_raw(base + vurl, timeout=300)
    (out / "final.mp4").write_bytes(raw)
    artifacts["video"] = str(out / "final.mp4")
    steps.append({"step": "video", "prompt": video_prompt,
                  "inputs": STEP_INPUTS["video"], "engine": "pcai",
                  "model": "wan2.2-ti2v-5b (I2V of combine)",
                  "mode": rv.get("mode", "i2v"), "elapsed_s": el})
    return timings, artifacts, notes, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo", required=True, help="front-view car photo")
    ap.add_argument("--provider", default="pcai", choices=["pcai", "gemini", "both"])
    ap.add_argument("--prompts", default="tuned", help="pack under prompts/ (tuned | customer_adm)")
    ap.add_argument("--scene", help="scene photo (REF 2). Omit for lab mock scene")
    ap.add_argument("--view", default="side", help="alt view angle: side | 3/4 front | 3/4 rear | rear")
    ap.add_argument("--color", default="silver", help="recolor target color (drives {color})")
    ap.add_argument("--requirement", default="", help="freeform style requirement (appended by backend)")
    ap.add_argument("--designer-edit", default=None, help="reserved: human PSD gate (app-UI path)")
    ap.add_argument("--out", default=None, help="output dir (default ./runs/run-<ts>)")
    ap.add_argument("--base-url", default=LAB)
    ap.add_argument("--skill-root", default=None)
    ap.add_argument("--image-model", default=None)
    ap.add_argument("--video-model", default=None)
    args = ap.parse_args()

    root = Path(args.skill_root or Path(__file__).resolve().parent.parent)
    pack = load_simple_yaml(root / "prompts" / f"{args.prompts}.yaml")
    # render every slot up front; the SAME rendered text goes to both engines
    slots = {"view_angle": args.view, "color": args.color or "",
             "requirement": args.requirement, "scene_desc": ""}
    prompts = {k: render(pack[k], slots) for k in STEP_ORDER if k in pack}
    missing = [k for k in STEP_ORDER[:5] if k not in prompts]
    if missing or "video" not in prompts:
        sys.exit(f"prompt pack '{args.prompts}' is missing steps: "
                 f"{missing or ['video']}")

    key = os.environ.get("GEMINI_API_KEY", "")
    if "gemini" in args.provider and not key:
        sys.exit("GEMINI_API_KEY env var not set — required for provider gemini/both")

    out_root = Path(args.out or (Path("runs") / time.strftime("run-%Y%m%d-%H%M%S")))
    out_root.mkdir(parents=True, exist_ok=True)

    overall = time.time()
    fallback = False
    provs = ["pcai", "gemini"] if args.provider == "both" else [args.provider]
    for prov in provs:
        out = out_root / prov
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if prov == "pcai":
            timings, artifacts, notes, steps = run_pcai(args.base_url, pack, prompts, args, out)
        else:
            timings, artifacts, notes, steps = run_gemini(key, pack, prompts, args, out, args.base_url)
        if any("fallback" in n or "composite-only" in n for n in notes):
            fallback = True
        summary_p = {"elapsed_s": round(time.time() - t0, 1),
                     "stage_timings_s": timings, "artifacts": artifacts,
                     "notes": notes, "steps": steps}
        (out / "run_summary.json").write_text(json.dumps(summary_p, indent=2), encoding="utf-8")

    top = {"photo": str(args.photo), "prompts": args.prompts,
           "rendered_prompts": prompts, "engine_differences": ENGINE_DIFFS,
           "total_s": round(time.time() - overall, 1), "fallback": fallback,
           "providers": provs}
    (out_root / "run_summary.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    first = provs[0]
    print(json.dumps({"out": str(out_root),
                      "video": str(out_root / first / "final.mp4") if (out_root / first / "final.mp4").exists() else None,
                      "fallback": fallback, "summary": str(out_root / "run_summary.json")}, indent=2))
    sys.exit(2 if fallback else 0)


if __name__ == "__main__":
    main()
