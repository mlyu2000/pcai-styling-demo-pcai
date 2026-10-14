"""PCAI Styling Demo — backend API.

Orchestrates the styling pipelines:
  - Quick generate   : image (+ style ref) -> vision-LLM prompt -> I2V video (MLIS)
  - 3-stage workflow : stage1 (base image -> two sets: base+opposite and style-variant+opposite)
                        -> stage2 (polish the picked set -> 2D stills of both views) -> stage3 video
  - Templates        : save/load a workflow state for one-click regeneration

Adaptive design: NO hardcoded URLs in the program. Every external endpoint and
the model name come from environment variables injected by the Helm chart
(values.yaml), so the same image + chart deploys on any PCAI platform:
  MLIS_VIDEO_GENERATION_URL, MLIS_API_KEY, MLIS_MODEL
  LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
  IMG_ENDPOINT, IMG_MODEL, IMG_API_KEY   (image model, optional — Z-Image-Turbo)
  DOMAIN, MODEL_DISPLAY_NAME, MAX_DURATION_SEC, SUPPORTED_RESOLUTIONS
"""

import base64
import colorsys
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import contextvars
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    REGISTRY,
)
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("styling-demo.backend")


def _require_env(name: str) -> str:
    """Fail fast if a required endpoint is not configured (no hardcoded fallback)."""
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(
            f"required environment variable {name} is not set. "
            f"It must be provided via the Helm chart values.yaml "
            f"(adaptive to the target PCAI platform — no hard-coded URL in the code)."
        )
    return v


# ---------------------------------------------------------------- config
# Required (adaptive, from values.yaml) — no hardcoded URLs.
try:
    MLIS_URL = _require_env("MLIS_VIDEO_GENERATION_URL").rstrip("/")
    LLM_URL = _require_env("LLM_BASE_URL").rstrip("/")
except RuntimeError as e:
    log.error("startup config error: %s", e)
    # Keep the process alive so /api/config can report the missing var; the
    # health endpoint will surface the failure to the operator.
    MLIS_URL = os.environ.get("MLIS_VIDEO_GENERATION_URL", "").rstrip("/")
    LLM_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")

MLIS_MODEL = os.environ.get("MLIS_MODEL", "").strip()
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "Styling Video Model")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-8-27b-int4")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
MLIS_API_KEY = os.environ.get("MLIS_API_KEY", "")

# Optional image model (Z-Image-Turbo via vLLM-Omni) for the 3-stage workflow.
# If unset, the image stages degrade to a clear 503 and the quick-generate path
# still works. Served over the OpenAI-compatible /v1/chat/completions endpoint
# (diffusion params in extra_body, image returned as a base64 data URL).
IMG_ENDPOINT = os.environ.get("IMG_ENDPOINT", "").strip().rstrip("/")
IMG_MODEL = os.environ.get("IMG_MODEL", "z-image").strip()
IMG_API_KEY = os.environ.get("IMG_API_KEY", "")
# stable-diffusion.cpp servers (e.g. Qwen-Image-Edit GGUF) ignore the plain
# strength/steps form fields on /v1/images/edits and instead read native JSON
# params embedded in the prompt as <sd_cpp_extra_args>{...}</sd_cpp_extra_args>.
# Set true in values.yaml when the image model is served by sdcpp-server.
IMG_SD_CPP_EXTRA = os.environ.get("IMG_SD_CPP_EXTRA", "").strip().lower() in ("1", "true", "yes")

# View-rotation strengths are model-dependent: Z-Image-Turbo rotates 180° at
# 0.8 without losing the car (the proven sweet spot), but Qwen-Image-Edit GGUF
# (sdcpp) treats 0.8 as 'keep the face' and only commits to the new view at full
# strength — verified by A/B sweep against the live service (0.8 -> front,
# 1.0 -> rear). Same reason the quick-viz alt-view needs more freedom on sdcpp.
ROT_STRENGTH = 1.0 if IMG_SD_CPP_EXTRA else 0.8
ALT_VIEW_STRENGTH = 0.85 if IMG_SD_CPP_EXTRA else 0.55

# ---------------------------------------------------------------- external API provider
# Optional, per-request, PER-ROLE overrides. Each of the 3 model roles can
# INDEPENDENTLY route to an external endpoint instead of its on-prem default:
#   - brain : prompt agent + style/scene vision  (OpenAI /v1/chat/completions)
#   - i2v   : image -> video                       (MLIS /v1/videos + poll + /content)
#   - image : image-edit / diffusion               (OpenAI /v1/images/edits)
# A role is "external" only if its base_url is set; otherwise it falls back to the
# on-prem default. Roles can be mixed (e.g. external brain + on-prem video). The
# provider is request-scoped (ContextVar) so concurrent requests can use different
# providers without touching the internal call sites. API keys are carried in the
# request only — never stored server-side.
@dataclass(frozen=True)
class RoleEndpoint:
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass(frozen=True)
class ExtProvider:
    brain: RoleEndpoint = field(default_factory=RoleEndpoint)
    i2v: RoleEndpoint = field(default_factory=RoleEndpoint)
    image: RoleEndpoint = field(default_factory=RoleEndpoint)


_ext_provider: contextvars.ContextVar[ExtProvider | None] = contextvars.ContextVar(
    "_EXT_PROVIDER", default=None)


def set_ext_provider(p: ExtProvider | None) -> None:
    _ext_provider.set(p)


def ext_provider() -> ExtProvider | None:
    return _ext_provider.get()


def _brain_endpoint() -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model) for an LLM/vision "brain" call.
    External (this request) if a brain base_url is set; else on-prem LiteLLM,
    honouring an optional on-prem MODEL override (pick another LiteLLM model)."""
    p = _ext_provider.get()
    if p and p.brain.base_url:
        return p.brain.base_url.rstrip("/"), p.brain.api_key, p.brain.model
    if p and p.brain.model:  # on-prem model override
        return LLM_URL, LLM_API_KEY, p.brain.model
    return LLM_URL, LLM_API_KEY, LLM_MODEL


# Back-compat alias for the 30+ existing LLM/vision call sites.
def _llm_endpoint_and_key() -> tuple[str, str, str]:
    return _brain_endpoint()


def _i2v_endpoint() -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model) for the I2V video job. External
    (this request) if configured, else the on-prem MLIS defaults."""
    p = _ext_provider.get()
    if p and p.i2v.base_url:
        return p.i2v.base_url.rstrip("/"), p.i2v.api_key, p.i2v.model
    return MLIS_URL, MLIS_API_KEY, MLIS_MODEL


def _image_endpoint() -> tuple[str, str, str, bool]:
    """Resolve (base_url, api_key, model, is_external) for image-edit calls.
    is_external gates the sdcpp <sd_cpp_extra_args> prompt embed — only the on-prem
    Qwen-Edit GGUF understands it; a standard external OpenAI server does not."""
    p = _ext_provider.get()
    if p and p.image.base_url:
        return (p.image.base_url.rstrip("/"), p.image.api_key, p.image.model, True)
    return (IMG_ENDPOINT, IMG_API_KEY, IMG_MODEL, False)


DOMAIN = os.environ.get("DOMAIN", "")
BUILD_TAG = os.environ.get("BUILD_TAG", "dev").strip() or "dev"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
MAX_IMAGES_MB = int(os.environ.get("MAX_IMAGES_MB", "15"))

# Model capabilities (set by values.yaml based on the deployed model).
MAX_DURATION_SEC = int(os.environ.get("MAX_DURATION_SEC", "120"))
MIN_DURATION_SEC = int(os.environ.get("MIN_DURATION_SEC", "4"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "121"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "9"))
MAX_FPS = int(os.environ.get("MAX_FPS", "24"))
MIN_FPS = int(os.environ.get("MIN_FPS", "4"))
SUPPORTED_RESOLUTIONS = [r.strip() for r in os.environ.get(
    "SUPPORTED_RESOLUTIONS", "1280x704,960x544,832x480").split(",") if r.strip()]
DEFAULT_RESOLUTION = os.environ.get("DEFAULT_RESOLUTION", SUPPORTED_RESOLUTIONS[0])

# Image-model (SDXL) generation caps. 768 (not 1024): the stage-1 UI shows four
# of these in 210px cells, so 768 keeps payloads ~2x smaller and renders crisp;
# still comfortably above Z-Image's 16-divisibility floor.
IMG_MAX_SIDE = int(os.environ.get("IMG_MAX_SIDE", "768"))
IMG_MAX_STEPS = int(os.environ.get("IMG_MAX_STEPS", "50"))

BASELINE_N = int(os.environ.get("BASELINE_N", "10"))
RECENT_WINDOW = 60 * 60  # throughput window: 1h
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------- metrics
GEN_TOTAL = Counter("styling_gen_total", "Total video generations", ["status"])
GEN_LATENCY = Histogram(
    "styling_gen_latency_seconds", "End-to-end generation latency (s)",
    buckets=(5, 15, 30, 60, 120, 300, 600))
STAGE_LATENCY = Histogram(
    "styling_stage_latency_seconds", "Per-stage latency (s)", ["stage"],
    buckets=(0.1, 0.5, 1, 3, 10, 30, 60, 180, 600))
REQUESTS = Counter("styling_requests_total", "HTTP requests", ["path", "status"])
GPU_UTIL = Gauge("styling_gpu_utilization_percent", "GPU utilization % during last job")
VRAM_GIB = Gauge("styling_vram_used_gib", "GPU VRAM used (GiB)")

latency_history: deque[tuple[float, float]] = deque(maxlen=1500)  # bounded: never OOM from history
lock = threading.Lock()
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline.json")


def _load_baseline() -> dict | None:
    try:
        with open(BASELINE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_baseline(b: dict) -> None:
    try:
        with open(BASELINE_FILE, "w") as f:
            json.dump(b, f, indent=2)
    except OSError as e:
        log.warning("could not persist baseline: %s", e)


# ---------------------------------------------------------------- models
class Params(BaseModel):
    style_strength: float = Field(0.7, ge=0.0, le=1.0)
    motion_strength: float = Field(0.5, ge=0.0, le=1.0)
    num_frames: int | None = Field(None, ge=MIN_FRAMES, le=MAX_FRAMES)
    fps: int | None = Field(None, ge=MIN_FPS, le=MAX_FPS)
    duration: int | None = Field(None, ge=1, le=MAX_DURATION_SEC)
    resolution: str | None = Field(None)
    guidance_scale: float | None = Field(None, ge=1.0, le=15.0)
    num_inference_steps: int | None = Field(None, ge=4, le=100)


class RoleOverride(BaseModel):
    """One model role's external endpoint (base_url + api_key + model)."""
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class ProviderOverride(BaseModel):
    """Per-request external API provider — one entry PER MODEL ROLE. Each of the
    3 roles (brain = LLM/vision, i2v = video, image = image-edit) can be routed to
    an external endpoint INDEPENDENTLY; a role left empty stays on-prem.

    Back-compat: a legacy single-endpoint payload `{base_url, api_key, model}` (no
    brain/i2v/image keys) is mapped to the `brain` role only — old clients keep
    working unchanged."""
    brain: RoleOverride | None = None
    i2v: RoleOverride | None = None
    image: RoleOverride | None = None
    # legacy flat fields (single-endpoint) — mapped to brain if no roles present
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    def as_provider(self) -> ExtProvider | None:
        """Build the request-scoped ExtProvider. Returns None if nothing is set —
        i.e. no external role (brain/i2v/image base_url) AND no on-prem brain model
        override — so callers can skip the ContextVar set entirely."""
        brain = self.brain or RoleOverride(
            base_url=self.base_url, api_key=self.api_key, model=self.model)
        i2v = self.i2v or RoleOverride()
        image = self.image or RoleOverride()
        ext = ExtProvider(
            brain=RoleEndpoint(base_url=brain.base_url.strip(),
                               api_key=brain.api_key, model=brain.model.strip()),
            i2v=RoleEndpoint(base_url=i2v.base_url.strip(),
                             api_key=i2v.api_key, model=i2v.model.strip()),
            image=RoleEndpoint(base_url=image.base_url.strip(),
                               api_key=image.api_key, model=image.model.strip()))
        if not (ext.brain.base_url or ext.brain.model
                or ext.i2v.base_url or ext.image.base_url):
            return None
        return ext


def _apply_provider(req) -> str:
    """Set the request-scoped provider from a request's `provider` field
    (back-compat: flat single-endpoint -> brain). Returns a short label for logs:
    which roles are external (plus any on-prem brain model override), or 'on-prem'."""
    p = getattr(req, "provider", None)
    ext = p.as_provider() if p is not None else None
    set_ext_provider(ext)
    if not ext:
        return "on-prem"
    external = [(n, r) for n, r in (("brain", ext.brain), ("i2v", ext.i2v),
                                    ("image", ext.image)) if r.base_url]
    if external:
        return "external[" + ", ".join(f"{n}:{(r.model or r.base_url)}" for n, r in external) + "]"
    # No external roles, but an on-prem brain model override was chosen.
    return f"on-prem[brain model {ext.brain.model}]"


class GenerateRequest(BaseModel):
    subject_image: str
    style_reference: str = ""
    # Quick Video: video direction from a loaded SAVED TEMPLATE (the template's
    # stage-5 scene direction + stage-3 video direction, pre-rendered client-side).
    # When set, the prompt agent is steered to follow it for the uploaded files.
    prompt_override: str = ""
    parameters: Params = Params()
    provider: ProviderOverride | None = None


class StageRequest(BaseModel):
    # 1 = base+opposite sets, 2 = polish picked set, 3 = video,
    # 4 = AI base sketch (front+rear+colorways), 5 = compose scene + quick viz.
    stage: int = Field(..., ge=1, le=5)
    # base64 (data: prefix tolerated). stage1/4: image. stage2/5: design. stage5: scene.
    image: str = ""
    design_image: str = ""
    reference_image: str = ""
    scene_image: str = ""                        # stage5: scene photo (REF 2); alias of reference_image
    requirement: str = ""
    colorways: int = Field(2, ge=1, le=4)        # stage4: number of colorway pairs
    color_targets: list[str] | None = None       # optional explicit colors (default contrast map)
    use_mock_design: bool = False                # stage5: simulate the Photoshop pass
    use_mock_scene: bool = False                 # stage5: use the bundled sample scene
    view_angle: str = "side"                     # stage5: "side" | "3/4 front" | "rear"
    # stage5 integration mode: "scene-gen" (default when car_photo is provided)
    # = single-ref i2i scene generation from the ORIGINAL vehicle photo — the
    # same working path as the pipeline client (grounded, contact shadow, no
    # paste rectangle). "composite" = deterministic paste + harmonize (fallback).
    mode: str = ""                              # stage5: "scene-gen" | "composite"
    car_photo: str = ""                         # stage5 scene-gen: original vehicle reference (single ref)
    width: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    height: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    resolution: str | None = Field(None)
    strength: float = Field(0.6, ge=0.0, le=1.0)
    seed: int | None = None
    # Agent-skill prompt overrides: when set, the stage uses these instead of the
    # built-in tuned prompts (prompt-pack from the styling-video-automation skill).
    prompt_override: str = ""        # stage3 video / stage4 front sketch / stage5 main
    prompt_override_rear: str = ""   # stage4 rear sketch (180deg)
    prompt_override_alt: str = ""    # stage5 another_view
    provider: ProviderOverride | None = None


class TemplateRequest(BaseModel):
    name: str
    description: str = ""
    stage1_images: dict = {}
    stage1_requirement: str = ""
    stage2_images: dict = {}
    stage2_requirement: str = ""
    stage3_image: str = ""
    stage3_requirement: str = ""
    # IDD 6-stage additions (v23): stage4 = AI base sketch sets, stage5 =
    # scene compose + quick viz; the designer gate carries its own fields.
    stage4_images: dict = {}
    stage4_requirement: str = ""
    stage5_images: dict = {}
    stage5_requirement: str = ""
    design_image: str = ""
    use_mock_design: bool = False
    scene_image: str = ""
    use_mock_scene: bool = False
    view_angle: str = "side"
    final: dict = {}


# ------------------------------------------------------------------ per-step flow API (v29)
# One endpoint per demo step. Each step is PROMPT-DRIVEN with exactly the
# declared inputs — the same semantics as the Gemini reference stream, so the
# two engines are functionally equal by construction (see
# skill/pcai-styling-video-automation for the reconciliation). Every step
# reports which path ran (model vs deterministic fallback) via `mode`.
class FlowSketchRequest(BaseModel):
    image: str
    prompt_front: str = ""       # empty -> built-in tuned sketch prompt
    prompt_rear: str = ""        # empty -> tuned rear (anti-top-down, color-anchored)
    requirement: str = ""
    width: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    height: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    seed: int | None = None


class FlowCombineRequest(BaseModel):
    design_image: str
    scene_image: str = ""
    prompt: str = ""
    requirement: str = ""
    use_mock_scene: bool = False
    mode: str | None = Field(
        None,
        description="combine integrator: 'composite+harmonize' (default — "
                    "deterministic geometry-locked paste + single-ref harmonize) "
                    "or 'scene-gen' (single-ref i2i scene GENERATION: the car photo "
                    "is the only reference and the scene is generated from "
                    "scene_prompt/requirement; falls back to composite on model "
                    "failure). scene-gen is the visually integrated path — a "
                    "deterministic paste of a studio product photo reads as a "
                    "floating rectangle even with the geometry fix.")
    scene_prompt: str | None = Field(
        None, description="scene-gen only: natural-language description of the "
                          "target scene (road, weather, time, lighting, props). "
                          "If omitted, `requirement` is used as the scene "
                          "description. Ignored in composite+harmonize mode.")
    depth: float = Field(0.5, ge=0.15, le=0.9,
                         description="Distance of the car on the road plane: "
                                     "0.15 = close (large), 0.9 = near horizon "
                                     "(small). 0.5 = mid-ground. (composite "
                                     "path only; scene-gen is perspective-free.)")
    width: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    height: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    seed: int | None = None


class FlowViewRequest(BaseModel):
    image: str
    prompt: str = ""
    view_angle: str = "side"
    requirement: str = ""
    width: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    height: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    seed: int | None = None


class FlowRecolorRequest(BaseModel):
    image: str
    prompt: str = ""
    color: str = ""
    requirement: str = ""
    width: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    height: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    seed: int | None = None


class FlowVideoRequest(BaseModel):
    image: str
    prompt: str = ""
    requirement: str = ""
    resolution: str | None = None
    style_strength: float = Field(0.7, ge=0.0, le=1.0)
    motion_strength: float = Field(0.6, ge=0.0, le=1.0)


class FlowSceneCleanRequest(BaseModel):
    image: str
    prompt: str = ""
    requirement: str = ""
    width: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    height: int | None = Field(None, ge=256, le=IMG_MAX_SIDE)
    seed: int | None = None
    strength: float = Field(0.7, ge=0.0, le=1.0)


def _decode_b64_image(data: str, max_mb: int = MAX_IMAGES_MB) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data, validate=False)
    if len(raw) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"image exceeds {max_mb}MB limit")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")
    return img


def _b64_image_to_pil(data: str) -> Image.Image | None:
    if not data:
        return None
    try:
        return _decode_b64_image(data)
    except HTTPException:
        return None


def _img_to_b64_jpeg(img: Image.Image, quality: int = 88, max_dim: int = 1280) -> str:
    img = img.copy()
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _img_to_pil_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------- MLIS I2V helpers
def _mlis_headers(api_key: str | None = None) -> dict:
    key = api_key if api_key is not None else MLIS_API_KEY
    return {"Authorization": f"Bearer {key}"} if key else {}


async def _create_video_job(client: httpx.AsyncClient, prompt: str, subject: Image.Image,
                           p: Params, resolution: str,
                           base_url: str = MLIS_URL, model: str = MLIS_MODEL) -> tuple[str, float]:
    """POST /v1/videos (multipart). Returns (job_id, guidance_scale_used)."""
    guidance = p.guidance_scale if p.guidance_scale is not None else 3.0 + 4.5 * p.motion_strength
    steps = p.num_inference_steps or 50
    try:
        rw, rh = resolution.lower().split("x")
        w, h = int(rw), int(rh)
    except Exception:
        w, h = 1280, 704
    jpeg = io.BytesIO()
    subject.resize((w, h), Image.LANCZOS).save(jpeg, format="JPEG", quality=90)
    jpeg.seek(0)

    form = {
        "prompt": prompt,
        "model": model or MLIS_MODEL,
        "image_reference": json.dumps(
            {"image_url": f"data:image/jpeg;base64,{base64.b64encode(jpeg.getvalue()).decode()}"}),
        "width": str(w),
        "height": str(h),
        "num_frames": str(p.num_frames or 121),
        "fps": str(p.fps or 8),
        "num_inference_steps": str(steps),
        "guidance_scale": f"{guidance:.1f}",
        "flow_shift": "5.0",
    }
    r = await client.post(f"{base_url}/v1/videos", data=form)
    if r.status_code >= 400:
        raise RuntimeError(f"I2V /v1/videos {r.status_code}: {r.text[:500]}")
    job = r.json()
    return str(job.get("id") or job.get("video_id")), guidance


async def _poll_job(client: httpx.AsyncClient, job_id: str, base_url: str = MLIS_URL,
                    timeout_s: float = 1800) -> dict:
    t0 = time.time()
    delay = 2.0
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"MLIS job {job_id} timed out after {timeout_s}s")
        r = await client.get(f"{base_url}/v1/videos/{job_id}")
        if r.status_code == 404:
            raise RuntimeError(f"MLIS job {job_id} not found: {r.text[:200]}")
        r.raise_for_status()
        job = r.json()
        status = str(job.get("status") or job.get("state") or "").lower()
        if status in ("completed", "succeeded", "success", "ready"):
            return job
        if status in ("failed", "error", "cancelled", "canceled"):
            raise RuntimeError(f"MLIS job {job_id} failed: {json.dumps(job)[:500]}")
        await _sleep(delay)
        delay = min(delay * 1.5, 15.0)


async def _sleep(s: float) -> None:
    import asyncio
    await asyncio.sleep(s)


def _faststart_mp4(mp4: bytes) -> bytes:
    """Move the `moov` atom to the front of an MP4 (faststart) so browsers can
    start <video> playback from a top-of-file Range request.

    MLIS i2v output is written with `moov` at the END (mdat first). Browsers
    fetch <video> via Range from offset 0; without the moov index up front the
    tile cannot begin decoding and shows as an empty/unplayable video even though
    the file is valid and the URL serves 206. Re-atomizes in pure stdlib:
    [ftyp,free,...] + moov + mdat, patching every stco/co64 chunk offset in moov
    by the delta mdat moved (delta == len(moov)). Recursive box-walk descends
    into container boxes (moov>trak>mdia>minf>stbl) to find stco. Never raises:
    any unexpected layout returns the input unchanged.
    """
    n = len(mp4)

    def parse_boxes(start, end, buf):
        i = start
        while i + 8 <= end:
            size = int.from_bytes(buf[i:i + 4], "big")
            typ = bytes(buf[i + 4:i + 8])  # bytes (hashable), not a bytearray slice
            if size == 1:
                size = int.from_bytes(buf[i + 8:i + 16], "big")
            elif size == 0:
                size = end - i
            if size < 8 or i + size > end:
                return
            yield typ, i, size
            i += size

    top = list(parse_boxes(0, n, mp4))
    types = [t for t, _, _ in top]
    if b"moov" not in types or b"mdat" not in types:
        return mp4
    moov = next(a for a in top if a[0] == b"moov")
    mdat = next(a for a in top if a[0] == b"mdat")
    if moov[1] < mdat[1]:
        return mp4  # already faststart
    _, moov_off, moov_size = moov
    _, mdat_off, mdat_size = mdat
    prefix = mp4[:mdat_off]                       # ftyp/free/etc
    moov_bytes = bytearray(mp4[moov_off:moov_off + moov_size])
    mdat_bytes = mp4[mdat_off:mdat_off + mdat_size]
    delta = len(moov_bytes)                       # mdat shifts forward by len(moov)

    CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl",
                  b"udta", b"ilst", b"edts"}

    def patch(start, end):
        for typ, i, size in parse_boxes(start, end, moov_bytes):
            if typ in (b"stco", b"co64"):
                cnt = int.from_bytes(moov_bytes[i + 12:i + 16], "big")
                w = 4 if typ == b"stco" else 8
                k = i + 16
                for _ in range(cnt):
                    v = int.from_bytes(moov_bytes[k:k + w], "big") + delta
                    moov_bytes[k:k + w] = v.to_bytes(w, "big")
                    k += w
            elif typ in CONTAINERS:
                patch(i + 8, i + size)            # descend into children

    try:
        patch(8, len(moov_bytes))                 # moov's direct children
    except Exception:
        return mp4
    return prefix + bytes(moov_bytes) + mdat_bytes


async def generate_video(prompt: str, subject: Image.Image, p: Params,
                         resolution: str) -> tuple[bytes, float, dict]:
    """Create job, poll, download MP4. Returns (mp4_bytes, guidance, job_meta).

    The I2V endpoint is resolved per-request via _i2v_endpoint() — external if the
    request set an i2v provider, else the on-prem MLIS default. The request/response
    shape is identical either way (MLIS /v1/videos + poll + /content)."""
    base_url, api_key, model = _i2v_endpoint()
    t0 = time.time()
    timeout = httpx.Timeout(60.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout, headers=_mlis_headers(api_key)) as client:
        job_id, guidance = await _create_video_job(
            client, prompt, subject, p, resolution, base_url=base_url, model=model)
        log.info("I2V job created: %s (guidance=%.1f, endpoint=%s)",
                 job_id, guidance, base_url)
        job = await _poll_job(client, job_id, base_url=base_url)
        r = await client.get(f"{base_url}/v1/videos/{job_id}/content")
        r.raise_for_status()
        mp4 = r.content
        if not mp4:
            raise RuntimeError(f"empty video content for job {job_id}")
        try:
            mp4 = _faststart_mp4(mp4)  # browsers need moov up front to play
        except Exception as e:
            log.warning("faststart re-atomize failed (%s); serving as-is", e)
    gen_ms = (time.time() - t0) * 1000
    meta = job.get("usage") or job.get("metadata") or {}
    return mp4, guidance, {"job_id": job_id, "gen_ms": gen_ms, "usage": meta}


# ---------------------------------------------------------------- vision LLM
SYSTEM_PROMPT = (
    "You are a senior automotive styling art director on an on-prem AI platform. "
    "You receive a SUBJECT image (a vehicle or car render) and a STYLE REFERENCE image "
    "(artistic style, scene, lighting or finish). Write ONE cinematic image-to-video prompt "
    "(max 40 words) that animates the subject in the style of the reference: describe motion "
    "(camera movement, the car's motion), lighting, color grade, and environment. "
    "Do not mention the word 'style' or 'reference'. Output ONLY the prompt text, no preamble."
)


async def run_prompt_agent(subject: Image.Image, style_ref: Image.Image | None,
                           style_strength: float, override: str = "") -> str:
    t0 = time.time()
    # A loaded SAVED TEMPLATE can carry the customer's exact scene + video
    # direction (stage-5 requirement + stage-3 requirement, pre-rendered on the
    # client). When present, instruct the agent to follow it verbatim for the
    # uploaded files — this is what makes Quick Video "follow the template".
    override = (override or "").strip()
    steer = (
        "A saved styling template was selected by the user. Its intended scene and "
        "video direction is below — FOLLOW IT as the primary creative direction for "
        "this render (keep its scene, mood, lighting, camera movement and any "
        "'strictly keep the car design' guardrails), adapting only the car that "
        "appears in the attached subject image.\nTEMPLATE DIRECTION:\n" + override
    ) if override else (
        "Describe the scene and camera direction for a cinematic car render."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": (
                f"Subject image (vehicle) attached first, style reference image attached second. "
                f"Style strength: {style_strength:.1f} (higher = follow the reference more closely). "
                f"{steer}")},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64_jpeg(subject)}"}},
            *([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64_jpeg(style_ref)}"}}]
               if style_ref is not None else []),
        ]},
    ]
    try:
        ext_url, ext_key, ext_model = _llm_endpoint_and_key()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{ext_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {ext_key}"},
                json={"model": ext_model, "messages": messages, "max_tokens": 2048,
                      "temperature": 0.8},
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            prompt = (msg.get("content") or msg.get("reasoning_content") or "").strip().strip('"')
            prompt = re.sub(r"\s+", " ", prompt)
            if not prompt:
                raise ValueError("empty prompt from LLM")
            log.info("prompt agent: %s", prompt)
            return prompt
    except Exception as e:
        log.warning("prompt agent failed (%s); using fallback prompt", e)
        return ("A sleek new car glides slowly forward on a polished studio floor, cinematic "
                "camera push-in, dramatic rim lighting, soft reflections, smooth steady motion.")
    finally:
        STAGE_LATENCY.labels(stage="prompt_agent").observe(time.time() - t0)


STYLE_DESC_PROMPT = (
    "You are given a STYLE REFERENCE image. Describe ONLY its visual style in one "
    "compact phrase (max 20 words) that can steer a car render: the color palette, "
    "lighting/mood, surface material/finish and overall artistic style (e.g. 'moody "
    "teal-and-orange studio look, matte carbon-fibre panels, dramatic rim light'). "
    "Do NOT describe the car or subject. Output ONLY the phrase, no preamble."
)


async def _describe_style(ref: Image.Image) -> str:
    """Use the vision LLM to turn a style-reference image into a short style
    descriptor (palette/lighting/material/mood) that can be folded into a render
    prompt. Returns '' if the LLM is unavailable or fails."""
    ext_url, ext_key, ext_model = _llm_endpoint_and_key()
    if not ext_url:
        return ""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{ext_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {ext_key}"},
                json={
                    "model": ext_model,
                    "messages": [
                        {"role": "system", "content": STYLE_DESC_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Style reference image attached."},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{_img_to_b64_jpeg(ref)}"}},
                        ]},
                    ],
                    "max_tokens": 200, "temperature": 0.4,
                },
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            desc = (msg.get("content") or msg.get("reasoning_content") or "").strip().strip('"')
            desc = re.sub(r"\s+", " ", desc)
            log.info("style descriptor: %s", desc)
            return desc
    except Exception as e:
        log.warning("style descriptor failed (%s); refining without style ref", e)
        return ""


# ---------------------------------------------------------------- Z-Image (vLLM-Omni) image model
def _img_headers(api_key: str | None = None) -> dict:
    key = api_key if api_key is not None else IMG_API_KEY
    return {"Authorization": f"Bearer {key}"} if key else {}


def _img_extract_data_url(content) -> str | None:
    """Pull a base64 data:image URL out of a chat-completion message content
    (vLLM-Omni diffusion returns the image as data:image/...;type=base64,<b64>
    inside choices[0].message.content — a list of items or a plain string)."""
    if isinstance(content, str):
        return content if content.startswith("data:image") else None
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                url = (item.get("image_url") or {}).get("url") or item.get("url")
                if url and url.startswith("data:image"):
                    return url
            else:
                url = getattr(item, "image_url", None)
                if url:
                    url = getattr(url, "url", None)
                    if url and url.startswith("data:image"):
                        return url
    return None


def _dims_from_image(img: Image.Image, req_w: int | None = None,
                     req_h: int | None = None,
                     max_side: int = IMG_MAX_SIDE) -> tuple[int, int]:
    """Output (w,h) that PRESERVES the input image's aspect ratio (capped at
    max_side, aligned to 8). Forcing a square (e.g. 1024x1024) on a landscape
    car photo makes the model squash/distort it (bloated body, oversized
    wheels), so we match the reference's own proportions instead of the
    requested dims. req_w/req_h act only as an optional cap, not a target."""
    iw, ih = img.size
    cap = max_side
    if req_w:
        cap = min(cap, req_w)
    if req_h:
        cap = min(cap, req_h)
    scale = cap / max(iw, ih)
    w = int(round(iw * scale))
    h = int(round(ih * scale))
    # Z-Image's /v1/images/edits requires BOTH dims divisible by 16.
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    return w, h


async def _img_generate(prompt: str, image_pil: Image.Image | None,
                        width: int, height: int, strength: float = 0.6,
                        steps: int = 8, guidance: float = 0.0,
                        seed: int | None = None,
                        negative: str | None = None) -> Image.Image:
    """Text-to-image (image_pil None) or image-to-image (image_pil set) via
    Z-Image-Turbo served by vLLM-Omni.

    t2i hits POST /v1/chat/completions (text-only message, diffusion params in
    extra_body, result as a base64 data:image URL in message content).

    i2i hits POST /v1/images/edits (multipart form, OpenAI-DALL-E-compatible,
    result as b64_json in data[0]). The /v1/chat/completions path does NOT
    honour `strength` for Z-Image (vLLM-Omni v0.28.x never forwards it into
    OmniDiffusionSamplingParams there), so img2img via that endpoint always
    returns a near-identical copy of the input; /v1/images/edits does apply
    strength to the denoise start timestep, which is what makes editing work.

    Z-Image-Turbo is a CFG-free distilled model: it is trained with
    guidance_scale=0 and runs fast in ~8 steps, so those are the defaults and
    the guidance argument is not forwarded (honouring a non-zero caller value
    would degrade quality and double the latency).
    """
    # Resolve the image endpoint per-request: external (this request) if the user
    # configured the image role, else the on-prem default. is_external gates the
    # sdcpp <sd_cpp_extra_args> prompt embed (only the on-prem Qwen-Edit GGUF reads it).
    img_base, img_key, img_model, img_external = _image_endpoint()
    if not img_base:
        raise HTTPException(
            503,
            "Image model is not configured (IMG_ENDPOINT unset and no external image "
            "provider). Set it in values.yaml or pick an external image-edit model in "
            "the Provider tab; deterministic fallbacks keep the demo running.")
    steps = min(max(int(steps), 1), IMG_MAX_STEPS)
    width = min(max(int(width), 256), IMG_MAX_SIDE)
    height = min(max(int(height), 256), IMG_MAX_SIDE)
    # Z-Image's /v1/images/edits requires BOTH dims divisible by 16 (8 is NOT
    # enough — height 520 returns a 500). Align to 16 so every i2i call is valid.
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    t0 = time.time()
    timeout = httpx.Timeout(30.0, read=300.0)
    gscale = guidance if guidance and guidance > 0 else 0.0  # CFG-free default

    # ---------------- text-to-image: /v1/chat/completions ----------------
    # (sd-cpp servers don't expose that route — IMG_SD_CPP_EXTRA diverts them to
    # POST /v1/images/generations instead, which works the same for us.)
    if image_pil is None and IMG_SD_CPP_EXTRA:
        body = {"prompt": prompt, "size": f"{width}x{height}", "n": 1,
                "num_inference_steps": steps, "guidance_scale": gscale or 4.0,
                "response_format": "b64_json"}
        if seed is not None:
            body["seed"] = int(seed)
        if negative and negative.strip():
            body["negative_prompt"] = negative.strip()
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=_img_headers(img_key)) as client:
                r = await client.post(f"{img_base}/v1/images/generations", json=body)
                if r.status_code != 200:
                    raise RuntimeError(f"sd-cpp gens {r.status_code}: {r.text[:300]}")
                b64_payload = r.json()["data"][0]["b64_json"]
                img = Image.open(io.BytesIO(base64.b64decode(b64_payload))).convert("RGB")
                STAGE_LATENCY.labels(stage="image_gen").observe(time.time() - t0)
                return img
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as e:
            raise HTTPException(502, f"image model request failed: {e}")
    if image_pil is None:
        body = {
            "model": IMG_MODEL or "z-image",
            "messages": [{"role": "user", "content": prompt}],
            "extra_body": {
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                # Z-Image-Turbo is a distill model: it is trained to run with
                # guidance_scale=0 (CFG-free). Honour a caller override only if > 0.
                "guidance_scale": gscale,
                "seed": int(seed) if seed is not None else None,
            },
        }
        if negative and negative.strip():
            body["extra_body"]["negative_prompt"] = negative.strip()
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=_img_headers(img_key)) as client:
                r = await client.post(f"{img_base}/v1/chat/completions", json=body)
                if r.status_code != 200:
                    raise RuntimeError(f"Z-Image {r.status_code}: {r.text[:300]}")
                data = r.json()
                try:
                    msg_content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError(f"Z-Image: no image in response: {json.dumps(data)[:300]}")
                data_url = _img_extract_data_url(msg_content)
                if not data_url:
                    raise RuntimeError(f"Z-Image: no data:image URL in content: {str(msg_content)[:300]}")
                b64_payload = data_url.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64_payload))).convert("RGB")
                STAGE_LATENCY.labels(stage="image_gen").observe(time.time() - t0)
                return img
        except httpx.HTTPError as e:
            raise HTTPException(502, f"image model request failed: {e}")

    # ---------------- image-to-image: /v1/images/edits (multipart) --------
    # On-prem the model is selected by the endpoint (single-model server), so no
    # model field is sent. An EXTERNAL OpenAI server may need the model name, so we
    # send it only when one was provided. strength controls the denoise start
    # timestep (0 = keep the input, 1 = full re-draw) — only /v1/images/edits
    # applies it (see the docstring above for why /v1/chat/completions can't i2i).
    #
    # SD-cpp servers (Qwen-Image-Edit GGUF via stable-diffusion.cpp) IGNORE the
    # plain strength/steps form fields — they read native params from a JSON payload
    # embedded in the prompt as <sd_cpp_extra_args>{...}</sd_cpp_extra_args>. The
    # embed is toggled by IMG_SD_CPP_EXTRA=1 in values AND only for the ON-PREM
    # endpoint: a standard external OpenAI server does not understand the embed, so
    # it is skipped when the image role is external (plain form fields are used).
    eff_prompt = prompt
    if IMG_SD_CPP_EXTRA and not img_external:
        extra = {"strength": float(min(max(strength, 0.0), 1.0)),
                 "sample_steps": steps}
        if seed is not None:
            extra["seed"] = int(seed)
        eff_prompt = f"{prompt}<sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"
    png_bytes = _img_to_pil_bytes(image_pil, "PNG")
    form = {
        "prompt": eff_prompt,
        "size": f"{width}x{height}",
        "num_inference_steps": str(steps),
        "strength": str(float(min(max(strength, 0.0), 1.0))),
    }
    if img_model:
        form["model"] = img_model
    # vLLM-Omni: send the (CFG-free 0) guidance explicitly. sd-cpp: omit unless
    # a non-zero value was requested, so the server's --cfg-scale default holds.
    if gscale > 0 or not (IMG_SD_CPP_EXTRA and not img_external):
        form["guidance_scale"] = str(gscale)
    if seed is not None:
        form["seed"] = str(int(seed))
    if negative and negative.strip():
        form["negative_prompt"] = negative.strip()
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_img_headers(img_key)) as client:
            r = await client.post(
                f"{img_base}/v1/images/edits",
                data=form,
                files={"image": ("image.png", png_bytes, "image/png")},
            )
            if r.status_code != 200:
                raise RuntimeError(f"Z-Image edits {r.status_code}: {r.text[:300]}")
            data = r.json()
            try:
                b64_payload = data["data"][0]["b64_json"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"Z-Image edits: no b64_json in response: {json.dumps(data)[:300]}")
            img = Image.open(io.BytesIO(base64.b64decode(b64_payload))).convert("RGB")
            STAGE_LATENCY.labels(stage="image_gen").observe(time.time() - t0)
            return img
    except httpx.HTTPError as e:
        raise HTTPException(502, f"image model request failed: {e}")


# ---------------------------------------------------------------- app
app = FastAPI(title="PCAI Styling Demo Backend", version="2.0.0")


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/api", "/healthz", "/readyz")):
        # Bound the label space: use the FastAPI route template, NOT the raw
        # URL path — /api/results/<job>.mp4 is unique per video and would add
        # an unbounded new Prometheus series (time series) per request,
        # leaking memory until the pod OOMs.
        p = request.url.path
        route = getattr(request, "route", None)
        p = route.path if route is not None and getattr(route, "path", None) else (
            "/api/results/{id}.mp4" if p.startswith("/api/results/") and p.endswith(".mp4")
            else p)
        REQUESTS.labels(path=p, status=str(response.status_code)).inc()
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not MLIS_URL:
        return JSONResponse({"ready": False, "reason": "MLIS_VIDEO_GENERATION_URL not set"},
                            status_code=503)
    try:
        async with httpx.AsyncClient(timeout=5, headers=_mlis_headers()) as client:
            r = await client.get(f"{MLIS_URL}/v1/models")
            return {"ready": r.status_code < 500, "mlis": r.status_code}
    except Exception as e:
        return JSONResponse({"ready": False, "mlis": str(e)[:200]}, status_code=503)


@app.get("/api/config")
async def api_config():
    """Dynamic config for the frontend — no hardcoded values in the UI."""
    return {
        "model": MLIS_MODEL,
        "modelDisplayName": MODEL_DISPLAY_NAME,
        "brainModel": LLM_MODEL,
        "imageModel": IMG_MODEL if IMG_ENDPOINT else "",
        "imageModelEnabled": bool(IMG_ENDPOINT),
        "domain": DOMAIN,
        "maxDurationSec": MAX_DURATION_SEC,
        "minDurationSec": MIN_DURATION_SEC,
        "supportedResolutions": SUPPORTED_RESOLUTIONS,
        "defaultResolution": DEFAULT_RESOLUTION,
        "minFrames": MIN_FRAMES,
        "maxFrames": MAX_FRAMES,
        "minFps": MIN_FPS,
        "maxFps": MAX_FPS,
        "mockSceneAvailable": MOCK_SCENE_FILE.exists(),
        "version": "2.0.0",
        "build": BUILD_TAG,
    }


class ProviderTestRequest(BaseModel):
    base_url: str
    api_key: str = ""
    model: str = ""
    # Which role to test: "brain" (default — chat completion), "i2v" (MLIS
    # /v1/videos reachability), or "image" (image-edit). Lets the UI validate each
    # per-role endpoint independently.
    role: str = "brain"


@app.post("/api/providers/test")
async def test_provider(req: ProviderTestRequest):
    """Live connectivity check for a per-role endpoint. Reports success/failure so
    the UI can validate URL + key before the user commits. Role-dependent probe:
      - brain : minimal chat completion to <base>/v1/chat/completions.
      - i2v   : reachability via <base>/v1/models (a real I2V job is slow/expensive,
                so we only confirm the service answers with a model list / 2xx).
      - image : reachability via <base>/v1/models (same rationale as i2v)."""
    base = (req.base_url or "").strip().rstrip("/")
    model = (req.model or "").strip()
    role = (req.role or "brain").strip().lower()
    if not base:
        raise HTTPException(400, "base_url is required")
    headers = {"Authorization": f"Bearer {req.api_key}"} if req.api_key else {}
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if role == "brain":
                r = await client.post(
                    f"{base}/v1/chat/completions", headers=headers,
                    json={"model": model,
                          "messages": [{"role": "user",
                                        "content": "Reply with the single word: ok"}],
                          "max_tokens": 8},
                )
                ok = r.status_code == 200
                sample = ""
                if ok:
                    try:
                        data = r.json()
                        sample = (data["choices"][0]["message"].get("content")
                                  or data["choices"][0]["message"].get("reasoning_content") or "")
                    except Exception:
                        sample = ""
            else:  # i2v / image — reachability via /v1/models
                r = await client.get(f"{base}/v1/models", headers=headers)
                ok = r.status_code == 200
                sample = ""
                if ok:
                    try:
                        ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)]
                        sample = ", ".join(str(i) for i in ids[:5])
                    except Exception:
                        pass
    except httpx.HTTPError as e:
        return {"ok": False, "role": role, "status": "unreachable",
                "error": f"could not reach {base}: {type(e).__name__}: {e}"}
    ms = int((time.time() - t0) * 1000)
    if not ok:
        return {"ok": False, "role": role, "status": r.status_code,
                "error": (r.text or "")[:400]}
    return {"ok": True, "role": role, "status": 200, "model": model,
            "latency_ms": ms, "sample": (sample or "").strip()[:200]}


@app.get("/api/providers/models")
async def list_provider_models(base_url: str, api_key: str = ""):
    """Server-side model discovery for a base URL (no browser CORS needed). Calls
    <base>/v1/models and returns the model ids so the UI can fill a dropdown.
    Best-effort: on any failure returns {ok: false, models: []} and the UI falls
    back to manual entry. Also used to enumerate the on-prem LiteLLM models."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(400, "base_url is required")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{base}/v1/models", headers=headers)
    except httpx.HTTPError as e:
        return {"ok": False, "models": [], "count": 0,
                "error": f"could not reach {base}: {type(e).__name__}: {e}"}
    ms = int((time.time() - t0) * 1000)
    if r.status_code != 200:
        return {"ok": False, "models": [], "count": 0, "status": r.status_code,
                "error": (r.text or "")[:300]}
    try:
        data = r.json()
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception as e:
        return {"ok": False, "models": [], "count": 0,
                "error": f"unparseable model list: {type(e).__name__}: {e}"}
    return {"ok": True, "models": sorted(ids), "count": len(ids), "latency_ms": ms}


@app.get("/api/providers/onprem-models")
async def list_onprem_models():
    """Model discovery for the ON-PREM brain (LiteLLM proxy). Server-side call to
    <LLM_URL>/v1/models so the UI can offer the on-prem model list (the browser
    can't know the proxy URL). Best-effort: {models: []} on failure."""
    if not LLM_URL:
        return {"ok": False, "models": [], "count": 0, "error": "LLM_BASE_URL not set"}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{LLM_URL}/v1/models", headers=headers)
    except httpx.HTTPError as e:
        return {"ok": False, "models": [], "count": 0,
                "error": f"could not reach {LLM_URL}: {type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"ok": False, "models": [], "count": 0, "status": r.status_code}
    try:
        ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception:
        return {"ok": False, "models": [], "count": 0, "error": "unparseable model list"}
    return {"ok": True, "models": sorted(ids), "count": len(ids), "default": LLM_MODEL}


def _resolve_frames_and_fps(p: Params) -> tuple[int, int]:
    """Resolve num_frames and fps, honoring duration (clamped to model limits)."""
    fps = p.fps or 8
    fps = min(max(fps, MIN_FPS), MAX_FPS)
    if p.num_frames is not None:
        frames = min(max(p.num_frames, MIN_FRAMES), MAX_FRAMES)
        return frames, fps
    if p.duration is not None:
        frames = int(round(p.duration * fps))
        frames = min(max(frames, MIN_FRAMES), MAX_FRAMES)
        return frames, fps
    return 121, fps


# ---------------------------------------------------------------- quick generate
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not MLIS_URL:
        raise HTTPException(503, "MLIS_VIDEO_GENERATION_URL not configured")
    job_id = uuid.uuid4().hex
    t_start = time.time()

    # Per-request external provider (PER-ROLE: brain / i2v / image — each may be
    # external independently). Sets the request-scoped ContextVar used by the
    # resolvers (_brain_endpoint / _i2v_endpoint / _image_endpoint).
    prov_label = _apply_provider(req)
    if prov_label != "on-prem":
        log.info("generate %s: external provider -> %s", job_id, prov_label)

    subject = _decode_b64_image(req.subject_image)
    style_ref = _decode_b64_image(req.style_reference) if req.style_reference else None

    resolution = (req.parameters.resolution or DEFAULT_RESOLUTION).lower()
    if resolution not in [r.lower() for r in SUPPORTED_RESOLUTIONS]:
        resolution = DEFAULT_RESOLUTION
    frames, fps = _resolve_frames_and_fps(req.parameters)
    log.info("generate %s: frames=%s fps=%s res=%s style=%.1f motion=%.1f",
             job_id, frames, fps, resolution,
             req.parameters.style_strength, req.parameters.motion_strength)

    t_prompt = time.time()
    prompt = await run_prompt_agent(subject, style_ref, req.parameters.style_strength,
                                    req.prompt_override)
    prompt_agent_ms = (time.time() - t_prompt) * 1000

    p = req.parameters.model_copy(update={"num_frames": frames, "fps": fps})
    mp4, guidance, gen_meta = await generate_video(prompt, subject, p, resolution)

    out_path = os.path.join(RESULTS_DIR, f"{job_id}.mp4")
    with open(out_path, "wb") as f:
        f.write(mp4)

    total_ms = (time.time() - t_start) * 1000
    GEN_TOTAL.labels(status="success").inc()
    GEN_LATENCY.observe(total_ms / 1000)
    STAGE_LATENCY.labels(stage="video_generation").observe(gen_meta["gen_ms"] / 1000)
    await _update_gpu_gauges()

    with lock:
        latency_history.append((time.time(), total_ms / 1000))
        if _load_baseline() is None and len(latency_history) >= BASELINE_N:
            _save_baseline(_make_baseline())

    return {
        "id": job_id,
        "video_url": f"/api/results/{job_id}.mp4",
        "prompt": prompt,
        "model": MLIS_MODEL,
        "provider": prov_label,
        "parameters": {
            "num_frames": frames,
            "fps": fps,
            "resolution": resolution,
            "duration_sec": round(frames / fps, 2),
            "guidance_scale": round(guidance, 2),
            "num_inference_steps": req.parameters.num_inference_steps or 30,
            "style_strength": req.parameters.style_strength,
            "motion_strength": req.parameters.motion_strength,
        },
        "metrics": {
            "total_latency_ms": round(total_ms),
            "stage_latencies": {
                "prompt_agent_ms": round(prompt_agent_ms),
                "video_generation_ms": round(gen_meta["gen_ms"]),
                "compilation_ms": round(total_ms - gen_meta["gen_ms"]),
            },
            "gpu_utilization": _gpu_gauge_snapshot(),
        },
    }


def _make_baseline() -> dict:
    with lock:
        lat = [v for _, v in latency_history]
        now = time.time()
        rpm = sum(1 for ts, _ in latency_history if now - ts <= RECENT_WINDOW) / (RECENT_WINDOW / 60)
        return {
            "throughput_rpm": round(rpm, 3),
            "avg_latency_ms": round(sum(lat) / len(lat) * 1000) if lat else None,
            "requests": len(lat),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }


async def _update_gpu_gauges() -> None:
    try:
        async with httpx.AsyncClient(timeout=5, headers=_mlis_headers()) as client:
            r = await client.get(f"{MLIS_URL}/metrics")
            m = re.search(r"DCGM_FI_DEV_GPU_UTIL\s+([\d.]+)", r.text)
            if m:
                GPU_UTIL.set(float(m.group(1)))
            m = re.search(r"DCGM_FI_DEV_FB_USED\s+([\d.]+)", r.text)
            if m:
                VRAM_GIB.set(round(float(m.group(1)) / 1024, 2))
    except Exception:
        pass


def _gpu_gauge_snapshot() -> dict:
    return {"gpu_utilization_pct": round(GPU_UTIL._value.get(), 1),
            "vram_used_gib": round(VRAM_GIB._value.get(), 2)}


def _counter_total(name: str) -> int:
    """Sum a (possibly labeled) counter across all label combos, via the
    Prometheus text exposition (robust regardless of client internals)."""
    try:
        total = 0
        for line in generate_latest(REGISTRY).decode().splitlines():
            if line.startswith("#") or not line.startswith(name + " "):
                continue
            # e.g. styling_gen_total{status="success"} 3.0
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
        return int(total)
    except Exception:
        return 0


@app.get("/api/metrics")
async def metrics():
    with lock:
        now = time.time()
        window = [v for ts, v in latency_history if now - ts <= RECENT_WINDOW]
        rpm = len(window) / (RECENT_WINDOW / 60)
        avg = sum(window) / len(window) * 1000 if window else None
    return {
        "live": {
            "throughput_rpm": round(rpm, 3),
            "avg_latency_ms": round(avg) if avg else None,
            "requests_total": _counter_total("styling_gen_total"),
            "gpu_utilization_pct": round(GPU_UTIL._value.get(), 1),
            "vram_used_gib": round(VRAM_GIB._value.get(), 2),
            "latency_history": [
                {"t": datetime.fromtimestamp(ts, timezone.utc).isoformat(), "ms": round(v * 1000)}
                for ts, v in list(latency_history)[-50:]
            ],
        },
        "baseline": _load_baseline(),
    }


@app.get("/api/results/{job_id}.mp4")
async def result(job_id: str):
    safe = re.sub(r"[^a-zA-Z0-9-]", "", job_id)
    path = os.path.join(RESULTS_DIR, f"{safe}.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, "result not found")
    # Serve the video INLINE, not as a download. A <video> element in the UI
    # must be able to play this in-page; `Content-Disposition: attachment`
    # (which Starlette's `filename=` arg produces) tells some browsers to
    # treat it as a download and refuse inline playback, leaving the tile
    # black / "no video". inline still carries a filename for save-as.
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="styling-{safe}.mp4"'},
    )


@app.get("/metrics")
async def prom_metrics():
    return JSONResponse(generate_latest(REGISTRY),
                        media_type="text/plain; version=0.0.4; charset=utf-8")


# ---------------------------------------------------------------- 3-stage workflow
# Stage 1: base image -> two output SETS (full images, no line-art):
#   Set 1: base image + its 180-degree opposite view (rear of car).
#   Set 2: a color/instruction-following variant + its 180-degree opposite view.
# Stage 2: pick one image from a set -> refined 2D still + alt view + color variant.
# Stage 3: pick a still -> generate video.
# Each stage accepts a free-text requirement and can regenerate.

STAGE1_OPPOSITE_PROMPT = (
    "This image shows a vehicle from one view angle. Generate a full-color image of the "
    "SAME vehicle from a viewpoint rotated 180 degrees around the car's vertical axis: "
    "the REAR of the car (taillights, trunk lid, rear bumper, exhaust) is now the dominant "
    "face facing the viewer, and the opposite side panel is visible at an angle. Keep the "
    "exact same car model, paint, proportions, wheels and body details. Only the camera "
    "has moved to the opposite side of the car. Photorealistic image, consistent lighting."
)
STAGE1_VARIANT_PROMPT = (
    "Re-render this same vehicle with a CLEARLY DIFFERENT body paint color: {target_color} "
    "finish. Keep the EXACT same car model, body shape, wheels, proportions, view angle and "
    "composition — only the paint color changes. The result must look like a different paint "
    "job on the same car, NOT a near-copy of the original. Photorealistic image, consistent "
    "lighting. {requirement}"
)
STAGE2_REFINE_PROMPT = (
    "Refine this vehicle design into a polished, photorealistic 2D still image: clean and "
    "crisp surfaces, consistent studio lighting, realistic materials and reflections. Keep "
    "the exact same car model, paint, view angle and composition. {style}{requirement}"
)
STAGE2_ALT_PROMPT = (
    "Render the same vehicle (same styling, paint and finish) from a different view "
    "angle (e.g. 3/4 front). Photorealistic still image, consistent lighting. "
    "{requirement}"
)
STAGE2_COLOR_PROMPT = (
    "Render the same vehicle with a completely different paint color scheme (e.g. if it "
    "is blue, make it metallic red or matte black). Keep all other styling identical. "
    "Photorealistic still image. {requirement}"
)
STAGE3_VIDEO_PROMPT = (
    "Generate a cinematic video of this vehicle. Slow camera orbit revealing the design "
    "details, studio lighting, professional automotive commercial style. {requirement}"
)

# --- 6-stage IDD workflow (Mario's "Exterior New Design"): stage 4 = AI base
# sketch (front + rear + colorways), stage 5 = scene composition + quick viz.
STAGE4_SKETCH_PROMPT = (
    "Redesign this vehicle reference into a clean automotive DESIGN SKETCH shown as a "
    "head-on FRONT view: crisp marker-render line art with subtle shading, clearly "
    "drawn headlights, front bumper and windshield (no red lamps), studio-neutral "
    "background, design-studio presentation style, same body stance, paint color and "
    "proportions as the reference. {requirement}")
# Prepended to the (already anti-top-down, color-anchored) opposite/rear prompt so
# the rear output is a SKETCH too, instead of drifting back to photoreal.
SKETCH_STYLE_PREFIX = (
    "Automotive DESIGN SKETCH — crisp marker-render line art with subtle shading on a "
    "studio-neutral background. ")
STAGE5_MAIN_PROMPT = (
    "Refine this car-into-scene composite into a polished photorealistic automotive "
    "visualization: seamlessly match the scene's lighting, reflections and color "
    "temperature on the car. Keep the exact same car model, paint, view angle, "
    "position and composition. {style}{requirement}")
# .format(view=...) — "another view" of the quick-viz set; must pin position,
# background and framing so the only change is the camera angle.
STAGE5_ALT_VIEW_PROMPT = (
    "Keep the EXACT same scene, environment, lighting and the car's position within the "
    "frame, and change ONLY the camera angle to a {view} view of the car. Same car model, "
    "same paint, same background. Photorealistic automotive visualization.")
# The generic prompt above (verified live 2026-10-05) was too WEAK: on the Qwen-Edit
# GGUF i2i the model kept the input car's orientation (a side-profile photo stayed a
# near-identical side/rear render) — "3/4 front" came back looking the same as "side".
# Fix: an explicit, direction-anchored prompt per view that names the EXACT car
# surfaces that must face the camera (front: grille+headlights+bumper; rear: tailgate+
# taillights; 3/4: side + the named end). Verified live: 3/4 front -> true 3/4 front,
# rear -> straight rear, at the same ALT_VIEW_STRENGTH.
_ALT_VIEW_EXPLICIT = {
    "side": (
        "Rotate the camera to a true straight SIDE profile of the car: both wheels in "
        "perfect side profile, the front and rear ends edge-on and NOT visible, only the "
        "side body panels facing the camera, the car's long axis running horizontally "
        "across the frame. Keep the EXACT same car model, paint, scene, environment, "
        "lighting, position and framing. Photorealistic automotive visualization."),
    "3/4 front": (
        "Rotate the camera to a three-quarter FRONT view of the car: the FRONT face "
        "(grille, both headlights, front bumper) must be clearly visible and angled "
        "toward the camera, together with one full side. Keep the EXACT same car model, "
        "paint, scene, environment, lighting, position and framing. Photorealistic "
        "automotive visualization."),
    "front": (
        "Rotate the camera to a straight FRONT view of the car: the grille, both "
        "headlights and front bumper must face the camera directly. Keep the EXACT same "
        "car model, paint, scene, environment, lighting, position and framing. "
        "Photorealistic automotive visualization."),
    "3/4 rear": (
        "Rotate the camera to a three-quarter REAR view of the car: the REAR face "
        "(tailgate, both taillights, rear bumper) must be clearly visible and angled "
        "toward the camera, together with one full side. Keep the EXACT same car model, "
        "paint, scene, environment, lighting, position and framing. Photorealistic "
        "automotive visualization."),
    "rear": (
        "Rotate the camera to a straight REAR view of the car: the tailgate, both "
        "taillights, rear bumper and rear window must face the camera directly. Keep "
        "the EXACT same car model, paint, scene, environment, lighting, position and "
        "framing. Photorealistic automotive visualization."),
}


def alt_view_prompt(view: str) -> str:
    """Explicit, direction-anchored re-angle prompt for a named view (see the
    _ALT_VIEW_EXPLICIT note above). Unknown views fall back to the generic
    STAGE5_ALT_VIEW_PROMPT."""
    return _ALT_VIEW_EXPLICIT.get((view or "").strip().lower(),
                                  STAGE5_ALT_VIEW_PROMPT.format(view=view or "side"))
# .format(color=...) — fallback prompt path; the primary colorway mechanism stays
# the deterministic _recolor_body (guarantees geometry preservation).
STAGE5_COLOR_PROMPT = (
    "Re-render the same car in the same scene with a CLEARLY DIFFERENT body paint color: "
    "{color}. Keep the exact same car model, view angle, composition, environment and "
    "lighting — only the paint color changes. Photorealistic automotive visualization.")


def _fill(tpl: str, requirement: str) -> str:
    req = f"Additional requirement: {requirement.strip()}." if requirement and requirement.strip() else ""
    return (tpl + " " + req).strip()


# ---------------------------------------------------------------- body-color anchor
# The opposite-view re-render and the set-2 color variant both drift to a generic
# (often white) car if the model is free to "choose". We anchor them by detecting
# the input car's actual body color and naming it in the prompt, so the re-render
# keeps the same paint (blue stays blue) instead of re-drawing a white car.
_BODY_COLOR_CONTRAST = {
    "blue": "red", "red": "blue", "green": "orange",
    "orange": "teal", "yellow": "purple", "black": "silver",
    "white": "black", "silver-grey": "red", "grey": "gold",
    "teal": "red", "purple": "lime", "magenta": "blue",
    "lime": "purple", "beige": "black",
}

# Color words we recognise in a free-text requirement, so a user's color request
# (e.g. "make it red") overrides the auto-contrast target for set 2.
_COLOR_WORDS = [
    "crimson", "magenta", "purple", "violet", "indigo", "teal", "cyan", "blue",
    "navy", "aqua", "green", "lime", "emerald", "olive", "yellow", "gold",
    "orange", "amber", "red", "maroon", "pink", "white", "black", "silver",
    "grey", "gray", "beige", "brown",
]


def _target_variant_color(base_color: str, requirement: str) -> str:
    """The set-2 target paint color. Use the user's requested color if one is
    named in the requirement, otherwise a contrasting color so set 2 is visibly
    different from the base."""
    req = (requirement or "").lower()
    for word in _COLOR_WORDS:
        if word in req:
            return word
    return _BODY_COLOR_CONTRAST.get(base_color, "red")


def _name_hsv(h: float, s: float, v: float) -> str:
    """Name a color from HSV (h in [0,360), s/v in [0,1])."""
    if s < 0.13:
        return "black" if v < 0.30 else ("white" if v > 0.82 else "silver-grey")
    if s < 0.30:
        return "beige" if v > 0.80 else "grey"
    if h < 14 or h >= 346:
        return "red"
    if h < 40:
        return "orange"
    if h < 66:
        return "yellow"
    if h < 90:
        return "lime"
    if h < 150:
        return "green"
    if h < 190:
        return "teal"
    if h < 250:
        return "blue"
    if h < 290:
        return "purple"
    if h < 346:
        return "magenta"
    return "red"


def _detect_body_color(img: Image.Image) -> str:
    """Detect the dominant car-body color, ignoring a uniform background.

    The background is estimated from the four corner patches; the most common
    saturated hue among the remaining (non-background) pixels is named. Returns
    e.g. "blue". Falls back to "a vivid" when it cannot tell."""
    try:
        a = np.asarray(img.convert("RGB")).astype(np.float32)
        h, w = a.shape[:2]
        patch = max(8, min(h, w) // 8)
        corners = [a[:patch, :patch], a[:patch, -patch:], a[-patch:, :patch], a[-patch:, -patch:]]
        bg = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)
        dist = np.linalg.norm(a - bg, axis=2)
        mask = dist > 45
        if mask.sum() < max(200, a.size // 20):
            mask = np.ones((h, w), bool)
        px = a[mask]
        r, g, b = px[:, 0] / 255, px[:, 1] / 255, px[:, 2] / 255
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        v = mx
        s = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0)
        d = mx - mn
        hh = np.zeros_like(v)
        rr, gg, bb = mx == r, mx == g, mx == b
        hh[rr] = (60 * ((g[rr] - b[rr]) / np.where(d[rr] > 0, d[rr], 1)) + 360) % 360
        hh[gg] = (60 * ((b[gg] - r[gg]) / np.where(d[gg] > 0, d[gg], 1)) + 120)
        hh[bb] = (60 * ((r[bb] - g[bb]) / np.where(d[bb] > 0, d[bb], 1)) + 240)
        colored = s > 0.12
        if colored.sum() < 50:
            return _name_hsv(0, float(s.mean()), float(v.mean()))
        bins = np.arange(0, 361, 15)
        hist, edges = np.histogram(hh[colored], bins=bins, weights=s[colored])
        hist[0] += hist[-1]
        best = int(np.argmax(hist))
        center = (edges[best] + edges[best + 1]) / 2
        if best == 0:
            center = 352
        return _name_hsv(center, float(np.median(s[colored])), float(np.median(v[colored])))
    except Exception as e:  # never let color detection block generation
        log.warning("body-color detection failed (%s); using generic anchor", e)
        return "a vivid"


def _opposite_prompt(color: str) -> str:
    return (
        "This image shows a vehicle from one view angle. Generate a full-color image of the "
        "SAME vehicle from a viewpoint rotated 180 degrees around the car's vertical axis: "
        "the REAR of the car (taillights, trunk lid, rear bumper, exhaust) is now the dominant "
        "face facing the viewer, with the opposite side panel visible at an angle. Keep the "
        "exact same car model, body shape, proportions, wheels and body details, and keep the "
        "same " + color + " body paint color — do NOT change the color or swap in a different "
        "car. The camera stays at the SAME height as the car, looking at its rear; do NOT "
        "switch to a top-down, overhead or bird's-eye view — no roof-only or floating-from-above "
        "shot. Only the camera has moved to the opposite side of the car. Photorealistic image, "
        "consistent lighting."
    )


# Paint targets for the deterministic set-2 recolor: name -> (hue deg, sat, val).
_NAMED_HSV = {
    "red": (0, 0.85, 0.72), "crimson": (355, 0.8, 0.6), "maroon": (350, 0.8, 0.4),
    "orange": (28, 0.85, 0.80), "amber": (38, 0.85, 0.75), "gold": (45, 0.8, 0.75),
    "yellow": (52, 0.85, 0.85), "lime": (85, 0.8, 0.75), "emerald": (160, 0.8, 0.5),
    "green": (140, 0.75, 0.55), "olive": (80, 0.4, 0.5), "teal": (185, 0.7, 0.55),
    "cyan": (190, 0.7, 0.6), "aqua": (185, 0.5, 0.7), "blue": (222, 0.8, 0.6),
    "navy": (225, 0.85, 0.28), "indigo": (245, 0.7, 0.5), "purple": (280, 0.6, 0.6),
    "violet": (270, 0.6, 0.6), "magenta": (320, 0.7, 0.65), "pink": (330, 0.5, 0.8),
    "white": (0, 0.0, 0.95), "black": (0, 0.0, 0.12), "silver": (0, 0.08, 0.8),
    "silver-grey": (0, 0.08, 0.8), "grey": (0, 0.06, 0.5), "gray": (0, 0.06, 0.5),
    "beige": (40, 0.25, 0.85), "brown": (25, 0.6, 0.4), "vivid": (222, 0.8, 0.6),
}


def _body_mask(img: Image.Image) -> np.ndarray:
    """Boolean mask of the car's painted-body pixels.

    Primary: distance-from-background (background estimated from the four
    corner patches) — right for a car photographed on a distinct background.

    Fallback (for full-frame photos whose corners are the car itself, not the
    background): the dominant *saturated* region, kept to the central part of
    the frame. Either way, wheels/glass/gloss (low saturation) are excluded.
    """
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    patch = max(8, min(h, w) // 8)
    corners = [a[:patch, :patch], a[:patch, -patch:], a[-patch:, :patch], a[-patch:, -patch:]]
    bg = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    mask = dist > 45
    frac = mask.mean()
    if not (0.03 < frac < 0.80):
        # corners are not background -> use the dominant saturated region,
        # restricted to the central part of the frame
        hsv = np.asarray(img.convert("HSV")).astype(np.float32) / 255.0
        S, V = hsv[..., 1], hsv[..., 2]
        yy, xx = np.mgrid[0:h, 0:w]
        center = (np.abs(yy - h / 2) <= h * 0.42) & (np.abs(xx - w / 2) <= w * 0.42)
        mask = (S > 0.35) & (V > 0.15) & center
    return mask


def _recolor_body(img: Image.Image, target: str) -> Image.Image:
    """Deterministic repaint of the car body to `target` WITHOUT touching
    geometry, view angle, wheels, details or background — so the result is the
    SAME car as the upload with only the paint changed. An image model (i2i)
    re-draws the whole car and swaps the model, which is exactly the defect
    this avoids.
    """
    try:
        mask = _body_mask(img)
        hsv = img.convert("HSV")
        arr = np.asarray(hsv).astype(np.int16)
        H, S, V = arr[..., 0].copy(), arr[..., 1].copy(), arr[..., 2].copy()
        th, ts, tv = _NAMED_HSV.get((target or "").lower(), (0, 0.85, 0.72))
        # current dominant body hue (only saturated body pixels count)
        s_body = S[mask]
        h_body = H[mask]
        sat_body = h_body[s_body > 30]
        cur_h = float(np.median(h_body)) if len(h_body) else 0.0
        if len(sat_body) > 50:
            cur_h = float(np.median(sat_body))
        achromatic_target = ts < 0.15
        if achromatic_target:
            # white/black/silver: desaturate body pixels + set the target value
            S[mask] = (S[mask] * 0.05).astype(np.int16)
            V[mask] = np.clip(V[mask] + (int(tv * 255) - V[mask]) * 0.85, 0, 255)
        else:
            # cur_h is on PIL's 0-255 hue scale; convert to degrees before
            # computing the shift (th is in degrees).
            cur_deg = cur_h * 360.0 / 255.0
            shift = int(((th - cur_deg) % 360) * 255.0 / 360.0)
            H[mask] = (H[mask] + shift) % 255
            # nudge saturation/value toward the target shade
            S[mask] = np.clip(S[mask] + (int(ts * 255) - S[mask]) * 0.5, 0, 255)
            V[mask] = np.clip(V[mask] + (int(tv * 255) - V[mask]) * 0.3, 0, 255)
        out = np.dstack([H.astype(np.uint8), S.astype(np.uint8), V.astype(np.uint8)])
        return Image.fromarray(out, "HSV").convert("RGB")
    except Exception as e:  # fall back to the original rather than block the stage
        log.warning("deterministic recolor failed (%s); using original", e)
        return img


def _apply_color_grade(img: Image.Image, ref: Image.Image) -> Image.Image:
    """Shift the image's color grade toward the reference's palette/mood
    (a deterministic L*a*b* offset), so the uploaded style reference is
    actually applied to the still's pixels — not only mentioned in the
    prompt. 35% strength keeps the car readable while adopting the look."""
    try:
        ref2 = ref.resize(img.size).convert("LAB")
        src = np.asarray(img.convert("LAB")).astype(np.float32)
        refv = np.asarray(ref2).astype(np.float32)
        off = (refv.reshape(-1, 3).mean(0) - src.reshape(-1, 3).mean(0)) * 0.35
        out = np.clip(src + off, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "LAB").convert("RGB")
    except Exception as e:
        log.warning("color grade failed (%s); returning original", e)
        return img


@app.post("/api/workflow/stage")
async def workflow_stage(req: StageRequest):
    # Per-request external provider (PER-ROLE: brain / i2v / image — each may be
    # external independently). Sets the request-scoped ContextVar used by the resolvers.
    prov_label = _apply_provider(req)
    if prov_label != "on-prem":
        log.info("workflow stage %s: external provider -> %s", req.stage, prov_label)
    if req.stage == 1:
        return await _stage1(req)
    if req.stage == 2:
        return await _stage2(req)
    if req.stage == 3:
        return await _stage3(req)
    if req.stage == 4:
        return await _stage4(req)
    if req.stage == 5:
        return await _stage5(req)
    raise HTTPException(400, "stage must be 1–5")


async def _stage1(req: StageRequest):
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "stage 1 requires an input image (image)")
    w, h = _dims_from_image(img, req.width, req.height)
    t0 = time.time()

    def norm(im):
        """All four stage-1 outputs share one (w,h) so the UI cells are uniform."""
        if im.size == (w, h):
            return im
        return im.resize((w, h), Image.LANCZOS)

    def enc(im):
        return "data:image/png;base64," + base64.b64encode(_img_to_pil_bytes(norm(im))).decode()

    # Detect the uploaded car's body color so the re-renders stay on-model and
    # keep the same paint (the #1 reported defect: the opposite view drifted to a
    # generic white car). Set 2 is a CONTRAST color so the two sets are clearly
    # different; the user's requirement can override the target color.
    base_color = _detect_body_color(img)
    target_color = _target_variant_color(base_color, req.requirement)
    log.info("stage1 base color=%s, set2 target=%s", base_color, target_color)

    # SET 1: the base image itself (normalized to the common size) + its
    # 180-degree opposite view (rear), anchored to the base color so it stays
    # blue-on-blue instead of drifting to white. Strength = ROT_STRENGTH
    # (model-dependent: 0.8 Z-Image sweet spot, 1.0 Qwen-GGUF rotation).
    set1_opposite = await _img_generate(_opposite_prompt(base_color), img, w, h,
                                        strength=ROT_STRENGTH, seed=req.seed)
    # SET 2 primary: a deterministic PAINT-ONLY recolor of the upload. We do NOT
    # use the image model here — i2i re-draws the whole car and swaps the model
    # (the reported defect). The HSV repaint keeps the EXACT same view, geometry,
    # wheels and details; only the body paint changes to the target color.
    variant = norm(_recolor_body(img, target_color))
    # SET 2 opposite: i2i on the recolored car (real model+view as input) to get
    # its 180-degree rear, anchored to the variant's NEW paint color so the rear
    # does not drift back to grey/white.
    set2_opposite = await _img_generate(_opposite_prompt(target_color), variant, w, h,
                                        strength=ROT_STRENGTH, seed=(req.seed or 0) + 2)
    elapsed = (time.time() - t0) * 1000
    return {
        "stage": 1,
        "results": {
            "base": enc(img),
            "base_opposite": enc(set1_opposite),
            "variant": enc(variant),
            "variant_opposite": enc(set2_opposite),
        },
        "base_color": base_color,
        "variant_color": target_color,
        "requirement": req.requirement,
        "size": {"w": w, "h": h},
        "elapsed_ms": round(elapsed),
    }


async def _stage2(req: StageRequest):
    # design_image = the PRIMARY image of the set picked in stage 1.
    design = _b64_image_to_pil(req.design_image)
    if design is None:
        raise HTTPException(400, "stage 2 requires the chosen set's primary image (design_image)")
    w, h = _dims_from_image(design, req.width, req.height)
    t0 = time.time()
    enc = lambda im: "data:image/png;base64," + base64.b64encode(_img_to_pil_bytes(im)).decode()

    # Optional style reference: (a) describe it (palette/lighting/material/mood)
    # and fold the descriptor into the refine prompt, AND (b) apply a real
    # deterministic L*a*b* color grade from the reference to the still's pixels,
    # so the uploaded style is visibly applied — not only mentioned in a prompt
    # that the model can ignore at low strength.
    style = ""
    ref_pil = None
    if req.reference_image:
        ref_pil = _b64_image_to_pil(req.reference_image)
        if ref_pil is not None:
            desc = await _describe_style(ref_pil)
            if desc:
                style = (f"Style: adopt this reference look — {desc}. Match its "
                         f"palette, lighting mood, materials and finish. ")

    # Polished 2D still of the chosen view. Moderate strength so the model may
    # clean surfaces/lighting but must not swap the car or reframe it.
    req_text = _fill("", req.requirement)
    main = await _img_generate(
        STAGE2_REFINE_PROMPT.format(style=style, requirement=req_text),
        design, w, h, strength=0.55, seed=req.seed)
    # Apply the reference's color grade to the main still (pixels, not prompt)
    # so the style upload changes the output even if the model ignored it.
    if ref_pil is not None:
        main = _apply_color_grade(main, ref_pil)
    # Polished 2D still of the same car from the 180-degree opposite view,
    # anchored to the chosen car's color so it stays on-paint (no white drift),
    # so the final set has both a primary and a rear still for the video.
    opposite = await _img_generate(
        _opposite_prompt(_detect_body_color(design)), main, w, h,
        strength=ROT_STRENGTH, seed=(req.seed or 0) + 1)
    if ref_pil is not None:
        opposite = _apply_color_grade(opposite, ref_pil)
    elapsed = (time.time() - t0) * 1000
    return {
        "stage": 2,
        "results": {
            "main": enc(main),
            "opposite": enc(opposite),
        },
        "style_reference": bool(req.reference_image),
        "requirement": req.requirement,
        "elapsed_ms": round(elapsed),
    }


async def _stage3(req: StageRequest):
    if not MLIS_URL:
        raise HTTPException(503, "MLIS_VIDEO_GENERATION_URL not configured")
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "stage 3 requires a final design image (image)")
    prompt = _fill(req.prompt_override or STAGE3_VIDEO_PROMPT, req.requirement)
    t0 = time.time()
    p = Params(style_strength=0.7, motion_strength=0.6, num_frames=121, fps=8)
    resolution = req.resolution or DEFAULT_RESOLUTION
    if resolution not in [r.lower() for r in SUPPORTED_RESOLUTIONS]:
        resolution = DEFAULT_RESOLUTION
    mp4, guidance, gen_meta = await generate_video(prompt, img, p, resolution)
    job_id = uuid.uuid4().hex
    out_path = os.path.join(RESULTS_DIR, f"{job_id}.mp4")
    with open(out_path, "wb") as f:
        f.write(mp4)
    elapsed = (time.time() - t0) * 1000
    GEN_TOTAL.labels(status="success").inc()
    return {
        "stage": 3,
        "results": {
            "video_url": f"/api/results/{job_id}.mp4",
            "job_id": job_id,
        },
        "prompt": prompt,
        "requirement": req.requirement,
        "guidance_scale": round(guidance, 2),
        "elapsed_ms": round(elapsed),
        "metrics": {
            "video_generation_ms": round(gen_meta["gen_ms"]),
            "gpu_utilization": _gpu_gauge_snapshot(),
        },
    }


# ---------------------------------------------------------------- stage 4/5 helpers
SAMPLE_DATA_DIR = Path(os.environ.get("SAMPLE_DATA_DIR",
                       str(Path(__file__).parent / "sample_data")))
MOCK_SCENE_FILE = SAMPLE_DATA_DIR / "sample_scene_dusk.png"


def _load_mock_scene() -> Image.Image | None:
    """Bundled dusk-street scene used when the demo has no customer REF 2 photo."""
    try:
        if MOCK_SCENE_FILE.exists():
            return Image.open(MOCK_SCENE_FILE).convert("RGB")
    except Exception as e:
        log.warning("mock scene load failed: %s", e)
    return None


def _pencil_sketch(img: Image.Image) -> Image.Image:
    """Deterministic pencil/marker sketch fallback (dodge-blend line art over a
    posterized color base) so the sketch stage never hard-fails without the
    image model."""
    try:
        a = np.asarray(ImageOps.grayscale(img)).astype(np.float32)
        inv = ImageOps.invert(Image.fromarray(a.astype(np.uint8)))
        blur = np.asarray(inv.filter(ImageFilter.GaussianBlur(6))).astype(np.float32)
        # classic dodge sketch: lines = where the blurred inverted luminance is bright
        lines = np.clip(a * 255.0 / np.maximum(255.0 - blur, 1.0), 0, 255)
        base = np.asarray(img.quantize(colors=16, method=2).convert("RGB")).astype(np.float32)
        # multiply the faint posterized color by the normalized line map
        # (lines is a single-channel HxW map -> add the channel axis so it
        # broadcasts over the HxWx3 posterized base)
        out = np.clip(base * (lines / 255.0 * 1.25 + 0.35)[..., None], 0, 255).astype(np.uint8)
        return ImageEnhance.Contrast(Image.fromarray(out)).enhance(1.15)
    except Exception as e:
        log.warning("pencil sketch fallback failed (%s); returning original", e)
        return img


def _colorway_targets(base_color: str, n: int, explicit: list[str] | None,
                      requirement: str) -> list[str]:
    """Up to n distinct paint targets for the stage-4 colorways. Explicit
    `color_targets` win; otherwise the first is the user/contrast color from
    the requirement, and the rest are contrast-map colors that differ from the
    base car and from each other."""
    if explicit:
        return [c for c in explicit if c][:n] or ["red"]
    out: list[str] = []
    first = _target_variant_color(base_color, requirement)
    out.append(first)
    order = list(_BODY_COLOR_CONTRAST.values())
    for c in order:
        if len(out) >= n:
            break
        if c != base_color and c not in out:
            out.append(c)
    return out[:max(1, n)]


def _default_font(size: int):
    for p in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _mock_designer_edit(img: Image.Image, accent: str = "crimson") -> Image.Image:
    """Deterministic PIL pass simulating the designer's Photoshop iteration on
    the AI sketch: a bold accent pinstripe following the beltline (the top edge
    of the detected body), a gloss highlight stroke over the hood, a slight
    warm grade, and a 'FINAL DESIGN — rev B' watermark. Same geometry always —
    only the designer overlay changes, so the demo gate never depends on luck."""
    try:
        w, h = img.size
        canvas = img.convert("RGB").copy()
        mask = _body_mask(canvas)
        ys, xs = np.nonzero(mask)
        if len(xs) > 200:
            x0, x1 = int(xs.min()), int(xs.max())
            # transparent overlay layer (drawing on a copy of the opaque canvas
            # and alpha_compositing it would both mismatch modes and re-paste the
            # whole image); only the stripe + gloss below are composited.
            top = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(top, "RGBA")
            # beltline: per-column top edge of the body, smoothed + offset down 15% of body height
            body_h = max(8, int(ys.max() - ys.min()))
            cols = range(x0, x1 + 1, max(1, (x1 - x0) // 120))
            pts = []
            for x in cols:
                col_ys = ys[xs == x]
                if len(col_ys):
                    y = int(col_ys.min()) + int(body_h * 0.30)
                    pts.append((x, y))
            if len(pts) > 3:
                ah, asat, av = _NAMED_HSV.get(accent, (355, 0.8, 0.6))
                _r, _g, _b = colorsys.hsv_to_rgb((ah % 360) / 360.0, asat, av)
                acol = (int(_r * 255), int(_g * 255), int(_b * 255))
                d.line(pts, fill=acol + (230,), width=max(3, body_h // 40), joint="curve")
                # gloss highlight over the hood/front third of the body
                gx0 = x0 + (x1 - x0) // 8
                gx1 = x0 + (x1 - x0) // 2
                gy = pts[min(4, len(pts) - 1)][1] - body_h // 8
                d.ellipse([gx0, gy - body_h // 14, gx1, gy + body_h // 14],
                          fill=(255, 255, 255, 70))
                canvas = Image.alpha_composite(canvas.convert("RGBA"), top).convert("RGB")
        # slight warm grade: boost R, cut B
        arr = np.asarray(canvas).astype(np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.06 + 4, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)
        canvas = Image.fromarray(arr.astype(np.uint8))
        # watermark
        d2 = ImageDraw.Draw(canvas, "RGBA")
        fnt = _default_font(max(14, h // 22))
        text = "FINAL DESIGN — rev B  (simulated designer edit)"
        tb = d2.textbbox((0, 0), text, font=fnt)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d2.rectangle([w - tw - 24, h - th - 28, w - 8, h - 8], fill=(10, 14, 24, 150))
        d2.text((w - tw - 16, h - th - 20), text, font=fnt, fill=(220, 230, 250, 235))
        return canvas
    except Exception as e:
        log.warning("mock designer edit failed (%s); using original design", e)
        return img


def _road_geometry(scene: Image.Image) -> dict:
    """Estimate the road plane of a one-point-perspective scene, pure-numpy.
    Strategy: the road must be a wide dark band at the frame bottom (a car or
    skyline fails this gate); its bright edge lines are tracked row-by-row from
    the bottom upward, then fitted — their intersection is the vanishing point.
    Returns {"ok", "vp", "horizon", "left", "right"} with each edge x = m*y + b;
    falls back to a centered heuristic when the road can't be detected."""
    W, H = scene.size
    g = np.asarray(ImageOps.grayscale(scene.resize((W, H))), dtype=np.float32)
    lower = g[int(H * 0.55):, :]
    dark_t = max(50.0, min(150.0, float(np.percentile(lower, 30)) + 25))
    bright_t = max(130.0, float(np.percentile(lower, 90)))
    # gate 1: the bottom row must carry a WIDE dark band (the road at close
    # range). A reference car (~<35% width) or a distant skyline fails this.
    row = g[int(H * 0.985), :]
    dark = row < dark_t
    cx = W // 2
    if not dark[cx]:
        cands = np.nonzero(dark[max(0, cx - int(W * 0.25)):cx + int(W * 0.25)])[0]
        if len(cands) == 0:
            return {"ok": False, "vp": (W / 2, H * 0.52), "horizon": H * 0.52,
                    "left": (0.0, W * 0.20), "right": (0.0, W * 0.80)}
        cx = max(0, cx - int(W * 0.25)) + int(cands[len(cands) // 2])
    x = cx
    while x > 0 and dark[x]:
        x -= 1
    xl = x + 1
    x = cx
    while x < W - 1 and dark[x]:
        x += 1
    xr = x - 1
    if (xr - xl) < W * 0.15:
        return {"ok": False, "vp": (W / 2, H * 0.52), "horizon": H * 0.52,
                "left": (0.0, W * 0.20), "right": (0.0, W * 0.80)}
    if (xr - xl) > W * 0.75:                 # the whole bottom is dark: not a road
        return {"ok": False, "vp": (W / 2, H * 0.52), "horizon": H * 0.52,
                "left": (0.0, W * 0.20), "right": (0.0, W * 0.80)}
    # track the bright edge lines from the bottom upward
    pts_left, pts_right = [], []
    bright = g > bright_t
    for y in range(int(H * 0.985), int(H * 0.50), -1):
        rowb = bright[y, :]
        # left edge: rightmost bright pixel at least 4% left of center
        cand = np.nonzero(rowb[max(0, cx - int(W * 0.45)):cx - int(W * 0.04)])[0]
        if len(cand):
            pts_left.append((float(y), float(max(0, cx - int(W * 0.45)) + cand.max())))
        cand = np.nonzero(rowb[cx + int(W * 0.04):min(W, cx + int(W * 0.45))])[0]
        if len(cand):
            pts_right.append((float(y), float(cx + int(W * 0.04) + cand.min())))
    if len(pts_left) < 6 or len(pts_right) < 6:
        return {"ok": False, "vp": (W / 2, H * 0.52), "horizon": H * 0.52,
                "left": (0.0, W * 0.20), "right": (0.0, W * 0.80)}
    def fit(pts):
        a = np.array(pts, dtype=np.float32)
        m, b = np.polyfit(a[:, 0], a[:, 1], 1)
        resid = np.abs(a[:, 1] - (m * a[:, 0] + b))
        keep = resid < max(8.0, 0.05 * W)
        if keep.sum() >= 5:
            m, b = np.polyfit(a[keep, 0], a[keep, 1], 1)
        return float(m), float(b)
    ml, bl = fit(pts_left)
    mr, br = fit(pts_right)
    # as rows go DOWN (y increases) the road widens: left edge slopes left
    # (m<0), right edge slopes right (m>0) — that is the convergence signature
    if not (ml < -0.02 and mr > 0.02):
        return {"ok": False, "vp": (W / 2, H * 0.52), "horizon": H * 0.52,
                "left": (ml, bl), "right": (mr, br)}
    # intersection of x = m*y + b lines: y = (br - bl)/(ml - mr)
    vpy = (br - bl) / (ml - mr)
    vpx = ml * vpy + bl
    vpy = max(H * 0.30, min(H * 0.62, float(vpy)))   # sanity clamp
    vpx = max(W * 0.2, min(W * 0.8, float(vpx)))
    return {"ok": True, "vp": (vpx, vpy), "horizon": vpy,
            "left": (ml, bl), "right": (mr, br)}


def _flat_road_strip(scene: Image.Image) -> int | None:
    """Detect a *flat side-on* road: a bright horizontal asphalt strip along the
    bottom of the frame (a roadside shot, NOT a one-point-perspective road —
    that is why _road_geometry's vanishing-point gate fails here). Returns the
    TOP row (y) of the bright bottom band, or None when the scene is not a
    flat roadside. The car's wheels are then placed just inside this strip."""
    W, H = scene.size
    g = np.asarray(ImageOps.grayscale(scene.resize((W, H))), dtype=np.float32)

    def row_mean(y: int) -> float:
        return float(g[y, :].mean())

    # the bottom row must be BRIGHT (asphalt), not dark (a one-point road is
    # dark at close range and is handled by _road_geometry instead).
    if row_mean(H - 1) < 105:
        return None
    # walk up from the bottom while the row stays bright -> top of the band
    top = H - 1
    while top > 0 and row_mean(top - 1) > 100:
        top -= 1
    strip_h = H - top
    # the strip is a thin band at the bottom (3%-45% of frame height), not the
    # whole frame and not a sliver
    if strip_h < H * 0.03 or strip_h > H * 0.45:
        return None
    # guard: the band must span the width (a road), not a small bright patch
    mid = top + strip_h // 2
    if (g[mid, :] > 100).mean() < 0.55:
        return None
    return int(top)


def _compose_scene(design: Image.Image, scene: Image.Image,
                   depth: float = 0.5) -> Image.Image:
    """Deterministic car-into-scene composition (geometry-locked):
    1. detect the scene's road plane (vanishing point + edge lines, Hough);
    2. place the car ON the road at a set depth (perspective-scaled, in-lane);
    3. ground it with a tight contact shadow + soft drop shadow + asphalt
       reflection, and match the car's light to the scene's road tone;
    4. the caller's model harmonize pass then refines a grounded car instead
       of rescuing a floating billboard. Falls back to a center-weighted
    brightness guess when the road can't be detected."""
    try:
        mask = _body_mask(design)
        ys, xs = np.nonzero(mask)
        if len(xs) < 200:
            raise ValueError("car not separable from design background")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # guard: if the mask spans the whole frame it did NOT separate the car
        # (gradient/photographic background) — restrict to a central crop so we
        # never paste a giant full-frame rectangle over the scene.
        dw, dh = design.width, design.height
        if (x1 - x0) > dw * 0.88 and (y1 - y0) > dh * 0.88:
            mx0, mx1 = int(dw * 0.15), int(dw * 0.85)
            my0, my1 = int(dh * 0.15), int(dh * 0.85)
            x0, x1 = max(x0, mx0), min(x1, mx1)
            y0, y1 = max(y0, my0), min(y1, my1)
        car = design.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGBA")
        m = mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255
        alpha = Image.fromarray(m).filter(ImageFilter.GaussianBlur(2))  # feather edge
        car.putalpha(alpha)
        W, H = scene.size
        geo = _road_geometry(scene)
        if geo["ok"]:
            vp, (ml, bl), (mr, br) = geo["vp"], geo["left"], geo["right"]
            d = max(0.2, min(0.9, float(depth)))
            # perspective: a point on the road plane at depth d (0=bottom edge,
            # 1=vanishing point) projects to row vp_y + d*(H - vp_y)
            gy = int(vp[1] + d * (H - vp[1]))
            # road left/right edges at that row, then inset to sit inside the lane
            ledge = ml * gy + bl
            redge = mr * gy + br
            road_w = max(4.0, redge - ledge)
            # car is ~78% of the road width at this depth, capped by the frame
            cw = int(road_w * 0.78)
            cw = max(32, min(cw, int(W * 0.62), int((H * 0.44) * (car.width / car.height))))
            ch = max(16, int(cw * car.height / car.width))
            car = car.resize((cw, ch), Image.LANCZOS)
            cx = (ledge + redge) / 2                      # lane center
            gx = int(cx - cw / 2)
        else:
            # No one-point road. Try a FLAT side-on roadside (bright bottom
            # strip): anchor the wheels inside that strip at a reference-like
            # side-on scale so the car sits ON the asphalt, not floating on
            # mid-frame grass.
            strip_top = _flat_road_strip(scene)
            if strip_top is not None:
                strip_h = H - strip_top
                # side-on roadside framing (like the reference photo): large
                # car, wheels near the bottom of the frame
                cw = max(32, min(int(W * 0.80), int((H * 0.55) * (car.width / car.height))))
                ch = max(16, int(cw * car.height / car.width))
                car = car.resize((cw, ch), Image.LANCZOS)
                # wheels ~35% down into the strip (on the asphalt, not its top edge)
                gy = int(min(H - 4, strip_top + max(4.0, strip_h * 0.35)))
                gx = (W - cw) // 2
            else:
                # fallback: old brightness-based road-band guess (center-weighted)
                g = np.asarray(ImageOps.grayscale(scene.resize((W // 4, H // 4)))).astype(np.float32)
                gh = g.shape[0]
                band = g[int(gh * 0.55):, :]
                cw = max(32, int(W * 0.45))
                ch = max(16, int(cw * car.height / car.width))
                car = car.resize((cw, ch), Image.LANCZOS)
                gy = int((vp_y := H * 0.5) + 0.5 * (H - H * 0.5))   # mid-ground row
                gx = (W - cw) // 2
        out = scene.convert("RGBA").copy()
        cx = gx + cw // 2
        # --- tight contact shadow under the wheel line (grounds the car) ---
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh, "RGBA")
        sd.ellipse([gx + cw // 10, gy - ch // 18, gx + cw - cw // 10,
                    gy + ch // 9], fill=(0, 0, 0, 130))
        out = Image.alpha_composite(out, sh.filter(ImageFilter.GaussianBlur(max(2, ch // 16))))
        # --- soft drop shadow slightly back (away from camera) for depth ---
        sh2 = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sd2 = ImageDraw.Draw(sh2, "RGBA")
        sd2.ellipse([gx + cw // 7, gy - ch // 6, gx + cw - cw // 7,
                     gy + ch // 12], fill=(0, 0, 0, 55))
        out = Image.alpha_composite(out, sh2.filter(ImageFilter.GaussianBlur(max(3, ch // 8))))
        # --- asphalt reflection: flipped, masked, fading into the road ---
        refl = car.transpose(Image.FLIP_TOP_BOTTOM)
        fade = np.zeros((ch, cw), dtype=np.uint8)
        for r in range(ch):
            fade[r, :] = int(150 * max(0.0, 1.0 - r / (ch * 0.55)))
        refl_alpha = np.array(refl)[:, :, 3].copy()
        refl.putalpha(Image.fromarray(np.minimum(refl_alpha, fade).astype(np.uint8)))
        out.paste(refl, (gx, gy + 2), refl)
        # --- scene light-match: nudge the car's white balance toward the road tone ---
        road = np.asarray(scene.crop((max(0, gx - cw // 2), int(gy - ch * 0.3),
                                      min(W, gx + cw * 1.5), min(H, gy + ch)))
                                .resize((8, 8))).astype(np.float32)
        if road.size:
            tmean = road.reshape(-1, 3).mean(0)
            arr = np.array(car).astype(np.float32)
            opaque = arr[:, :, 3] > 16
            for c in range(3):
                arr[:, :, c] = np.where(opaque,
                                        np.clip(arr[:, :, c] + (tmean[c] - 128) * 0.16, 0, 255),
                                        arr[:, :, c])
            car = Image.fromarray(arr.astype(np.uint8))
        # --- paste the car, wheels anchored on the road row ---
        out.paste(car, (gx, gy - ch), car)
        return out.convert("RGB")
    except Exception as e:
        log.warning("scene composition failed (%s); using design over scene center", e)
        out = scene.convert("RGB").copy()
        cw = int(scene.width * 0.5)
        car = design.resize((cw, max(16, int(cw * design.height / design.width))), Image.LANCZOS)
        gx, gy = (scene.width - cw) // 2, int(scene.height * 0.6)
        out.paste(car, (max(0, gx), max(0, gy - car.height)))
        return out


async def _harmonize(img: Image.Image, w: int, h: int,
                     seed: int | None) -> Image.Image:
    """One low-strength i2i pass to melt the pasted car into the scene lighting.
    Falls back to the raw composite if the image model is unavailable — the
    demo must never hard-fail on the polish pass."""
    try:
        return await _img_generate(
            "Harmonize this car into the scene: match the scene's lighting, "
            "reflections and color temperature on the car body and glass; keep "
            "the exact same car, position and background. Photorealistic.",
            img, w, h, strength=0.45, seed=seed)
    except HTTPException as e:
        log.warning("harmonize skipped (%s)", e.detail)
        return img


# ---------------------------------------------------------------- stage 4: AI base sketch
async def _stage4(req: StageRequest):
    """Mario's 'BASE SKETCH: AI' box — front + rear DESIGN SKETCH of the uploaded
    vehicle plus N paint-ONLY colorways (deterministic recolor, so every colorway
    keeps the exact same view and geometry; per user v21 rule)."""
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "stage 4 requires a vehicle reference image (image)")
    w, h = _dims_from_image(img, req.width, req.height)
    t0 = time.time()

    def norm(im):
        if im.size == (w, h):
            return im
        return im.resize((w, h), Image.LANCZOS)

    def enc(im):
        return "data:image/png;base64," + base64.b64encode(_img_to_pil_bytes(norm(im))).decode()

    base_color = _detect_body_color(img)
    targets = _colorway_targets(base_color, req.colorways, req.color_targets,
                                req.requirement)
    sketch_tmpl = req.prompt_override or STAGE4_SKETCH_PROMPT
    if "{requirement}" in sketch_tmpl:
        sketch_prompt = sketch_tmpl.format(requirement=_fill("", req.requirement).strip())
    else:
        sketch_prompt = _fill(sketch_tmpl, req.requirement)

    # 1) front sketch (i2i at the proven 0.8 strength; pencil fallback if the
    #    image model is down or errors)
    try:
        front = await _img_generate(sketch_prompt, img, w, h, strength=0.8, seed=req.seed)
        sketch_mode = "model"
    except HTTPException as e:
        log.warning("stage4 sketch model failed (%s); pencil fallback", e.detail)
        front, sketch_mode = _pencil_sketch(img), "pencil-fallback"
    # 2) rear sketch: sketch wording prepended to the anti-top-down, color-anchored
    #    opposite prompt. Strength = ROT_STRENGTH: on Qwen-GGUF, 0.8 anchors the
    #    model to the SAME face (user report: photoreal front sketch came back as
    #    a second front view); only full strength commits the 180° rotation.
    rear_prompt = (req.prompt_override_rear
                   or SKETCH_STYLE_PREFIX + _opposite_prompt(base_color))
    try:
        rear = await _img_generate(rear_prompt, front, w, h,
                                   strength=ROT_STRENGTH, seed=(req.seed or 0) + 2)
    except HTTPException as e:
        log.warning("stage4 rear sketch model failed (%s); pencil(rear-of-front) fallback",
                    e.detail)
        rear = _opposite_sketch_fallback(front, w, h)
    # 3) colorways: deterministic paint-only recolors of BOTH sketches
    colorways = []
    for t in targets:
        colorways.append({
            "color": t,
            "front": enc(_recolor_body(front, t)),
            "rear": enc(_recolor_body(rear, t)),
        })
    elapsed = (time.time() - t0) * 1000
    return {
        "stage": 4,
        "results": {
            "base_sketch": enc(front),
            "base_sketch_rear": enc(rear),
            "colorways": colorways,
        },
        "base_color": base_color,
        "colorways_targets": targets,
        "sketch_mode": sketch_mode,
        "requirement": req.requirement,
        "size": {"w": w, "h": h},
        "prompt": sketch_prompt,
        "elapsed_ms": round(elapsed),
    }


def _opposite_sketch_fallback(front: Image.Image, w: int, h: int) -> Image.Image:
    """No-model rear sketch: mirrored, mildly perspective-squashed pencil pass —
    visually a 'concept flip', clearly not the front sketch."""
    flipped = ImageOps.mirror(_pencil_sketch(front))
    fh = int(flipped.height * 0.92)
    canvas = Image.new("RGB", flipped.size, "white")
    canvas.paste(flipped.resize((flipped.width, fh), Image.LANCZOS),
                 (0, (flipped.height - fh) // 2))
    return canvas


# ---------------------------------------------------------------- stage 5: compose + quick viz
async def _stage5(req: StageRequest):
    """Mario's 'REF 2 + DESIGN combined' and 'QUICK VIZ' boxes in one call:
    compose the (optionally mock-designer-edited) design into a scene, then
    produce main / another_view / another_color stills from it."""
    design = _b64_image_to_pil(req.design_image) or _b64_image_to_pil(req.image)
    if design is None:
        raise HTTPException(400, "stage 5 requires the chosen design (design_image)")
    scene = None
    scene_src = req.scene_image or req.reference_image
    if scene_src:
        scene = _b64_image_to_pil(scene_src)
    if scene is None:
        if req.use_mock_scene:
            scene = _load_mock_scene()
        if scene is None:
            raise HTTPException(400, "stage 5 requires a scene: send scene_image "
                                     "(or reference_image), or use_mock_scene=true "
                                     "(bundled sample_scene_dusk.png)")
    t0 = time.time()
    if req.use_mock_design:
        design = _mock_designer_edit(design)

    # ---- scene-gen: single-ref i2i scene generation from the ORIGINAL vehicle
    # photo — the same working path as the pipeline client (run_pipeline.py).
    # The studio product photo is the ONLY reference; the model generates the
    # car already integrated into the described scene (grounded, contact shadow,
    # lighting-matched, no paste rectangle). Verified live 2026-10-01 at
    # strength 0.85. Composite+harmonize remains the fallback (model down, or
    # no car_photo supplied).
    mode_wanted = (req.mode or "").strip().lower()
    car_photo = _b64_image_to_pil(req.car_photo)
    if (mode_wanted == "scene-gen" or (mode_wanted == "" and car_photo is not None)) \
            and car_photo is not None:
        sw, sh = _dims_from_image(scene, req.width, req.height)
        scene_desc = (req.requirement or "").strip()
        gen_prompt = (
            "Photorealistic automotive photography of THIS EXACT car (same model, "
            "same paint, same design, same wheels, same proportions, same view "
            "angle) integrated into this scene: "
            + (scene_desc or "a realistic road environment")
            + ". The car is on the road, wheels on the surface, correct "
              "perspective and grounding, matching the scene's lighting, "
              "reflections and color temperature. Full scene, not a studio shot. "
              "STRICTLY KEEP ALL THE CAR DESIGN ELEMENTS, DO NOT IMPROVIZE WITH "
              "THE CAR DESIGN.")
        mode5 = "scene-gen"
        try:
            composed = await _img_generate(gen_prompt, car_photo, sw, sh,
                                           strength=0.85, seed=req.seed)
        except HTTPException as e:
            log.warning("stage5 scene-gen failed (%s); deterministic composite fallback",
                        e.detail)
            composed = _compose_scene(design, scene)
            mode5 = "composite-fallback"
            try:
                composed = await _harmonize(composed, sw, sh, req.seed)
            except HTTPException:
                pass
        w, h = _dims_from_image(composed, req.width, req.height)
        def nenc(im):
            if im.size == (w, h):
                return im
            return im.resize((w, h), Image.LANCZOS)
        def eenc(im):
            return "data:image/png;base64," + base64.b64encode(_img_to_pil_bytes(nenc(im))).decode()
        composed = nenc(composed)
        # main: refine via harmonize + L*a*b* grade from the scene (same as
        # composite path) so the quick-viz main matches the pipeline look.
        try:
            main = await _harmonize(composed, w, h, req.seed)
        except HTTPException:
            main = composed
        main = _apply_color_grade(main, scene)
        # another view: prompt-driven re-angle of the composed scene (pinned).
        view = (req.view_angle or "side").strip() or "side"
        alt_prompt = (req.prompt_override_alt or alt_view_prompt(view))
        try:
            another_view = await _img_generate(alt_prompt, main, w, h,
                                               strength=ALT_VIEW_STRENGTH,
                                               seed=(req.seed or 0) + 5)
        except HTTPException as e:
            log.warning("stage5 scene-gen alt view failed (%s); returning main", e.detail)
            another_view = main
        body_color = _detect_body_color(main)
        color_target = _target_variant_color(body_color, req.requirement)
        if color_target == body_color:
            color_target = _BODY_COLOR_CONTRAST.get(body_color, "red")
        another_color = _recolor_body(main, color_target)
        return {
            "stage": 5,
            "results": {"composed": eenc(composed), "main": eenc(main),
                        "another_view": eenc(another_view),
                        "another_color": eenc(another_color)},
            "mode": mode5,
            "mock_design_applied": bool(req.use_mock_design),
            "mock_scene_used": bool(req.use_mock_scene and not scene_src),
            "view_angle": view,
            "base_color": body_color,
            "color_target": color_target,
            "viz_mode": "scene-gen",
            "requirement": req.requirement,
            "size": {"w": w, "h": h},
            "prompt": gen_prompt,
            "elapsed_ms": round((time.time() - t0) * 1000),
        }

    composed = _compose_scene(design, scene)
    w, h = _dims_from_image(composed, req.width, req.height)

    def norm(im):
        if im.size == (w, h):
            return im
        return im.resize((w, h), Image.LANCZOS)

    def enc(im):
        return "data:image/png;base64," + base64.b64encode(_img_to_pil_bytes(norm(im))).decode()

    composed = norm(composed)
    # main: refine + L*a*b* grade from the scene (deterministic look-match on top
    # of the model harmonize pass) — fall back through harmonize -> raw composite
    style = ""
    desc = ""
    if _llm_endpoint_and_key()[0]:
        desc = await _describe_style(scene)
        if desc:
            style = (f"Match the scene look — {desc}. ")
    if req.prompt_override:
        main_prompt = _fill(req.prompt_override, req.requirement)
    else:
        main_prompt = STAGE5_MAIN_PROMPT.format(style=style,
                                                requirement=_fill("", req.requirement).strip())
    try:
        main = await _img_generate(main_prompt, composed, w, h, strength=0.45,
                                   seed=req.seed)
        viz_mode = "model"
    except HTTPException as e:
        log.warning("stage5 main model failed (%s); composite+grade fallback", e.detail)
        main = await _harmonize(composed, w, h, req.seed)
        viz_mode = "composite-only"
    main = _apply_color_grade(main, scene)

    # another view: prompt-driven re-angle of the composed scene (pinned position)
    view = (req.view_angle or "side").strip() or "side"
    alt_prompt = (req.prompt_override_alt
                  or alt_view_prompt(view))
    try:
        another_view = await _img_generate(alt_prompt, main, w, h, strength=ALT_VIEW_STRENGTH,
                                           seed=(req.seed or 0) + 5)
    except HTTPException as e:
        log.warning("stage5 alt view model failed (%s); returning main", e.detail)
        another_view = main

    # another color: deterministic paint-only recolor of the composed still
    body_color = _detect_body_color(main)
    color_target = _target_variant_color(body_color, req.requirement)
    if color_target == body_color:
        color_target = _BODY_COLOR_CONTRAST.get(body_color, "red")
    another_color = _recolor_body(main, color_target)

    elapsed = (time.time() - t0) * 1000
    return {
        "stage": 5,
        "results": {
            "composed": enc(composed),
            "main": enc(main),
            "another_view": enc(another_view),
            "another_color": enc(another_color),
        },
        "mock_design_applied": bool(req.use_mock_design),
        "mock_scene_used": bool(req.use_mock_scene and not scene_src),
        "view_angle": view,
        "base_color": body_color,
        "color_target": color_target,
        "viz_mode": viz_mode,
        "requirement": req.requirement,
        "size": {"w": w, "h": h},
        "prompt": main_prompt,
        "elapsed_ms": round(elapsed),
    }


# ================================================================ per-step flow API (v29)
# Gemini-parity semantics: each step = (prompt, declared inputs) -> model.
# Deterministic machinery is a DECLARED FALLBACK (reported in `mode`), never
# the default path — the same rule the Gemini reference stream obeys.

def _flow_enc(im: Image.Image, w: int, h: int) -> str:
    """Normalized data-URI encoder (all flow outputs share one size per call)."""
    if im.size == (w, h):
        out = im
    else:
        out = im.resize((w, h), Image.LANCZOS)
    return "data:image/png;base64," + base64.b64encode(_img_to_pil_bytes(out)).decode()


@app.post("/api/flow/sketch")
async def flow_sketch(req: FlowSketchRequest):
    """Step 1 — front AND rear design sketches, EACH generated directly from
    the uploaded photo (no front->rear chaining — the Gemini reference does
    exactly this: one prompt per view, the photo as the only input)."""
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "sketch requires a vehicle photo (image)")
    w, h = _dims_from_image(img, req.width, req.height)
    t0 = time.time()
    base_color = _detect_body_color(img)
    front_prompt = (_fill(req.prompt_front, req.requirement) if req.prompt_front
                    else STAGE4_SKETCH_PROMPT.format(requirement=_fill("", req.requirement).strip()))
    rear_prompt = (req.prompt_rear or SKETCH_STYLE_PREFIX + _opposite_prompt(base_color))
    modes = {}
    try:
        front = await _img_generate(front_prompt, img, w, h, strength=0.8, seed=req.seed)
        modes["front"] = "model"
    except HTTPException as e:
        log.warning("flow.sketch front model failed (%s); pencil fallback", e.detail)
        front, modes["front"] = _pencil_sketch(img), "pencil-fallback"
    try:
        rear = await _img_generate(rear_prompt, img, w, h,
                                   strength=ROT_STRENGTH, seed=(req.seed or 0) + 2)
        modes["rear"] = "model"
    except HTTPException as e:
        log.warning("flow.sketch rear model failed (%s); fallback", e.detail)
        rear, modes["rear"] = _opposite_sketch_fallback(img, w, h), "pencil-fallback"
    return {
        "step": "sketch",
        "results": {"front": _flow_enc(front, w, h),
                    "rear": _flow_enc(rear, w, h)},
        "mode": modes,
        "base_color": base_color,
        "requirement": req.requirement,
        "size": {"w": w, "h": h},
        "prompt_front": front_prompt,
        "prompt_rear": rear_prompt,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


@app.post("/api/flow/combine")
async def flow_combine(req: FlowCombineRequest):
    """Step 3 — integrate the design into the scene.

    ENGINE CONSTRAINT (verified in-pod 2026-09-29): the deployed Qwen-Image-Edit
    GGUF is a SINGLE-reference editor. A second image is parsed (HTTP 200) but
    SILENTLY DISCARDED — so the model cannot integrate [design, scene] directly.
    Therefore the actual integration is the deterministic geometry-locked
    composite (the ONLY thing that uses both images on this engine), and the
    model is applied where it genuinely works: a low-strength harmonize pass
    that melts the pasted car into the scene's lighting. This is the PRIMARY
    combine path on PCAI, NOT a fallback. Gemini's native two-image integration
    is a declared engine difference (see run_summary engine_differences).
    """
    design = _b64_image_to_pil(req.design_image)
    if design is None:
        raise HTTPException(400, "combine requires the design (design_image)")
    scene = _b64_image_to_pil(req.scene_image)
    if scene is None and req.use_mock_scene:
        scene = _load_mock_scene()
    if scene is None:
        raise HTTPException(400, "combine requires the scene (scene_image), or use_mock_scene=true")
    t0 = time.time()
    sw, sh = _dims_from_image(scene, req.width, req.height)
    # ---- mode=scene-gen: single-ref i2i SCENE GENERATION (PRIMARY integration
    # path for studio product photos). The car photo is the ONLY reference; the
    # model generates the car already integrated into the described scene
    # (grounded, contact shadow, lighting-matched — no paste rectangle). At
    # strength 0.6 the model keeps the studio background (scene ignored);
    # 0.85 generates the scene (verified live 2026-10-01). Falls back to the
    # deterministic composite when the image model is down.
    if (req.mode or "").strip().lower() == "scene-gen":
        scene_desc = (req.scene_prompt or "").strip() or (req.requirement or "").strip()
        gen_prompt = req.prompt or (
            "Photorealistic automotive photography of THIS EXACT car (same model, "
            "same paint, same design, same wheels, same proportions, same view "
            "angle) integrated into this scene: "
            + (scene_desc or "a realistic road environment")
            + ". The car is on the road, wheels on the surface, correct "
              "perspective and grounding, matching the scene's lighting, "
              "reflections and color temperature. Full scene, not a studio shot. "
              "STRICTLY KEEP ALL THE CAR DESIGN ELEMENTS, DO NOT IMPROVIZE WITH "
              "THE CAR DESIGN.")
        gen_prompt = _fill(gen_prompt, req.requirement if req.scene_prompt else "")
        try:
            composed = await _img_generate(gen_prompt, design, sw, sh,
                                           strength=0.85, seed=req.seed)
            mode = "scene-gen"
        except HTTPException as e:
            log.warning("flow.combine scene-gen failed (%s); "
                        "deterministic composite fallback", e.detail)
            composed = _compose_scene(design, scene, depth=float(req.depth))
            mode = "composite-fallback"
            gen_prompt = f"scene-gen prompt (model unavailable): {gen_prompt}"
        return {
            "step": "combine",
            "results": {"composed": _flow_enc(composed, sw, sh)},
            "mode": mode,
            "requirement": req.requirement,
            "size": {"w": sw, "h": sh},
            "prompt": gen_prompt,
            "elapsed_ms": round((time.time() - t0) * 1000),
        }
    # ---- default mode=composite+harmonize (legacy path, unchanged) ----
    prompt = req.prompt or STAGE5_MAIN_PROMPT.format(
        style="", requirement=_fill("", req.requirement).strip())
    prompt = _fill(prompt, req.requirement)
    # Deterministic composite = the integrator (uses BOTH images, geometry-locked).
    composed = _compose_scene(design, scene, depth=float(req.depth))
    # Model pass (single-ref, where it genuinely works): harmonize the car into
    # the scene lighting. Degrades to the raw composite if the model is down.
    mode = "composite+harmonize"
    try:
        composed = await _harmonize(composed, sw, sh, req.seed)
    except HTTPException as e:
        log.warning("flow.combine harmonize skipped (%s); returning raw composite", e.detail)
        mode = "composite"
    return {
        "step": "combine",
        "results": {"composed": _flow_enc(composed, sw, sh)},
        "mode": mode,
        "requirement": req.requirement,
        "size": {"w": sw, "h": sh},
        "prompt": prompt,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


@app.post("/api/flow/view")
async def flow_view(req: FlowViewRequest):
    """Step 4 — prompt-driven re-angle of a still: same car, same scene,
    a new viewing angle (Gemini alt_view semantics)."""
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "view requires an input image (image)")
    w, h = _dims_from_image(img, req.width, req.height)
    t0 = time.time()
    view = (req.view_angle or "side").strip() or "side"
    prompt = (req.prompt if req.prompt
              else alt_view_prompt(view))
    prompt = _fill(prompt, req.requirement)
    try:
        out = await _img_generate(prompt, img, w, h,
                                  strength=ALT_VIEW_STRENGTH, seed=req.seed)
        mode = "model"
    except HTTPException as e:
        log.warning("flow.view model failed (%s); returning input", e.detail)
        out, mode = img, "input-fallback"
    return {
        "step": "view",
        "results": {"view": _flow_enc(out, w, h)},
        "mode": mode,
        "view_angle": view,
        "requirement": req.requirement,
        "size": {"w": w, "h": h},
        "prompt": prompt,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


@app.post("/api/flow/recolor")
async def flow_recolor(req: FlowRecolorRequest):
    """Step 5 — prompt-driven body recolor: `re-color the car body <color>,
    keep every design element` (Gemini recolor semantics). The model is
    primary (moderate strength to anchor geometry); the deterministic HSV
    repaint is the DECLARED FALLBACK when the model is unavailable."""
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "recolor requires an input image (image)")
    w, h = _dims_from_image(img, req.width, req.height)
    t0 = time.time()
    target = (req.color or "").strip() or _target_variant_color(
        _detect_body_color(img), req.requirement)
    prompt = (req.prompt
              or f"Re-color the car body to {target}. Keep EXACTLY the same car: "
                 f"same model, same view angle, same proportions, same wheels and "
                 f"details, same background — only the body paint changes. "
                 f"Photorealistic.")
    prompt = _fill(prompt, req.requirement)
    try:
        out = await _img_generate(prompt, img, w, h, strength=0.6, seed=req.seed)
        mode = "model"
    except HTTPException as e:
        log.warning("flow.recolor model failed (%s); deterministic repaint fallback", e.detail)
        out, mode = _recolor_body(img, target), "deterministic-fallback"
    return {
        "step": "recolor",
        "results": {"recolor": _flow_enc(out, w, h)},
        "mode": mode,
        "color_target": target,
        "requirement": req.requirement,
        "size": {"w": w, "h": h},
        "prompt": prompt,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


@app.post("/api/flow/scene-clean")
async def flow_scene_clean(req: FlowSceneCleanRequest):
    """Step 0 — turn a reference photo that CONTAINS a car into a CLEAN scene:
    remove the vehicle, keep the road and environment intact. Single-reference
    i2i (the image model is single-reference on PCAI). This is the PCAI way to
    use a 'style/scene reference' photo that happens to show a car: clean it
    first, then feed the result into /api/flow/combine as the scene."""
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "scene-clean requires an input image (image)")
    w, h = _dims_from_image(img, req.width, req.height)
    t0 = time.time()
    prompt = (req.prompt
              or "Remove the car (and any vehicles) from this photo completely. "
                 "Keep EXACTLY the same road, pavement, lane markings, horizon, "
                 "buildings, trees, sky and lighting — only the vehicle is "
                 "erased, replaced by the surrounding road and environment. "
                 "Photorealistic, no car, no vehicle, no ghost of the car.")
    prompt = _fill(prompt, req.requirement)
    out = await _img_generate(prompt, img, w, h, strength=req.strength, seed=req.seed)
    return {
        "step": "scene-clean",
        "results": {"clean": _flow_enc(out, w, h)},
        "mode": "model",
        "size": {"w": w, "h": h},
        "prompt": prompt,
        "requirement": req.requirement,
        "strength": req.strength,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


@app.post("/api/flow/video")
async def flow_video(req: FlowVideoRequest):
    """Step 6 — final I2V of a still (prompt-driven). Wan2.2 has no
    first/last-frame mode, so the Gemini P6 two-frame variant is not
    expressible on this engine — a model-inherent difference, declared in
    `mode`."""
    if not MLIS_URL:
        raise HTTPException(503, "MLIS_VIDEO_GENERATION_URL not configured")
    img = _b64_image_to_pil(req.image)
    if img is None:
        raise HTTPException(400, "video requires a design image (image)")
    prompt = _fill(req.prompt or STAGE3_VIDEO_PROMPT, req.requirement)
    t0 = time.time()
    p = Params(style_strength=req.style_strength, motion_strength=req.motion_strength)
    resolution = req.resolution or DEFAULT_RESOLUTION
    if resolution not in [r.lower() for r in SUPPORTED_RESOLUTIONS]:
        resolution = DEFAULT_RESOLUTION
    mp4, guidance, gen_meta = await generate_video(prompt, img, p, resolution)
    job_id = uuid.uuid4().hex
    out_path = os.path.join(RESULTS_DIR, f"{job_id}.mp4")
    with open(out_path, "wb") as f:
        f.write(mp4)
    GEN_TOTAL.labels(status="success").inc()
    return {
        "step": "video",
        "results": {"video_url": f"/api/results/{job_id}.mp4", "job_id": job_id},
        "mode": "i2v",
        "prompt": prompt,
        "requirement": req.requirement,
        "guidance_scale": round(guidance, 2),
        "size": {"w": img.width, "h": img.height},
        "elapsed_ms": round((time.time() - t0) * 1000),
        "metrics": {"video_generation_ms": round(gen_meta["gen_ms"]),
                    "gpu_utilization": _gpu_gauge_snapshot()},
    }


@app.get("/api/mock/scene")
async def mock_scene():
    """Bundled sample scene (REF 2 mock) for the frontend dropzone preview."""
    if not MOCK_SCENE_FILE.exists():
        raise HTTPException(404, "sample scene not found")
    return FileResponse(MOCK_SCENE_FILE, media_type="image/png")


# ---------------------------------------------------------------- templates
TEMPLATES_DIR = Path(os.path.join(RESULTS_DIR, "templates"))
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/templates")
async def list_templates():
    templates = []
    for f in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                t = json.load(fh)
            # Don't return the heavy image payloads in the listing.
            templates.append({
                "id": t.get("id", f.stem),
                "name": t.get("name", f.stem),
                "description": t.get("description", ""),
                "created": t.get("created", ""),
                "has_stage1": bool(t.get("stages", {}).get("1", {}).get("images")),
                "has_stage2": bool(t.get("stages", {}).get("2", {}).get("images")),
                "has_stage3": bool(t.get("stages", {}).get("3", {}).get("image")),
                "final": t.get("final", {}),
            })
        except Exception:
            continue
    return {"templates": templates}


@app.post("/api/templates")
async def save_template(req: TemplateRequest):
    if not req.name.strip():
        raise HTTPException(400, "template name required")
    template_id = uuid.uuid4().hex[:12]
    template = {
        "id": template_id,
        "name": req.name.strip(),
        "description": req.description,
        "created": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "1": {"images": req.stage1_images, "requirement": req.stage1_requirement},
            "2": {"images": req.stage2_images, "requirement": req.stage2_requirement},
            "3": {"image": req.stage3_image, "requirement": req.stage3_requirement},
            "4": {"images": req.stage4_images, "requirement": req.stage4_requirement},
            "5": {"images": req.stage5_images, "requirement": req.stage5_requirement},
            "gate": {"design_image": req.design_image, "use_mock_design": req.use_mock_design,
                     "scene_image": req.scene_image, "use_mock_scene": req.use_mock_scene,
                     "view_angle": req.view_angle},
        },
        "final": req.final,
    }
    with open(os.path.join(TEMPLATES_DIR, f"{template_id}.json"), "w") as f:
        json.dump(template, f)
    log.info("template saved: %s (%s)", req.name, template_id)
    return {"id": template_id, "name": req.name, "created": template["created"]}


@app.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    safe = re.sub(r"[^a-zA-Z0-9-]", "", template_id)
    path = os.path.join(TEMPLATES_DIR, f"{safe}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "template not found")
    with open(path) as f:
        return json.load(f)


@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: str):
    safe = re.sub(r"[^a-zA-Z0-9-]", "", template_id)
    path = TEMPLATES_DIR / f"{safe}.json"
    if path.exists():
        os.remove(path)
        return {"deleted": True}
    raise HTTPException(404, "template not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
