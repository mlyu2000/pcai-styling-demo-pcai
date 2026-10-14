# PCAI Styling Demo — API reference (for the automation skill)

Base (self-hosted): `https://styling-demo.aie.cs1.ctc.sg.lab` (lab ingress, **self-signed** → unverified SSL).

**Primary: the per-step flow API (backend v32+)** — one endpoint per step,
prompt-driven with exactly the declared inputs (Gemini reference semantics).
The legacy stage API (`/api/workflow/stage`, prompt_override* fields) still
exists for the app UI; the skill no longer uses it.

**Reconciled step model** (identical on both engines): sketch_front, sketch_rear,
combine, alt_view, recolor, video → `/api/flow/sketch`, `/combine`, `/view`,
`/recolor`, `/video`.

## 1. Flow endpoints — `POST /api/flow/<step>`

One call per step; **synchronous** (blocks for the whole step — minutes). Long
timeouts required. Every response carries `mode` (which path actually ran —
model vs declared deterministic fallback), the exact rendered `prompt`, and
`elapsed_ms`.

### `POST /api/flow/sketch` — steps `sketch_front` + `sketch_rear`
```json
{
  "image": "<data: URI of front-view photo>",
  "prompt_front": "<sketch_front, rendered>",
  "prompt_rear": "<sketch_rear, rendered>",
  "requirement": "freeform (appended by backend _fill)"
}
```
Response: `results.front`, `results.rear`; `mode: {front: ..., rear: ...}`
(each `model` | `pencil-fallback`), `prompt_front`, `prompt_rear`.
- **Gemini parity**: front AND rear are each generated DIRECTLY from the photo
  (independent model calls — no chaining).
- **Rear-view gate**: `results.rear` must be an actual rear (180°).
  If the model returns a second front, that is a model-quality issue, not a
  pipeline mechanism — verify visually.

### `POST /api/flow/combine` — step `combine`
```json
{
  "design_image": "<data: URI of car photo (base) or front sketch>",
  "scene_image": "<data: URI of scene (REF 2)>",
  "mode": "composite+harmonize | scene-gen   (default: composite+harmonize)",
  "scene_prompt": "<scene-gen only: scene description>",
  "prompt": "<optional; scene-gen default builds a 'THIS EXACT car in scene' prompt>",
  "requirement": "freeform"
}
```
Response: `results.composed`; `mode`: `composite+harmonize` | `composite`
(harmonize model down) | `scene-gen` | `composite-fallback` (scene-gen model
down → deterministic composite).
- **PCAI engine constraint (verified in-pod)**: the Qwen-Image-Edit GGUF is
  SINGLE-REFERENCE — a second image (repeated `image` field or `images[]`)
  returns HTTP 200 but is silently discarded. So the MODEL cannot integrate
  [design, scene] directly.
- **`mode=composite+harmonize`** (default, backward-compat): geometry-locked
  deterministic mask-paste composite (the only path that uses both images),
  then a single-reference model harmonize pass. A deterministic paste of a
  STUDIO product photo reads as a floating rectangle even with the v33/v35
  geometry fix.
- **`mode=scene-gen`** (visually integrated path, verified live 2026-10-01):
  single-ref i2i SCENE GENERATION — the car photo is the ONLY reference and
  the model generates the car already integrated into the scene described by
  `scene_prompt`/`requirement` (grounded, contact shadow, lighting-matched, no
  paste edge). Strength 0.85 (0.6 keeps the studio background; 0.95 is for
  car erasure via /api/flow/scene-clean). This mirrors the customer's Gemini
  workflow (one car ref + scene prompt → generated scene). `scene_image` is
  only used on the model-failure fallback. The `run_pipeline.py` client uses
  `mode=scene-gen` by default.
- **Gemini parity**: gemini does native two-image model integration of
  [design, scene] (Mario's P1/P3 pattern) — a declared engine difference,
  recorded in `run_summary.json → engine_differences`.

### `POST /api/flow/view` — step `alt_view`
```json
{
  "image": "<data: URI of combined still>",
  "prompt": "<alt_view, rendered>",
  "view_angle": "side | 3/4 front | 3/4 rear | rear",
  "requirement": "freeform"
}
```
Response: `results.view`; `mode` (`model` | `input-fallback`).
- **Explicit per-view prompts (v44/v45)**: the backend re-angles using
  `alt_view_prompt(view)` — explicit direction-anchored prompts naming the
  exact car surfaces that must face the camera (e.g. side = "both wheels in
  perfect side profile, front/rear ends edge-on and NOT visible"). A generic
  "change camera angle" prompt drifts because the model anchors to the input
  orientation. Keep strength 0.85 (`ALT_VIEW_STRENGTH`).

### `POST /api/flow/recolor` — step `recolor`
```json
{
  "image": "<data: URI of combined still>",
  "prompt": "<recolor, rendered>",
  "color": "silver",
  "requirement": "freeform"
}
```
Response: `results.recolor`; `mode` (`model` | `deterministic-fallback`),
`color_target`.
- **Gemini parity**: prompt-driven model repaint ("re-color the body to X,
  keep every design element", strength 0.6); deterministic HSV repaint is a
  DECLARED FALLBACK only.

### `POST /api/flow/video` — step `video`
```json
{
  "image": "<data: URI of combined still>",
  "prompt": "<video, rendered>",
  "requirement": "freeform",
  "resolution": "480p | 720p"
}
```
Response: `results.video_url` = `/api/results/<job>.mp4` → **strip leading
`/api/`** before fetching; `mode: i2v`; `guidance_scale`; `metrics`.
- **Engine difference (model-inherent)**: I2V of the combined still (Wan2.2
  has no first/last-frame mode, so the Gemini P6 two-frame morph is not
  expressible here).
- Video gate: > 1 MB, contains `ftyp` and `avcC` (H.264).

## 2. Mock scene — `GET /api/mock/scene`
Returns the bundled dusk street scene (PNG). Use when no `--scene` is supplied (the REF-2 box).

## 3. Config — `GET /api/config`
Health/feature probe: `{"imageGen": {"sdCppExtra": true, ...}, "video": {...}, ...}`. Use to confirm the service is up before a run.

## 3b. Per-role providers (backend v43+)

Three model roles — **brain** (LLM/vision), **i2v** (video), **image** (image-edit) —
each routable independently to an external OpenAI-compatible endpoint. Keys are
**request-scoped only** (never persisted server-side). Only `GenerateRequest`
(`/api/generate`) and `StageRequest` (`/api/workflow/stage`) accept the
`provider` field; the `/api/flow/*` endpoints this skill drives do **NOT**
(always on-prem).

```json
// provider field value (per-role)
{
  "brain": {"base_url": "...", "api_key": "***", "model": "..."},
  "i2v":   {"base_url": "...", "api_key": "***", "model": "..."},
  "image": {"base_url": "...", "api_key": "***", "model": "..."}
}
```
- Leave a role empty → that role stays **on-prem** (MLIS).
- Legacy flat `{base_url, api_key, model}` (no role keys) → mapped to `brain`
  only (back-compat).
- Endpoints: `GET /api/providers/onprem-models` (on-prem catalog),
  `GET /api/providers/models`, `POST /api/providers/test?role=<brain|i2v|image>`
  (echoes role + graceful unreachable error shape).

## 4. Gemini provider (Google AI Studio)

Base: `https://generativelanguage.googleapis.com/v1beta`. Auth: header `x-goog-api-key: $GEMINI_API_KEY` (env only — never print/write).

### Image edit (Nano Banana) — `POST /models/{image_model}:generateContent`
```json
{"contents":[{"role":"user","parts":[
  {"text":"<prompt>"},
  {"inline_data":{"mime_type":"image/jpeg","data":"<b64 img 1>"}},
  {"inline_data":{"mime_type":"image/jpeg","data":"<b64 img 2>"} }
]}]}
```
- Image out in `candidates[0].content.parts[].inline_data.data` (base64).
- **Two-image combine** = Mario's P1/P3 pattern: [design, scene] in one call.

### Video (Veo) — async
1. `POST /models/{video_model}:predictLongRunning` with
   ```json
   {"instances":[{"first_frame":{"image_bytes_b64":"<b64 front>","mime_type":"image/jpeg"},
                    "last_frame":{"image_bytes_b64":"<b64 rear>","mime_type":"image/jpeg"},
                    "prompt":"<video prompt>"}]}
   ```
2. Response `name` = op name → `GET /{op_name}` until `done`.
3. `response.generateVideoResponse.generatedVideos[0].video.uri` → download the MP4.
- `first_frame` = sketch_front, `last_frame` = sketch_rear → matches P6 "camera movement from first image until to second image".

## Model defaults (cheap tier — user decision)
- Gemini image: `gemini-3.1-flash-image`
- Gemini video: `veo-3.1-fast-generate-preview` (3 s)
- Self-hosted image: Qwen-Image-Edit GGUF (MLIS) · self-hosted video: Wan2.2-TI2V-5B (MLIS)
