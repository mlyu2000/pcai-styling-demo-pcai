# PCAI Styling Demo — Project Description

**Customer:** Astra Daihatsu Motor (ADM) — exterior automotive styling
**Platform:** HPE Private Cloud AI (PCAI) on CS1 — HPE Ezmeral Kubernetes 1.32 (airgapped)
**Owner:** HPE Presales / Solution Architecture (Eric Yu, HPE)
**Code:** `github.com/mlyu2000/pcai-styling-demo` — branch `feat/external-api-provider`
**Current build:** backend **v45** · frontend **v23** (live: `styling-demo.aie.cs1.ctc.sg.lab`)
**Date:** 2026-10-09 · **Status:** validated PoC / live demo

---

## 1. What this project is (one paragraph)

The PCAI Styling Demo is a generative-AI styling tool that takes **Mario's actual design
workflow** — the IDD ("Exterior New Design") process ADM's designers use today in
Photoshop + Gemini — and runs it **end-to-end on HPE Private Cloud AI**: a front-view
car photo goes in, and a polished **styling video** comes out, through six model-driven
steps (sketch → design → scene → view/recolor → video). It demonstrates that ADM's
styling workflow can be accelerated, made repeatable, and run **fully on-prem**
(data never leaves the datacenter), while also showing the exact same workflow can be
executed against public models (Google Gemini) through a single API — proving the
platform is model-agnostic, not vendor-locked.

## 2. Business perspective

### 2.1 The customer's problem

Today, ADM's design team (Mario and colleagues) produce styling explorations by hand:
sketch the car in Photoshop, combine it into a scene reference, re-angle, recolor, and
animate a camera sweep. They run this in **Gemini (Nano Banana + Veo) with hand-written
prompts** (six verbatim prompts, P1–P6, captured from their actual workflow — see
`skill/.../prompts/customer_adm.yaml`). Pain points:

- **Time.** Each exploration is manual: hours of Photoshop work per concept, per view.
- **Inconsistency.** Prompt quality drives output quality; the same request produces
  different results depending on the exact wording.
- **Data governance.** Designs are customer IP. Sending them to a public model API is a
  governance question ADM must manage for every asset.
- **No repeatability.** A good result can't be saved, shared, or re-run by another
  designer with the same settings.

### 2.2 What the demo proves (the "so what")

| Demo capability | Business value |
|---|---|
| Full IDD flow on-prem (Qwen-Image-Edit GGUF + Wan2.2 I2V + qwen3-8-27b vision LLM, all on CS1 GPUs) | **Design IP stays in ADM's datacenter** — no egress to public APIs required |
| Same flow on Google Gemini with the same prompts | **Model-agnostic workflow** — ADM keeps the option to use public or private models per asset class |
| Per-role provider routing (brain / image / video independently on-prem or external) | **Hybrid is a first-class citizen** — e.g. external vision brain + on-prem video |
| Request-scoped external keys (never persisted server-side) | **Meets ADM's security posture**: no stored credentials in the platform |
| Saved templates (one-click re-run of an entire workflow run) | **Repeatable, shareable design exploration** — a "style" becomes a team asset |
| Prompt packs: `tuned` (our production prompts) vs `customer_adm` (Mario's verbatim) | **Like-for-like comparison** of their current process vs the platform process — the evidence for the pitch |
| OpenClaw agent skill (`pcai-styling-video-automation`) | **The workflow becomes an agent-driven service** — "make me the styling video for this photo" with no UI |
| Deterministic fallbacks with declared `mode` on every step | The demo **never hard-fails** in front of a customer — every degradation is visible and honest |

### 2.3 The sizing story (why ADM buys more)

The production scaling model is deliberately simple and was the driver of the hardware
conversation: **1 GPU ≈ 1 concurrent stylist** (vLLM-Omni queues the rest). With four
A100-40GB GPUs on CS1 today, the demo runs; a production rollout with N concurrent
stylists needs N video GPUs. The 40GB cards cap the model choices (Wan2.2-5B, not
HunyuanVideo-1.5 which needs ~100 GiB), which is the honest basis for the 80GB-class
hardware ask in the roadmap (Phase 2).

### 2.4 Position in the HPE account

This is a **reference PoC for PCAI** (Private Cloud AI): it exercises the full
Ezmeral stack — MLIS/Aioli for model serving, Knative for scale-to-0, Istio for
routing, NFS PVCs for weights and results, LiteLLM for the LLM proxy, EzApps tile
registration, and an OpenClaw agent — so it doubles as the technical proof that PCAI
can host a real GenAI creative workload, and as the commercial proof for ADM's styling
organization.

## 3. Technical perspective

### 3.1 System overview

> Diagram: **`docs/diagrams/architecture.html`** (open in a browser)

```
Browser SPA (4 tabs)                 OpenClaw agent (skill: run_pipeline.py)
        │  /api (same-origin)                    │  POST /api/flow/*
        ▼                                         │
Istio ezaf-gateway (VirtualService)              │
        │                                         │
        ▼                                         ▼
Backend FastAPI (v45, 26 routes) ◄──────────────────┘
  ├── per-role provider router (brain | i2v | image; request-scoped)
  ├── flow engine (6-step IDD)      ├── workflow stages 1–5 (UI editor)
  ├── prompt agent (vision LLM)     ├── templates CRUD
  ├── video job client              └── metrics / health / config
        │ brain (LiteLLM)  │ image (sdcpp)  │ i2v (vLLM-Omni)
        ▼                  ▼                ▼
  ns/project-user-aieadmin — MLIS (Aioli) + Knative predictors, 4× A100-40GB
        │
        └── NFS: pcai-wan-store (weights) · ns/styling-demo: results PVC (MP4 + templates)
```

### 3.2 Components

| Component | Tech | Location | Role |
|---|---|---|---|
| **Frontend** | Single-file SPA, vanilla JS, nginx | `ns/styling-demo` | 4 tabs: **Quick Video** (one-click I2V of a photo + style ref), **Styling Workflow** (the 6-step IDD editor), **Templates** (save/load/re-run), **Provider** (per-role on-prem/external config). No build step; all endpoints relative (`location.origin + "/api"`); no-cache headers (a stale cached JS once rendered the wrong UI); version badge reads `/api/config` live. |
| **Backend** | FastAPI + httpx + Pillow (Python), 2,876 lines, 26 routes | `ns/styling-demo`, 1 replica | The brain of the demo. Zero hardcoded URLs — every endpoint/model comes from env (set by the Helm chart). Per-role provider routing, the flow engine, prompt agent, video job client, templates, metrics, and deterministic fallbacks on every step. |
| **brain** | qwen3-8-27b-int4 (dflash2-r2) via LiteLLM proxy | mks-wrk-01 (2× A100) | Vision LLM: describes style/scene references into prompt text, optimizes the I2V prompt, detects the car's body color (used to anchor colorway recolors). |
| **image** | Qwen-Image-Edit GGUF via sdcpp-server (stable-diffusion.cpp) | mks-wrk-02 (1× A100) | Image-to-image editing: sketches, scene-gen, re-angle, recolor. **Single-reference editor** (a second image is parsed but silently discarded — verified in-pod), which shaped the combine design. `IMG_MAX_SIDE=768`; frontend must not send w/h (422). |
| **i2v** | Wan2.2-TI2V-5B via vLLM-Omni (OpenAI-compatible `/v1/videos`) | currently **paused** on CS1 (re-pin via MLIS `node_selectors`) | Image-to-video: the final still → cinematic MP4 (up to 121 frames @ 8 fps, 480p–720p). 34.2 GB weights pre-seeded on NFS. |
| **External providers** | Any OpenAI-compatible endpoint (e.g. Gemini Nano Banana + Veo) | outside the cluster | Per-role substitution: brain, image, or video can each run externally with keys supplied **per request only** — nothing is stored server-side. |
| **MLIS (Aioli)** | PackagedModel + Deployment via `POST /api/v1` | control plane | Materializes the Knative predictors. Node moves = Postgres `node_selectors` UPDATE + `status=Deploying` → ListenLoop creates a new revision. Weights mount via `pvc://` (zero copy). |
| **Storage** | NFS PVCs | `pcai-wan-store` (weights, shared, no node affinity), `results` (MP4 + artifacts + templates) | Durability: weights survive node moves without re-seed; results survive pod restarts. |
| **OpenClaw agent** | `nemoclaw-openclaw` (v2026.3.11), skill on PVC | `ns/project-user-aieadmin` | The agent-side integration: the `pcai-styling-video-automation` skill drives the 6-step flow with a stdlib-only Python client (`run_pipeline.py`), supports `pcai | gemini | both` and `tuned | customer_adm` prompt packs, and writes a `run_summary.json` per run. |

> Diagram: **`docs/diagrams/topology.html`** — the live CS1 topology (verified against
> the cluster: pods, nodes, PVCs, paused i2v, OpenClaw).

### 3.3 The 6-step IDD flow (the core)

> Diagram: **`docs/diagrams/pipeline.html`**

The flow is the machine encoding of Mario's P2–P6 process. Each step is a synchronous
`POST /api/flow/<step>` call with base64 image(s) + prompt, returning the result image(s)
as data-URIs plus the exact `prompt` used, the `mode`, and `elapsed_ms`:

| # | Step | Endpoint | Prompt pack key | PCAI engine | Gemini engine | Quality gate |
|---|------|----------|-----------------|-------------|---------------|--------------|
| 1 | **sketch** | `/api/flow/sketch` | `sketch_front` + `sketch_rear` | Qwen-Edit GGUF i2i, front @0.8 / rear @1.0 — one call, two **independent** generations from the photo (no chaining — Gemini does exactly this) | two `generateContent` calls (P4/P2 verbatim) | rear must show the **rear** (taillights/trunk) |
| 2 | **design gate** | — (human) | — | — | — | the designer's Photoshop edit is the input to step 3 (mock available for demo) |
| 3 | **combine** | `/api/flow/combine` | `combine` | **single-ref i2i scene-gen @0.85** from the original car photo (primary; the model is single-reference); deterministic composite paste + harmonize as fallback (0.95 erases the car — 0.85 is the tuned value) | native two-image integration (true P3) | car visibly inside the scene |
| 4 | **view** / **recolor** | `/api/flow/view`, `/api/flow/recolor` | `alt_view` / `recolor` | i2i @0.85 with **explicit per-view prompts** (v44/v45 fix — see pitfall below) | same | same car, only camera/paint changes |
| 5 | **video** | `/api/flow/video` | `video` | Wan2.2 I2V of the combined still — cinematic orbit, studio lighting, anti-drift guardrail | Veo **first/last-frame morph** sketch_front → sketch_rear (that *is* P6: "3s camera movement from first image to second") | MP4 > 1 MB with `ftyp` + `avcC` |

The UI's **Styling Workflow** tab bundles these: stage 4 = step 1 (AI base sketch:
front + rear + paint-only colorways), the design gate = step 2, stage 5 = steps 3+4
(scene compose + quick viz: main / another_view / another_color in one call), stage 3 =
step 5 (video). **Quick Video** is the one-click version of step 5 for any photo.

### 3.4 Per-role provider routing (v43+ — the architectural centerpiece)

Each model role resolves independently per request:

```
provider: {
  brain: {mode: external, base_url: ..., api_key: ..., model: ...},   # or on-prem model override
  i2v:   {mode: external, ...},                                        # or on-prem default
  image: {mode: external, ...}                                         # or on-prem default
}
```

- A request-scoped `ContextVar` carries the override through the call stack; the
  resolvers (`_brain_endpoint`, `_i2v_endpoint`, `_image_endpoint`) pick external vs
  on-prem per role. `is_external` on the image role gates the sdcpp
  `<sd_cpp_extra_args>` prompt-embed trick (only the on-prem Qwen-Edit GGUF understands
  it — a standard OpenAI server does not).
- **Keys are sent with the request and never persisted** (ADM's explicit requirement).
- `/api/providers/test` + `/api/providers/models` + `/api/providers/onprem-models`
  power the Provider tab's on-prem model detection (e.g. pick a different LiteLLM model
  for the brain).
- Note: `/api/flow/*` is **always on-prem** (the agent-skill contract); provider
  overrides apply to `/api/generate` and `/api/workflow/stage`.

### 3.5 Resilience model (why the demo never hard-fails)

Every step has a **deterministic fallback** that declares itself in `mode`:

| Step | Model path | Fallback (mode) |
|---|---|---|
| sketch | GGUF i2i | `pencil-fallback` — deterministic line-art extraction |
| combine | scene-gen i2i @0.85 | `composite` — geometry-locked paste + low-strength harmonize |
| view | i2i re-angle @0.85 | `input-fallback` — return the input (honest: no fake re-angle) |
| video | Wan2.2 I2V | 503 (video has no deterministic substitute — the MP4 gate is real) |
| scene | user's REF-2 | bundled mock dusk scene (`use_mock_scene`) |
| design | designer's Photoshop | mock sample design (`use_mock_design`) |

The agent protocol (in the skill) requires **exit code 2 = a fallback was used → report
which one**. This turns "the model was down" from a silent quality bug into visible,
auditable degradation.

### 3.6 Deployment & operations

> Diagram: **`docs/diagrams/topology.html`**

- **Source of truth:** `helm-chart/` (backend/frontend/VirtualService/results-PVC),
  `values.yaml` = the only place a domain, namespace, endpoint URL, model name, or
  secret reference lives. `manifests/` is a frozen reference; deploy =
  `helm upgrade --install`. Secrets (litellm-key, mlis-key, styling-api-key) are created
  by `scripts/deploy-secrets.sh` and **never committed**.
- **Airgapped build:** bastion `docker build` → `registry.ctc.sg.lab:5000`
  (`scripts/build-deploy.sh`); unique tag per build (retag, no `--no-pull` surprises).
- **CS1 gotchas (operational knowledge, all verified live):**
  - `KUBECONFIG` must be the **native Windows path** (`C:/Users/ml/hermes/cs1_platform/
    kubeconfig-cs1.conf`) — MSYS `/c/...` is not translated for `kubectl.exe`.
  - `curl` to `*.aie.cs1.ctc.sg.lab` needs `-k` (internal CA).
  - All MLIS GPU predictors share `priorityClass mlis-gpu` (8000) → equal-priority
    preemption is **refused**; a zombie `Terminating` pod (dead istio preStop sidecar)
    blocks GPU release → `kubectl delete pod --grace-period=0 --force`.
  - A model can be re-pinned to another node via the MLIS Postgres
    (`UPDATE deployments SET node_selectors=..., status='Deploying'`) — no re-seed
    (weights on shared NFS), ~15–20 min NFS load.
- **Testing:** `scripts/ui_e2e.py` (CDP, headless Chrome against the real UI: upload →
  stages → video, screenshot evidence), `test_flow_api.py`, `test_generate.py`,
  `smoke_stage45_local.py`, plus the agent-skill's 4 quality gates.

### 3.7 Key engineering decisions (and why)

1. **One image-edit model (Qwen-Image-Edit GGUF) for all four image steps** — the demo
   is honest about its engine: the same single-reference model does sketch, scene-gen,
   re-angle and recolor, with per-step strengths tuned (0.8 / 0.85 / 1.0).
2. **scene-gen, not paste, as the primary combine** — a single-reference model cannot
   integrate [design, scene] (second image silently discarded). Generating the car
   already-in-scene from the original photo gives grounded contact shadow and no paste
   rectangle; the deterministic composite remains as the declared fallback.
3. **Explicit per-view re-angle prompts (v44/v45)** — the v44 attempt ("change camera
   angle to 3/4 front") came back as a 3/4 **rear**: Qwen-Edit GGUF anchors to the
   input image's orientation, so a generic prompt drifts. The fix names the exact car
   surfaces that must face the camera per view (`side` / `3/4 front` / `front` /
   `3/4 rear` / `rear`). Do not "simplify" these prompts back.
4. **Recolor by color-anchor, not free generation** — stage-4 colorways are
   deterministic paint-only recolors (exact same view/geometry; per ADM v21 rule), so
   "same car, new paint" is a guarantee, not a hope.
5. **Per-role providers as a request-scoped ContextVar** — zero changes to the 30+ LLM
   call sites (back-compat alias `_llm_endpoint_and_key`), keys never persisted, and
   the flow API stays a clean on-prem contract for the agent skill.
6. **The agent skill is stdlib-only Python** — it must run in the OpenClaw sandbox
   (python 3.13, no pip). No dependencies = no supply-chain, no install step.
7. **MP4 faststart + inline** — the browser `<video>` tile only plays if the `moov`
   atom is at the front; served with `Content-Disposition: inline` (v39) so the result
   renders in place, not as a download.

## 4. The agent integration (OpenClaw)

The `pcai-styling-video-automation` skill (in the repo at
`skill/pcai-styling-video-automation/`, imported into both CS1 OpenClaw instances)
is the "no UI" face of the demo:

- **`run_pipeline.py`** (442 lines, stdlib only) runs the 6 steps against
  `/api/flow/*` (pcai) or the Gemini REST APIs (gemini) or both in parallel, using a
  prompt pack (`tuned` = our production prompts with the 180° rear lock + anti-drift
  video guardrails; `customer_adm` = Mario's verbatim P2–P6 for like-for-like).
- **Output contract:** per-provider `run_summary.json` (rendered prompts,
  `engine_differences`, per-step model/mode/elapsed) + artifacts
  (`sketch_front.png` … `final.mp4`).
- **Verification gates** (the skill mandates checking before declaring success):
  rear-view, composite, and video (`ftyp`+`avcC`) gates; **exit 2** = a deterministic
  fallback was used → report which.
- **CS1 import gotcha (both instances):** the OpenClaw gateway scans its skill registry
  **once at startup** — the UI reads the in-memory registry via WS `skills.status`.
  On the user-facing instance (`nemoclaw-openclaw`, PVC-backed skills dir) a plain file
  copy + `rollout restart` is durable; on `oc-gw` (emptyDir home) durability requires
  the ConfigMap `oc-skills` + initContainer pattern. Full procedure: the skill's
  `SKILL.md` §"Import to CS1 OpenClaw".

## 5. Current state (verified 2026-10-09)

- Live: `styling-demo.aie.cs1.ctc.sg.lab` — backend **v45**, frontend **v23**, pods
  `2/2 Running`; `/api/config` reports `build: v45`, model `Wan2.2-TI2V-5B (I2V)`,
  brain `qwen3-8-27b-int4-dflash2-r2`, image `pcai-qwen-image-edit-gguf`.
- Predictors: brain (wrk-01, 17d), image GGUF (wrk-02, 23h) **running**; i2v
  (Wan2.2) **currently paused** on CS1 — re-pin via MLIS when video is needed;
  hunyuan3d-2.1 (wrk-03) belongs to the separate 3D Viz PoC (decision 2026-10-09:
  it does **not** replace Wan — wrong modality, GLB ≠ MP4).
- OpenClaw: skill visible in the UI's live registry (53 skills, `pcai-styling-video-automation`
  `eligible: true`, `source: openclaw-managed`) on the `nemoclaw-openclaw` instance.
- Git: branch `feat/external-api-provider`, latest commit `d7a92e4`. Not merged to
  `main` (per standing instruction).

## 6. Known limitations (honest list)

| Limitation | Status / mitigation |
|---|---|
| Prompt-driven re-angle is an **approximation** of true multi-view consistency | Works for the 4 demo views (explicit prompts); true multi-view = Phase 3 model work |
| Single-reference image model (no two-image i2i) | scene-gen design around it; Gemini path has native two-image |
| 40GB GPUs cap the video model at Wan2.2-5B (480p–720p, ≤121 frames) | 80GB-class upgrade path documented (Phase 2) |
| Sync step calls up to 900 s (no queue) | Fine for a single-user demo; async job model = Phase 1 |
| No auth on `/api/*` (lab-only) | Phase 1 (OIDC/JWT + per-user keys) |
| One backend replica, no HPA | Phase 1 |
| i2v currently paused on CS1 | Re-pin before the video step; Quick Video needs it live |
| Back-half geometry of the car is inferred from one photo | Inherent to 2D generation; Hunyuan3D 3D-preview (Phase 3) partially addresses |

## 7. Production expansion path

> Diagram: **`docs/diagrams/roadmap.html`**

The architecture is deliberately built so **production is a values.yaml change plus
phased work, not a re-architecture**: the 6-step contract, the per-role providers, and
the Helm chart are stable; what changes is operational maturity and model fidelity.

### Phase 1 — Service hardening (≈ 2–3 weeks): make the PoC a real service
- **AuthN/Z:** OIDC/JWT via Istio on the domain; per-user API keys; RBAC over templates
  (today `/api/*` is unauthenticated — acceptable only in the airgapped lab).
- **Async job model:** the 6 flow steps become a queue (Redis or K8s Jobs) with
  `GET job status`, progress, cancel, and retry — removes the 900 s synchronous HTTP
  ceiling and makes long I2V jobs a first-class resource.
- **Scale-out:** backend is stateless (job state externalized) → multi-replica + HPA;
  predictor `minReplicas: 1` (or warm pool) to kill multi-minute scale-to-0 cold starts.
- **Observability:** structured logs + trace ID across the 6 steps; Prometheus scrape
  into Ezmeral monitoring; per-step SLOs; the 4 quality gates become CI checks on every
  run artifact.

### Phase 2 — Scale & throughput (≈ 4–6 weeks): GPU economics for a real team
- **Capacity model:** N concurrent stylists ≈ N video GPUs (the sizing story for the
  ADM conversation); fair-share queue, off-peak scheduling, per-project GPU metering.
- **Model fleet manager:** automate today's manual node_selectors re-pinning into an
  API (move model → watch NFS load → clear zombie pods).
- **80GB-class upgrade:** HunyuanVideo-1.5 (480p) or Wan2.2-14B via a one-line
  `values.yaml` change (already documented in `DESIGN.md` §6); 720p via FP8 + VAE
  tiling.
- **Multi-platform:** the chart is already platform-adaptive — a
  `values.platform-<env>.yaml` deploys the same code to any PCAI/Ezmeral cluster.

### Phase 3 — Production-grade AI (≈ 8–12 weeks): from demo approximations to fidelity
- **True multi-view consistency** (multi-view/3D-consistent model) — fixes the
  view-drift class of problems structurally, not with prompt engineering.
- **ControlNet object preservation** (lineart/canny) — "do not improvize with the car
  design" enforced by the model, not the prompt.
- **Hunyuan3D-2.1 as an additive "3D preview" stage** (photo → GLB → orbit viewer) —
  separate from video per the 2026-10-09 decision; Blender turntable MP4 as an
  alternative video mode.
- **Prompt packs as versioned data:** A/B against the quality gates; per-model prompt
  banks (Qwen-Edit and Nano Banana want different wording).
- **Supervised agent loop:** the OpenClaw skill grows a design-critic loop — auto-retry
  with a new seed when a gate fails, human-in-the-loop approval, full run audit.

### Already production-safe (no phase needed)
Zero hardcoded URLs (platform-adaptive) · secrets only in K8s, external keys
request-scoped · durable NFS state (weights/results/templates) · deterministic,
declared fallbacks · airgapped build pipeline · E2E UI tests.

## 8. Repository map

```
pcai-styling-demo/
├── backend/            main.py (2,876 lines, 26 routes) · Dockerfile · requirements.txt
│   └── sample_data/    bundled mock dusk scene (REF-2 substitute)
├── frontend/           index.html (SPA, v23) · nginx.conf · Dockerfile
├── helm-chart/         THE deploy source of truth (values.yaml = config spine)
├── chart/              EzApps tile registration (separate small chart)
├── manifests/          frozen k8s reference (superseded by helm-chart/)
├── inference/          vLLM-Omni entrypoint (image is the official vllm-omni)
├── scripts/            build-deploy.sh (bastion) · deploy-secrets.sh · ui_e2e.py (CDP)
│                       test_flow_api.py · test_generate.py · smoke_stage45_local.py
├── skill/pcai-styling-video-automation/
│   ├── SKILL.md        the agent skill (workflow, gates, import procedure)
│   ├── scripts/run_pipeline.py   stdlib-only 6-step client (pcai|gemini|both)
│   ├── prompts/        tuned.yaml · customer_adm.yaml (Mario's verbatim)
│   └── references/api.md         flow API + Gemini API contract
├── docs/
│   ├── PROJECT.md      ← this document
│   ├── enhancement-plan.md       the v6-era enhancement plan (superseded by v43+)
│   ├── ezapps-tile.md
│   └── diagrams/       architecture.html · pipeline.html · topology.html · roadmap.html
├── DESIGN.md           original architecture (single-model vLLM-Omni decision)
└── cache_scratch/      dev probes/analysis (not shipped)
```

## 9. How to run it (quick reference)

```bash
# deploy (from the bastion with registry access)
KUBECONFIG=C:/Users/ml/hermes/cs1_platform/kubeconfig-cs1.conf \
  helm upgrade --install pcai-styling ./helm-chart -f helm-chart/values.yaml

# probe
curl -sk https://styling-demo.aie.cs1.ctc.sg.lab/api/config          # build, models
curl -sk https://styling-demo.aie.cs1.ctc.sg.lab/healthz

# e2e (headless Chrome CDP against the real UI)
python scripts/ui_e2e.py

# agent-side (from an OpenClaw sandbox with network to the lab ingress)
python skill/pcai-styling-video-automation/scripts/run_pipeline.py \
  --photo car.jpg --provider both --prompts customer_adm --out runs/demo
```

---
*Generated from a live code + cluster review on 2026-10-09. All versions, node
assignments, and route counts were verified against the running system, not assumed.*
