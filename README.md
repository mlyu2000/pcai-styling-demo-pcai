# Automotive Styling Video

> **Personal standalone copy** of the *Automotive Styling Video* demo (generative-AI exterior-styling workflow (photo → styling video)), validated on HPE Private Cloud AI (CS1). Restructured to the `ai-solution-eng/ai-solution-demos` contribution layout. See the [contribution guidelines](https://github.com/ai-solution-eng/ai-solution-demos/blob/main/CONTRIBUTING.md).
*HPE Private Cloud AI reference demo for Astra Daihatsu Motor (ADM) — a generative-AI styling tool that turns a front-view car photo into a cinematic styling video, end-to-end on-prem. Author: Eric Yu (HPE).*

| Owner                 | Name    | Email            |
| ----------------------|---------|------------------|
| Use Case Owner        | Eric Yu | eric.yu@hpe.com  |
| PCAI Deployment Owner | Eric Yu | eric.yu@hpe.com  |

*Use Case Owner: brought and designed this demo for ADM. Contact for more information.*

*PCAI Deployment Owner: ported and validated the demo on PCAI (CS1 / Ezmeral). Contact on deployment issues.*

## Abstract

A generative-AI styling tool that reproduces **ADM's actual exterior-design (IDD) workflow** —
today done by hand in Photoshop + Gemini — and runs it **end-to-end on HPE Private Cloud AI**.
A front-view car photo goes in; a polished **styling video** comes out, through six model-driven
steps (sketch → design → scene → view/recolor → video).

- **Accelerates** styling exploration — a multi-hour manual concept becomes a repeatable pipeline run.
- **Keeps design IP in the datacenter** — every step can run fully on-prem (Qwen-Image-Edit GGUF + Wan2.2 I2V + a vision LLM), no egress to public APIs.
- **Model-agnostic** — the identical workflow also runs against public models (Google Gemini) through a single API, proving the platform is not vendor-locked.
- **Per-role provider routing** — brain / image / video can each be on-prem or external, independently, per request.
- **Agent-driven** — an OpenClaw skill (`pcai-styling-video-automation`) runs the whole flow with no UI.
- **Never hard-fails in front of a customer** — every step has a deterministic fallback that declares itself.

Features:
- 6-step IDD flow (sketch, design, scene, view/recolor, video) with a 4-tab web UI: Quick Video, Styling Workflow, Templates, Provider.
- Per-role provider routing (brain / image / video: on-prem vs external) with request-scoped keys that are **never persisted server-side**.
- Saved templates — one-click re-run of an entire workflow; a "style" becomes a team asset.
- Two prompt packs for like-for-like comparison: `tuned` (production) vs `customer_adm` (the designer's verbatim prompts).
- Deterministic fallbacks with a declared `mode` on every step; the agent protocol reports which fallback was used.
- OpenClaw / NemoClaw agent skill — "make me the styling video for this photo" with no UI.

Recordings:
- Demo video is **not hosted on GitHub** per repo policy; available on request / at review.

## Description

### Overview

The demo is a single-origin web app (FastAPI backend + a no-build vanilla-JS SPA served by nginx)
fronting a 6-step generative pipeline. An Istio VirtualService routes the public hostname to the
frontend/backend in the `styling-demo` namespace; the backend calls MLIS (Aioli) / Knative
InferenceServices for the on-prem models, or external OpenAI-compatible endpoints when a role is
configured external. Results (MP4 + templates) land on an NFS PVC.

```
Browser SPA (4 tabs)                 OpenClaw agent (skill: run_pipeline.py)
        │  /api (same-origin)                    │  POST /api/flow/*
        ▼                                         │
Istio ezaf-gateway (VirtualService)              │
        │                                         │
        ▼                                         ▼
Backend FastAPI (v45) ◄────────────────────────────┘
  ├── per-role provider router (brain | i2v | image; request-scoped)
  ├── flow engine (6-step IDD)      ├── workflow stages 1–5 (UI editor)
  ├── prompt agent (vision LLM)     ├── templates CRUD
  ├── video job client              └── metrics / health / config
        │ brain (LiteLLM)  │ image (sdcpp)  │ i2v (vLLM-Omni)
        ▼                  ▼                ▼
  MLIS (Aioli) + Knative predictors on CS1 GPUs (4× A100-40GB)
        │
        └── NFS: pcai-wan-store (weights) · ns/styling-demo: results PVC (MP4 + templates)
```

Interactive diagram (open in a browser): [`doc/diagrams/architecture.html`](doc/diagrams/architecture.html).
Additional interactive diagrams: [`topology.html`](doc/diagrams/topology.html),
[`pipeline.html`](doc/diagrams/pipeline.html), [`process-flow.html`](doc/diagrams/process-flow.html),
[`roadmap.html`](doc/diagrams/roadmap.html).

| Component | Tech | Role |
|---|---|---|
| **Frontend** | Single-file SPA, vanilla JS, nginx | 4 tabs: Quick Video (one-click I2V), Styling Workflow (6-step editor), Templates, Provider. No build step; version badge reads `/api/config` live. |
| **Backend** | FastAPI + httpx + Pillow | The brain: per-role provider routing, flow engine, prompt agent, video job client, templates, deterministic fallbacks. Every endpoint/model comes from env (set by the chart). |
| **brain** | qwen3-8-27b (vision LLM) via LiteLLM proxy | Describes style/scene references into prompt text, optimizes the I2V prompt, detects body color for colorway recolor. |
| **image** | Qwen-Image-Edit GGUF via sdcpp-server (stable-diffusion.cpp) | Image-to-image editing: sketch, scene-gen, re-angle, recolor. Single-reference editor; `IMG_MAX_SIDE=768`. |
| **i2v** | Wan2.2-TI2V-5B via vLLM-Omni (OpenAI `/v1/videos`) | Image-to-video: final still → cinematic MP4 (up to 121 frames @ 8 fps, 480p–720p). |
| **Agent skill** | `pcai-styling-video-automation` (stdlib-only Python) | Runs the 6 steps against `/api/flow/*` (on-prem), Gemini REST (external), or both, with quality gates. |

### Workflow

1. **Input** — a front-view car photo (or a sample scene). The designer may supply design/scene references.
2. **brain** (vision LLM) reads the references and writes the prompt text that drives each image step and the final video prompt.
3. **image** (Qwen-Image-Edit GGUF, single-reference) runs the image steps: **sketch** → **scene-gen** (car composited into the scene) → **view/re-angle** → **recolor** (deterministic paint-only colorway on an anchored view). Every step has a deterministic fallback declared in `mode`.
4. **i2v** (Wan2.2 I2V) turns the final still into a cinematic MP4 (faststart, served inline).
5. **Landing** — the MP4 and saved templates persist to the results PVC; the frontend renders the video in place.
6. **Agent path** — the OpenClaw skill drives the same `/api/flow/*` contract with no UI, emitting a per-provider `run_summary.json` (rendered prompts, engine differences, per-step model/mode/elapsed) and artifacts (`sketch_front.png` … `final.mp4`), plus verification gates (rear-view, composite, video; exit 2 = a fallback was used).

Interactive pipeline diagram: [`doc/diagrams/pipeline.html`](doc/diagrams/pipeline.html).

## Deployment

### Prerequisites

- **HPE Private Cloud AI** (Ezmeral Kubernetes) with **MLIS / Aioli** for model serving, Knative for scale-to-0, Istio for routing, NFS PVCs for weights + results, and LiteLLM for the LLM proxy.
- **GPUs** — the demo ran on **4× A100-40GB** (one brain LLM on 2 GPUs, one image model, one video model). Production sizing: **1 GPU ≈ 1 concurrent stylist** (the inference server queues the rest).
- **Models** — qwen3-8-27b (brain), Qwen-Image-Edit GGUF (image, via sdcpp-server), Wan2.2-TI2V-5B (i2v, via vLLM-Omni). These are platform InferenceServices configured in `values.yaml`.
- An internal registry for the airgapped build, and a hostname routed through the Ezmeral VirtualService.

All environment-specific values (domain, namespaces, registry, endpoint URLs, model identity, secret
**names** — never secret values) live in [`deploy/config/values.yaml`](deploy/config/values.yaml).

### Installation and configuration

1. **Build & push images** (airgapped bastion):
   ```bash
   bash source/code/scripts/build-deploy.sh
   ```
   This builds backend/frontend/inference images and pushes them to your internal registry. Use a unique tag per build.

2. **Create cluster secrets** (read live from the cluster — never committed):
   ```bash
   bash source/code/scripts/deploy-secrets.sh
   ```
   Creates `regcred` (registry), `litellm-key` (LiteLLM master key read from the litellm deployment), and a generated `styling-api-key`.

3. **Create the results PVC** (once):
   ```bash
   kubectl apply -f source/code/manifests/mlis-pvc.yaml
   ```

4. **Deploy the workload** with the demo chart:
   ```bash
   helm upgrade --install pcai-styling deploy/chart/pcai-styling-1.0.0.tgz \
     -f deploy/config/values.yaml -n styling-demo --create-namespace
   ```

5. **Register the EzApps tile** — apply the registration ConfigMap in [`deploy/config/chart-registration/`](deploy/config/chart-registration/) so the demo appears in the Ezmeral EzApps tile.

6. **(Optional) Agent skill** — import `source/code/skill/pcai-styling-video-automation/` into an OpenClaw / NemoClaw instance to drive the flow with no UI.

> CS1 operational notes (verified live): set `KUBECONFIG` to the **native** path (`C:/Users/.../kubeconfig-cs1.conf` — MSYS `/c/...` is not translated for `kubectl.exe`); `curl` to the internal hostname needs `-k` (internal CA). See [`doc/PROJECT.md`](doc/PROJECT.md) §Deployment for the full runbook.

## Running the demo

### Demo 1 — Styling Workflow (the full 6-step IDD, in the UI)

Open the hostname (e.g. `https://styling-demo.aie.cs1.ctc.sg.lab`) → **Styling Workflow** tab:

1. Upload a front-view car photo.
2. Step through **sketch → design → scene → view → recolor**, adjusting prompts; watch each result render.
3. Click **Generate video** on the final still → the MP4 renders inline in the page.
4. Save the run as a **template**, then re-run it later with one click.

### Demo 2 — Agent-driven (no UI) + provider comparison

- From the OpenClaw instance with the `pcai-styling-video-automation` skill: *"make me the styling video for this photo."* The skill runs all six steps, emits `run_summary.json` with per-step model/mode/elapsed, and reports any fallback (exit 2).
- In the **Provider** tab, switch a role (brain / image / video) to an external model (e.g. Google Gemini) and re-run — the identical workflow runs against public models, proving model-agnosticism.

### Validation / tests

- `python source/code/scripts/test_flow_api.py` — flow API contract.
- `python source/code/scripts/ui_e2e.py` — headless-Chrome end-to-end (upload → stages → video) with screenshot evidence.
- The agent skill's quality gates (rear-view, composite, video) verify a real run.

## Limitations

- **Single image-edit engine** — Qwen-Image-Edit GGUF is a *single-reference* model (a second image is parsed but discarded), which is why combine uses scene-generation rather than a two-image paste.
- **Re-angle drift** — Qwen-Edit GGUF anchors to the input orientation, so re-angle prompts must name the exact car surfaces facing the camera; a generic "change camera angle" prompt drifts to the wrong view.
- **40 GB cards cap model choice** — Wan2.2-5B (not larger video models needing ~100 GB) on 40 GB GPUs; an 80 GB-class ask is the roadmap path for bigger models.
- **Video has no deterministic fallback** — if the i2v model is down, the video step returns 503 (the MP4 quality gate is real); the other steps degrade gracefully and declare `mode`.
- **Internal CA / airgapped** — the hostname uses an internal certificate (`curl -k`) and the build assumes an internal registry; not a public-internet demo.

See [`doc/PROJECT.md`](doc/PROJECT.md) for the full project description, architecture, and engineering decisions.