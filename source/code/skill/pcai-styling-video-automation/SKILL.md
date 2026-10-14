---
name: pcai-styling-video-automation
description: Use when the user wants to turn a front-view car photo into a styling video (sketch -> scene -> quick viz -> recolor -> video) via the PCAI Styling Demo — on the self-hosted lab service, on Google Gemini (Nano Banana + Veo), or both side-by-side — using one shared prompt pack. Automates the full flow with one command; no app rebuild.
---

# PCAI Styling Video Automation

Automate Mario's IDD flow (reference → base sketch → design into scene → quick viz / color → video) end-to-end. **This skill drives the deployed service / public APIs with a thin stdlib client — it does NOT rebuild or modify the application.** It relies on the `prompt_override*` fields on the styling backend (v28+).

## Reconciled workflow (both engines, same prompts)

The **reference semantics are the Gemini stream: every step is exactly
`(user-provided prompt, declared inputs) → model`** — no hidden machinery in the
step logic. Both engines run the identical step sequence with the identical
rendered prompt text (from one YAML pack):

| # | Step | Prompt (pack key) | Declared inputs |
|---|------|-------------------|-----------------|
| 1 | `sketch_front` | `sketch_front` | `[photo]` |
| 2 | `sketch_rear` | `sketch_rear` | `[photo]` |
| 3 | `combine` | `combine` | `[front, scene]` |
| 4 | `alt_view` | `alt_view` | `[combine]` |
| 5 | `recolor` | `recolor` (slot `{color}`) | `[combine]` |
| 6 | `video` | `video` | engine-specific source (below) |

### Engine differences (post-v32 rebuild: only these remain)

After the v32 rebuild (native per-step flow API), the PCAI stream is
functionally equal to the Gemini reference: same step sequence, same rendered
prompts, same declared inputs. What remains is recorded in
`run_summary.json → engine_differences` and in each step's `mode`:

- **`combine`** — pcai: the Qwen-Image-Edit GGUF is SINGLE-REFERENCE (a 2nd
  image is parsed but silently discarded, verified in-pod). Two integrator
  modes: `composite+harmonize` (default — geometry-locked deterministic
  composite + single-ref harmonize; the only path that uses both images, but
  a paste of a studio product photo reads as a floating rectangle) and
  `scene-gen` (verified live 2026-10-01 — single-ref i2i SCENE GENERATION: the
  car photo is the ONLY reference and the model generates the car already
  integrated into the described scene; grounded, lighting-matched, no paste
  edge; strength 0.85; mirrors the customer's Gemini one-ref + scene-prompt
  workflow; deterministic composite = fallback when the model is down). The
  `run_pipeline.py` client uses `mode=scene-gen` by default. gemini: native
  two-image model integration of `[design, scene]`.
- **`video`** — pcai: I2V of the combined still (scene video; Wan2.2 has no
  first/last-frame mode — model-inherent). gemini: first/last-frame morph
  `sketch_front → sketch_rear` (matches P6 "camera movement from first image
  until to second image").
- **Fault tolerance** — each pcai `/api/flow/*` step falls back
  deterministically and declares which path ran in `mode` (never hard-fail);
  gemini is fail-fast.

**Backend v32 note**: since the v32 rebuild the pcai provider drives the
NATIVE per-step flow API (`/api/flow/sketch|combine|view|recolor|video`),
which implements the Gemini reference semantics directly — prompt-driven,
exactly the declared inputs, deterministic machinery only as declared
fallbacks (combine's composite is the declared primary integrator on this
engine, see above). The old "chained rear" and "pre-composite combine"
engine differences no longer exist. The legacy stage API (`/api/workflow/stage`)
remains for the app UI only.

## Per-role providers (backend v43+)

Each model role — **brain** (LLM), **i2v** (video), **image** (i2i/texture) —
can independently run **on-prem** (MLIS) or an **external** OpenAI-compatible
endpoint (request-scoped; keys are never persisted server-side). The
**app UI** and the legacy **stage API** accept the per-role `provider` payload
(`{role: {mode, base_url, api_key, model}}`) and test via
`/api/providers/test?role=<brain|i2v|image>`. **IMPORTANT: the `/api/flow/*`
endpoints that this skill drives are ALWAYS on-prem** — they do not accept a
`provider` field, so a pipeline run here uses the lab's MLIS models regardless
of the per-role provider setting. Use the flow API for the automated pipeline;
use the stage API / app UI only when you need an external provider for a step.

## The one command

```bash
python scripts/run_pipeline.py --photo <front-view car jpg> \
    [--prompts tuned | customer_adm] \
    [--provider pcai | gemini | both] \
    [--scene <REF2 jpg>] [--view side | "3/4 front" | "3/4 rear" | rear] \
    [--color silver] [--requirement "red metallic, aggressive stance"] \
    [--out runs/run-<name>]
```

- **`--prompts`**: the shared prompt pack (YAML, one template per step).
  `tuned` = our production prompts (180° rear lock, harmonize wording,
  anti-drift video guardrails — default). `customer_adm` = Mario's verbatim
  Gemini one-liners from the 2026-09-29 meeting (P2/P3/P4/P5/P6) for
  like-for-like comparison.
- **`--provider`**: `pcai` = deployed lab service (`https://styling-demo.aie.cs1.ctc.sg.lab`).
  `gemini` = Google AI Studio (Nano Banana image steps + Veo video; needs
  `GEMINI_API_KEY` in env — check presence only, never print it). `both` = same
  prompts, both engines, side-by-side folders for comparison.
- **Cheap defaults** (user decision): Gemini image = `gemini-3.1-flash-image`,
  video = `veo-3.1-fast-generate-preview`. Override with `--image-model` /
  `--video-model`.
- **`--scene`**: omit → the lab's bundled mock dusk scene is used automatically
  (the REF-2 box in the customer flow).
- **`--color`** (default `silver`): the recolor target, substituted into the
  `recolor` prompt's `{color}` slot.

## How to run it (agent protocol)

1. **Input gate (WARN-only)**: confirm the photo is a **front view** of the car
   (the flow generates the rear). If unsure, ask; do not block on it.
2. **Check Gemini key presence** only if provider includes `gemini`/`both`:
   `python -c "import os;print(bool(os.environ.get('GEMINI_API_KEY')))"` — never echo the value.
3. **Run in the background — the pipeline is long.** Do NOT run foreground with a
   short timeout and do NOT poll in a tight loop.
   - Hermes: `terminal(background=true, notify=true)`.
   - OpenClaw: equivalent background task; resume on completion notification.
   - Budgets (measured): pcai ≈ 3–15 min (video 5–8 min on Wan2.2);
     gemini ≈ 2–6 min on the fast models.
4. **On completion, VERIFY before declaring success** (verification gates — this
   is what makes it real automation, not blind replay):
   - `run_summary.json` exists in `--out`; read per-provider `notes` for
     `sketch_mode` / `viz_mode` / fallbacks.
   - **Rear-view gate**: vision-check `sketch_rear.*` — it must show the car's
     REAR (taillights/trunk). A second front view = regression of the 180° bug;
     do not deliver, report it.
   - **Composite gate**: `combine.*` shows the car visibly inside the scene
     (not floating on a blank background).
   - **Video gate**: `final.mp4` — size > 1 MB and contains `ftyp` + `avcC`
     (H.264): `python -c "d=open('final.mp4','rb').read(4096);print('ftyp' in d and b'avcC' in d)"`
5. **Deliver**: `final.mp4` + the run folder + `run_summary.json` (contains the
   exact rendered prompts per step and the engine differences — cite them when
   the user asks "what prompt was used?"). For `--provider both`: deliver both
   videos + side-by-side `combine.*` stills + the `engine_differences` block.
6. **Exit code 2** = a deterministic fallback was used (pencil sketch or
   composite-only) — artifacts still produced; report WHICH fallback (from
   notes).

## Pitfalls (learned live — do not rediscover them)

- **View-angle drift (v44/v45 fix)**: the Qwen-Image-Edit GGUF anchors to the
  input image's orientation, so a generic "change camera angle to X" prompt
  drifts (3/4 front came back as 3/4 rear; side drifted to 3/4 rear). The
  backend now uses explicit per-view prompts (`alt_view_prompt(view)`) naming
  the exact facing car surfaces — do NOT "simplify" them back to a generic
  prompt. Valid angles: `side`, `3/4 front`, `3/4 rear`, `rear`. Verify the
  output angle with vision before delivering the `alt_view` still.
- **Flow API request fields** (v29+): `/api/flow/sketch` uses `image` +
  `prompt_front` + `prompt_rear`; `/api/flow/combine` uses `design_image` +
  `scene_image` + `prompt`; `/api/flow/view` + `/api/flow/recolor` use `image`
  + `prompt` (+ `view_angle` / `color`); `/api/flow/video` uses `image` +
  `prompt`. All responses are nested under `results` (`front`, `rear`,
  `composed`, `view`, `recolor`, `video_url`) plus a top-level `mode`.
  Images are `data:` URIs.
- **No `width/height` in any flow payload** — the backend 422s on them.
- **`video_url` comes back as `/api/results/...`** — strip the leading `/api/`
  before fetching.
- **Long synchronous step calls**: every flow call blocks until the step
  finishes (minutes). Video step timeout = 900 s.
- **Gemini**: image output is in `candidates[0].content.parts[].inline_data`
  (base64). Video is async: `predictLongRunning` → poll the op name until
  `done` → `generatedVideos[0].video.uri` → download.
- **Veo first/last frame**: client sends `sketch_front` as first and
  `sketch_rear` as last — that IS Mario's P6. Do not swap.
- **Lab ingress is self-signed** — unverified SSL context is used for lab URLs
  only (never for Gemini).
- **`_fill` only appends requirements** — it does NOT resolve `{slots}`. The
  client pre-renders all slots; never send a pack template containing raw
  `{...}` to the backend expecting substitution.
- The **designer gate** (Mario's Photoshop box) is intentionally NOT in the
  reconciled pipeline (reference = user-prompt-only). It remains available in
  the app UI and via `--designer-edit` (reserved).

## Output contract

```
<out>/
├── run_summary.json        # rendered prompts, engine_differences, totals (THE artifact)
├── pcai/                   # self-hosted run (and/or gemini/ for that provider)
│   ├── run_summary.json    # per-step: prompt, inputs, engine, model, mode, elapsed
│   ├── sketch_front.png    # step 1
│   ├── sketch_rear.png     # step 2  ← rear-view gate
│   ├── combine.png         # step 3  ← composite gate
│   ├── alt_view.png        # step 4
│   ├── recolor.png         # step 5
│   └── final.mp4           # step 6  ← video gate
└── gemini/                 # when --provider both (same names, .jpg)
```

## Boundaries

- This skill never calls models directly except through the two documented
  providers; never re-hosts, re-prompts, or edits backend code. Prompt changes =
  edit `prompts/*.yaml` (shared, provider-agnostic) — that's the whole point of
  the pack.
- The `pcai` stream's deterministic mechanics are now only DECLARED FALLBACKS
  (backend v29): each flow step reports which path ran via `mode`, so nothing
  is hidden — the default path on both engines is the model, prompt-driven.
- Secrets: `GEMINI_API_KEY` lives in the environment only. Never write it to
  disk, logs, or summaries.

## Import to CS1 OpenClaw

There are **TWO OpenClaw deployments on CS1 — do not confuse them.** Both
scan skills **once at startup** (no live-reload, no reload RPC) and the
Control UI reads the gateway's in-memory registry via the `skills.status` WS
RPC (port 18789), so after importing you must **restart the deployment** for
the UI to show the skill.

### Instance A — the one your UI (`nemoclaw-openclaw.aie.cs1.ctc.sg.lab`) uses
`ns/project-user-aieadmin` deploy `nemoclaw-openclaw` (OpenClaw 2026.3.11,
container `nemoclaw`, istio sidecar; token secret
`nemoclaw-openclaw-gateway-token`, config cm
`nemoclaw-openclaw-openclaw-config`). Its `/sandbox/.openclaw`
(=`managedSkillsDir`) is a **persistent PVC** (`nemoclaw-openclaw-agent-state`,
NFS) → a plain copy is **durable**; no ConfigMap/initContainer needed.

```bash
K=kubectl.exe; export KUBECONFIG=<native C:/.../kubeconfig-cs1.conf>
cd <repo>/skill
tar czf /tmp/ocskill.tgz --exclude=__pycache__ --exclude=runs pcai-styling-video-automation
POD=$($K -n project-user-aieadmin get pod -l app.kubernetes.io/name=nemoclaw -o jsonpath='{.items[0].metadata.name}')
$K -n project-user-aieadmin exec -i $POD -c nemoclaw -- sh -c \
  'mkdir -p /sandbox/.openclaw/skills && tar xzf - -C /sandbox/.openclaw/skills \
   && chmod +x /sandbox/.openclaw/skills/pcai-styling-video-automation/scripts/run_pipeline.py' < /tmp/ocskill.tgz
# gateway must re-boot to re-scan (safe — skill is on the PVC, not wiped)
$K -n project-user-aieadmin rollout restart deploy/nemoclaw-openclaw
$K -n project-user-aieadmin rollout status deploy/nemoclaw-openclaw --timeout=240s
# verify (disk check, then the UI's own source)
$K -n project-user-aieadmin exec $POD -c nemoclaw -- openclaw skills info pcai-styling-video-automation   # ✓ Ready, openclaw-managed
```

### Instance B — `ns/oc-test` deploy `oc-gw` (OpenClaw 2026.3.1)
Its `home` volume (`/sandbox/.openclaw`) is an **`emptyDir`, NOT a PVC** —
hand-drops vanish on every pod restart. Durable method here: keep the skill
files in **ConfigMap `oc-skills`** (etcd) and add **initContainer
`install-skills`** that copies them into the `home` emptyDir on every pod
start.

```bash
K=kubectl.exe; export KUBECONFIG=<native C:/.../kubeconfig-cs1.conf>
python - <<'PY'
import json, subprocess
SRC="C:/Users/ml/hermes/pcai-styling-demo/skill/pcai-styling-video-automation"
data={"SKILL.md":open(f"{SRC}/SKILL.md").read(),
      "run_pipeline.py":open(f"{SRC}/scripts/run_pipeline.py").read(),
      "tuned.yaml":open(f"{SRC}/prompts/tuned.yaml").read(),
      "customer_adm.yaml":open(f"{SRC}/prompts/customer_adm.yaml").read(),
      "api.md":open(f"{SRC}/references/api.md").read()}
cm={"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"oc-skills",
     "namespace":"oc-test","labels":{"app":"oc-gw"}},"data":data}
subprocess.run(["kubectl","-n","oc-test","apply","-f","-"],
               input=json.dumps(cm), check=True)
img=subprocess.run(["kubectl","-n","oc-test","get","deploy","oc-gw",
    "-o","jsonpath={.spec.template.spec.containers[0].image}"],
    capture_output=True, text=True, check=True).stdout.strip()
cmd=("set -e; mkdir -p /home-out/skills/pcai-styling-video-automation/scripts "
     "/home-out/skills/pcai-styling-video-automation/prompts "
     "/home-out/skills/pcai-styling-video-automation/references "
     "&& cp /skills-cm/SKILL.md /home-out/skills/pcai-styling-video-automation/ "
     "&& cp /skills-cm/run_pipeline.py /home-out/skills/pcai-styling-video-automation/scripts/ "
     "&& cp /skills-cm/tuned.yaml /home-out/skills/pcai-styling-video-automation/prompts/ "
     "&& cp /skills-cm/customer_adm.yaml /home-out/skills/pcai-styling-video-automation/prompts/ "
     "&& cp /skills-cm/api.md /home-out/skills/pcai-styling-video-automation/references/ "
     "&& chmod +x /home-out/skills/pcai-styling-video-automation/scripts/run_pipeline.py "
     "&& echo SKILL_INSTALLED")
patch={"spec":{"template":{"spec":{
  "initContainers":[{"name":"install-skills","image":img,
    "imagePullPolicy":"IfNotPresent","command":["sh","-c",cmd],
    "volumeMounts":[{"name":"skills-cm","mountPath":"/skills-cm","readOnly":True},
                    {"name":"home","mountPath":"/home-out"}]}],
  "volumes":[{"name":"skills-cm","configMap":{"name":"oc-skills"}}]}}}}
subprocess.run(["kubectl","-n","oc-test","patch","deploy","oc-gw",
    "--type=strategic","-p",json.dumps(patch)], check=True)
subprocess.run(["kubectl","-n","oc-test","rollout","status","deploy/oc-gw",
    "--timeout=180s"], check=True)
PY
```

### Verify against what the UI reads (either instance)
The Control UI lists skills via the gateway's `skills.status` WS RPC on
:18789. Connect with the gateway token (instance A: secret
`nemoclaw-openclaw-gateway-token` `.token`; instance B: cm `oc-cfg`
`.data.openclaw.json` → `gateway.auth.token`), send an `Origin` header
(e.g. the UI's host), `connect` (minProtocol/maxProtocol 3, role
`operator`), then `skills.status` → the skill appears with
`source: openclaw-managed`, `eligible: true`. (`openclaw skills info`
in-pod is a fresh disk scan — it proves the file exists but NOT that the
running gateway sees it.)

- **To update the skill later**: instance A — re-copy into the PVC +
  `rollout restart`; instance B — re-`apply` the ConfigMap +
  `rollout restart` (the initContainer re-copies on the fresh start).
- The sandbox python is `/sandbox/.venv/bin/python` (3.13); the stdlib-only
  `run_pipeline.py` runs there as-is (lab ingress is self-signed — the client
  uses an unverified SSL context for lab URLs only).
- `pcai-styling-video-automation` is imported to **both** instances
  (2026-10-08).
