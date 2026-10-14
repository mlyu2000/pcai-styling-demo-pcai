#!/usr/bin/env python3
"""Generate the demo's mock assets — reproducible, no customer input needed.

  sample_scene_dusk.png : REF 2 scene photo (Mario's "car into scene" input).
    1) tries the Z-Image-Turbo t2i endpoint (needs IMG_ENDPOINT reachable —
       from the bastion/cluster only),
    2) otherwise draws a deterministic stylised dusk street (pure PIL),
    3) caches: skips work unless --force, so repeat runs are instant.

Usage (from the repo root):
  python scripts/make_mock_samples.py [--force] [--url http://... --key-env IMG_API_KEY]
"""
import argparse
import base64
import io
import math
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "backend" / "sample_data" / "sample_scene_dusk.png"
W, H = 1024, 768
PROMPT = ("An empty city street at dusk, wet dark asphalt with soft reflections, "
          "glowing street lamps, distant bokeh shop lights, moody teal-and-orange "
          "cinematic color grade, wide road in the foreground, photorealistic, "
          "no cars, no people, low camera eye view of an open road")


def try_model(url: str, key: str) -> Image.Image | None:
    """t2i via vLLM-Omni /v1/chat/completions (same contract as backend)."""
    if not url:
        return None
    body = {
        "model": os.environ.get("IMG_MODEL", "z-image") or "z-image",
        "messages": [{"role": "user", "content": PROMPT}],
        "extra_body": {"height": 768, "width": 1024, "num_inference_steps": 8,
                       "guidance_scale": 0.0, "seed": 42},
    }
    import json as _json
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = _json.loads(r.read().decode())
        content = data["choices"][0]["message"]["content"]
        urls = []
        if isinstance(content, str):
            urls = [content] if content.startswith("data:image") else []
        elif isinstance(content, list):
            for it in content:
                u = (it.get("image_url") or {}).get("url") if isinstance(it, dict) else None
                if u and u.startswith("data:image"):
                    urls.append(u)
        if not urls:
            raise RuntimeError("no data:image in response")
        b64 = urls[0].split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        print(f"model path unavailable ({e}); drawing deterministic scene instead")
        return None


def draw_dusk_street() -> Image.Image:
    """Deterministic stylised dusk-street scene (fixed seed -> reproducible):
    sky gradient + buildings with lit windows + street lamps with halos + wet
    asphalt road with perspective lane lines and reflected lamp smears."""
    random.seed(7)
    rng = np.random.default_rng(11)
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    horizon = int(H * 0.52)

    # --- sky: dusk gradient warm low -> deep teal high (smooth, no seam)
    top = np.array([18, 26, 48], float)
    warm = np.array([150, 90, 70], float)
    low = np.array([232, 132, 74], float)
    sky = np.zeros((horizon, W, 3), np.float32)
    for y in range(horizon):
        t = y / max(1, horizon - 1)
        if t < 0.7:                       # top -> warm midpoint
            k = t / 0.7
            c = top * (1 - k) + warm * k
        else:                             # warm -> orange horizon (smoothstep)
            k = (t - 0.7) / 0.3
            k = k * k * (3 - 2 * k)
            c = warm * (1 - k) + low * k
        sky[y] = c
    img.paste(Image.fromarray(sky.clip(0, 255).astype(np.uint8)), (0, 0))
    for _ in range(60):
        x, y = random.randrange(W), random.randrange(int(horizon * 0.6))
        d.point((x, y), fill=(210, 220, 255))
    # --- buildings both sides: dark silhouettes with warm lit windows
    for side in (0, 1):
        x = 0 if side == 0 else int(W * 0.62)
        jumped = side == 1
        while x < W and (x < W * 0.42 or (x > W * 0.60 and side == 1)):
            bwd = random.randint(60, 130)
            bht = random.randint(int(H * 0.16), int(H * 0.42))
            bx = min(x, W - bwd)
            d.rectangle([bx, horizon - bht, bx + bwd, horizon + 4],
                        fill=(14, 17, 28))
            for wy in range(horizon - bht + 8, horizon - 8, 16):
                for wx in range(bx + 6, bx + bwd - 8, 14):
                    if random.random() < 0.35:
                        c = random.choice([(255, 196, 120), (255, 224, 170),
                                           (180, 200, 235)])
                        d.rectangle([wx, wy, wx + 5, wy + 8], fill=c)
            x += bwd + random.randint(4, 18)
            if side == 0 and not jumped and x > W * 0.30:
                x = int(W * 0.62)
                jumped = True  # one-shot: leave the road corridor open
    # --- wet asphalt: road trapezoid with curbs + sidewalks, dark gradient + grain
    road = np.zeros((H - horizon, W, 3), np.float32)
    for y in range(H - horizon):
        t = y / max(1, H - horizon - 1)
        c = np.array([26, 30, 42], float) * (1 - t) + np.array([12, 14, 20], float) * t
        road[y] = c
    road += rng.normal(0, 2.5, road.shape).astype(np.float32)
    img.paste(Image.fromarray(np.clip(road, 0, 255).astype(np.uint8)), (0, horizon))
    d = ImageDraw.Draw(img, "RGBA")
    # sidewalks + curbs converging to the vanishing point (grounds the road)
    vp = (W // 2, horizon - 6)
    for side in (-1, 1):
        curb_far_x = vp[0] + side * int(W * 0.045)
        curb_near_x = W // 2 + side * int(W * 0.475)
        d.polygon([(vp[0], horizon), (curb_far_x, horizon),
                   (curb_near_x, H), (W if side > 0 else 0, H)],
                  fill=(40, 44, 56, 255))  # sidewalk slab beyond the curb
        d.line([(curb_far_x, horizon + 2), (curb_near_x, H)],
               fill=(86, 94, 116, 255), width=5)   # lit curb edge
    # lane edge lines (white, converging) so the car's lane is legible
    for side in (-1, 1):
        far_x = vp[0] + side * int(W * 0.034)
        near_x = W // 2 + side * int(W * 0.36)
        d.line([(far_x, horizon + 3), (near_x, H)], fill=(205, 205, 195, 140), width=3)
    # perspective lane dashes toward the vanishing point — only in the FAR two
    # thirds of the road so the hero-car zone (lower-center) stays clean asphalt
    for i in range(9):
        t0, t1 = 0.06 + i * 0.09, 0.06 + i * 0.09 + 0.05
        if t1 > 0.62:
            continue
        y0 = int(vp[1] + (H - vp[1]) * t0)
        y1 = int(vp[1] + (H - vp[1]) * t1)
        wdt = max(1, int(1 + 7 * t1))
        alpha = int(40 + 90 * t1)
        d.polygon([(vp[0] - int(6 * t0), y0), (vp[0] + int(6 * t0), y0),
                   (vp[0] + int(8 * t1) + wdt, y1), (vp[0] - int(8 * t1) - wdt, y1)],
                  fill=(225, 225, 215, alpha))
    # --- street lamps: visible pole + arm + head, halo, wet-road reflections
    for lx, s in ((int(W * 0.16), 1.0), (int(W * 0.84), 1.0),
                  (int(W * 0.30), 0.78), (int(W * 0.70), 0.78)):
        ly = horizon - int(150 * s)
        base_y = horizon + int(30 * s)
        pw = max(4, int(8 * s))
        pole = (70, 78, 100, 255)  # lighter than building silhouettes -> readable
        d.line([lx, base_y, lx, ly + 26], fill=pole, width=pw)
        arm_dir = 1 if lx < W // 2 else -1  # arm reaches toward the road
        arm = int(26 * s)
        d.line([lx, ly + 26, lx + arm_dir * arm, ly + 12], fill=pole,
               width=max(3, int(6 * s)))
        hx = lx + arm_dir * arm
        d.rectangle([hx - int(12 * s), ly + 4, hx + int(12 * s), ly + int(14 * s)],
                    fill=(96, 104, 126, 255))  # lamp head housing
        # light cone from head to ground
        cone = Image.new("RGBA", img.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(cone)
        cd.polygon([(hx - 6 * s, ly + 12), (hx + 6 * s, ly + 12),
                    (hx + int(70 * s), base_y + int(90 * s)),
                    (hx - int(70 * s), base_y + int(90 * s))],
                   fill=(255, 214, 150, 26))
        img = Image.alpha_composite(img.convert("RGBA"),
                                    cone.filter(ImageFilter.GaussianBlur(7))).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        lx = hx  # glow + reflection hang off the head, not the pole
        bulb = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bulb)
        bd.ellipse([lx - 10 * s, ly + 6 * s, lx + 10 * s, ly + 22 * s],
                   fill=(255, 224, 160, 255))
        bd.ellipse([lx - 40 * s, ly - 16 * s, lx + 40 * s, ly + 48 * s],
                   fill=(255, 200, 130, 65))
        bulb = bulb.filter(ImageFilter.GaussianBlur(6 * s))
        img.paste(Image.alpha_composite(img.convert("RGBA"), bulb).convert("RGB"),
                  (0, 0))
        refl = Image.new("RGBA", img.size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(refl)
        ry0 = horizon + int(30 * s)
        rd.rectangle([lx - 10 * s, ry0, lx + 10 * s, min(H, ry0 + int(220 * s))],
                     fill=(255, 205, 140, 70))
        refl = refl.filter(ImageFilter.GaussianBlur(9))
        img = Image.alpha_composite(img.convert("RGBA"), refl).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
    # horizon haze + vignette for cinematic mood
    haze = Image.new("L", img.size, 0)
    hd = ImageDraw.Draw(haze)
    hd.rectangle([0, horizon - 30, W, horizon + 60], fill=90)
    glow = Image.new("RGB", img.size, (255, 190, 150))
    img = Image.composite(glow, img, haze.filter(ImageFilter.GaussianBlur(24)))
    vig = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    dark = Image.new("RGB", (W, H), (0, 0, 0))
    mask = Image.fromarray(np.clip((rr - 0.75) * 130, 0, 110).astype(np.uint8))
    img = Image.composite(dark, img, mask.filter(ImageFilter.GaussianBlur(40))
                          .point(lambda v: int(v * 0.8)))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--url", default="", help="image model base URL (cluster-internal)")
    ap.add_argument("--key-env", default="IMG_API_KEY")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and not a.force:
        print(f"cached: {OUT} (use --force to regenerate)")
        return 0
    im = try_model(a.url, os.environ.get(a.key_env, "")) \
        if a.url else try_model(os.environ.get("IMG_ENDPOINT", ""),
                                os.environ.get(a.key_env, ""))
    src = "z-image-t2i"
    if im is None:
        im, src = draw_dusk_street(), "pil-deterministic"
    im = im.resize((W, H), Image.LANCZOS) if im.size != (W, H) else im
    im.save(OUT)
    print(f"wrote {OUT} (1024x768, source={src})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
