#!/usr/bin/env python3
"""Unit tests for the stage-4/5 deterministic image helpers of the PCAI styling demo.

Covers backend/main.py: _body_mask, _recolor_body, _pencil_sketch,
_mock_designer_edit, _compose_scene, _colorway_targets, _opposite_sketch_fallback.

Plain-python, no pytest: each test runs its asserts and reports PASS/FAIL;
the process exits nonzero if anything fails.

    python scripts/test_stage_helpers.py

All inputs are SYNTHETIC images built here with PIL, so the tests are
self-contained and reproducible (no dependency on scratch assets).
"""

import importlib.util
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend" / "main.py"


# ------------------------------------------------------------------ import
def load_backend():
    """Import backend/main.py by path with a throwaway RESULTS_DIR.

    The module logs a config error when MLIS/LLM env vars are missing, so we
    point them at an unroutable localhost (never contacted by these helpers)
    and blank IMG_ENDPOINT so any accidental _img_generate call fails fast."""
    tmp = tempfile.mkdtemp(prefix="pcai-stage-tests-")
    os.environ["RESULTS_DIR"] = tmp
    os.environ["SAMPLE_DATA_DIR"] = str(Path(tmp) / "no_sample_data")
    os.environ["IMG_ENDPOINT"] = ""
    os.environ.setdefault("MLIS_VIDEO_GENERATION_URL", "http://127.0.0.1:1")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:1")
    os.environ["MLIS_VIDEO_GENERATION_URL"] = "http://127.0.0.1:1"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1"
    spec = importlib.util.spec_from_file_location("styling_backend_under_test", BACKEND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BACKEND}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- synth assets
BLUE = (40, 90, 200)
GREY_BG = (168, 172, 176)


def synth_car(w=480, h=300, body=BLUE, bg=GREY_BG, dark_block=False):
    """A blue car (cabin + body + two black wheels) on a uniform grey studio
    background — the corner-based background estimate in _body_mask works on it."""
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([int(w * 0.34), int(h * 0.30), int(w * 0.66), int(h * 0.50)],
                        radius=12, fill=(30, 65, 150))                     # cabin
    d.rectangle([int(w * 0.38), int(h * 0.34), int(w * 0.62), int(h * 0.47)],
                fill=(210, 225, 235))                                      # glass
    d.rounded_rectangle([int(w * 0.10), int(h * 0.50), int(w * 0.90), int(h * 0.72)],
                        radius=18, fill=body)                              # painted body
    for cx in (int(w * 0.28), int(w * 0.72)):
        cy = int(h * 0.72)
        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(25, 25, 28))  # wheels
    if dark_block:
        # heavy marker block on the FRONT (left) of the car only, hugging the
        # left edge so a horizontal mirror is unambiguously detectable.
        d.rectangle([int(w * 0.025), int(h * 0.52), int(w * 0.10), int(h * 0.70)],
                    fill=(8, 8, 10))
    return img


def synth_scene(w=600, h=380):
    """Dusk street: bright sky over the top 55%, asphalt that darkens toward the
    viewer (the road-band guess in _compose_scene needs the near field darkest)."""
    a = np.zeros((h, w, 3), np.float64)
    for y in range(h):
        t = y / h
        if t < 0.55:
            v = 150 + 60 * (1 - t / 0.55)
            a[y] = (v, v * 0.8, v * 0.65)
        else:
            k = (t - 0.55) / 0.45
            base = 78 - 50 * k
            a[y] = (base, base * 1.02, base * 1.08)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(h * 0.55) - 2, w, int(h * 0.55) + 2], fill=(96, 96, 102))
    return img


def arr(img):
    return np.asarray(img.convert("RGB")).astype(np.float64)


def mean_abs_diff(a, b):
    return float(np.abs(a - b).mean())


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def hue_deg(hsv_img, mask):
    """Circular-mean PIL-HSV hue (0-255) of masked pixels, in degrees.
    A plain mean is wrong here: a hue shift wraps blue pixels to both sides of
    red, so arithmetic averaging over 0/360 cancels the signal out."""
    h = np.asarray(hsv_img.convert("HSV")).astype(np.float64)
    sel = mask & (h[..., 1] > 30)          # only saturated pixels have a hue
    if not sel.any():
        return None
    ang = np.deg2rad(h[..., 0][sel] * 360.0 / 255.0)
    mean_vec = np.angle(np.exp(1j * ang).mean())
    return float((np.rad2deg(mean_vec) + 360.0) % 360.0)


def circ_dist_deg(a, b):
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def laplacian_energy(img):
    """Mean |Laplacian| of the grayscale = edge energy, built from 1-D numpy
    convolve along both axes (kernel [1, -2, 1] twice, summed)."""
    g = np.asarray(ImageOps.grayscale(img)).astype(np.float64)
    k = np.array([1.0, -2.0, 1.0])
    dxx = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, g)
    dyy = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, g)
    return float(np.abs(dxx + dyy).mean())


# ------------------------------------------------------------------- tests
def test_body_mask_fraction(m, notes):
    car = synth_car()
    mask = m._body_mask(car)
    assert mask.dtype == bool, f"mask dtype {mask.dtype}, expected bool"
    assert mask.shape == (car.height, car.width), f"mask shape {mask.shape}"
    frac = float(mask.mean())
    notes.append(f"mask fraction={frac:.3f}")
    assert 0.03 < frac < 0.8, f"mask fraction {frac:.3f} outside (0.03, 0.8)"
    # the mask must actually cover painted body and exclude the grey background
    a = arr(car)
    body_px = np.all(np.abs(a - np.array(BLUE)) < 30, axis=2)
    coverage = float(mask[body_px].mean())
    notes.append(f"body-pixel coverage={coverage:.2f}")
    assert coverage > 0.8, f"mask misses the blue body (coverage {coverage:.2f})"
    bg_px = np.all(np.abs(a - np.array(GREY_BG)) < 8, axis=2)
    bg_leak = float(mask[bg_px].mean())
    notes.append(f"background leak={bg_leak:.4f}")
    assert bg_leak < 0.02, f"mask leaks into the background ({bg_leak:.3f})"


def test_recolor_body_preserves_geometry(m, notes):
    car = synth_car()
    out = m._recolor_body(car, "red")
    assert out.size == car.size, f"size changed: {out.size} != {car.size}"
    assert out.mode == "RGB", f"mode {out.mode}, expected RGB"

    mask = m._body_mask(car)
    a, b = arr(car), arr(out)
    outside = float(np.abs(a[~mask] - b[~mask]).mean())
    notes.append(f"outside-mask mean abs diff={outside:.4f}")
    assert outside < 2.0, f"background/geometry changed (outside diff {outside:.3f} >= 2.0)"

    inside = float(np.abs(a[mask] - b[mask]).mean())
    notes.append(f"inside-mask mean abs diff={inside:.1f}")
    assert inside > 10.0, f"body paint barely changed (inside diff {inside:.2f})"

    h_in, h_out = hue_deg(car, mask), hue_deg(out, mask)
    assert h_in is not None and h_out is not None, "no saturated body hue to measure"
    d_in, d_out = circ_dist_deg(h_in, 0.0), circ_dist_deg(h_out, 0.0)
    notes.append(f"body hue {h_in:.0f}deg -> {h_out:.0f}deg; dist-to-red {d_in:.0f} -> {d_out:.0f}")
    assert d_in - d_out > 60.0, f"hue did not shift toward red ({d_in:.0f}deg -> {d_out:.0f}deg)"
    rb_in = a[..., 0].mean() - a[..., 2].mean()
    rb_out = b[..., 0].mean() - b[..., 2].mean()
    notes.append(f"R-B {rb_in:.1f} -> {rb_out:.1f}")
    assert rb_out > rb_in + 20, f"body did not get redder (R-B {rb_in:.1f} -> {rb_out:.1f})"
    # wheels stay black: geometry/details untouched (wheel pixels are not body)
    assert outside < 2.0


def test_pencil_sketch(m, notes):
    car = synth_car()
    sk = m._pencil_sketch(car)
    assert sk.size == car.size, f"size {sk.size} != {car.size}"
    assert sk.mode == "RGB", f"mode {sk.mode}, expected RGB (docstring promises a colour sketch)"
    e_in, e_sk = laplacian_energy(car), laplacian_energy(sk)
    notes.append(f"edge energy {e_in:.3f} -> {e_sk:.3f} (x{e_sk / max(e_in, 1e-9):.2f})")
    assert e_sk > e_in, f"sketch has no extra line art (edge energy {e_sk:.3f} <= {e_in:.3f})"
    # the fallback must not silently return the original
    assert mean_abs_diff(arr(car), arr(sk)) > 5.0, "pencil sketch returned the input unchanged"


def test_mock_designer_edit(m, notes):
    car = synth_car()
    edited = m._mock_designer_edit(car)
    assert edited.size == car.size, f"size {edited.size} != {car.size}"
    assert edited.mode == "RGB", f"mode {edited.mode}, expected RGB"
    a, c = arr(car), arr(edited)
    diff = mean_abs_diff(a, c)
    notes.append(f"mean abs diff vs input={diff:.2f}")
    assert diff > 1.0, f"designer edit is a no-op (mean diff {diff:.3f})"
    rb_in = a[..., 0].mean() - a[..., 2].mean()
    rb_out = c[..., 0].mean() - c[..., 2].mean()
    notes.append(f"warm grade: R-B {rb_in:.1f} -> {rb_out:.1f}")
    assert rb_out > rb_in, f"no detectable warm grade (R-B {rb_in:.1f} -> {rb_out:.1f})"
    assert (rb_out - rb_in) > 5.0, f"warm grade too weak (delta {rb_out - rb_in:.1f})"
    # the crimson beltline stripe must actually land on the car
    mask = m._body_mask(car)
    red_dom = (c[..., 0] > c[..., 2] + 40) & (c[..., 0] > c[..., 1] + 30)
    stripe = float((red_dom & mask).sum())
    notes.append(f"accent-stripe pixels on body={int(stripe)}")
    assert stripe > 200, f"accent pinstripe missing ({int(stripe)} body pixels)"


def test_compose_scene(m, notes):
    car, scene = synth_car(), synth_scene()
    out = m._compose_scene(car, scene)
    assert out.size == scene.size, f"output {out.size} != scene {scene.size}"
    assert out.mode == "RGB", f"mode {out.mode}, expected RGB"

    s, o = arr(scene), arr(out)
    h, w = scene.height, scene.width
    r0, c0, c1 = int(h * 2 / 3), int(w / 3), int(w * 2 / 3)
    band = mean_abs_diff(s[r0:h, c0:c1], o[r0:h, c0:c1])
    notes.append(f"center-bottom-third mean abs diff={band:.1f}")
    assert band > 5.0, f"car was not pasted into the scene (band diff {band:.2f})"
    top_band = mean_abs_diff(s[:int(h * 0.5)], o[:int(h * 0.5)])
    notes.append(f"sky region unchanged={top_band:.4f}")
    assert top_band < 1.0, f"scene sky was overwritten (diff {top_band:.2f})"
    # blue car pixels appear in the composite
    blueish = (o[..., 2] > o[..., 0] + 40) & (o[..., 2] > o[..., 1] + 20)
    notes.append(f"blue body pixels in composite={int(blueish.sum())}")
    assert blueish[r0:h, c0:c1].sum() > 500, "the recolored body is not visible in the scene"

    out2 = m._compose_scene(car, scene)
    assert png_bytes(out) == png_bytes(out2), "_compose_scene is not deterministic"
    notes.append("deterministic=True")


def test_colorway_targets(m, notes):
    got = m._colorway_targets("blue", 2, None, "")
    notes.append(f"blue/2/auto -> {got}")
    assert isinstance(got, list) and len(got) == 2, f"expected 2 targets, got {got}"
    assert len(set(got)) == 2, f"targets not distinct: {got}"
    assert "blue" not in got, f"a target equals the base color: {got}"
    assert all(isinstance(c, str) and c for c in got), f"non-empty strings expected: {got}"

    n3 = m._colorway_targets("blue", 3, None, "")
    notes.append(f"blue/3/auto -> {n3}")
    assert len(n3) == 3 and len(set(n3)) == 3 and "blue" not in n3, f"3 targets: {n3}"

    exp = m._colorway_targets("blue", 2, ["green", "gold", "purple"], "")
    notes.append(f"explicit override -> {exp}")
    assert exp == ["green", "gold"], f"explicit color_targets must win/truncate to n: {exp}"

    req = m._colorway_targets("blue", 1, None, "make it crimson")
    notes.append(f"requirement colour -> {req}")
    assert req == ["crimson"], f"requirement colour should be the first target: {req}"

    more = m._colorway_targets("blue", 6, None, "")
    notes.append(f"n=6 -> {more}")
    assert len(more) == len(set(more)) and "blue" not in more, f"targets must stay distinct: {more}"


def test_opposite_sketch_fallback(m, notes):
    car = synth_car(dark_block=True)
    out = m._opposite_sketch_fallback(car, car.width, car.height)
    assert out.size == car.size, f"size {out.size} != {car.size}"
    assert out.mode == "RGB", f"mode {out.mode}, expected RGB"

    sketch = m._pencil_sketch(car)
    mirrored = ImageOps.mirror(sketch)
    o_g = np.asarray(ImageOps.grayscale(out)).astype(np.float64)
    ink_profile = 255.0 - o_g
    col = ink_profile.mean(axis=0)
    ref_col = (255.0 - np.asarray(ImageOps.grayscale(mirrored)).astype(np.float64)).mean(axis=0)
    raw_col = (255.0 - np.asarray(ImageOps.grayscale(sketch)).astype(np.float64)).mean(axis=0)
    corr_ref = float(np.corrcoef(col, ref_col)[0, 1])
    corr_raw = float(np.corrcoef(col, raw_col)[0, 1])
    notes.append(f"corr(mirrored)={corr_ref:.3f} corr(unmirrored)={corr_raw:.3f}")
    assert corr_ref > 0.8, f"output is not the mirrored sketch (corr {corr_ref:.3f})"
    # the front-only marker block must end up on the opposite side
    band = car.width // 10
    left, right = col[:band].mean(), col[-band:].mean()
    notes.append(f"ink left={left:.1f} right={right:.1f}")
    assert right > left, f"image was not flipped horizontally (left {left:.1f} >= right {right:.1f})"
    assert mean_abs_diff(arr(out), arr(sketch)) > 5.0, "fallback returned the front sketch"


def test_load_mock_scene_missing(m, notes):
    """SAMPLE_DATA_DIR points at an empty temp dir, so the bundled scene must
    degrade to None instead of raising (the stage then 400s with a clear hint)."""
    assert m._load_mock_scene() is None, "missing mock scene must return None"
    assert not m.MOCK_SCENE_FILE.exists(), f"unexpected mock scene at {m.MOCK_SCENE_FILE}"
    notes.append("missing bundled scene -> None")


TESTS = [
    ("1. _body_mask: mask fraction in (0.03, 0.8)", test_body_mask_fraction),
    ("2. _recolor_body: geometry preserved + hue -> red", test_recolor_body_preserves_geometry),
    ("3. _pencil_sketch: same-size RGB + higher edge energy", test_pencil_sketch),
    ("4. _mock_designer_edit: RGB + differs + warm grade", test_mock_designer_edit),
    ("5. _compose_scene: scene size + car pasted + deterministic", test_compose_scene),
    ("6. _colorway_targets: distinct, no base, explicit wins", test_colorway_targets),
    ("7. _opposite_sketch_fallback: mirrored, same size", test_opposite_sketch_fallback),
    ("8. _load_mock_scene: graceful when asset absent", test_load_mock_scene_missing),
]


def main() -> int:
    print(f"backend under test: {BACKEND}")
    try:
        m = load_backend()
    except Exception as e:
        print(f"FATAL: could not import backend/main.py: {type(e).__name__}: {e}")
        return 2
    print(f"RESULTS_DIR -> {m.RESULTS_DIR}\n")

    failed = 0
    for name, fn in TESTS:
        notes: list[str] = []
        try:
            fn(m, notes)
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}")
            for n in notes:
                print(f"        {n}")
            print(f"      -> {e}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}  ({type(e).__name__}: {e})")
        else:
            print(f"PASS  {name}")
            for n in notes:
                print(f"        {n}")
    total = len(TESTS)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
