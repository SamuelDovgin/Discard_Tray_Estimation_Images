#!/usr/bin/env python3
"""
detect_and_crop.py – Detect the discard-tray shoe in each frame via
computer vision, then crop every image to a common frame with fixed
margins around the shoe.

Detection strategy (tailored to clear acrylic tray on a table):
  1. Find the dark BASE plate  (thresholding – very reliable)
  2. From the base, search UPWARD for the acrylic wall edges
     (Canny + row-wise edge density) to find the tray top.
  3. Fall back to an estimated height ratio if edges are too faint.

Alignment is anchored on the shoe's *base* (bottom edge) by default,
because the base position is the most stable feature across frames —
cards grow upward while the base stays put.

Margin convention (fractions of the detected shoe WIDTH):
  --margin-left 0.05   ->  5 %  of shoe width, added to left
  --margin-right 0.05  ->  5 %  of shoe width, added to right
  --margin-top 0.10    ->  10 % of shoe width, added to top
  --margin-bottom 0.10 ->  10 % of shoe width, added to bottom

Legacy options (--h-margin, --v-margin) set all sides symmetrically.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ───────────────────────── image listing ──────────────────────────────

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _list_images(folder: Path) -> List[Path]:
    """Return image paths sorted by the last numeric group in the stem."""
    paths = [p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS]
    if not paths:
        raise SystemExit(f"No images found in {folder}")

    def _num(p: Path) -> int:
        m = re.search(r"(\d+)(?!.*\d)", p.stem)
        return int(m.group(1)) if m else 0

    return sorted(paths, key=_num)


# ───────────────────────── shoe detection ─────────────────────────────


def _find_dark_base(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find the dark base plate of the shoe. Returns (x, y, w, h) or None."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold for dark pixels (the base is black / very dark)
    _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Restrict search to the centre-lower region of the image
    mask = np.zeros_like(dark)
    mask[int(h * 0.30):int(h * 0.92), int(w * 0.12):int(w * 0.88)] = 255
    dark = cv2.bitwise_and(dark, mask)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close, iterations=3)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_open, iterations=1)

    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Tuple[int, int, int, int]] = None
    best_area = 0

    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        # Base is wider than tall
        if bh > bw * 0.8:
            continue
        # Size constraints (relative to image)
        if bw < w * 0.05 or bw > w * 0.40:
            continue
        if bh < h * 0.02 or bh > h * 0.18:
            continue
        # Roughly centred
        cx = bx + bw / 2
        if abs(cx - w / 2) > w * 0.30:
            continue

        if area > best_area:
            best_area = area
            best = (bx, by, bw, bh)

    return best


def _find_tray_rim_y(
    img: np.ndarray,
    base_rect: Tuple[int, int, int, int],
    tray_ratio: float = 1.30,
) -> Optional[int]:
    """Find the tray rim by looking for a strong horizontal edge near the
    expected rim position (derived from *tray_ratio* x base-width).

    The rim is a distinctive horizontal line.  We use Sobel-Y in a narrow
    search window centred on the expected position so background features
    are unlikely to interfere.

    Returns the y-coordinate of the rim, or None.
    """
    ih, iw = img.shape[:2]
    bx, by, bw, bh = base_rect
    base_bottom = by + bh

    expected_rim_y = base_bottom - int(bw * tray_ratio)

    # Search ±25 % of base width around the expected position
    margin = int(bw * 0.25)
    y0 = max(0, expected_rim_y - margin)
    y1 = min(by, expected_rim_y + margin)  # never below base top

    if y1 <= y0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    sobel_y = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))

    # Sum horizontal-edge strength across the shoe's column range
    roi = sobel_y[y0:y1, bx:bx + bw]
    row_sum = roi.sum(axis=1)

    if row_sum.max() < bw * 3:  # too weak → give up
        return None

    # Light smoothing
    if len(row_sum) > 5:
        k = np.ones(5) / 5.0
        row_sum = np.convolve(row_sum, k, mode="same")

    peak_idx = int(np.argmax(row_sum))
    return y0 + peak_idx


def _detect_foreground_fallback(
    img: np.ndarray,
    prev: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Last-resort foreground segmentation (background colour modelling)."""
    h, w = img.shape[:2]
    m = max(1, int(min(w, h) * 0.06))
    strips = [img[:m], img[-m:], img[:, :m], img[:, -m:]]
    bg = np.vstack([s.reshape(-1, 3) for s in strips]).astype(np.float64)
    mu = bg.mean(0)
    sigma = bg.std(0) + 1.0

    diff = np.abs(img.astype(np.float64) - mu)
    fg = (np.max(diff / sigma, axis=2) > 2.5).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=4)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=2)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Tuple[int, int, int, int]] = None
    best_s = 0.0

    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < w * 0.06 or bh < h * 0.10:
            continue
        cx = bx + bw / 2
        cy = by + bh / 2
        area = (bw * bh) / (w * h)
        dx = abs(cx / w - 0.5) * 2
        dy = abs(cy / h - 0.5) * 2
        cent = max(0.01, 1.0 - (dx + dy) / 2)
        s = area * cent
        if prev is not None:
            pcx, pcy = prev
            d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            s *= max(0.05, 1.0 - d / (w * 0.12))
        if s > best_s:
            best_s = s
            best = (bx, by, bw, bh)

    return best


def detect_shoe(
    img: np.ndarray,
    prev_anchor: Optional[Tuple[float, float]] = None,
    height_ratio: float = 1.55,
) -> Optional[Tuple[int, int, int, int]]:
    """Detect the full shoe (base + walls + card headroom).

    Returns (x, y, w, h) where (x,y) is the top-left and (w,h) spans
    from the top of the crop region to the base bottom.

    Strategy:
      1. Find the dark base (most reliable anchor).
      2. Try to locate the tray *rim* via horizontal-edge (Sobel-Y) search
         near the expected position (~1.30 x base-width above the base).
         If found, add card headroom above the rim.
      3. Fall back to ``height_ratio * base_width`` if rim detection fails.

    *height_ratio*: total shoe-height / base-width ratio (includes card
    headroom above the rim).  Default 1.55.
    """
    # The acrylic-only ratio (base bottom → rim) is about 1.30.
    TRAY_RATIO = 1.30

    base = _find_dark_base(img)

    if base is not None:
        bx, by, bw, bh = base
        base_bottom = by + bh

        # Try to find the rim for precise positioning
        rim_y = _find_tray_rim_y(img, base, tray_ratio=TRAY_RATIO)

        if rim_y is not None:
            # Add headroom above the rim for cards extending past it
            headroom_ratio = max(0.0, height_ratio - TRAY_RATIO)
            headroom_px = int(round(bw * headroom_ratio))
            tray_top = rim_y - headroom_px
        else:
            # Fall back to ratio-estimated height
            shoe_h = int(round(bw * height_ratio))
            tray_top = base_bottom - shoe_h

        tray_top = max(0, tray_top)
        shoe_h = base_bottom - tray_top
        return (bx, tray_top, bw, shoe_h)

    # If base detection fails entirely, try generic foreground segmentation
    return _detect_foreground_fallback(img, prev_anchor)


# ───────────────────────── anchor smoothing ───────────────────────────


def _smooth_1d(arr: np.ndarray, win: int) -> np.ndarray:
    """Median-filter a 1-D float array, filling NaN gaps first."""
    out = arr.copy()
    # forward fill
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    # back fill
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    # median filter
    half = win // 2
    smoothed = np.empty_like(out)
    for i in range(len(out)):
        lo = max(0, i - half)
        hi = min(len(out), i + half + 1)
        smoothed[i] = np.nanmedian(out[lo:hi])
    return smoothed


# ───────────────────────── crop helper ────────────────────────────────


def _extract_crop(
    img: np.ndarray, x0: int, y0: int, cw: int, ch: int,
) -> np.ndarray:
    """Extract a (cw x ch) region, padding with border replication."""
    h, w = img.shape[:2]
    pl = max(0, -x0)
    pt = max(0, -y0)
    pr = max(0, x0 + cw - w)
    pb = max(0, y0 + ch - h)
    if pl or pt or pr or pb:
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        x0 += pl
        y0 += pt
    return img[y0 : y0 + ch, x0 : x0 + cw]


# ───────────────────────── CLI ────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", type=Path, default=Path("1024x"),
        help="Input image folder (default: 1024x)",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("cropped_hq"),
        help="Output folder for high-quality crops (default: cropped_hq)",
    )
    ap.add_argument(
        "--margin-left", type=float, default=None,
        help="Left margin as a fraction of shoe width (default: 0.05)",
    )
    ap.add_argument(
        "--margin-right", type=float, default=None,
        help="Right margin as a fraction of shoe width (default: 0.05)",
    )
    ap.add_argument(
        "--margin-top", type=float, default=None,
        help="Top margin as a fraction of shoe width (default: 0.10)",
    )
    ap.add_argument(
        "--margin-bottom", type=float, default=None,
        help="Bottom margin as a fraction of shoe width (default: 0.10)",
    )
    # Legacy options for backward compatibility
    ap.add_argument(
        "--h-margin", type=float, default=0.05,
        help="[Legacy] Horizontal margin (sets left & right if not specified individually)",
    )
    ap.add_argument(
        "--v-margin", type=float, default=0.10,
        help="[Legacy] Vertical margin (sets top & bottom if not specified individually)",
    )
    ap.add_argument(
        "--anchor", choices=["base", "center"], default="base",
        help="Vertical anchor point for alignment (default: base). "
             "'base' keeps the shoe bottom fixed; 'center' centres the shoe.",
    )
    ap.add_argument(
        "--height-ratio", type=float, default=1.55,
        help="Fallback shoe-height / base-width ratio when edge detection "
             "cannot find the tray top (default: 1.55).",
    )
    ap.add_argument(
        "--smooth", type=int, default=9,
        help="Temporal median-filter window for anchor smoothing (default: 9)",
    )
    ap.add_argument(
        "--name-pattern", default="{index:03d}.png",
        help="Output filename pattern. Fields: index, src_num. (default: {index:03d}.png)",
    )
    ap.add_argument(
        "--debug-dir", type=Path, default=None,
        help="If set, save debug images with drawn bounding boxes to this folder",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    paths = _list_images(args.input)
    if args.limit:
        paths = paths[: args.limit]
    n = len(paths)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Found {n} images in {args.input}")

    # ── pass 1: detect shoe in every frame ────────────────────────────
    raw: List[Optional[Tuple[int, int, int, int]]] = []
    prev: Optional[Tuple[float, float]] = None

    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Cannot read {p}")

        det = detect_shoe(img, prev, height_ratio=args.height_ratio)
        raw.append(det)

        if det is not None:
            cx = det[0] + det[2] / 2.0
            base_y = det[1] + det[3]          # base y (bottom of detection)
            prev = (cx, base_y)
            if not args.quiet and (i % 25 == 0 or i == n - 1):
                print(
                    f"  [{i + 1:>3}/{n}] {p.name}  "
                    f"shoe {det[2]}x{det[3]} @ ({det[0]},{det[1]})"
                )
        else:
            if not args.quiet:
                print(f"  [{i + 1:>3}/{n}] {p.name}  *** detection failed ***")

        # optional debug visualisation
        if args.debug_dir:
            dbg = img.copy()
            if det is not None:
                x, y, dw, dh = det
                cv2.rectangle(dbg, (x, y), (x + dw, y + dh), (0, 255, 0), 2)
                # Red dot = base centre (anchor)
                cv2.circle(
                    dbg,
                    (int(x + dw / 2), int(y + dh)),
                    5, (0, 0, 255), -1,
                )
            cv2.imwrite(
                str(args.debug_dir / f"debug_{i:03d}.jpg"),
                dbg,
                [cv2.IMWRITE_JPEG_QUALITY, 60],
            )

    ok = sum(1 for d in raw if d is not None)
    if ok == 0:
        raise SystemExit("Shoe detection failed on every image.")
    if not args.quiet:
        fail = n - ok
        print(
            f"Detected shoe in {ok}/{n} frames"
            + (f" ({fail} will be interpolated)" if fail else "")
        )

    # ── build anchor arrays & smooth ──────────────────────────────────
    anchor_x = np.full(n, np.nan)
    anchor_y = np.full(n, np.nan)
    widths = np.full(n, np.nan)
    heights = np.full(n, np.nan)

    for i, d in enumerate(raw):
        if d is None:
            continue
        bx, by, bw, bh = d
        anchor_x[i] = bx + bw / 2.0   # horizontal centre
        anchor_y[i] = by + bh          # base y (bottom of shoe)
        widths[i] = bw
        heights[i] = bh

    anchor_x = _smooth_1d(anchor_x, args.smooth)
    anchor_y = _smooth_1d(anchor_y, args.smooth)

    # ── reference dimensions & margins ────────────────────────────────
    valid_w = widths[~np.isnan(widths)]
    valid_h = heights[~np.isnan(heights)]

    ref_w = int(round(float(np.median(valid_w))))
    # Use P97 height so the tallest card stacks still fit inside the crop
    ref_h = int(round(float(np.percentile(valid_h, 97))))

    # Resolve margins: use individual if set, else fall back to legacy h/v
    m_left = args.margin_left if args.margin_left is not None else args.h_margin
    m_right = args.margin_right if args.margin_right is not None else args.h_margin
    m_top = args.margin_top if args.margin_top is not None else args.v_margin
    m_bottom = args.margin_bottom if args.margin_bottom is not None else args.v_margin

    left_px = int(round(ref_w * m_left))
    right_px = int(round(ref_w * m_right))
    top_px = int(round(ref_w * m_top))
    bottom_px = int(round(ref_w * m_bottom))

    out_w = ref_w + left_px + right_px
    out_h = ref_h + top_px + bottom_px
    out_w += out_w % 2   # make even
    out_h += out_h % 2

    if not args.quiet:
        print(
            f"Ref shoe: {ref_w}x{ref_h}  "
            f"margins L={left_px} R={right_px} T={top_px} B={bottom_px}  "
            f"output {out_w}x{out_h}"
        )

    # ── pass 2: crop & save ───────────────────────────────────────────
    manifest_images: list[dict] = []

    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        ax = float(anchor_x[i])
        ay = float(anchor_y[i])

        # Horizontal: center the shoe, applying left/right margins
        shoe_left = ax - ref_w / 2
        crop_x = int(round(shoe_left - left_px))

        if args.anchor == "base":
            # place shoe base at (out_h - bottom_px) from the top
            crop_y = int(round(ay - out_h + bottom_px))
        else:
            # centre the shoe vertically
            shoe_top = ay - ref_h
            crop_y = int(round(shoe_top - top_px))

        crop = _extract_crop(img, crop_x, crop_y, out_w, out_h)

        src_num_m = re.search(r"(\d+)(?!.*\d)", p.stem)
        src_num = int(src_num_m.group(1)) if src_num_m else i
        out_name = args.name_pattern.format(index=i, src_num=src_num)
        cv2.imwrite(
            str(args.output / out_name),
            crop,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],   # fast, lossless
        )

        manifest_images.append({
            "index": i,
            "source": p.name,
            "output": out_name,
            "anchor": {"x": round(ax, 1), "y": round(ay, 1)},
            "crop_origin": {"x": crop_x, "y": crop_y},
        })

        if not args.quiet and (i % 50 == 0 or i == n - 1):
            print(f"  [{i + 1:>3}/{n}] saved {out_name}")

    # ── manifest ──────────────────────────────────────────────────────
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "count": len(manifest_images),
        "anchor_mode": args.anchor,
        "height_ratio": args.height_ratio,
        "ref_shoe": {"w": ref_w, "h": ref_h},
        "margins": {
            "left_frac": m_left,
            "right_frac": m_right,
            "top_frac": m_top,
            "bottom_frac": m_bottom,
            "left_px": left_px,
            "right_px": right_px,
            "top_px": top_px,
            "bottom_px": bottom_px,
        },
        "output_size": {"w": out_w, "h": out_h},
        "smooth_window": args.smooth,
        "images": manifest_images,
    }
    out_json = args.output / "manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"\nDone. {len(manifest_images)} images -> {args.output}/")
        print(f"Manifest: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
