#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FixReportItem:
    index: int
    filename: str
    selected: bool
    red_pixels_before: int
    red_pixels_after: int
    changed: bool


def _extract_number(path: Path) -> Optional[int]:
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(m.group(1)) if m else None


def _iter_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        raise SystemExit(f"No images found in {folder}")

    def sort_key(p: Path):
        n = _extract_number(p)
        return (0 if n is not None else 1, n if n is not None else 0, p.name)

    return sorted(paths, key=sort_key)


def _parse_ranges(value: str) -> Set[int]:
    # Accept: "14-16,42-54,78,100-200"
    out: Set[int] = set()
    if not value.strip():
        return out
    parts = re.split(r"[,\s]+", value.strip())
    for part in parts:
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            if b < a:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def _hsv_red_mask(hsv: np.ndarray, *, min_sat: int, min_val: int) -> np.ndarray:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    red = ((h <= 10) | (h >= 170)) & (s >= min_sat) & (v >= min_val)
    return red.astype(np.uint8) * 255


def _hsv_blue_mask(hsv: np.ndarray, *, min_h: int, max_h: int, min_sat: int, min_val: int) -> np.ndarray:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue = (h >= min_h) & (h <= max_h) & (s >= min_sat) & (v >= min_val)
    return blue.astype(np.uint8) * 255


def _median_blue_hue(img_bgr: np.ndarray) -> Optional[int]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = _hsv_blue_mask(hsv, min_h=90, max_h=140, min_sat=80, min_val=40)
    if int(mask.sum()) < 255 * 500:
        return None
    h = hsv[:, :, 0]
    vals = h[mask.astype(bool)]
    return int(round(float(np.median(vals))))


def _pick_reference_blue_hue(paths: List[Path]) -> int:
    # Scan a handful of early frames and pick the first with a decent amount of blue pixels.
    for p in paths[:60]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        hue = _median_blue_hue(img)
        if hue is not None:
            return hue
    return 115  # reasonable default for typical "Bicycle blue" in OpenCV HSV


def _load_manifest_margins(input_dir: Path) -> Optional[Tuple[int, int, int, int]]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        margins = manifest.get("margins") or {}
        left_px = int(margins.get("left_px", 0))
        right_px = int(margins.get("right_px", 0))
        top_px = int(margins.get("top_px", 0))
        bottom_px = int(margins.get("bottom_px", 0))
        return left_px, right_px, top_px, bottom_px
    except Exception:
        return None


def _margin_mask(shape_hw: Tuple[int, int], margins_px: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape_hw
    left_px, right_px, top_px, bottom_px = margins_px
    top_px = max(0, int(top_px))
    left_px = max(0, int(left_px))
    right_px = max(0, int(right_px))
    bottom_px = max(0, int(bottom_px))

    mask = np.zeros((h, w), dtype=bool)
    if left_px:
        mask[:, :left_px] = True
    if right_px:
        mask[:, w - right_px :] = True
    if top_px:
        mask[:top_px, :] = True
    if bottom_px:
        mask[h - bottom_px :, :] = True
    return mask


def _median_background_for_mask(images_bgr: List[np.ndarray], mask: np.ndarray) -> np.ndarray:
    # Compute the per-pixel median only for the masked pixels (fast + memory-safe).
    h, w = images_bgr[0].shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    n = len(images_bgr)
    p = len(xs)
    samples = np.empty((n, p, 3), dtype=np.uint8)
    for i, img in enumerate(images_bgr):
        samples[i] = img[ys, xs]
    med = np.median(samples, axis=0).astype(np.uint8)  # (p, 3)

    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[ys, xs] = med
    return out


def _apply_red_to_blue(img_bgr: np.ndarray, *, target_hue: int, red_min_sat: int, red_min_val: int, restrict_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red_mask = _hsv_red_mask(hsv, min_sat=red_min_sat, min_val=red_min_val).astype(bool)
    if restrict_mask is not None:
        red_mask &= restrict_mask

    red_pixels_before = int(red_mask.sum())
    if red_pixels_before == 0:
        return img_bgr, 0, 0

    h = hsv[:, :, 0].copy()
    h[red_mask] = int(target_hue) % 180
    hsv[:, :, 0] = h

    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    red_mask_after = _hsv_red_mask(hsv_after, min_sat=red_min_sat, min_val=red_min_val).astype(bool)
    if restrict_mask is not None:
        red_mask_after &= restrict_mask
    red_pixels_after = int(red_mask_after.sum())
    return out, red_pixels_before, red_pixels_after


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Fix inconsistent red card backs by recoloring them to match the blue set. "
        "Optionally cleans the crop margins to hide 'half-out' cards that protrude into the margins."
    )
    ap.add_argument("--input", type=Path, default=Path("cropped_hq4"), help="Input folder (default: cropped_hq4)")
    ap.add_argument("--output", type=Path, default=Path("cropped_hq4_fixed"), help="Output folder")
    ap.add_argument(
        "--ranges",
        type=str,
        default="",
        help='Comma-separated inclusive ranges of indices to process (e.g. "14-16,42-54,276-364"). '
        "If omitted, processes all frames (red pixels only).",
    )
    ap.add_argument("--one-based", action="store_true", help="Interpret provided --ranges as 1-based and convert to 0-based indices.")
    ap.add_argument("--target-blue-hue", type=int, default=None, help="Override target blue hue (OpenCV HSV: 0-179).")
    ap.add_argument(
        "--blue-ref",
        type=Path,
        default=None,
        help="Optional reference image used to infer blue hue (if --target-blue-hue not provided).",
    )
    ap.add_argument("--red-min-sat", type=int, default=80, help="Red mask minimum saturation (default: 80)")
    ap.add_argument("--red-min-val", type=int, default=40, help="Red mask minimum value/brightness (default: 40)")
    ap.add_argument(
        "--clean-margins",
        action="store_true",
        help="Replace crop margin pixels with the median background (requires manifest.json in input folder).",
    )
    ap.add_argument("--quiet", action="store_true", help="Less console output")
    args = ap.parse_args(argv)

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_images(input_dir)
    selected = _parse_ranges(args.ranges)
    if args.one_based:
        selected = {i - 1 for i in selected}
    selected = {i for i in selected if i >= 0}

    images: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read {p}")
        images.append(img)

    h, w = images[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in images):
        raise SystemExit("All input images must have the same dimensions.")

    if args.target_blue_hue is not None:
        target_hue = int(args.target_blue_hue) % 180
    elif args.blue_ref is not None:
        ref_img = cv2.imread(str(args.blue_ref), cv2.IMREAD_COLOR)
        if ref_img is None:
            raise SystemExit(f"Failed to read --blue-ref {args.blue_ref}")
        hue = _median_blue_hue(ref_img)
        target_hue = hue if hue is not None else 115
    else:
        target_hue = _pick_reference_blue_hue(paths)

    margins_px = _load_manifest_margins(input_dir)
    margin_mask = None
    median_margin = None
    if args.clean_margins:
        if margins_px is None:
            raise SystemExit("--clean-margins requested but no readable manifest.json was found in the input folder.")
        margin_mask = _margin_mask((h, w), margins_px)
        median_margin = _median_background_for_mask(images, margin_mask)

    # Restrict recolor to non-margin area when we have a margin mask
    recolor_restrict = None
    if margin_mask is not None:
        recolor_restrict = ~margin_mask

    if not args.quiet:
        print(f"Input: {input_dir} ({len(paths)} images, {w}x{h})")
        print(f"Output: {output_dir}")
        print(f"Target blue hue: {target_hue}")
        if args.clean_margins:
            left_px, right_px, top_px, bottom_px = margins_px  # type: ignore[misc]
            print(f"Clean margins: left={left_px}px right={right_px}px top={max(0, top_px)}px bottom={bottom_px}px")
        if selected:
            print(f"Selected indices: {len(selected)}")
        else:
            print("Selected indices: (all)")

    report: List[FixReportItem] = []

    for idx, (p, img) in enumerate(zip(paths, images)):
        out = img.copy()
        if median_margin is not None and margin_mask is not None:
            out[margin_mask] = median_margin[margin_mask]

        is_selected = (not selected) or (idx in selected)
        red_before = 0
        red_after = 0
        changed = False
        if is_selected:
            fixed, red_before, red_after = _apply_red_to_blue(
                out,
                target_hue=target_hue,
                red_min_sat=int(args.red_min_sat),
                red_min_val=int(args.red_min_val),
                restrict_mask=recolor_restrict,
            )
            changed = red_before > 0 and (red_after < red_before)
            out = fixed

        out_path = output_dir / p.name
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            raise SystemExit(f"Failed to write {out_path}")

        report.append(
            FixReportItem(
                index=idx,
                filename=p.name,
                selected=bool(is_selected),
                red_pixels_before=int(red_before),
                red_pixels_after=int(red_after),
                changed=bool(changed),
            )
        )

    out_manifest_src = input_dir / "manifest.json"
    if out_manifest_src.exists():
        try:
            (output_dir / "manifest.json").write_text(out_manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    fix_report = {
        "input": str(input_dir),
        "output": str(output_dir),
        "target_blue_hue": int(target_hue),
        "ranges": args.ranges,
        "one_based": bool(args.one_based),
        "clean_margins": bool(args.clean_margins),
        "red_mask": {"min_sat": int(args.red_min_sat), "min_val": int(args.red_min_val)},
        "images": [asdict(r) for r in report],
    }
    (output_dir / "fix_report.json").write_text(json.dumps(fix_report, indent=2), encoding="utf-8")

    if not args.quiet:
        changed_cnt = sum(1 for r in report if r.changed)
        print(f"Done. Wrote {len(report)} images. Recolored frames: {changed_cnt}.")
        print(f"Report: {output_dir / 'fix_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

