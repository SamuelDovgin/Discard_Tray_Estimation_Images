#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    source: str
    output: str
    index: int
    cards: int
    ecc: Optional[float]
    warp: list
    failed: bool


def _parse_size(value: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if not m:
        raise argparse.ArgumentTypeError("Expected WxH, e.g. 1024x768")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Width/height must be > 0")
    return w, h


def _parse_roi(value: str) -> Tuple[int, int, int, int]:
    parts = re.split(r"[,\s]+", value.strip())
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected x,y,w,h (4 integers)")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width/height must be > 0")
    return x, y, w, h


def _extract_number(path: Path) -> Optional[int]:
    # Use the *last* run of digits in the stem (works for e.g. foo_001.png).
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(m.group(1)) if m else None


def _iter_images(input_dir: Path, exts: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for ext in exts:
        paths.extend(input_dir.glob(f"*{ext}"))
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")

    def sort_key(p: Path):
        n = _extract_number(p)
        return (0 if n is not None else 1, n if n is not None else 0, p.name)

    return sorted(paths, key=sort_key)


def _preprocess_for_ecc(img_bgr: np.ndarray, mode: str, blur_sigma: float) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=blur_sigma)

    if mode == "gray":
        out = gray.astype(np.float32) / 255.0
        return out

    if mode == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
        return mag.astype(np.float32)

    if mode == "canny":
        edges = cv2.Canny(gray, 80, 160)
        return (edges.astype(np.float32) / 255.0).astype(np.float32)

    raise ValueError(f"Unknown preprocess mode: {mode}")


def _motion_type(name: str) -> int:
    name = name.lower().strip()
    if name == "translation":
        return cv2.MOTION_TRANSLATION
    if name == "euclidean":
        return cv2.MOTION_EUCLIDEAN
    if name == "affine":
        return cv2.MOTION_AFFINE
    if name == "homography":
        return cv2.MOTION_HOMOGRAPHY
    raise ValueError(f"Unknown motion model: {name}")


def _identity_warp(motion: int) -> np.ndarray:
    if motion == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, 3, dtype=np.float32)
    return np.eye(2, 3, dtype=np.float32)


def _roi_mask(shape_hw: Tuple[int, int], roi: Optional[Tuple[int, int, int, int]], border_frac: float) -> Optional[np.ndarray]:
    h, w = shape_hw
    if roi is None and border_frac <= 0:
        return None

    if roi is None:
        x0, y0, rw, rh = 0, 0, w, h
    else:
        x0, y0, rw, rh = roi
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        rw = max(1, min(w - x0, rw))
        rh = max(1, min(h - y0, rh))

    mask = np.zeros((h, w), dtype=np.uint8)
    if border_frac <= 0:
        mask[y0 : y0 + rh, x0 : x0 + rw] = 255
        return mask

    border_frac = float(border_frac)
    if not (0 < border_frac < 0.5):
        raise ValueError("--mask-border must be between 0 and 0.5 (exclusive)")

    bx = max(1, int(round(rw * border_frac)))
    by = max(1, int(round(rh * border_frac)))

    # Outer ROI on, inner ROI off => a "frame" around the ROI.
    mask[y0 : y0 + rh, x0 : x0 + rw] = 255
    ix0 = x0 + bx
    iy0 = y0 + by
    ix1 = x0 + rw - bx
    iy1 = y0 + rh - by
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = 0
    return mask


def _resize_gray_and_mask(
    gray: np.ndarray, mask: Optional[np.ndarray], scale: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if scale == 1.0:
        return gray, mask
    if not (0 < scale <= 1.0):
        raise ValueError("--coarse-scale must be in (0, 1]")
    h, w = gray.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    gray_s = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_s = None
    if mask is not None:
        mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return gray_s, mask_s


def _scale_warp_to_full(warp: np.ndarray, motion: int, scale: float) -> np.ndarray:
    if scale == 1.0:
        return warp
    if motion == cv2.MOTION_HOMOGRAPHY:
        s = float(scale)
        s_mat = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)
        s_inv = np.array([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]], dtype=np.float32)
        return (s_inv @ warp @ s_mat).astype(np.float32)

    warp_full = warp.astype(np.float32).copy()
    warp_full[0, 2] /= float(scale)
    warp_full[1, 2] /= float(scale)
    return warp_full


def _warp_image(img_bgr: np.ndarray, warp: np.ndarray, motion: int, out_size_wh: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = out_size_wh
    if motion == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(
            img_bgr,
            warp,
            dsize=(out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return cv2.warpAffine(
        img_bgr,
        warp,
        dsize=(out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _warp_corners(w: int, h: int, warp: np.ndarray, motion: int) -> np.ndarray:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    if motion == cv2.MOTION_HOMOGRAPHY:
        corners_h = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
        mapped = (warp @ corners_h.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        return mapped

    a = warp.astype(np.float32)
    mapped = corners @ a[:, :2].T + a[:, 2]
    return mapped


def _intersection_crop(
    base_w: int, base_h: int, warps: list[np.ndarray], motion: int
) -> Optional[Tuple[int, int, int, int]]:
    left = 0.0
    top = 0.0
    right = float(base_w)
    bottom = float(base_h)
    for w in warps:
        pts = _warp_corners(base_w, base_h, w, motion)
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        left = max(left, x_min)
        top = max(top, y_min)
        right = min(right, x_max)
        bottom = min(bottom, y_max)

    x0 = int(math.ceil(max(0.0, left)))
    y0 = int(math.ceil(max(0.0, top)))
    x1 = int(math.floor(min(float(base_w), right)))
    y1 = int(math.floor(min(float(base_h), bottom)))

    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return None
    return x0, y0, x1, y1


def _format_output_name(pattern: str, *, cards: int, index: int, src_num: Optional[int], ext: str) -> str:
    try:
        return pattern.format(cards=cards, index=index, src_num=src_num, ext=ext)
    except Exception as e:
        raise SystemExit(f"Invalid --name-pattern {pattern!r}: {e}") from e


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Align sequential discard-tray photos to a common reference and export resized/renamed copies."
    )
    parser.add_argument("--input", type=Path, default=Path("1024x"), help="Input folder of images (default: 1024x)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aligned_out"),
        help="Output folder (default: aligned_out)",
    )
    parser.add_argument(
        "--output-size",
        type=_parse_size,
        default=None,
        help="Force output size WxH (e.g. 1024x768). If omitted, keeps aligned size (or resizes by width/height).",
    )
    parser.add_argument("--output-width", type=int, default=1024, help="Output width (default: 1024)")
    parser.add_argument("--output-height", type=int, default=None, help="Output height (default: keep aspect ratio)")
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument("--jpg-quality", type=int, default=90, help="JPG quality (default: 90)")
    parser.add_argument(
        "--name-pattern",
        default="{cards:03d}{ext}",
        help="Python format pattern for output filenames (default: {cards:03d}{ext}). "
        "Available fields: cards, index, src_num, ext",
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=0,
        help="Index in the sorted list to use as the alignment template (default: 0)",
    )
    parser.add_argument(
        "--motion",
        choices=["translation", "euclidean", "affine", "homography"],
        default="euclidean",
        help="Warp model used for alignment (default: euclidean)",
    )
    parser.add_argument(
        "--preprocess",
        choices=["gray", "sobel", "canny"],
        default="gray",
        help="Preprocess used for alignment (default: gray)",
    )
    parser.add_argument("--blur-sigma", type=float, default=1.0, help="Gaussian blur sigma (default: 1.0)")
    parser.add_argument(
        "--roi",
        type=_parse_roi,
        default=None,
        help="Optional ROI to emphasize during alignment: x,y,w,h in template coordinates",
    )
    parser.add_argument(
        "--mask-border",
        type=float,
        default=0.0,
        help="If set (0-0.5), only use a border 'frame' of the ROI for alignment (helps ignore changing card pile).",
    )
    parser.add_argument(
        "--coarse-scale",
        type=float,
        default=0.5,
        help="Downscale factor for coarse alignment (default: 0.5). Use 1 to disable.",
    )
    parser.add_argument("--coarse-iterations", type=int, default=60, help="ECC iterations for coarse pass (default: 60)")
    parser.add_argument("--fine-iterations", type=int, default=80, help="ECC iterations for fine pass (default: 80)")
    parser.add_argument("--ecc-eps", type=float, default=1e-6, help="ECC epsilon stop threshold (default: 1e-6)")
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Crop all aligned images to the shared valid intersection region (reduces edge replication artifacts).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (debug)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on any alignment failure")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    args = parser.parse_args(argv)

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [".png", ".jpg", ".jpeg", ".webp"]
    paths = _iter_images(input_dir, exts=exts)
    if args.limit is not None:
        paths = paths[: max(0, int(args.limit))]
    if not paths:
        raise SystemExit("No images selected.")

    if not (0 <= args.template_index < len(paths)):
        raise SystemExit(f"--template-index must be in [0, {len(paths)-1}]")

    template_path = paths[args.template_index]
    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise SystemExit(f"Failed to read template image: {template_path}")

    base_h, base_w = template_bgr.shape[:2]
    motion = _motion_type(args.motion)

    mask = _roi_mask((base_h, base_w), roi=args.roi, border_frac=float(args.mask_border))
    template_proc = _preprocess_for_ecc(template_bgr, args.preprocess, args.blur_sigma)

    if not args.quiet:
        print(f"Found {len(paths)} images")
        print(f"Template: {template_path.name} ({base_w}x{base_h})")
        if args.roi is not None:
            print(f"ROI: {args.roi}  mask-border: {args.mask_border}")
        print(f"Motion: {args.motion}  preprocess: {args.preprocess}  coarse-scale: {args.coarse_scale}")

    warps: list[np.ndarray] = []
    ecc_scores: list[Optional[float]] = []
    failed: list[bool] = []

    for idx, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read image: {p}")
        if img.shape[:2] != (base_h, base_w):
            raise SystemExit(f"All images must have the same size as the template. {p} is {img.shape[1]}x{img.shape[0]}.")

        if idx == args.template_index:
            warps.append(_identity_warp(motion))
            ecc_scores.append(1.0)
            failed.append(False)
            continue

        input_proc = _preprocess_for_ecc(img, args.preprocess, args.blur_sigma)
        warp = _identity_warp(motion)
        score: Optional[float] = None

        try:
            if args.coarse_scale != 1.0:
                tpl_s, mask_s = _resize_gray_and_mask(template_proc, mask, float(args.coarse_scale))
                inp_s, _ = _resize_gray_and_mask(input_proc, None, float(args.coarse_scale))
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(args.coarse_iterations),
                    float(args.ecc_eps),
                )
                score, warp_s = cv2.findTransformECC(
                    tpl_s, inp_s, warp, motion, criteria, inputMask=mask_s, gaussFiltSize=5
                )
                warp = _scale_warp_to_full(warp_s, motion, float(args.coarse_scale))

            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(args.fine_iterations),
                float(args.ecc_eps),
            )
            score, warp = cv2.findTransformECC(
                template_proc, input_proc, warp, motion, criteria, inputMask=mask, gaussFiltSize=5
            )
            warps.append(warp.astype(np.float32))
            ecc_scores.append(float(score) if score is not None else None)
            failed.append(False)
            if not args.quiet and (idx % 25 == 0 or idx == len(paths) - 1):
                print(f"[{idx+1:>3}/{len(paths)}] {p.name}  ecc={score:.5f}")
        except cv2.error as e:
            if args.fail_fast:
                raise
            warps.append(_identity_warp(motion))
            ecc_scores.append(None)
            failed.append(True)
            if not args.quiet:
                print(f"[{idx+1:>3}/{len(paths)}] {p.name}  ECC FAILED: {e}")

    crop: Optional[Tuple[int, int, int, int]] = None
    if args.auto_crop:
        crop = _intersection_crop(base_w, base_h, warps, motion)
        if crop is None:
            if not args.quiet:
                print("Auto-crop: could not compute a valid intersection crop; skipping crop.")
        else:
            if not args.quiet:
                x0, y0, x1, y1 = crop
                print(f"Auto-crop: x0={x0} y0={y0} x1={x1} y1={y1} (size {x1-x0}x{y1-y0})")

    out_ext = ".png" if args.format == "png" else ".jpg"
    if args.output_size is not None:
        out_w, out_h = args.output_size
    else:
        out_w = int(args.output_width) if args.output_width is not None else None
        out_h = int(args.output_height) if args.output_height is not None else None

    results: list[AlignmentResult] = []
    for idx, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        warp = warps[idx]
        aligned = _warp_image(img, warp, motion, out_size_wh=(base_w, base_h))

        if crop is not None:
            x0, y0, x1, y1 = crop
            aligned = aligned[y0:y1, x0:x1]

        if args.output_size is not None:
            aligned = cv2.resize(aligned, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            if out_w is not None and out_h is not None:
                aligned = cv2.resize(aligned, (out_w, out_h), interpolation=cv2.INTER_AREA)
            elif out_w is not None:
                h, w = aligned.shape[:2]
                scale = out_w / float(w)
                new_h = max(1, int(round(h * scale)))
                aligned = cv2.resize(aligned, (out_w, new_h), interpolation=cv2.INTER_AREA)
            elif out_h is not None:
                h, w = aligned.shape[:2]
                scale = out_h / float(h)
                new_w = max(1, int(round(w * scale)))
                aligned = cv2.resize(aligned, (new_w, out_h), interpolation=cv2.INTER_AREA)

        src_num = _extract_number(p)
        cards = idx  # assumes 1st image is empty tray => card-count index
        out_name = _format_output_name(args.name_pattern, cards=cards, index=idx, src_num=src_num, ext=out_ext)
        out_path = output_dir / out_name

        if args.format == "png":
            ok = cv2.imwrite(str(out_path), aligned)
        else:
            q = int(args.jpg_quality)
            q = max(1, min(100, q))
            ok = cv2.imwrite(str(out_path), aligned, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise SystemExit(f"Failed to write: {out_path}")

        results.append(
            AlignmentResult(
                source=str(p.name),
                output=str(out_name),
                index=idx,
                cards=cards,
                ecc=ecc_scores[idx],
                warp=warp.astype(float).tolist(),
                failed=bool(failed[idx]),
            )
        )

    manifest = {
        "input": str(input_dir),
        "output": str(output_dir),
        "count": len(results),
        "template": str(template_path.name),
        "template_index": int(args.template_index),
        "base_size": {"width": int(base_w), "height": int(base_h)},
        "output_ext": out_ext,
        "output_size": {"width": int(aligned.shape[1]), "height": int(aligned.shape[0])},
        "auto_crop": bool(args.auto_crop),
        "crop": None if crop is None else {"x0": crop[0], "y0": crop[1], "x1": crop[2], "y1": crop[3]},
        "motion": args.motion,
        "preprocess": args.preprocess,
        "blur_sigma": float(args.blur_sigma),
        "coarse_scale": float(args.coarse_scale),
        "coarse_iterations": int(args.coarse_iterations),
        "fine_iterations": int(args.fine_iterations),
        "ecc_eps": float(args.ecc_eps),
        "roi": None if args.roi is None else {"x": args.roi[0], "y": args.roi[1], "w": args.roi[2], "h": args.roi[3]},
        "mask_border": float(args.mask_border),
        "images": [asdict(r) for r in results],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.quiet:
        failures = sum(1 for r in results if r.failed)
        print(f"Done. Wrote {len(results)} images to {output_dir} (failures: {failures})")
        print(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

