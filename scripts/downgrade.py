#!/usr/bin/env python3
"""
downgrade.py – Batch-reduce image quality: resize and/or compress.

Typical workflow:
  1. Run detect_and_crop.py  -> cropped_hq/  (lossless PNGs)
  2. Run this script         -> cropped_sm/  (smaller JPGs/WebPs)

Usage examples:
  # Resize to 256 px wide, JPEG quality 80:
  python scripts/downgrade.py --input cropped_hq --output cropped_sm --width 256 --format jpg --jpg-quality 80

  # Half-size WebP at quality 75:
  python scripts/downgrade.py --input cropped_hq --output cropped_webp --scale 0.5 --format webp --webp-quality 75

  # Keep original size, just convert to JPEG quality 70:
  python scripts/downgrade.py --input cropped_hq --output cropped_jpg --format jpg --jpg-quality 70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch-downgrade images: resize and/or compress.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, required=True, help="Input folder of images")
    ap.add_argument("--output", type=Path, required=True, help="Output folder")
    ap.add_argument(
        "--width", type=int, default=None,
        help="Target width in pixels (maintains aspect ratio). Overrides --scale.",
    )
    ap.add_argument(
        "--scale", type=float, default=None,
        help="Scale factor, e.g. 0.5 for half size. Ignored when --width is set.",
    )
    ap.add_argument(
        "--format", choices=["png", "jpg", "webp"], default="jpg",
        help="Output format (default: jpg)",
    )
    ap.add_argument("--jpg-quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    ap.add_argument("--webp-quality", type=int, default=75, help="WebP quality 1-100 (default: 75)")
    ap.add_argument("--png-compression", type=int, default=6, help="PNG compression 0-9 (default: 6)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    if not paths:
        raise SystemExit(f"No images found in {args.input}")

    ext_map = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}
    out_ext = ext_map[args.format]

    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {p}, skipping.", file=sys.stderr)
            continue

        # Resize
        if args.width is not None:
            h, w = img.shape[:2]
            s = args.width / w
            new_h = max(1, int(round(h * s)))
            img = cv2.resize(img, (args.width, new_h), interpolation=cv2.INTER_AREA)
        elif args.scale is not None:
            h, w = img.shape[:2]
            new_w = max(1, int(round(w * args.scale)))
            new_h = max(1, int(round(h * args.scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_name = p.stem + out_ext
        out_path = args.output / out_name

        if args.format == "jpg":
            q = max(1, min(100, args.jpg_quality))
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        elif args.format == "webp":
            q = max(1, min(100, args.webp_quality))
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_WEBP_QUALITY, q])
        else:
            c = max(0, min(9, args.png_compression))
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, c])

        if not args.quiet and (i % 50 == 0 or i == len(paths) - 1):
            print(
                f"[{i + 1:>3}/{len(paths)}] {p.name} -> {out_name}  "
                f"({img.shape[1]}x{img.shape[0]})"
            )

    if not args.quiet:
        print(f"Done. {len(paths)} images -> {args.output}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
