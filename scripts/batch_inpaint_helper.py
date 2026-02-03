#!/usr/bin/env python3
"""
batch_inpaint_helper.py – Helper for ComfyUI inpainting workflow

Manages masks and prepares frame lists for manual inpainting in ComfyUI.
Since you're doing manual inpainting, this script helps organize which
frames need fixing and tracks your progress.

Usage:
  # List all frames that need fixing
  python scripts/batch_inpaint_helper.py --list-todo

  # Mark a frame as done
  python scripts/batch_inpaint_helper.py --mark-done 42

  # Generate a template mask for a specific frame
  python scripts/batch_inpaint_helper.py --create-mask-template 42

  # Show next N frames to work on
  python scripts/batch_inpaint_helper.py --next 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

import cv2
import numpy as np

# Problem frame ranges from COMFYUI_INPAINT_REQUEST.md
PROBLEM_RANGES = [
    (14, 16),
    (42, 54),
    (56, 60),
    (68, 78),
    (131, 135),
    (157, 182),
    (184, 188),
    (193, 208),
    (235, 270),
    (276, 364),
]

# Additional specific frames
PROBLEM_SINGLES = [244, 31]


def get_problem_indices() -> Set[int]:
    """Return set of all frame indices that need fixing."""
    indices = set()
    for start, end in PROBLEM_RANGES:
        indices.update(range(start, end + 1))
    indices.update(PROBLEM_SINGLES)
    return indices


def load_progress(progress_file: Path) -> Set[int]:
    """Load completed frame indices from progress file."""
    if not progress_file.exists():
        return set()
    data = json.loads(progress_file.read_text())
    return set(data.get("completed", []))


def save_progress(progress_file: Path, completed: Set[int]):
    """Save progress to JSON file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    data = {"completed": sorted(completed)}
    progress_file.write_text(json.dumps(data, indent=2))


def create_mask_template(
    frame_idx: int,
    input_dir: Path,
    mask_dir: Path,
    card_region_only: bool = True,
) -> Path:
    """Create a blank mask template for manual editing.

    If card_region_only=True, fills the approximate card-back region
    with white (top half of the tray) to give you a starting point.
    """
    img_path = input_dir / f"{frame_idx:03d}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Frame not found: {img_path}")

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Create blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if card_region_only:
        # Fill top 60% of image (where card backs typically are)
        # You'll refine this manually in an image editor
        mask[: int(h * 0.6), :] = 255

    mask_path = mask_dir / f"{frame_idx:03d}_mask.png"
    mask_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    return mask_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("cropped_hq4"),
        help="Input image directory (default: cropped_hq4)",
    )
    ap.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("masks"),
        help="Mask directory (default: masks)",
    )
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=Path("inpaint_progress.json"),
        help="Progress tracking file (default: inpaint_progress.json)",
    )
    ap.add_argument(
        "--list-todo",
        action="store_true",
        help="List all frames that still need inpainting",
    )
    ap.add_argument(
        "--mark-done",
        type=int,
        metavar="IDX",
        help="Mark frame IDX as completed",
    )
    ap.add_argument(
        "--create-mask-template",
        type=int,
        metavar="IDX",
        help="Create a template mask for frame IDX",
    )
    ap.add_argument(
        "--next",
        type=int,
        metavar="N",
        help="Show next N frames to work on",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Show progress statistics",
    )
    args = ap.parse_args(argv)

    problem_frames = get_problem_indices()
    completed = load_progress(args.progress_file)
    remaining = problem_frames - completed

    if args.stats:
        total = len(problem_frames)
        done = len(completed)
        todo = len(remaining)
        pct = (done / total * 100) if total > 0 else 0
        print(f"Progress: {done}/{total} frames completed ({pct:.1f}%)")
        print(f"Remaining: {todo} frames")
        return 0

    if args.list_todo:
        if not remaining:
            print("All frames completed!")
        else:
            print(f"Frames still needing inpainting ({len(remaining)} total):")
            for idx in sorted(remaining):
                mask_exists = (args.mask_dir / f"{idx:03d}_mask.png").exists()
                status = "[READY]" if mask_exists else "[NO MASK]"
                print(f"  {idx:03d}.png - {status}")
        return 0

    if args.mark_done is not None:
        idx = args.mark_done
        if idx not in problem_frames:
            print(f"Warning: {idx} is not in the problem frame list")
        completed.add(idx)
        save_progress(args.progress_file, completed)
        print(f"Marked {idx:03d}.png as completed")
        todo = len(problem_frames - completed)
        print(f"Remaining: {todo} frames")
        return 0

    if args.create_mask_template is not None:
        idx = args.create_mask_template
        mask_path = create_mask_template(idx, args.input, args.mask_dir)
        print(f"Created mask template: {mask_path}")
        print(
            f"Edit this mask in an image editor (white=inpaint, black=keep original)"
        )
        return 0

    if args.next is not None:
        n = args.next
        if not remaining:
            print("All frames completed!")
        else:
            todo_list = sorted(remaining)[:n]
            print(f"Next {len(todo_list)} frames to work on:")
            for idx in todo_list:
                mask_path = args.mask_dir / f"{idx:03d}_mask.png"
                mask_exists = mask_path.exists()
                status = "ready" if mask_exists else "needs mask"
                print(f"  {idx:03d}.png - {status}")
                if not mask_exists:
                    print(f"    > Run: python scripts/batch_inpaint_helper.py --create-mask-template {idx}")
        return 0

    # Default: show stats
    total = len(problem_frames)
    done = len(completed)
    todo = len(remaining)
    pct = (done / total * 100) if total > 0 else 0
    print(f"Inpaint Progress: {done}/{total} ({pct:.1f}%)")
    print(f"\nUse --list-todo to see all remaining frames")
    print(f"Use --next N to see next N frames to work on")
    print(f"Use --create-mask-template IDX to create a mask")
    print(f"Use --mark-done IDX after you've fixed a frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
