# Discard Tray Photo Cleanup – Request + Plan (ComfyUI Inpaint)

## Context

- Dataset: 365 sequential frames of a discard tray (empty → full), currently cropped via `scripts/detect_and_crop.py`.
- Goal: ingest these frames into an app and have them look **pixel-perfect** (consistent alignment/crop, consistent card-back appearance, no distracting anomalies).

## The Problems To Fix

### Wrong card backs / wrong colors

- Roughly ~50 frames have a different card-back color/type (red backs, etc.) and should be standardized to the same **blue** back used in the main set.
- My current HSV-based recolor attempt (red → blue hue shift) **did not do enough**:
  - some backs are still not the right color
  - the *pattern/type* of the back is still wrong (recoloring alone can’t create the correct texture)

Problem indices (0-based, matching `cropped_hq4/000.png` … `364.png`):

- `14-16`
- `42-54`
- `56-60`
- `68-78`
- `131-135`
- `157-182`
- `184-188`
- `193-208`
- `235-270`
- `276-364` (this last set is mainly red backs; I still want the same blue back)

Additional specific callouts:

- `244` is wrong back type (and also needs centering review).

### “Half-out of the tray” frames

- A few frames have a card that is partially out of the tray.
- I want to correct these so the sequence looks clean and consistent.

### Alignment/centering anomalies

- `31` is slightly off-center (needs correction).

## What I Want (Written In My Own Words)

> I want to correct the inconsistent-backed cards and the half-out cards nearly perfect.  
> The color switch didn’t do enough and some backs are just not the right color.  
> I’d rather use AI masking/inpainting so I can replace only the incorrect card back part of the picture with a correct blue card back.  
> I want to use ComfyUI and have a streamlined image-to-image inpaint workflow that’s quick and lets me fix only the frames I want.

## Proposed Best Approach (High-Level)

### 1) Keep the deterministic pipeline for geometry

- Continue using `detect_and_crop.py` for consistent crop/anchor across all frames.
- (Optional) Apply alignment on the cropped set if needed for the “off-center” frames.

### 2) Use AI inpainting for the “content” fixes (card backs + half-out cards)

Recoloring is not enough because card backs are not just “red vs blue”; the *pattern* matters.

Instead, use an inpaint workflow where:

- Input = the problematic frame
- Mask = only the wrong-back region (or the half-out card region)
- Conditioning = “keep everything else identical” + “replace masked area with the correct blue back”
- Output = fixed frame, saved with the same filename so the sequence stays intact

## ComfyUI Workflow Outline (What I Want Built Next)

I have a local ComfyUI setup already. I want a workflow that supports:

- **Single-frame fix**: load one image + one mask, inpaint, save.
- **Batch fix**: iterate through a list of indices (e.g. the ranges above), apply corresponding masks, write outputs to a folder.
- **Reference-driven back type**: I want to provide a reference image of the *exact* blue card back pattern and have the model match it.

Conceptual components (node-level, not final JSON yet):

- Load Image (frame)
- Load Mask (hand-drawn / precomputed mask)
- Inpaint model (image-to-image inpaint)
- Optional structure guidance to preserve edges/details (e.g., edge/line guidance so the tray/cards don’t “melt”)
- Optional reference conditioning (use a reference blue-back image so the texture/type matches)
- Save Image (same naming convention)

## Acceptance Criteria

- The fixed frames are indistinguishable from the main sequence:
  - correct blue back type/pattern (not just hue)
  - no obvious seams at mask boundaries
  - tray edges + lighting preserved
  - no new artifacts (extra edges, smears, warped geometry)
- The sequence remains consistent when you flip quickly through frames in order.

## Next Deliverables (Not Implemented Yet)

- A ComfyUI workflow JSON (image-to-image inpaint) tailored to this dataset.
- A small helper script to:
  - generate/organize per-frame masks
  - run only selected indices
  - keep output naming stable (`000.png` … `364.png`)

