# Discard Tray Estimation – Photo Standardization

365 sequential photos of a clear acrylic discard tray (shoe) sitting on a table.
Image 001 is the empty tray; image 365 shows a full shoe.
Each successive frame adds roughly one card to the tray, so the sequence
captures the full range from empty to full.

## Scripts

### 1. `detect_and_crop.py` – CV-based shoe detection & cropping

Uses computer vision (edge detection + foreground segmentation) to locate
the shoe in every frame, then crops each image to a fixed output size with
configurable margins. Alignment is anchored on the shoe's **base** so the
tray bottom stays in the same place across all frames and only the rising
card stack changes.

```powershell
# Default: 5% shoe-width margin left/right, 10% top/bottom
python scripts/detect_and_crop.py --input 1024x --output cropped_hq

# Custom margins (all four sides independently)
python scripts/detect_and_crop.py --input 1024x --output cropped_hq --margin-left 0.05 --margin-right 0.05 --margin-top 0.02 --margin-bottom 0.10

# Finalize (current best settings)
python scripts/detect_and_crop.py --input 1024x --output cropped_hq4 --margin-left 0.08 --margin-right 0.05 --margin-top -.12 --margin-bottom 0.03

# Legacy symmetric margins (sets left+right and top+bottom)
python scripts/detect_and_crop.py --input 1024x --output cropped_hq --h-margin 0.08 --v-margin 0.15

# Save debug images showing detected bounding boxes
python scripts/detect_and_crop.py --input 1024x --output cropped_hq --debug-dir debug_vis

# Centre-anchored instead of base-anchored
python scripts/detect_and_crop.py --input 1024x --output cropped_hq --anchor center
```

Outputs:

- High-quality lossless PNGs in `cropped_hq/`
- `cropped_hq/manifest.json` with per-image anchor positions, crop origins, and reference dimensions

### 2. `downgrade.py` – Batch resize & compress

Takes the high-quality crops (or any folder of images) and produces
smaller or more compressed copies.

```powershell
# Resize to 256 px wide, JPEG quality 80
python scripts/downgrade.py --input cropped_hq --output cropped_sm --width 256 --format jpg --jpg-quality 80

# Half-size WebP
python scripts/downgrade.py --input cropped_hq --output cropped_webp --scale 0.5 --format webp --webp-quality 75

# Keep size, just convert to JPEG
python scripts/downgrade.py --input cropped_hq --output cropped_jpg --format jpg --jpg-quality 70
```

### 3. `align_discard_tray_photos.py` – ECC whole-image alignment (legacy)

Aligns every photo to a template using OpenCV's ECC algorithm to correct
small camera shifts/rotations, then exports resized copies with consistent
naming. This operates on the full image rather than cropping to the shoe.

```powershell
python scripts/align_discard_tray_photos.py --input 1024x --output out_aligned --auto-crop --output-size 512x384
```

## Naming

By default `detect_and_crop.py` names files by **index** (000–364).
Use `--name-pattern` to customise:

| Field     | Description                               |
|-----------|-------------------------------------------|
| `index`   | 0-based frame index                       |
| `src_num` | Number parsed from the source filename    |

Example (keep source numbering):

```powershell
python scripts/detect_and_crop.py --input 1024x --output cropped_hq --name-pattern "{src_num:03d}.png"
```

## 4. Inpainting Workflow (ComfyUI) – Fix Card Backs

For frames with wrong card backs or anomalies, use the ComfyUI inpainting workflow:

See **[INPAINT_WORKFLOW_GUIDE.md](INPAINT_WORKFLOW_GUIDE.md)** for detailed setup and usage.

**Quick start:**
```powershell
# Show which frames need fixing
python scripts/batch_inpaint_helper.py --next 10

# Create a mask template for frame 42
python scripts/batch_inpaint_helper.py --create-mask-template 42

# Edit the mask in Paint.NET/GIMP (white=inpaint, black=keep)
# Then run the ComfyUI workflow (see guide)

# Mark as done after fixing
python scripts/batch_inpaint_helper.py --mark-done 42
```

## Requirements

- Python 3.8+
- `opencv-python` >= 4.5
- `numpy` >= 1.19
- (Optional) ComfyUI with SDXL-inpaint or SD1.5-inpaint model for card back fixing


FINAL COMMANDS

 python scripts/detect_and_crop.py --input 1024x --output cropped_hq4 --margin-left 0.08 --margin-right 0.05 --margin-top -.12 --margin-bottom 0.03
python scripts/downgrade.py --input cropped_hq4 --output downgraded_webp5 --scale 1 --format webp --webp-quality 100