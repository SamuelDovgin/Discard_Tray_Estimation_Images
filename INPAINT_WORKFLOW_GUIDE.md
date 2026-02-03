# ComfyUI Inpaint Workflow Guide – Card Back Fixing

## Setup (One-Time)

### 1. Install ComfyUI Dependencies

If you haven't already, ensure you have the SDXL inpainting model:

```bash
# In your ComfyUI/models/checkpoints/ folder, download:
# sd_xl_base_1.0_inpainting_0.1.safetensors
# From: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
```

**For LOW VRAM systems**, consider using SD 1.5 inpaint instead:
- Model: `sd-v1-5-inpainting.ckpt` (smaller, ~4GB VRAM vs SDXL's 8GB+)
- Change in workflow: Edit node #3 to use this model instead

### 2. Load the Workflow

1. Open ComfyUI in your browser (usually `http://127.0.0.1:8188`)
2. Click "Load" button
3. Navigate to: `C:\Users\samue\Developer\Discard_Tray_Estimation\comfyui_workflows\inpaint_card_backs.json`
4. The workflow should load with all nodes connected

## Workflow Overview

**Nodes (left to right):**

1. **LoadImage** (top left) - Load the problematic frame (e.g., `042.png`)
2. **LoadImage (Mask)** (below it) - Load the mask for that frame (e.g., `042_mask.png`)
3. **CheckpointLoader** - SDXL inpainting model (or SD1.5 for low VRAM)
4. **CLIPTextEncode (Positive)** - Describes what you want: "blue bicycle card back"
5. **CLIPTextEncode (Negative)** - What to avoid: "red card, blurry"
6. **VAEEncode** - Encodes the image to latent space
7. **SetLatentNoiseMask** - Applies your mask to the latents
8. **KSampler** - The actual inpainting (denoise=0.85)
9. **VAEDecode** - Decodes back to pixel space
10. **SaveImage** - Saves the fixed image

## Manual Inpainting Workflow

### Step 1: Create a Mask

```bash
# From project root
cd C:\Users\samue\Developer\Discard_Tray_Estimation

# Create a template mask for frame 42
python scripts/batch_inpaint_helper.py --create-mask-template 42
```

This creates `masks/042_mask.png` with the top 60% filled white (card area).

### Step 2: Refine the Mask

Open `masks/042_mask.png` in any image editor (Paint.NET, GIMP, Photoshop, etc.):

- **White (255)** = "Inpaint this region" (the wrong card back)
- **Black (0)** = "Keep original" (tray, table, everything else)

Paint white ONLY over the incorrect card back region. Be precise around edges.

### Step 3: Run Inpainting in ComfyUI

1. In ComfyUI, click on the **LoadImage** node (top left)
2. Click "Choose File" and select `cropped_hq4/042.png`
3. Click on the **LoadImage (Mask)** node
4. Click "Choose File" and select `masks/042_mask.png`
5. (Optional) Adjust the **Positive Prompt** to describe the specific blue card back pattern
6. Click **Queue Prompt** (top right)

The inpainted image will be saved to `ComfyUI/output/fixed_XXXXX.png`

### Step 4: Review and Replace

1. Check the output in `ComfyUI/output/`
2. If it looks good, copy it back to `cropped_hq4/042.png` (replacing the original)
3. Mark it as done:

```bash
python scripts/batch_inpaint_helper.py --mark-done 42
```

### Step 5: Repeat for All Problem Frames

```bash
# See next 10 frames to work on
python scripts/batch_inpaint_helper.py --next 10

# Check overall progress
python scripts/batch_inpaint_helper.py --stats
```

## Low VRAM Tips

### If you run out of VRAM:

1. **Use SD 1.5 instead of SDXL:**
   - Download: `sd-v1-5-inpainting.ckpt`
   - In workflow node #3, select this model
   - Requires ~4GB VRAM instead of 8GB+

2. **Reduce image resolution:**
   - Add a **ImageScale** node before VAEEncode
   - Resize to 512x512 or smaller for inpainting
   - Add another **ImageScale** node after VAEDecode to restore original size

3. **Use --lowvram launch flag:**
   ```bash
   python main.py --lowvram
   ```

4. **Close other GPU applications** (browsers, games, etc.)

## Prompt Tuning

### For better card back matching:

**Positive prompt examples:**
- `"blue bicycle playing card back, detailed ornate pattern, white border, centered design, studio photo"`
- `"high quality photograph of blue poker card back, intricate swirl pattern, photorealistic"`

**Negative prompt:**
- `"red, different color, blurry, distorted, wrong pattern, watermark, text"`

### Denoise strength (node #8):

- `0.5-0.7` - Subtle changes, preserves more original
- `0.85` (default) - Good balance for card back replacement
- `0.95-1.0` - More creative, but may lose details

## Reference Image Approach (Advanced)

If you have a perfect blue card back reference image:

1. Add an **IPAdapter** or **ControlNet** node
2. Load your reference blue card back
3. This guides the inpainting to match the exact pattern

(Note: Requires additional ComfyUI custom nodes - `IPAdapter` or `ControlNet Tile`)

## Batch Processing (Future Enhancement)

The helper script tracks which frames need work. For full automation:

1. Pre-create all masks manually (most time-consuming part)
2. Use ComfyUI API mode to process all frames in one go
3. Or use the ComfyUI batch features with wildcards

For now, manual review per frame ensures quality.

## Troubleshooting

**"Out of memory" error:**
- Use SD 1.5 instead of SDXL
- Reduce resolution before inpainting
- Use `--lowvram` flag

**Seams visible around mask edges:**
- Feather/blur your mask edges slightly (5-10px Gaussian blur)
- Increase mask slightly beyond the card border

**Card back doesn't match reference:**
- Make prompt more specific
- Try different seeds (node #8)
- Provide a reference image with IPAdapter/ControlNet

**Tray/table details changed:**
- Reduce denoise strength to 0.6-0.7
- Make mask more precise (only card, not surroundings)

---

## Quick Command Reference

```bash
# List all frames needing work
python scripts/batch_inpaint_helper.py --list-todo

# See next 10 frames
python scripts/batch_inpaint_helper.py --next 10

# Create mask template for frame 42
python scripts/batch_inpaint_helper.py --create-mask-template 42

# Mark frame 42 as done
python scripts/batch_inpaint_helper.py --mark-done 42

# Check progress
python scripts/batch_inpaint_helper.py --stats
```
