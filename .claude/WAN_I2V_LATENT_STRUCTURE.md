# Wan I2V Latent Structure and Mask Handling

## Temporal Compression Basics

Wan video models use 4:1 temporal compression with a special case for the first frame:

- Pixel frame 0 maps to latent frame 0 (1:1 encoding)
- Pixel frames 1-4 map to latent frame 1 (4:1 encoding)
- Pixel frames 5-8 map to latent frame 2 (4:1 encoding)
- And so on...

The formula for calculating latent temporal dimension from pixel frames:
```python
temporal_latent = ((pixel_frames - 1) // 4) + 1
# Example: 81 pixel frames -> ((81-1) // 4) + 1 = 21 latent frames
```

The reverse formula:
```python
pixel_frames = (temporal_latent - 1) * 4 + 1
# Example: 21 latent frames -> (21-1) * 4 + 1 = 81 pixel frames
```

## The 4-Channel Mask Structure

The concat_mask for Wan I2V models has shape `[batch, 4, temporal_latent, height, width]`.

The 4 channels are NOT just duplicates. Each channel represents one of the 4 pixel frames that were compressed into that latent frame:

For latent frame 0 (special case):
- All 4 channels represent pixel frame 0 (because frame 0 is 1:1 encoded)

For latent frame t > 0:
- Channel 0 represents pixel frame (t-1)*4 + 1
- Channel 1 represents pixel frame (t-1)*4 + 2
- Channel 2 represents pixel frame (t-1)*4 + 3
- Channel 3 represents pixel frame (t-1)*4 + 4

Example for 81 pixel frames (21 latent frames):
```
Latent 0: channels [0,1,2,3] all = pixel frame 0
Latent 1: channel 0 = pixel frame 1, channel 1 = pixel frame 2, channel 2 = pixel frame 3, channel 3 = pixel frame 4
Latent 2: channel 0 = pixel frame 5, channel 1 = pixel frame 6, channel 2 = pixel frame 7, channel 3 = pixel frame 8
...
Latent 20: channel 0 = pixel frame 77, channel 1 = pixel frame 78, channel 2 = pixel frame 79, channel 3 = pixel frame 80
```

## How the Mask is Used in the Model

In `comfy/model_base.py`, the mask is concatenated with the latent image along the channel dimension:

```python
# From nodes_sampler.py in WanVideoWrapper
image_cond = torch.cat([image_cond_mask, image_cond])  # [4 + 16, T, H, W] = [20, T, H, W]
```

The model receives 20 channels total: 4 mask channels followed by 16 latent channels. This allows the model to apply different masking to each of the 4 pixel frames within each latent frame.

## Converting Pixel-Space Masks to Latent-Space

When you have a mask at pixel temporal resolution (e.g., 81 frames), you must convert it properly:

```python
# Input: mask at pixel resolution [B, pixel_frames, H, W]
# e.g., [1, 81, 60, 104]

# Step 1: Repeat the first frame 4 times (for 1:1 encoding)
start_mask_repeated = mask[:, 0:1].repeat(1, 4, 1, 1)  # [B, 4, H, W]

# Step 2: Keep remaining frames as-is
mask_middle = mask[:, 1:]  # [B, 80, H, W]

# Step 3: Concatenate
mask = torch.cat([start_mask_repeated, mask_middle], dim=1)  # [B, 84, H, W]

# Step 4: Reshape into groups of 4
num_groups = mask.shape[1] // 4  # 21
mask = mask[:, :num_groups * 4]  # Trim to multiple of 4
mask = mask.view(B, num_groups, 4, H, W)  # [B, 21, 4, H, W]

# Step 5: Transpose to get channels first
mask = mask.transpose(1, 2)  # [B, 4, 21, H, W]

# Result: [1, 4, 21, H, W] - proper latent mask format
```

After this conversion:
- `mask[:, :, 0, :, :]` contains 4 copies of pixel frame 0
- `mask[:, 0, 1, :, :]` contains pixel frame 1
- `mask[:, 1, 1, :, :]` contains pixel frame 2
- `mask[:, 2, 1, :, :]` contains pixel frame 3
- `mask[:, 3, 1, :, :]` contains pixel frame 4
- And so on...

## Mask Polarity

In ComfyUI's Wan implementation:
- 0.0 = use the provided latent (don't generate)
- 1.0 = generate new content

This is consistent with ComfyUI's `WanFirstLastFrameToVideo` node in `comfy_extras/nodes_wan.py`:

```python
mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
if start_image is not None:
    mask[:, :, :start_image.shape[0] + 3] = 0.0  # 0 = use provided
```

Note: The model code in `model_base.py` line 1149 inverts the mask before use (`mask = 1.0 - mask`), but you should provide masks with the polarity described above.

## BindWeave Special Case

BindWeave models prepend 4 reference frames before the I2V sequence:

```
[ref0, ref1, ref2, ref3, i2v_frame0, i2v_frame1, ...]
```

Each reference frame is exactly 1 latent frame (not 4:1 compressed). The reference masks have shape `[1, 4, 4, H, W]` where:
- First dimension (4) = the 4 mask channels
- Second dimension (4) = the 4 reference frame slots

For reference masks:
- 0.0 = reference slot has an actual image
- 1.0 = reference slot is empty (zero-padded)

The full BindWeave mask structure:
```python
reference_mask = [1, 4, 4, H, W]      # 4 reference frames
i2v_mask = [1, 4, T_latent, H, W]     # I2V sequence
full_mask = torch.cat([reference_mask, i2v_mask], dim=2)  # [1, 4, 4+T_latent, H, W]
```

## Common Pitfalls

1. Simple channel repetition loses per-frame granularity. If you have a pixel-space mask and just repeat it to 4 channels, you lose the ability to mask individual frames within a latent frame.

2. Interpolating pixel-space masks to latent temporal resolution also loses granularity. The proper conversion described above preserves per-frame control.

3. The first frame is special. All 4 channels of latent frame 0 represent the same pixel frame 0. Don't treat it like subsequent frames.

4. Reference frames in BindWeave are NOT temporally compressed. Each reference is exactly 1 latent frame with all 4 channels representing that single image.

## Reference Implementations

ComfyUI native implementation: `comfy_extras/nodes_wan.py` - `WanFirstLastFrameToVideo`

WanVideoWrapper implementation: `custom_nodes/ComfyUI-WanVideoWrapper/nodes.py` - `WanVideoImageToVideoEncode` (lines 960-970)

Model handling: `comfy/model_base.py` - `concat_cond()` method around line 1143
