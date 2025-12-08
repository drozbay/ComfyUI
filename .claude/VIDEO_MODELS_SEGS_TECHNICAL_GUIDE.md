# Video Models & SEGS Technical Guide for ComfyUI

## Executive Summary

The SEGS (Segments) system in ComfyUI's Impact Pack provides targeted detail enhancement by processing individual regions of images or videos. Modern video architectures (Mochi, LTX-Video, SVD, Wan) use 5-dimensional latent tensors with temporal compression, which differs from the 4-dimensional tensors used by traditional image models and AnimateDiff.

**Key insight: Despite the difference in latent dimensions, the SEGS system now works correctly with modern video models.** The sampling process is model-agnostic - it operates on latent tensors regardless of whether they're 4D or 5D. The model's forward pass handles the dimensional complexity internally, while SEGS operations work at the pixel/frame level with 3D temporal masks.

## Understanding the SEGS System

### Core Architecture

SEGS is a data structure designed to track and process individual regions of interest within images or videos. Each SEGS object contains:

```python
# From modules/impact/core.py
SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence',
                  'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])
```

**Components explained:**
- `cropped_image`: The extracted region from the original image/video
- `cropped_mask`: Binary mask defining the exact area of interest
- `confidence`: Detection confidence score (0.0 to 1.0)
- `crop_region`: Tuple of (x1, y1, x2, y2) defining the crop boundaries
- `bbox`: Bounding box coordinates
- `label`: Semantic label (e.g., "person", "face")
- `control_net_wrapper`: Optional ControlNet conditioning

A SEGS object is a tuple: `(dimensions, list[SEG])` where dimensions is typically `(height, width)`.

### SEGS Processing Pipeline

The typical SEGS workflow:
1. **Detection**: Identify regions of interest (via SAM, YOLO, etc.)
2. **Segmentation**: Create individual SEG objects for each region
3. **Enhancement**: Process each segment independently (upscaling, inpainting)
4. **Composition**: Paste enhanced segments back to original

## Video Model Latent Formats

### The Fundamental Difference

Modern video models introduce a paradigm shift in how temporal information is encoded:

```python
# AnimateDiff-style (4D tensor)
# From SEGSDetailerForAnimateDiff
latent_animatediff = torch.zeros([batch_size * frames, channels, height, width])

# Modern video models (5D tensor)
# From nodes_mochi.py - Mochi model
latent_mochi = torch.zeros([batch_size, 12, ((length - 1) // 6) + 1, height // 8, width // 8])

# From nodes_lt.py - LTX-Video model
latent_ltx = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32])
```

### Dimension Comparison Table

| Model Type | Pixel Space | Latent Space | Temporal Compression | Spatial Compression |
|------------|-------------|--------------|---------------------|-------------------|
| **Image (SD)** | [B, H, W, 3] | [B, 4, H/8, W/8] | N/A | 8x |
| **AnimateDiff** | [B*F, H, W, 3] | [B*F, 4, H/8, W/8] | None | 8x |
| **SVD** | [B*F, H, W, 3] | [B, 4, F, H/8, W/8] | None | 8x |
| **Mochi** | [B*F, H, W, 3] | [B, 12, F/6, H/8, W/8] | 6x | 8x |
| **LTX-Video** | [B*F, H, W, 3] | [B, 128, F/8, H/32, W/32] | 8x | 32x |

*Where B=batch, F=frames, H=height, W=width*

## Pixel Space vs Latent Space

### The Transformation Pipeline

The relationship between pixel and latent space in video models involves two types of compression:

1. **Spatial Compression**: Reduces height and width dimensions
2. **Temporal Compression**: Reduces the number of frames in latent space

```python
# Example: 25 frames of 512x768 video in LTX-Video

# Pixel space
pixel_video = torch.zeros([25, 512, 768, 3])  # 25 frames, RGB

# After VAE encoding to latent space
latent_video = torch.zeros([1, 128, 4, 16, 24])
# Breakdown:
# - Batch: 1
# - Channels: 128 (model-specific)
# - Temporal: 4 (25 frames / 8 + padding)
# - Height: 16 (512 / 32)
# - Width: 24 (768 / 32)
```

### Why This Matters for SEGS

**Important: The SEGS system works with both 4D and 5D latent formats because the sampling process is model-agnostic.** SEGS operations work at the pixel/frame level, and the VAE handles conversion between pixel and latent space. The model's forward pass during sampling handles the dimensional complexity transparently.

```python
# SEGS workflow (works with all model types)
# 1. Crop frames at pixel level (works with any frame count)
cropped_frames = crop_frames(video_frames, crop_region)

# 2. VAE encodes to latent (4D or 5D depending on model)
latent = vae.encode(cropped_frames)  # Model-specific format

# 3. Sample (model-agnostic - just passes latent to model)
enhanced_latent = sample(model, latent, conditioning)

# 4. VAE decodes back to pixels
enhanced_frames = vae.decode(enhanced_latent)
```

## MakeTileSEGS Implementation Analysis

### Original MakeTileSEGS (Image-focused)

The original implementation creates tile segments for detailed processing:

```python
# From segs_nodes.py - MakeTileSEGS (lines 2122-2172)
def doit(images, bbox_size, crop_factor, min_overlap, ...):
    # Calculate tile positions
    for j in range(0, n_vertical):
        for i in range(0, n_horizontal):
            # Proper relative coordinate calculation
            bbox = x1, y1, x2, y2
            crop_region = utils.make_crop_region(iw, ih, bbox, crop_factor)
            cx1, cy1, cx2, cy2 = crop_region

            # Create mask with correct relative coordinates
            rel_left = x1 - cx1
            rel_top = y1 - cy1
            rel_right = x2 - cx1
            rel_bot = y2 - cy1

            # Apply mask
            mask[rel_top:rel_bot, rel_left:rel_right] = 1.0
```

### MakeTileSEGSForVideo Implementation (Fixed)

**Status: FIXED** - The coordinate bug has been corrected in the current implementation:

```python
# From segs_nodes.py - MakeTileSEGSForVideo (lines 1995-2019)
# Proper relative coordinate calculation
rel_left = x1 - cx1
rel_top = y1 - cy1
rel_right = x2 - cx1
rel_bot = y2 - cy1

if mask_irregularity > 0:
    if batch_size > 1:
        # Handle 3D temporal masks - apply to all frames
        if mask_cache is not None:
            for frame_idx in range(batch_size):
                # FIXED: Using relative coordinates
                core.adaptive_mask_paste(mask[frame_idx], mask_cache,
                                        (rel_left, rel_top, rel_right, rel_bot))
        else:
            # Generate one random mask and apply to all frames
            random_mask_2d = np.zeros_like(mask[0])
            core.random_mask(random_mask_2d, (rel_left, rel_top, rel_right, rel_bot),
                           factor=mask_irregularity, size=mask_quality, fast=fast, seed=seed)
            for frame_idx in range(batch_size):
                mask[frame_idx] = random_mask_2d
```

## Practical Workflow Examples

### Detecting Model Type

Impact Pack detects video models to allow batch (multi-frame) processing:

```python
# From modules/impact/utils.py (lines 16-45)
def is_known_image_model(model):
    """Returns True for traditional image models, False for video models"""
    image_model_classes = (
        comfy.model_base.SD15,
        comfy.model_base.SD20,
        comfy.model_base.SDXL,
        comfy.model_base.SD3,
        comfy.model_base.Flux,
        comfy.model_base.PixArt,
        comfy.model_base.HunyuanDiT,
        # ... other image models
    )
    return isinstance(model.model.model_sampling, image_model_classes)

# Usage in SEGSDetailer (segs_nodes.py line 222-223)
if utils.is_known_image_model(model) and len(image) > 1:
    raise Exception('SEGSDetailer does not support batch processing with standard image models.')
# Note: Video models (Mochi, LTX-Video, Wan, SVD, etc.) are NOT in this list,
# so they support batch processing (multiple frames)
```

### Handling Temporal Masks

Video masks need special handling for temporal consistency:

```python
# From utils.py - Enhanced mask dilation for video
def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return mask

    # Handle temporal masks (3D)
    if isinstance(mask, torch.Tensor) and len(mask.shape) == 3:
        # Process each frame independently
        dilated_frames = []
        kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

        for i in range(mask.shape[0]):
            frame_mask = mask[i].cpu().numpy()
            if dilation_factor > 0:
                dilated = cv2.dilate(frame_mask, kernel, iterations=iter)
            else:
                dilated = cv2.erode(frame_mask, kernel, iterations=iter)
            dilated_frames.append(dilated)

        return torch.from_numpy(np.stack(dilated_frames))
```

## Compatibility Considerations

### Model Support Matrix

| Feature | Image Models | AnimateDiff | Modern Video Models (5D) |
|---------|--------------|-------------|--------------------------|
| Standard SEGS | ✅ Full | ✅ Full | ✅ Full |
| MakeTileSEGS | ✅ Full | ✅ Full | ✅ Full (fixed) |
| MakeTileSEGSForVideo | N/A | ✅ Full | ✅ Full (fixed) |
| SEGSDetailer | ✅ Single frame only | ✅ Full (multi-frame) | ✅ Full (multi-frame) |
| SEGSPaste | ✅ Full | ✅ Full | ✅ Full (3D mask support) |
| Temporal masks (3D) | N/A | ✅ Supported | ✅ Supported |
| Latent operations | ✅ 4D | ✅ 4D | ✅ 5D (model handles internally) |

### Common Issues and Solutions

1. **Issue**: "SEGSDetailer does not support batch processing" error
   - **Cause**: Using an image model (SD1.5, SDXL, Flux) with multiple frames
   - **Solution**: Use single frames with image models, or use video models for multi-frame processing

2. **Issue**: Temporal inconsistency in enhanced segments
   - **Cause**: Model doesn't maintain temporal coherence (depends on model architecture)
   - **Solution**: Use video models designed for temporal consistency (Mochi, LTX-Video, Wan)

3. **Issue**: Mask dimensions don't match video frames
   - **Cause**: 2D mask provided when 3D temporal mask needed
   - **Solution**: SEGS system automatically handles this - masks can be 2D (applied to all frames) or 3D (per-frame masks)

## Current Status & Working Features

### ✅ Completed Features

1. **MakeTileSEGSForVideo coordinate fix** - Fixed and working correctly with relative coordinates
2. **3D temporal mask support** - SEGSPaste handles 3D masks correctly for video
3. **Video model detection** - `is_known_image_model()` automatically identifies video vs image models
4. **SEGSDetailer video support** - Works with all video models (Mochi, LTX-Video, Wan, SVD)

### 🎯 How It Works

The SEGS system is now fully compatible with modern video models because:

1. **Sampling is model-agnostic**: The diffusion sampling process works identically with 4D or 5D latents
2. **VAE handles format conversion**: Encoding/decoding between pixel and latent space is model-specific
3. **Temporal masks supported**: Both 2D (applied to all frames) and 3D (per-frame) masks work
4. **Batch processing enabled**: Video models support processing multiple frames as a batch

### 💡 Potential Future Enhancements

1. **TemporalMaskInterpolation**: Smooth mask transitions across frames for animated effects
2. **Optimized tiling strategies**: Better tile overlap handling for very long videos
3. **Frame range selection**: Process specific frame ranges within a video segment

## Conclusion

**The SEGS system now fully supports modern video models.** Despite using 5-dimensional latent tensors with temporal compression, video models (Mochi, LTX-Video, Wan, SVD) work seamlessly with SEGS because:

1. **Sampling abstraction**: The diffusion process is model-agnostic - it operates on latent tensors regardless of dimensionality
2. **Model encapsulation**: The model's forward pass handles 4D vs 5D complexity internally
3. **Pixel-level operations**: SEGS works at the pixel/frame level, with VAE handling format conversion
4. **Fixed bugs**: Coordinate calculation and temporal mask issues have been resolved

### Current Capabilities

- ✅ High-quality video upscaling with regional control
- ✅ Targeted video inpainting and enhancement
- ✅ Efficient processing of high-resolution video content
- ✅ Seamless integration with existing ComfyUI workflows
- ✅ Support for all modern video architectures

### Understanding Latent Dimensions

While modern video models use 5D latents internally, this is **transparent to SEGS operations**:
- SEGS crops and masks work at the **pixel level** (frames × height × width)
- VAE encoding converts pixels → latents (model-specific format)
- Sampling operates on latents (model handles dimensions internally)
- VAE decoding converts latents → pixels (any model format → standard frames)

This abstraction is what makes SEGS work universally across all model types without special adaptation.

---

*Document Version: 2.0*
*Last Updated: October 3, 2025*
*Status: Current implementation verified against Impact Pack codebase*