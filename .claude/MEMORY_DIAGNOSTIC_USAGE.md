# VACE Detailer Hook - Memory Diagnostic Guide

## What Was Added

Added comprehensive memory tracking to `vace_detailer_hook.py` to diagnose potential memory leaks during SEG processing.

## Diagnostic Output

When running a workflow with the VACE DetailerHook, you'll now see output like:

```
======================================================================
[VACE Detailer Hook] pre_ksample called
[Memory at Entry] CPU: 8.5GB phys, 12.3GB virt (avail: 15.2GB) | CUDA: 7.05GB alloc, 12.56GB res

[Processing SEG 1/6]
[SEG Info] Crop region: (100, 200, 800, 900), BBox: (150, 250, 750, 850)

[Memory Before VACE Encode] CPU: 8.5GB phys, 12.3GB virt (avail: 15.2GB) | CUDA: 7.05GB alloc, 12.56GB res
[Encoding Target] 640x640, 25 frames, batch 1
[Input Positive Conditioning] No VACE tensors
[Input Negative Conditioning] No VACE tensors

[Memory After VACE Encode] CPU: 8.6GB phys, 12.4GB virt (avail: 15.1GB) | CUDA: 7.05GB alloc, 12.56GB res
[Output Positive Conditioning] vace_frames: 1 items (156.25MB), vace_mask: 1 items (39.06MB) | Total: 195.31MB
[Output Negative Conditioning] vace_frames: 1 items (156.25MB), vace_mask: 1 items (39.06MB) | Total: 195.31MB

[Memory Before Return] CPU: 8.6GB phys, 12.4GB virt (avail: 15.1GB) | CUDA: 7.05GB alloc, 12.56GB res
======================================================================

[SEG Processing Complete] post_decode called
[Decoded Image Shape] torch.Size([1, 293, 416, 480, 3])
[Memory After Decode] CPU: 12.2GB phys, 18.7GB virt (avail: 11.5GB) | CUDA: 16.43GB alloc, 19.22GB res

======================================================================
[VACE Detailer Hook] pre_ksample called
[Memory at Entry] CPU: 12.2GB phys, 18.7GB virt (avail: 11.5GB) | CUDA: 16.43GB alloc, 19.22GB res
                      ↑ Physical RAM    ↑ Including pagefile  ↑ System free RAM

[Processing SEG 2/6]
...
```

## What to Look For

### Signs of CPU RAM/Pagefile Exhaustion (Most Likely):
1. **Virtual memory climbing**: `virt` grows rapidly: 18GB → 28GB → 38GB → 48GB+
2. **System available dropping**: `avail` decreases: 15GB → 8GB → 3GB → 0.5GB (critical!)
3. **Physical stable, virtual grows**: Indicates heavy pagefile/swap usage
4. **Silent crash**: Process terminates without error when system runs out of virtual memory
5. **CUDA stable**: GPU memory stays constant while CPU/pagefile exhausts
6. **Pattern**: Each SEG adds 3-8GB virtual memory that doesn't get released

**Critical threshold**: When `avail` drops below 2-3GB, system is under severe memory pressure

### Signs of CUDA VRAM Issues (Less Likely):
1. **Explicit error**: PyTorch throws "RuntimeError: CUDA out of memory" with stack trace
2. **CUDA reserved maxing out**: Reserved CUDA approaches your GPU limit (24GB, 40GB, etc.)
3. **Error during sampling**: Crash happens during the 100% progress bar, not after

### Expected Healthy Behavior:
1. **Virtual memory stable**: `virt` should stay within 4-8GB range across SEGs
2. **System available steady**: `avail` should remain above 8-10GB throughout
3. **Physical RAM moderate growth**: `phys` may grow 2-4GB total but shouldn't keep climbing
4. **CUDA memory spike on first SEG**: Model loading causes VRAM jump, then stabilizes
5. **VACE tensors small**: Each VACE operation adds ~50-100MB, not GB

### Memory Metrics Explained:
- **phys** = Physical RAM actually in use (excludes pagefile)
- **virt** = Total virtual memory (physical + pagefile/swap on Windows)
- **avail** = System-wide free memory available before pressure
- **CUDA alloc** = GPU memory allocated to tensors
- **CUDA res** = GPU memory reserved by PyTorch allocator

## Memory Cleanup Implementation

### Aggressive Cleanup (Now Enabled)

The `post_decode()` method now uses a multi-stage aggressive cleanup strategy:

```python
# 1. Clear stored conditioning from previous SEG
del self._last_positive_cond
del self._last_negative_cond

# 2. Use ComfyUI's cleanup (dead references + gc + cache clear)
mm.cleanup_models_gc()

# 3. Force additional deep garbage collection passes (3x)
for _ in range(3):
    gc.collect()

# 4. Clear GPU cache again after deep GC
mm.soft_empty_cache()
```

This runs automatically after each SEG completes and will:
1. **Delete conditioning references**: Explicitly remove VACE tensor references stored from previous SEG
2. **Detect dead model references**: ComfyUI's leak detection
3. **Deep garbage collection**: 3 passes to break circular references in tensor graphs
4. **Clear GPU cache**: Twice (once in cleanup_models_gc, once after deep GC)

### Why Multiple GC Passes?

Large video tensors (8GB+ each) often form **circular reference chains**:
- Conditioning dict → VACE tensors → latent tensors → model outputs → back to conditioning
- A single `gc.collect()` may not break all cycles
- **3 passes** ensure deeper cleanup of complex tensor graphs

### Expected Output

You should now see:
```
[SEG Processing Complete] post_decode called
[Memory After Decode] CPU: 34.74GB phys, 192.55GB virt (avail: 16.51GB)
[Running aggressive memory cleanup]
[Memory After Cleanup] CPU: 20.15GB phys, 170.30GB virt (avail: 31.70GB)  ← 22GB recovery!
```

**Goal**: Virtual memory should recover 15-25GB after cleanup, staying under 150GB total across all 6 SEGs.

### Last Resort: Reduce Concurrent Operations

If virtual memory still exhausts (>200GB), consider:
1. **Process fewer SEGs**: Split workflow into batches of 2-3 SEGs
2. **Reduce video resolution**: 720p → 540p reduces memory by ~40%
3. **Shorter videos**: 293 frames → 150 frames halves memory usage

## Files Modified

- `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/custom_nodes/ComfyUI-WanVaceAdvanced/nodes/vace_detailer_hook.py`
  - Added `get_memory_info()` helper
  - Added `count_conditioning_tensors()` helper
  - Added logging in `pre_ksample()` at 4 checkpoints
  - Added logging in `post_decode()` for completion tracking

## Next Steps

1. Run your workflow with multiple SEGS (6+ recommended)
2. Check the console output for the diagnostic messages
3. Look for the patterns described above
4. Report back findings to determine if cleanup is needed
