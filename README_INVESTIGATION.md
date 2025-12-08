# ComfyUI Model Loading Investigation - Complete Analysis

## Overview

This investigation answers the question: **Can you load a model in ComfyUI that has layers/parameters not expected by the model class definition WITHOUT manually modifying checkpoint files?**

**Answer: YES - ComfyUI has 6 native mechanisms specifically designed for this.**

---

## Documents Included

1. **INVESTIGATION_REPORT.md** - Comprehensive 474-line report with:
   - Executive summary
   - All 5 mechanisms with code examples
   - Real-world examples from ComfyUI
   - Comparison of approaches
   - Limitations and best practices

2. **QUICK_REFERENCE.md** - Quick-start guide with:
   - TL;DR of each mechanism
   - Decision table
   - Real examples from ComfyUI
   - Limitations
   - Best practice path forward

3. **IMPLEMENTATION_PATHS.md** - Detailed technical guide with:
   - Exact file paths and line numbers
   - Code flow diagrams
   - Real code examples
   - Decision tree for choosing approach
   - Complete working example

4. **README_INVESTIGATION.md** - This file

---

## The 6 Native Mechanisms

### 1. **State Dict Preprocessing** (Cleanest for structure changes)
- Rename/remap checkpoint keys before model creation
- Location: `comfy/supported_models.py` - override `process_*_state_dict()`
- Example: SD1.5 fixing old CLIP key formats

### 2. **Constructor Injection** (Best for conditional layer creation)
- Inspect checkpoint in `get_model()`, pass params to model.__init__
- Conditionally create layers based on checkpoint contents
- Example: Stable Zero123, SDXL dynamic model type detection

### 3. **Object Patching** (Best for post-load module swapping)
- Replace entire submodules after loading with enhanced versions
- Location: `comfy/model_patcher.py` - call `add_object_patch()`
- Example: dtype casting, custom operations

### 4. **Weight Wrapping** (Best for dynamic transformation)
- Wrap weight access with custom functions
- Lazy evaluation, no extra memory until accessed
- Location: `comfy/model_patcher.py` - call `add_weight_wrapper()`
- Example: Quantization, on-the-fly scaling

### 5. **Model Patching** (Works with existing keys only)
- Apply LoRA-style patches to existing parameters
- Cannot create new keys, only modify existing ones
- Location: `comfy/model_patcher.py` - call `add_patches()`
- Example: LoRA, fine-tuning weights

### 6. **Model Injection** (Best for runtime hooking)
- Inject custom behavior at model runtime
- Full control over model behavior dynamically
- Location: `comfy/model_patcher.py` - call `set_injections()`
- Example: Custom layer hooks, feature injection

---

## Quick Decision Guide

| Need | Solution | Mechanism | Effort |
|------|----------|-----------|--------|
| Rename checkpoint keys | Remap in preprocessing | #1 | Low |
| Create layers conditionally | Inspect in get_model(), create in __init__ | #2 | Medium |
| Swap module after load | add_object_patch() | #3 | Low |
| Transform weights on access | add_weight_wrapper() | #4 | Medium |
| Apply LoRA-style patches | add_patches() | #5 | Low |
| Intercept model behavior | Implement PatcherInjection | #6 | High |
| Completely new model feature | Combination of #1 + #2 | Hybrid | High |

---

## Key Findings

### Finding 1: All Loading Uses `strict=False`
Every model load in ComfyUI uses `load_state_dict(strict=False)`:
- Location: `model_base.py:307`
- Effect: Unexpected keys are logged as warnings but don't prevent loading

### Finding 2: State Dict Processing Is the Hook
Every model config has processing methods:
```python
def process_unet_state_dict(self, state_dict):  # BASE class
    return state_dict
```
Override these to transform the checkpoint before loading.

### Finding 3: Model Creation Happens After Detection
The full pipeline:
```
Load → Detect Model Config → Transform State Dict → Create Model → Load Weights → Wrap Patcher
```
You can intercept at:
- Transform stage (override process_*_state_dict)
- Create stage (override get_model)
- Post-wrap stage (use model_patcher methods)

### Finding 4: No Magic - Just Flexible APIs
ComfyUI doesn't "automatically" handle extra layers. Instead:
- You explicitly handle them in your model config
- Or you explicitly patch them after loading
- No surprises, no hidden behaviors

---

## Real-World Examples Analyzed

### Example 1: SD1.5 - Key Remapping
**Problem:** Old checkpoints have different CLIP key structure
**Solution:** `process_clip_state_dict()` renames keys before loading
**File:** `comfy/supported_models.py:50-62`

### Example 2: Stable Zero123 - Extract & Inject
**Problem:** Model needs cc_projection weights from checkpoint
**Solution:** Extract in `get_model()`, pass to constructor
**File:** `comfy/supported_models.py:376-387`

### Example 3: SDXL - Dynamic Model Type
**Problem:** Different checkpoints need different sampling methods
**Solution:** Inspect state_dict, set sampling_settings, create appropriate model type
**File:** `comfy/supported_models.py:197-217`

### Example 4: ControlNet - Graceful Handling
**Problem:** ControlNet might have extra keys
**Solution:** Use `strict=False` loading, log unexpected keys, continue
**File:** `comfy/controlnet.py:441-448`

---

## How to Implement: Step-by-Step

### For Models with Extra Layers:

1. **Create model config in `supported_models.py`**:
   ```python
   class MyModel(supported_models_base.BASE):
       unet_config = { /* your config */ }
       
       def get_model(self, state_dict, prefix="", device=None):
           # Extract extra params
           extra = {}
           for key in ["custom.weight", "custom.bias"]:
               if key in state_dict:
                   extra[key] = state_dict.pop(key)
           
           # Create model with extra params
           return model_base.MyDiffusion(self, device=device, **extra)
   ```

2. **Create model class in `model_base.py`**:
   ```python
   class MyDiffusion(BaseModel):
       def __init__(self, config, device=None, custom_weight=None, custom_bias=None, ...):
           super().__init__(config, device=device)
           
           # Conditionally create custom layer
           if custom_weight is not None:
               self.custom = torch.nn.Linear(...)
               self.custom.weight = torch.nn.Parameter(custom_weight)
   ```

3. **Register in model detection** (optional):
   Add to `model_detection.py` so checkpoints auto-match

4. **Test**:
   ```python
   sd = load_checkpoint("model.safetensors")
   config = MyModel({})
   model = config.get_model(sd, device="cuda")
   ```

---

## Code Location Reference

| Task | File | Line | Method |
|------|------|------|--------|
| Load entry point | `sd.py` | 1193 | load_torch_file |
| Model detection | `sd.py` | 1198 | load_state_dict_guess_config |
| State dict processing | `supported_models.py` | per-model | process_*_state_dict |
| Model creation | `supported_models.py` | per-model | get_model |
| Strict=False loading | `model_base.py` | 307 | load_model_weights |
| Add patches | `model_patcher.py` | 555 | add_patches |
| Object patch | `model_patcher.py` | 471 | add_object_patch |
| Weight wrapper | `model_patcher.py` | 480 | add_weight_wrapper |
| Injections | `model_patcher.py` | 1021 | set_injections |

---

## Limitations to Understand

1. **`add_patches()` Only Modifies Existing Keys**
   - Cannot create entirely new layer parameters
   - Used for LoRA, fine-tuning (which modify existing weights)

2. **Extra Layers Must Be in nn.Module Hierarchy**
   - To appear in `state_dict()`, they must be registered modules
   - Use `self.layer = nn.Module(...)` not just `self.layer_weight = tensor`

3. **Model Detection Is Deterministic**
   - Based on `unet_config` structure, not state_dict inspection
   - Must manually register model or handle in custom loader

4. **No Automatic Discovery**
   - Extra layers won't magically appear
   - You must explicitly handle them in your code

---

## Common Patterns Used in ComfyUI

### Pattern 1: Conditional Layer Creation
```python
def __init__(self, ..., extra_param=None):
    super().__init__(...)
    if extra_param is not None:
        self.extra_layer = CustomLayer(extra_param)
    else:
        self.extra_layer = nn.Identity()
```

### Pattern 2: State Dict Inspection & Remapping
```python
def process_unet_state_dict(self, state_dict):
    if "old_key" in state_dict:
        state_dict["new_key"] = state_dict.pop("old_key")
    return state_dict
```

### Pattern 3: Extract & Pass Constructor Args
```python
def get_model(self, state_dict, prefix="", device=None):
    custom_param = state_dict.get("custom_param", None)
    return CustomModel(self, custom_param=custom_param, device=device)
```

### Pattern 4: Object Patching for Replacement
```python
patcher.add_object_patch("module.name", new_module_object)
```

---

## Recommended Path for Your Use Case

If you want to load models with completely new parameters:

1. **Extend `supported_models_base.BASE`** with your config
2. **Override `process_unet_state_dict()`** to handle checkpoint structure
3. **Override `get_model()`** to inspect and extract custom params
4. **Create custom model class** accepting extra parameters
5. **Conditionally create layers** in model.__init__

This mirrors exactly how ComfyUI handles:
- Stable Zero123 (cc_projection)
- SDXL (dynamic model type)
- Flux (guidance embedder)
- HunyuanDiT (custom embedders)

And it requires ZERO checkpoint modification - the checkpoint is read as-is and intelligently interpreted by your code.

---

## Related Files in This Analysis

- `INVESTIGATION_REPORT.md` - Full 474-line technical report
- `QUICK_REFERENCE.md` - Quick-start guide
- `IMPLEMENTATION_PATHS.md` - Detailed with line numbers and code
- This file - Overview and summary

## Conclusion

ComfyUI's design philosophy is:
1. **Never modify checkpoint files** - Just read and interpret them
2. **Flexible model config system** - Detect and adapt to checkpoint structure
3. **Powerful patcher system** - Modify loaded models on-the-fly
4. **Conditional layer creation** - Create layers only when needed

You can use any combination of these to load models with "unexpected" layers, making ComfyUI remarkably flexible for custom models and variants.

