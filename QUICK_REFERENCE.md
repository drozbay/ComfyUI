# ComfyUI Model Loading - Quick Reference Guide

## TL;DR - How to Load Models with Extra Layers

**ComfyUI can load checkpoints with unexpected layers WITHOUT modification.** Here are the 5 native mechanisms:

---

## 1. State Dict Preprocessing (Recommended for structure differences)

Convert unexpected keys into expected ones BEFORE model creation:

```python
# In supported_models.py
class MyModel(supported_models_base.BASE):
    def process_unet_state_dict(self, state_dict):
        # Rename or filter keys
        if "custom_layer.weight" in state_dict:
            state_dict["standard_layer.weight"] = state_dict.pop("custom_layer.weight")
        return state_dict
```

**Used in ComfyUI for:** SD1.5 (transformer key fixes), SDXL (clip key handling)

---

## 2. Constructor Injection (Recommended for layer creation control)

Inspect checkpoint and pass parameters to model __init__:

```python
# In supported_models.py
class MyModel(supported_models_base.BASE):
    def get_model(self, state_dict, prefix="", device=None):
        # Extract custom weights
        extra_params = {
            k: state_dict.get(k) 
            for k in ["custom_layer.weight", "custom_layer.bias"]
            if k in state_dict
        }
        
        # Pass to model that conditionally creates layers
        return model_base.MyCustomModel(self, device=device, **extra_params)
```

**Used in ComfyUI for:** Stable Zero123 (cc_projection), SDXL (dynamic model type)

---

## 3. Object Patching (For post-load module replacement)

Replace entire submodules after loading:

```python
# After model is loaded
new_module = create_custom_module(...)
patcher.add_object_patch("diffusion_model.custom_layer", new_module)
```

**Used in ComfyUI for:** dtype casting, custom operations

---

## 4. Weight Wrapping (For dynamic weight transformation)

Modify weights as they're accessed:

```python
def transform_weight(weight):
    return custom_operation(weight)

patcher.add_weight_wrapper("diffusion_model.layer.weight", transform_weight)
```

**Used in ComfyUI for:** Quantization, dynamic scaling

---

## 5. Model Injection (For runtime behavior modification)

Inject custom behavior into model operations:

```python
from comfy.patcher_extension import PatcherInjection

class CustomInjection(PatcherInjection):
    def inject(self, patcher):
        # Apply custom behavior
        pass

patcher.set_injections("key", [CustomInjection()])
```

**Used in ComfyUI for:** Custom layer hooking, feature injection

---

## Key Insights

| Pattern | When to Use | Pros | Cons |
|---------|------------|------|------|
| **State Dict Preprocessing** | Checkpoint keys differ from model expectations | Clean, before loading | Needs code modification |
| **Constructor Injection** | Extra layers based on checkpoint content | Flexible layer creation | Custom model class needed |
| **Object Patching** | Want to swap entire modules | Works with existing models | Module path brittle |
| **Weight Wrapping** | Need per-weight transformation | Lazy evaluation | Runtime overhead |
| **Model Injection** | Custom behavior/hooking | Full control | Complex API |

---

## How Load Works in ComfyUI

```
1. Load checkpoint → state_dict
2. Model detection (matches to config class)
3. process_*_state_dict() → transform keys
4. get_model() → inspect state_dict, create model with custom params
5. Model loads state_dict with strict=False (unexpected keys ignored)
6. Optional: add_patches/add_object_patch/add_weight_wrapper
```

---

## Real Examples from ComfyUI

### Example 1: Stable Zero123 - Extract & Inject
**File:** `comfy/supported_models.py:376-387`
- Extracts `cc_projection.weight` and `cc_projection.bias` from checkpoint
- Passes them to `model_base.Stable_Zero123()` constructor
- Model __init__ uses them to initialize projection layers

### Example 2: SDXL - Dynamic Model Type
**File:** `comfy/supported_models.py:197-214`
- Inspects checkpoint for `edm_mean`, `v_pred`, `edm_vpred.sigma_max`
- Sets `sampling_settings` based on detected keys
- Creates different model type (EPS, V_PREDICTION, EDM)

### Example 3: SD1.5 - Key Remapping
**File:** `comfy/supported_models.py:50-62`
- Detects old CLIP key structure
- Remaps `cond_stage_model.transformer.*` to `cond_stage_model.transformer.text_model.*`
- Makes "unexpected" keys match "expected" structure

---

## Limitations

1. **`add_patches()` can't create new keys** - Only modifies existing parameters (like LoRA)
2. **Extra layers must be in nn.Module hierarchy** - To appear in state_dict()
3. **Model detection is deterministic** - Based on unet_config, not state_dict inspection
4. **No auto-discovery** - You must explicitly handle extra layers in your code

---

## Best Practice Path Forward

If loading models with completely new layers:

1. **Create custom model config** → extends `supported_models_base.BASE`
2. **Override `process_unet_state_dict()`** → handle checkpoint transformation
3. **Override `get_model()`** → inspect state_dict, pass extra params
4. **Create custom model class** → in `model_base.py`, accepts extra params, conditionally creates layers
5. **Register model** → add to `sd.py:load_state_dict_guess_config()` or custom loader

This mirrors exactly how Stable Zero123, Flux, HunyuanDiT, and other ComfyUI models handle custom parameters.

---

## File Reference

- **Model configs:** `comfy/supported_models.py`
- **Base model logic:** `comfy/model_base.py`
- **Model patcher:** `comfy/model_patcher.py`
- **State dict loading:** `comfy/sd.py`
- **LoRA/patch patterns:** `comfy/lora.py`
- **ControlNet patterns:** `comfy/controlnet.py`
