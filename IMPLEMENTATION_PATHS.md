# ComfyUI Model Loading - Implementation Paths & Code References

## Investigation Summary

This document maps exactly where and how ComfyUI handles loading models with "unexpected" layers, with precise file paths and line numbers.

---

## Path 1: State Dict Preprocessing

### Where It Happens
- **File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/supported_models_base.py`
- **Lines:** 90-98
- **Method:** `process_clip_state_dict()` and `process_unet_state_dict()`

### Code Flow
```
sd.py:1198 (load_state_dict_guess_config)
  ↓
supported_models.py:XXX (model_config.process_unet_state_dict)
  ↓
model_base.py:306 (diffusion_model.load_state_dict(to_load, strict=False))
```

### Real Example: SD1.5
**File:** `comfy/supported_models.py:47-62`

```python
def process_clip_state_dict(self, state_dict):
    k = list(state_dict.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            state_dict[y] = state_dict.pop(x)  # FIX: Rename old key format to new
    return state_dict
```

### Real Example: SDXL
**File:** `comfy/supported_models.py:222-232`

```python
def process_clip_state_dict(self, state_dict):
    keys_to_replace = {}
    replace_prefix = {}
    
    replace_prefix["conditioner.embedders.0.transformer.text_model"] = "clip_l.transformer.text_model"
    replace_prefix["conditioner.embedders.1.model."] = "clip_g."
    state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
    # ... more transformations
    return state_dict
```

### How to Implement
1. Extend model class in `supported_models.py`
2. Override `process_unet_state_dict()` or `process_clip_state_dict()`
3. Rename/remap/filter keys as needed
4. Return transformed state_dict

---

## Path 2: Constructor Injection via State Dict Inspection

### Where It Happens
- **File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/supported_models.py`
- **Pattern:** Override `get_model(self, state_dict, prefix="", device=None)`
- **Called from:** `sd.py:1198+` (load_state_dict_guess_config)

### Code Flow
```
sd.py:1198 (load_state_dict_guess_config)
  ↓
model_config.get_model(state_dict, prefix, device)  [YOUR OVERRIDE]
  ↓
Inspect state_dict, extract custom params
  ↓
Create custom model class with extra parameters
  ↓
model.load_model_weights(sd, unet_prefix)  [model_base.py:299-314]
```

### Real Example 1: Stable Zero123
**File:** `comfy/supported_models.py:376-387`

```python
class Stable_Zero123(supported_models_base.BASE):
    required_keys = {
        "cc_projection.weight": None,
        "cc_projection.bias": None,
    }
    
    def get_model(self, state_dict, prefix="", device=None):
        # INSPECT: Extract custom weights from state_dict
        cc_projection_weight = state_dict.get("cc_projection.weight", None)
        cc_projection_bias = state_dict.get("cc_projection.bias", None)
        
        # INJECT: Pass to custom model class
        out = model_base.Stable_Zero123(
            self, 
            device=device, 
            cc_projection_weight=cc_projection_weight, 
            cc_projection_bias=cc_projection_bias
        )
        return out
```

### Real Example 2: SDXL - Dynamic Model Type
**File:** `comfy/supported_models.py:197-217`

```python
def model_type(self, state_dict, prefix=""):
    if 'edm_mean' in state_dict and 'edm_std' in state_dict:
        self.latent_format = latent_formats.SDXL_Playground_2_5()
        self.sampling_settings["sigma_data"] = 0.5
        return model_base.ModelType.EDM
    elif "edm_vpred.sigma_max" in state_dict:
        self.sampling_settings["sigma_max"] = float(state_dict["edm_vpred.sigma_max"].item())
        return model_base.ModelType.V_PREDICTION_EDM

def get_model(self, state_dict, prefix="", device=None):
    out = model_base.SDXL(self, model_type=self.model_type(state_dict, prefix), device=device)
    return out
```

### How to Implement
1. Create model class in `supported_models.py` extending `BASE`
2. Override `get_model(state_dict, prefix, device)`
3. Inspect state_dict for custom keys/parameters
4. Create custom model in `model_base.py` that accepts these parameters
5. In model.__init__, conditionally create layers based on passed parameters

---

## Path 3: Post-Load Object Patching

### Where It Happens
- **File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py`
- **Lines:** 471-472 (add_object_patch)
- **Lines:** 791-794 (apply during injection)

### Code Flow
```
Model loaded and wrapped with ModelPatcher
  ↓
patcher.add_object_patch("model.path.to.submodule", new_module_object)
  ↓
patcher.inject_model()  [Line 1069]
  ↓
For each object_patch: set_attr(model, key, new_object)  [Line 792]
```

### Implementation
```python
# After model is loaded
patcher = model_patcher.ModelPatcher(model, load_device, offload_device)

# Create enhanced module with extra parameters
enhanced_module = MyCustomModule(extra_param=value)

# Replace existing module
patcher.add_object_patch("diffusion_model.custom_layer", enhanced_module)
```

### Code Reference
```python
# Line 471-472: Add object patch
def add_object_patch(self, name, obj):
    self.object_patches[name] = obj

# Line 791-794: Apply during injection
for k in self.object_patches:
    old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
    if k not in self.object_patches_backup:
        self.object_patches_backup[k] = old
```

---

## Path 4: Weight Wrapping

### Where It Happens
- **File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py`
- **Lines:** 480-482 (add_weight_wrapper)
- **Lines:** 736-739 (apply during weight patching)

### Code Flow
```
patcher.add_weight_wrapper("diffusion_model.layer.weight", transform_func)
  ↓
During inference: weight_function is called for each forward pass
  ↓
transform_func(weight) returns modified weight
```

### Implementation
```python
def weight_transform_function(weight):
    # Apply custom transformation
    return modified_weight

patcher.add_weight_wrapper("diffusion_model.custom_layer.weight", weight_transform_function)
```

### Code Reference
```python
# Line 480-482: Add weight wrapper
def add_weight_wrapper(self, name, function):
    self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]

# Line 736-739: Apply during inference
if weight_key in self.weight_wrapper_patches:
    m.weight_function.extend(self.weight_wrapper_patches[weight_key])
```

---

## Path 5: Model Patcher add_patches

### Where It Happens
- **File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py`
- **Lines:** 555-577
- **Important:** Only works with keys that ALREADY EXIST in model.state_dict()

### Code Flow
```
patcher.add_patches(patch_dict, strength_patch, strength_model)
  ↓
For each patch in patch_dict:
  ↓
  If key exists in model.state_dict():
    Add to self.patches[key]
  Else:
    Skip (returns list of successfully added patches)
  ↓
During inference: patches are applied to existing weights
```

### Implementation
```python
# LoRA example
patch_dict = {
    "diffusion_model.layer1.weight": lora_patch_1,
    "diffusion_model.layer2.weight": lora_patch_2,
}
patcher.add_patches(patch_dict, strength_patch=1.0, strength_model=1.0)
```

### Code Reference
```python
# Line 555-577: add_patches method
def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
    with self.use_ejected():
        p = set()
        model_sd = self.model.state_dict()
        for k in patches:
            if key in model_sd:  # CRITICAL: Key must exist
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches
        return list(p)  # Returns which patches were actually added
```

---

## Path 6: Model Injection System

### Where It Happens
- **File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py`
- **Lines:** 1021-1022 (set_injections)
- **Lines:** 1069-1088 (inject_model / eject_model)

### Code Flow
```
Create PatcherInjection subclass
  ↓
patcher.set_injections("key", [injection_instance])
  ↓
patcher.inject_model()  [During model load]
  ↓
For each injection: inj.inject(patcher)
  ↓
Injection modifies patcher.model behavior
```

### Implementation
```python
from comfy.patcher_extension import PatcherInjection

class CustomLayerInjection(PatcherInjection):
    def inject(self, patcher):
        # Modify patcher.model or patcher settings
        patcher.model.custom_flag = True
    
    def eject(self, patcher):
        # Restore original state
        patcher.model.custom_flag = False

patcher.set_injections("custom_behavior", [CustomLayerInjection()])
```

### Code Reference
```python
# Line 1021-1022: Set injections
def set_injections(self, key: str, injections: list[PatcherInjection]):
    self.injections[key] = injections

# Line 1069-1075: Apply injections
def inject_model(self):
    if self.is_injected or self.skip_injection:
        return
    for injections in self.injections.values():
        for inj in injections:
            inj.inject(self)
            self.is_injected = True
```

---

## How the Full Load Pipeline Works

### Step 1: Load Checkpoint
**File:** `sd.py:1193`
```python
sd = comfy.utils.load_torch_file(ckpt_path)
```

### Step 2: Detect Model Type
**File:** `sd.py:1198-1250` (load_state_dict_guess_config)
```python
# Match checkpoint structure to model config class
for model_config in comfy.model_detection.detect_models(sd):
    # Try each detected config
```

### Step 3: Preprocess State Dict
**File:** Various model configs in `supported_models.py`
```python
to_load = model_config.process_unet_state_dict(to_load)
```

### Step 4: Create Model
**File:** Model config's `get_model()`
```python
out = model_config.get_model(state_dict, prefix, device)
```

### Step 5: Load Weights
**File:** `model_base.py:299-314`
```python
to_load = self.model_config.process_unet_state_dict(to_load)
m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
# m = missing keys (logged as warning)
# u = unexpected keys (logged as warning)
```

### Step 6: Wrap with ModelPatcher
**File:** `sd.py:1220+`
```python
out_model = comfy.model_patcher.ModelPatcher(out, ...)
```

### Step 7: Optional - Apply Patches
**File:** User code
```python
patcher.add_patches(lora_patches, ...)
patcher.add_object_patch("layer", new_module)
patcher.add_weight_wrapper("weight", transform_func)
patcher.set_injections("key", [injection])
```

---

## Key File Locations Summary

| Purpose | File | Key Lines |
|---------|------|-----------|
| State dict preprocessing | `comfy/supported_models.py` | Per-model `process_*_state_dict()` |
| Model creation | `comfy/supported_models.py` | Per-model `get_model()` |
| Model detection | `comfy/model_detection.py` | detect_models() |
| Model loading | `comfy/model_base.py` | 299-314 (load_model_weights) |
| Model patcher | `comfy/model_patcher.py` | 218+ (class ModelPatcher) |
| Add patches | `comfy/model_patcher.py` | 555-577 |
| Object patch | `comfy/model_patcher.py` | 471-472 |
| Weight wrapper | `comfy/model_patcher.py` | 480-482 |
| Injection | `comfy/model_patcher.py` | 1021-1088 |
| Load entry point | `comfy/sd.py` | 1193-1250 |
| LoRA loading | `comfy/lora.py` | 37-95 |
| ControlNet loading | `comfy/controlnet.py` | 441-448 |

---

## Decision Tree: Which Path to Use?

```
Do you want to:

1. Rename/remap checkpoint keys to match model expectations?
   → Use PATH 1: State Dict Preprocessing

2. Create layers conditionally based on checkpoint content?
   → Use PATH 2: Constructor Injection + Custom Model Class

3. Replace existing modules after loading?
   → Use PATH 3: Object Patching

4. Transform weights on-access (quantization, etc)?
   → Use PATH 4: Weight Wrapping

5. Modify existing parameters with LoRA/adapter style?
   → Use PATH 5: add_patches (if keys exist in model)

6. Intercept model behavior at runtime?
   → Use PATH 6: Model Injection

7. Add completely new parameters not in checkpoint?
   → Combination of PATH 1 + PATH 2
```

---

## Complete Example: Custom Model with Extra Layers

This example implements a model that loads extra parameters from checkpoint:

### Step 1: Model Config
**File:** `comfy/supported_models.py`
```python
class MyCustomModel(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        # ... your config
    }
    
    def process_unet_state_dict(self, state_dict):
        # Rename any keys if needed
        return state_dict
    
    def get_model(self, state_dict, prefix="", device=None):
        # Extract custom parameters from checkpoint
        custom_params = {}
        for key in ["custom_feature.weight", "custom_feature.bias"]:
            if key in state_dict:
                custom_params[key] = state_dict.pop(key)  # Remove from state_dict
        
        # Create custom model class
        out = model_base.MyCustomDiffusion(
            self,
            device=device,
            **custom_params
        )
        return out
```

### Step 2: Model Class
**File:** `comfy/model_base.py`
```python
class MyCustomDiffusion(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None, custom_feature_weight=None, custom_feature_bias=None):
        super().__init__(model_config, model_type, device)
        
        # Conditionally create custom layers if weights provided
        if custom_feature_weight is not None:
            self.custom_feature = torch.nn.Linear(768, 512)
            self.custom_feature.weight = torch.nn.Parameter(custom_feature_weight)
            if custom_feature_bias is not None:
                self.custom_feature.bias = torch.nn.Parameter(custom_feature_bias)
        else:
            self.custom_feature = None
```

### Step 3: Model Detection (Optional)
**File:** `comfy/model_detection.py` or custom loader
```python
# Add your model config to detection logic
# So checkpoints are automatically matched to MyCustomModel config
```

---

## Testing Your Implementation

```python
# 1. Load checkpoint
sd = comfy.utils.load_torch_file("model.safetensors")

# 2. Try to load with your custom model config
from comfy import supported_models
config = supported_models.MyCustomModel({})

# 3. Create model
model = config.get_model(sd, device="cuda")

# 4. Check that extra parameters were loaded
print(model.custom_feature)  # Should not be None if weights existed

# 5. Wrap with patcher and test
patcher = comfy.model_patcher.ModelPatcher(model, "cuda", "cpu")
patcher.load_model()
```

