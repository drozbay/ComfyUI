# ComfyUI Model Loading Investigation Report
## Can you load models with "unexpected" layers without checkpoint modification?

### Executive Summary

**YES - ComfyUI provides MULTIPLE native mechanisms to load models with unexpected/extra layers without modifying checkpoint files.**

The answer is nuanced: While `load_state_dict(strict=False)` allows loading models with unexpected keys, ComfyUI has several higher-level systems specifically designed to inject, patch, and extend model parameters AFTER loading. These are the "ComfyUI-native" ways to handle additional weights.

---

## Key Findings

### 1. **STATE DICT LOADING WITH `strict=False`**

**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_base.py:307`

```python
m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
if len(m) > 0:
    logging.warning("unet missing: {}".format(m))
if len(u) > 0:
    logging.warning("unet unexpected: {}".format(u))
```

**Key Point:** All model loading uses `strict=False`, which returns both `missing` and `unexpected` keys. The unexpected keys are simply logged as warnings but don't prevent model loading.

**Used Everywhere:**
- `comfy/model_base.py:307` - Main diffusion model loading
- `comfy/sd.py:259` - CLIP loading
- `comfy/controlnet.py:442` - ControlNet loading
- `comfy/clip_vision.py:69` - Vision model loading

### 2. **STATE DICT PREPROCESSING - TRANSFORMING "UNEXPECTED" INTO "EXPECTED"**

**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/supported_models_base.py:90-98`

```python
def process_clip_state_dict(self, state_dict):
    state_dict = utils.state_dict_prefix_replace(state_dict, {k: "" for k in self.text_encoder_key_prefix}, filter_keys=True)
    return state_dict

def process_unet_state_dict(self, state_dict):
    return state_dict
```

**Pattern:** EVERY model has `process_*_state_dict()` methods that transform the checkpoint BEFORE loading. This is where "unexpected" keys can be renamed, filtered, or restructured to match what the model expects.

**Real Examples:**

#### A. Stable Zero123 - Inspects and Injects Parameters
**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/supported_models.py:376-387`

```python
class Stable_Zero123(supported_models_base.BASE):
    required_keys = {
        "cc_projection.weight": None,
        "cc_projection.bias": None,
    }
    
    def get_model(self, state_dict, prefix="", device=None):
        # Extract weights from state_dict and PASS THEM to model init
        out = model_base.Stable_Zero123(
            self, 
            device=device, 
            cc_projection_weight=state_dict["cc_projection.weight"], 
            cc_projection_bias=state_dict["cc_projection.bias"]
        )
        return out
```

**Pattern:** Inspect state_dict, extract specific keys, and pass them as constructor arguments. The model can then use these to conditionally create or initialize layers.

#### B. SDXL - Dynamic Model Type Detection
**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/supported_models.py:197-217`

```python
def model_type(self, state_dict, prefix=""):
    if 'edm_mean' in state_dict and 'edm_std' in state_dict:  # Playground V2.5
        self.latent_format = latent_formats.SDXL_Playground_2_5()
        self.sampling_settings["sigma_data"] = 0.5
        return model_base.ModelType.EDM
    elif "edm_vpred.sigma_max" in state_dict:
        self.sampling_settings["sigma_max"] = float(state_dict["edm_vpred.sigma_max"].item())
        return model_base.ModelType.V_PREDICTION_EDM
    elif "v_pred" in state_dict:
        return model_base.ModelType.V_PREDICTION
```

**Pattern:** Inspect unexpected keys to determine model capabilities and modify model config accordingly. The "unexpected" keys inform the model how to behave.

#### C. State Dict Key Remapping
**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/supported_models.py:50-62`

```python
def process_clip_state_dict(self, state_dict):
    k = list(state_dict.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            state_dict[y] = state_dict.pop(x)  # Rename "unexpected" key
    return state_dict
```

**Pattern:** Rename checkpoint keys to match expected model structure.

---

### 3. **MODEL PATCHER SYSTEM - POST-LOAD PARAMETER INJECTION**

**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py`

#### A. `add_patches()` - Parameter Modification After Loading

**Location:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py:555-577`

```python
def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
    with self.use_ejected():
        p = set()
        model_sd = self.model.state_dict()
        for k in patches:
            if key in model_sd:  # ONLY patches existing keys
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches
        return list(p)
```

**Key Limitation:** `add_patches()` only works with keys that ALREADY EXIST in the model's state_dict. It modifies existing parameters but cannot add entirely new ones.

**Use Case:** LoRA and fine-tuning weights.

#### B. `add_object_patch()` - Replace Entire Module Objects

**Location:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py:471-472`

```python
def add_object_patch(self, name, obj):
    self.object_patches[name] = obj
```

**What it does:** Replaces an entire submodule with a new object. Used for:
- Changing dtype: `add_object_patch("manual_cast_dtype", dtype)`
- Replacing any nested module reference

**Example from codebase:**
```python
def set_model_compute_dtype(self, dtype):
    self.add_object_patch("manual_cast_dtype", dtype)
```

**Key Point:** This lets you REPLACE entire submodules with modified versions that have additional parameters.

#### C. `set_injections()` - Runtime Behavior Modification

**Location:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py:1021-1022`

```python
def set_injections(self, key: str, injections: list[PatcherInjection]):
    self.injections[key] = injections
```

**What it does:** Injects custom behavior into the model at runtime:
- `inject_model()` - Applies injections
- `eject_model()` - Removes injections

**Use Case:** Temporary modifications to model behavior without changing weights.

#### D. Weight Wrapper Patches - Modify How Weights Are Accessed

**Location:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/model_patcher.py:480-482`

```python
def add_weight_wrapper(self, name, function):
    self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]
```

**What it does:** Wraps weight retrieval with custom functions. Used for:
- Quantization
- Dynamic weight calculation
- Parameter transformation on-the-fly

---

### 4. **LORA/CONTROLNET PATTERNS - HOW EXTERNAL WEIGHTS ARE INTEGRATED**

**File:** `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/lora.py` and `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/controlnet.py`

#### Pattern: Create Patches for Existing Keys

**LoRA Example - `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/lora.py:37-95`**

```python
def load_lora(lora, to_load, log_missing=True):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        # ... check for various adapter types ...
        # Returns a patch_dict mapping MODEL_KEYS to ADAPTER_OBJECTS
        patch_dict[to_load[x]] = adapter
    return patch_dict
```

**Key Pattern:**
1. LoRA files reference model parameters (layer names)
2. A key mapping translates LoRA key names to actual model parameter names
3. Patches are created for each existing parameter
4. Patches are applied via `model_patcher.add_patches()`

**ControlNet Example - `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/controlnet.py:441-448`**

```python
def controlnet_load_state_dict(control_model, sd):
    missing, unexpected = control_model.load_state_dict(sd, strict=False)
    
    if len(missing) > 0:
        logging.warning("missing controlnet keys: {}".format(missing))
    
    if len(unexpected) > 0:
        logging.debug("unexpected controlnet keys: {}".format(unexpected))
```

**Key Pattern:** ControlNet uses `strict=False` loading and logs unexpected keys but continues.

---

### 5. **CONDITIONAL LAYER CREATION - MODELS THAT ADD LAYERS BASED ON STATE_DICT**

**Pattern Example: Flux Model - `/mnt/h/AppsDir/git/ComfyBase/ComfyUI/comfy/ldm/flux/model.py:43-90`**

```python
def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
    super().__init__()
    # ... setup ...
    self.guidance_in = (
        MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) 
        if params.guidance_embed  # Conditional creation
        else nn.Identity()
    )
    
    if final_layer:  # Another conditional
        self.final_layer = LastLayer(...)
```

**Pattern:** Model __init__ can take parameters that control which layers are created. Combined with state_dict inspection, you can dynamically enable/disable layers.

---

## COMPREHENSIVE SOLUTION: Making "Unexpected" Weights "Expected"

### Strategy 1: State Dict Preprocessing (Cleanest)

**Use:** When checkpoint has extra keys that need renaming or restructuring.

```python
class CustomModel(supported_models_base.BASE):
    def process_unet_state_dict(self, state_dict):
        # Rename unexpected keys to expected ones
        if "custom_extra_layer.weight" in state_dict:
            state_dict["model.extra_layer.weight"] = state_dict.pop("custom_extra_layer.weight")
        return state_dict
```

**Pros:**
- Happens before model creation
- Model sees only "expected" keys
- No runtime overhead
- Clean separation of concerns

**Cons:**
- Requires modifying ComfyUI codebase
- Must handle all variants yourself

### Strategy 2: Constructor Injection (State Dict Inspection)

**Use:** When you need to extract parameters from checkpoint and pass to model constructor.

```python
class CustomModel(supported_models_base.BASE):
    def get_model(self, state_dict, prefix="", device=None):
        # Inspect for extra parameters
        extra_weight = state_dict.get("custom_feature.weight", None)
        extra_bias = state_dict.get("custom_feature.bias", None)
        
        # Create model with these parameters
        out = model_base.CustomDiffusionModel(
            self,
            device=device,
            extra_weight=extra_weight,
            extra_bias=extra_bias
        )
        return out
```

**Pros:**
- Model __init__ controls layer creation
- Can conditionally create layers based on checkpoint contents
- Flexible and powerful

**Cons:**
- Requires custom model classes
- Model's __init__ must be flexible

### Strategy 3: Object Patching (Post-Load Modification)

**Use:** When you need to replace entire submodules with enhanced versions.

```python
# After model loading
patcher.add_object_patch("diffusion_model.custom_module", new_module)
```

**Pros:**
- Happens after model is loaded
- Can swap entire modules
- Works with existing models

**Cons:**
- Requires creating the new module object
- Module paths must be correct
- Can break if model structure changes

### Strategy 4: Weight Wrapping (On-Access Transformation)

**Use:** When you want to dynamically compute or transform weights at access time.

```python
def weight_wrapper_function(weight):
    # Transform weight on access
    return apply_custom_operation(weight)

patcher.add_weight_wrapper("diffusion_model.custom_layer.weight", weight_wrapper_function)
```

**Pros:**
- Lazy evaluation
- Can create complex transformations
- No extra memory until accessed

**Cons:**
- Overhead on every weight access
- More complex to debug
- State is not saved

### Strategy 5: Model Injection (Behavior Modification)

**Use:** When you want to intercept and modify model behavior at runtime.

```python
from comfy.patcher_extension import PatcherInjection

class CustomInjection(PatcherInjection):
    def inject(self, patcher):
        # Modify patcher.model at runtime
        pass
    
    def eject(self, patcher):
        # Restore original state
        pass

patcher.set_injections("custom_key", [CustomInjection()])
```

**Pros:**
- Full control over model behavior
- Can add/remove features dynamically
- Clean separation

**Cons:**
- Complex to implement
- Requires understanding of PatcherInjection API
- Harder to debug

---

## Recommended Approach for Your Use Case

If you want to load models with completely new parameters that aren't in the original class definition:

### Option A: Custom Model Class + State Dict Inspection (RECOMMENDED)

1. **Extend `supported_models_base.BASE`** with your custom model config
2. **Override `process_unet_state_dict()`** to handle checkpoint structure
3. **Override `get_model()`** to inspect state_dict and pass extra params to custom model
4. **Create custom model class** that accepts extra parameters in __init__

```python
# In supported_models.py
class MyCustomModel(supported_models_base.BASE):
    unet_config = { /* ... */ }
    
    def process_unet_state_dict(self, state_dict):
        # Transform checkpoint keys if needed
        return state_dict
    
    def get_model(self, state_dict, prefix="", device=None):
        # Extract custom parameters from checkpoint
        custom_params = {}
        for key in ["custom_layer1", "custom_layer2"]:
            if key in state_dict:
                custom_params[key] = state_dict[key]
        
        # Create model with custom params
        out = model_base.MyCustomDiffusionModel(
            self,
            device=device,
            **custom_params
        )
        return out
```

### Option B: Object Patching After Load

```python
# After loading model normally
custom_module = create_custom_module(...)
model_patcher.add_object_patch("diffusion_model.custom_layer", custom_module)
```

### Option C: Hybrid - Remapping in State Dict

```python
def process_unet_state_dict(self, state_dict):
    # Detect custom layers and remap them
    for key in list(state_dict.keys()):
        if "custom_feature" in key:
            # Remap to a standard layer name that your model expects
            new_key = key.replace("custom_feature", "feature")
            state_dict[new_key] = state_dict.pop(key)
    return state_dict
```

---

## Key Code Locations

| Component | File | Line | Purpose |
|-----------|------|------|---------|
| Load state dict | `model_base.py` | 307 | Uses `strict=False` |
| State dict processing | `supported_models_base.py` | 90-98 | Hook to transform checkpoint |
| Model config lookup | `sd.py` | 1198+ | Matches checkpoint to model config |
| LoRA/patch system | `model_patcher.py` | 555-577 | Applies weight patches |
| Object patching | `model_patcher.py` | 471-472 | Replace module objects |
| Weight wrapping | `model_patcher.py` | 480-482 | Transform weight access |
| Injection system | `model_patcher.py` | 1069-1088 | Runtime behavior modification |

---

## Important Limitations

1. **`add_patches()` only works with existing keys** - You cannot create entirely new parameters with LoRA-style patching. The patch target must already exist in the model.

2. **Model detection happens early** - The checkpoint is inspected to select the model class before detailed state_dict processing.

3. **No automatic discovery of extra layers** - Extra layers won't be included in `state_dict()` unless they're part of the nn.Module hierarchy.

4. **Strict mode is never used for main loading** - All core loading uses `strict=False`, so unexpected keys are always allowed but ignored.

---

## Conclusion

**The answer is YES - you can load models with extra layers WITHOUT modifying checkpoint files.**

The native ComfyUI way is:
1. **Leverage state dict preprocessing** (`process_*_state_dict()`) to transform the checkpoint
2. **Inspect state_dict in `get_model()`** to extract custom parameters
3. **Pass custom parameters to model constructors** to conditionally create layers
4. **Use object patching** if you need to swap entire modules post-load

This is how ComfyUI handles model variants, custom models, and extensions. You don't need to touch the checkpoint file itself - you just need to create custom model configuration classes that understand your extended model structure.
