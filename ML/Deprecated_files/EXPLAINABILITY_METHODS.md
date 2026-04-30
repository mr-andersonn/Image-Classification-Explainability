# Explainability Methods for Image Classification

This document explains the four explainability methods implemented for analyzing the fish classification model.
# This is pure chaptgpt, and not quality controlled
---

## 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

### What is it?
Grad-CAM uses the gradients flowing into the final convolutional layer to understand which regions of the image were important for a particular classification decision.

### How it works:
1. **Forward pass**: Image goes through the network to get a prediction
2. **Backward pass**: Compute gradient of the target class score with respect to the feature maps in the final convolutional layer
3. **Weight pooling**: Average the gradients across spatial dimensions to get importance weights for each channel
4. **Weighted combination**: Multiply feature maps by their importance weights and sum them
5. **ReLU & normalization**: Apply ReLU (keep only positive contributions) and normalize to [0, 1]

### Strengths:
- ✅ **Fast**: Single forward and backward pass
- ✅ **Class-discriminative**: Can explain any class, not just the predicted one
- ✅ **Localization**: Shows precise spatial regions the model focuses on
- ✅ **Uses model's actual logic**: Based on real gradients from the network

### Weaknesses:
- ❌ **Requires gradients**: Won't work with black-box models
- ❌ **Coarse resolution**: Limited by the final convolutional layer resolution (7×7 for MobileNetV2)
- ❌ **Single layer**: Only looks at the last convolutional layer, might miss earlier features

### Best used for:
- Quick visualization of what regions matter
- Understanding spatial attention patterns
- Comparing which parts of an image activate different classes

---

## 2. Integrated Gradients

### What is it?
Integrated Gradients attributes the prediction to input features by accumulating gradients along a path from a baseline (usually black image) to the actual input image.

### How it works:
1. **Define baseline**: Usually an all-black image (no information)
2. **Create path**: Generate interpolated images from baseline to actual input (e.g., 0%, 25%, 50%, 75%, 100%)
3. **Compute gradients**: Calculate gradient at each point along the path
4. **Integrate**: Average all gradients and multiply by the difference (input - baseline)
5. **Attribution map**: Sum across color channels to get pixel importance

### Strengths:
- ✅ **Theoretically grounded**: Satisfies axioms (sensitivity, implementation invariance)
- ✅ **Smooth attributions**: More refined and less noisy than vanilla gradients
- ✅ **Pixel-level precision**: Full resolution (224×224) attribution map
- ✅ **Path-based**: Considers the entire journey from "nothing" to "something"

### Weaknesses:
- ❌ **Computationally expensive**: Requires 30-50+ forward/backward passes
- ❌ **Requires gradients**: Needs model access, won't work with black-box models
- ❌ **Baseline dependent**: Results can vary based on choice of baseline
- ❌ **Slower for multiple classes**: Need to run separately for each class

### Best used for:
- High-quality explanations when you have time
- Research and detailed analysis
- When you need mathematically rigorous attributions

---

## 3. LIME (Local Interpretable Model-agnostic Explanations)

### What is it?
LIME explains predictions by approximating the complex model with a simple, interpretable model (linear regression) in the local neighborhood of the prediction.

### How it works:
1. **Segment image**: Divide image into "superpixels" (coherent regions)
2. **Generate perturbations**: Create thousands of variations by randomly hiding/showing superpixels
3. **Get predictions**: Run all perturbed images through the model
4. **Weight by similarity**: Closer perturbations get higher weights
5. **Fit linear model**: Train a Ridge regression to predict model output from superpixel presence
6. **Extract importance**: Linear model coefficients show which superpixels matter most

### Strengths:
- ✅ **Model-agnostic**: Works with ANY classifier (black-box models included!)
- ✅ **Interpretable**: Explains in terms of image regions humans can understand
- ✅ **"What-if" analysis**: Shows what happens when parts are removed
- ✅ **No gradient access needed**: Only needs prediction API

### Weaknesses:
- ❌ **Slow**: Requires 500-1000+ model predictions per explanation
- ❌ **Stochastic**: Different runs give slightly different results (randomness in sampling)
- ❌ **Segmentation dependent**: Quality depends on superpixel algorithm
- ❌ **Approximation**: Linear model might not perfectly capture complex model behavior
- ❌ **Very slow for all classes**: 9 classes × 1000 samples = 9000 predictions

### Best used for:
- Black-box models where you don't have gradient access
- When you want human-interpretable explanations (regions, not pixels)
- Debugging predictions by understanding "what if I removed this part?"

---

## 4. Occlusion Sensitivity

### What is it?
Occlusion Sensitivity measures how the prediction changes when different parts of the image are systematically hidden (occluded). It's one of the most intuitive explainability methods - if covering a region drops the confidence significantly, that region must be important.

### How it works:
1. **Get baseline**: Predict on the original image
2. **Slide occlusion patch**: Move a gray square patch across the image (like a sliding window)
3. **Predict each occluded version**: For each position, predict with that patch covered
4. **Measure drop in confidence**: Calculate how much the target class probability decreased
5. **Create sensitivity map**: Map showing which regions, when occluded, caused the biggest drops

### Strengths:
- ✅ **Highly intuitive**: Easy to explain to anyone - "when we cover this, prediction drops"
- ✅ **Model-agnostic**: Works with any classifier
- ✅ **No gradients needed**: Only needs prediction API
- ✅ **Direct causality**: Shows direct cause-and-effect relationships
- ✅ **Robust**: Not affected by gradient saturation or vanishing gradients

### Weaknesses:
- ❌ **Very slow**: Requires hundreds of forward passes (one per patch position)
- ❌ **Patch size dependent**: Results vary based on occlusion patch size
- ❌ **Not pixel-precise**: Limited by patch granularity (trade-off: smaller patches = more patches)
- ❌ **Occlusion artifact**: Gray patches might create unnatural patterns the model never saw during training
- ❌ **Extremely slow for all classes**: 9 classes with 64 patches = 576 forward passes

### Best used for:
- Presenting to non-technical audiences (very easy to understand)
- Validating gradient-based methods
- Black-box models where you can only query predictions
- When you want to show "what happens when we remove X"

---

## Comparison Table

| Method | Speed | Resolution | Requires Gradients | Model-Agnostic | Deterministic | Intuitive |
|--------|-------|------------|-------------------|----------------|---------------|-----------|
| **Grad-CAM** | ⚡ Fast | 🟡 Low (7×7) | ❌ Yes | ❌ No | ✅ Yes | 🟡 Medium |
| **Integrated Gradients** | 🐢 Slow | ✅ High (224×224) | ❌ Yes | ❌ No | ✅ Yes | 🟡 Medium |
| **LIME** | 🐌 Very Slow | 🟡 Medium (superpixels) | ✅ No | ✅ Yes | ❌ No | ✅ High |
| **Occlusion** | 🐌🐌 Extremely Slow | 🟡 Patch-based | ✅ No | ✅ Yes | ✅ Yes | ✅✅ Very High |

---

## When to Use Each Method

### Use Grad-CAM when:
- You need **quick** explanations for many images
- You want to see **where** the model is looking
- You're working with **convolutional neural networks**
- You want to compare attention across different classes

### Use Integrated Gradients when:
- You need **high-quality**, **pixel-precise** attributions
- You're doing **research** or need **rigorous** explanations
- You have **computational resources** and time
- You want **smooth**, **less noisy** gradient-based explanations

### Use LIME when:
- You have a **black-box model** (no gradient access)
- You want **human-interpretable** explanations (regions, not gradients)
- You're explaining to **non-technical stakeholders**
- You want to understand **"what-if" scenarios** (what if this region wasn't there?)

### Use Occlusion when:
- You need the **most intuitive** explanation possible
- You're presenting to **completely non-technical audiences**
- You want to **validate** other methods
- You have **time** and computational resources
- You want **direct causal evidence** ("hiding X caused Y% drop")

---

## Observations from Fish Classification

Based on our experiments with the fish classification model:

### Misclassifications provide the most insight:
- **True class: Striped Red Mullet** → **Predicted: Red Mullet (50.45%)**
- All four methods help understand why the model confused these similar fish

### Grad-CAM:
- Quickly shows the model focuses on the fish body and distinctive features
- Coarse resolution sometimes misses fine details

### Integrated Gradients:
- Provides finer detail, highlighting specific edges and texture patterns
- Shows the model uses both color and shape information

### LIME:
- Segments the fish into interpretable parts (head, body, tail)
- Shows which superpixels positively/negatively contribute to classification
- Slower but more interpretable to humans unfamiliar with neural networks

### Occlusion:
- Most intuitive - clearly shows that covering the fish body drops confidence
- Validates that the model isn't using background features
- Slower but provides clear cause-and-effect evidence

---

## Implementation Files

- `gradcam_single_image.py` - Single image Grad-CAM
- `gradcam_all_classes.py` - 3×3 grid of Grad-CAM for all 9 classes
- `integrated_gradients_single_image.py` - Single image Integrated Gradients
- `integrated_gradients_all_classes.py` - 3×3 grid of IG for all 9 classes
- `lime_single_image.py` - Single image LIME
- `lime_all_classes.py` - 3×3 grid of LIME for all 9 classes
- `occlusion_single_image.py` - Single image Occlusion Sensitivity
- `occlusion_all_classes.py` - 3×3 grid of Occlusion for all 9 classes
- `gather_image.py` - Utilities for finding random and misclassified images

---

## References

1. **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **Integrated Gradients**: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
3. **LIME**: Ribeiro et al. "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (KDD 2016)
4. **Occlusion Sensitivity**: Zeiler & Fergus "Visualizing and Understanding Convolutional Networks" (ECCV 2014)
