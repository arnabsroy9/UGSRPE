# UGSRPE Framework — Walkthrough for New Readers

**Uncertainty-Guided Self-Referring Prompt Evolution for One-Shot Polyp Segmentation with SAM**

This document provides an accessible, step-by-step explanation of how UGSRPE works. It is intended for readers new to one-shot segmentation, vision foundation models, or the specific combination of DINOv2 and SAM2. For the formal methodology, ablation studies, and experimental results, see the [paper](./README.md#citation).

The framework is **fully training-free**: no model is fine-tuned, no weights are updated, and no dataset-specific tuning is performed. All hyperparameters remain fixed across the four evaluated benchmarks.

---

## 1. Starting Point — Inputs

The framework begins with two inputs. The first is a **support image**: a single colonoscopy image in which the polyp has already been manually annotated by a clinician. This annotated image–mask pair serves as the sole reference for the entire pipeline. The second is a **query image**: an unseen colonoscopy image in which the system must automatically locate and segment the polyp.

Crucially, the support image is not chosen at random. An automatic support selection mechanism evaluates up to 1,000 candidate images from the dataset. For each candidate, DINOv2 extracts the polyp prototype — a feature vector obtained by average-pooling features within the annotated polyp region. The candidate whose prototype exhibits the highest mean cosine similarity to all other prototypes is selected as the support. This ensures that the chosen reference is maximally representative of the dataset's polyp diversity, providing the broadest possible feature compatibility with unseen query images.

Only **one** annotated image is required per dataset. This is the "one-shot" aspect of the framework.

---

## 2. Feature Extraction — DINOv2 ViT-L/14 (Frozen)

Both the support and query images are resized to 560×560 pixels and passed through a frozen DINOv2 ViT-L/14 encoder. DINOv2 is a self-supervised vision foundation model that produces semantically rich patch-level feature embeddings without requiring any task-specific training.

The encoder divides each image into a 40×40 grid of patches, where each patch corresponds to a 14×14 pixel region. Every patch is represented as a 1,024-dimensional feature vector encoding its semantic content. This yields a total of 1,600 feature vectors per image, denoted $f^s, f^q \in \mathbb{R}^{1600 \times 1024}$ for the support and query respectively. These features form the shared foundation upon which all subsequent modules operate.

The encoder is completely frozen — no fine-tuning or adaptation is performed. The framework relies entirely on DINOv2's pre-trained representations.

---

## 3. Hybrid Correlation-based Prior Generation (CPG)

The CPG module is responsible for producing an initial heat map — referred to as the **prior** — that indicates where the polyp is likely located in the query image. It accomplishes this through five sequential steps.

### Step 1: Cross-Correlation

A cross-correlation matrix is computed between every query patch and every support patch:

$$S_{\text{corr}} = \frac{f^q \cdot (f^s)^T}{\sqrt{D}}$$

where $D = 1024$ is the feature dimension. Each entry $S_{\text{corr}}[i, j]$ measures how similar query patch $i$ is to support patch $j$ in semantic space. The division by $\sqrt{D}$ is a standard scaling factor that prevents the dot product from growing too large as feature dimensionality increases.

This similarity matrix is then multiplied by the vectorised support mask $m_r^s \in \mathbb{R}^{1600 \times 1}$ to produce the initial prior:

$$p_{\text{init}} = S_{\text{corr}} \cdot m_r^s$$

In effect, each query patch receives a score equal to the sum of its similarities to all polyp patches in the support. The prior is then min-max normalised to the $[0, 1]$ range.

The resulting prior highlights regions of the query that semantically resemble the polyp region of the support. However, it is often overly diffuse because normal mucosal tissue and polyp tissue share similar low-level features in DINOv2's representation space.

### Step 2: Prototype-Residual Gate

To address the diffuseness problem, the module computes two prototype vectors from the support image — one representing the polyp (foreground), the other representing everything else (background):

$$p_{\text{fg}} = \text{normalize}\left(\frac{\sum_i f^s_i \cdot m_i}{\sum_i m_i}\right) \qquad p_{\text{bg}} = \text{normalize}\left(\frac{\sum_i f^s_i \cdot (1 - m_i)}{\sum_i (1 - m_i)}\right)$$

These are simply masked averages of the support feature map: the foreground prototype averages features inside the polyp mask, while the background prototype averages features outside it. The result is two compact 1,024-dimensional vectors that summarise what "polyp" and "non-polyp" look like for this particular support image.

For each query patch, a residual score is then computed:

$$r(x) = \text{ReLU}\bigl(\cos(f_x^q, p_{\text{fg}}) - \cos(f_x^q, p_{\text{bg}})\bigr)$$

This score is high for query patches that resemble the polyp more than the background, and near zero (clipped by ReLU) for patches that resemble both equally. Mucosal patches that share texture with both the polyp and the surrounding tissue receive low residual scores and are effectively suppressed.

The residual is then used as a multiplicative gate on the initial prior:

$$p_{\text{gated}} = p_{\text{init}} \cdot \bigl((1 - w) + w \cdot \hat{r}\bigr)$$

where $w = 0.5$ controls the gating strength and $\hat{r}$ is the normalised residual. The $(1 - w)$ term ensures that patches with zero residual are not entirely zeroed out — they are merely down-weighted by 50%. This preserves the prior's spatial coverage while sharpening it toward discriminative regions.

### Step 3: Self-Correlation Refinement

The gated prior is further refined using the query image's own self-similarity structure:

$$S_{\text{self}} = \text{softmax}\left(\frac{\hat{f}^q \cdot (\hat{f}^q)^T}{\tau}\right)$$

where $\hat{f}^q$ denotes L2-normalised query features and $\tau = 0.1$ is a temperature parameter. The softmax normalisation along each row converts raw similarities into a probability distribution: each query patch is now associated with a weighted neighbourhood of semantically similar patches.

The prior is then propagated through this matrix via $p \leftarrow S_{\text{self}} \cdot p$ for $\rho = 1$ round. The effect is that high-scoring patches spread their activation to semantically similar neighbouring patches, filling in gaps within the polyp region and reinforcing coherent clusters. This step exploits the fact that polyp tissue tends to look similar to itself across nearby patches.

### Step 4: Top-Percentile Thresholding

The bottom 70% of prior values are suppressed, retaining only the top 30% most confident activations. This focuses the prior on high-confidence regions and reduces its spatial coverage from approximately 50–60% of the image to 15–25%, which better matches typical polyp sizes in colonoscopy images.

### Step 5: Spatial Smoothing

The thresholded prior is smoothed using a 5×5 Gaussian kernel ($\sigma = 1.5$) at the 40×40 patch level to eliminate grid artefacts, followed by bicubic upsampling to 560×560. A second Gaussian blur (31×31, $\sigma = 10$) and morphological closing (25×25 elliptical kernel) at the full resolution produce a spatially coherent, smooth prior map suitable for downstream prompting.

---

## 4. Scale-Cascaded Prior Fusion (SPF)

A fundamental challenge in one-shot segmentation is that the support image contains a polyp of one particular size, whereas query polyps may be significantly larger or smaller. SPF addresses this scale variation by generating three versions of the support image through lesion-region scaling.

The three scales are: the original support (scale factor ×1.0), a reduced version with the polyp region scaled to ×0.5 (simulating smaller polyps), and an enlarged version scaled to ×1.5 (simulating larger polyps). Each version is processed independently through the CPG module, producing three separate prior maps.

To determine how much weight each scale should receive, the framework performs a **reverse transfer validation**. For each scale, the query features selected by that scale's prior are projected back onto the support image. The degree of overlap between this reverse projection $p_{\text{rev}}$ and the known support mask $m^s$ is measured using Confidence IoU (cIoU):

$$\text{cIoU} = \frac{\sum(\hat{p}_{\text{rev}} \cdot m^s \cdot p_{\text{rev}})}{\sum \hat{p}_{\text{rev}} + \sum m^s - \sum (\hat{p}_{\text{rev}} \cdot m^s)}$$

This is a confidence-weighted variant of standard IoU. The numerator combines the binarised reverse projection $\hat{p}_{\text{rev}}$ (indicating *where* the prior says polyp is), the ground-truth support mask $m^s$ (indicating *where* polyp truly is), and the continuous reverse projection $p_{\text{rev}}$ (indicating *how confident* the prior is at each location). A scale that recovers the correct support mask with high confidence receives a high cIoU; one that drifts or guesses receives a low score.

The three priors are then fused through a cIoU-weighted average:

$$p_{\text{avg}} = \sum_{\text{sz}} \omega_{\text{sz}} \cdot p_{\text{sz}}, \qquad \omega_{\text{sz}} = \frac{\text{cIoU}_{\text{sz}}}{\sum_{\text{sz}} \text{cIoU}_{\text{sz}}}$$

The scale that best explains the query–support relationship contributes most to the final combined prior, while less reliable scales are down-weighted. The result is a single fused prior map that is robust to polyp size variation.

---

## 5. Uncertainty-Guided Multi-Prompt Evolution (UGMPE)

UGMPE constitutes the central innovation of the framework. Rather than committing to a single prompt and relying on SAM2 to produce a correct mask, this module generates multiple diverse prompts, evaluates their collective agreement, and iteratively refines the segmentation based on uncertainty.

### 5a. Diverse Prompt Generation

Seven complementary prompt types are derived from the fused prior map. Each captures a different geometric interpretation of "where the polyp is":

- **Prompt 1 — EDT centre point**: the Euclidean Distance Transform centre of the prior's largest connected component, representing the point maximally distant from the boundary. This identifies the most confident interior point.
- **Prompt 2 — Tight bounding box**: the axis-aligned bounding box enclosing the prior's largest connected component.
- **Prompt 3 — Padded bounding box**: the tight box expanded by 10% on each side to capture boundary regions that may have been clipped by thresholding.
- **Prompt 4 — Combined**: the EDT centre point and tight bounding box provided together, giving SAM2 both positional and extent cues.
- **Prompts 5–6 — Jittered boxes**: the tight box with random ±8% perturbation on each edge, introducing spatial diversity to test sensitivity.
- **Prompt 7 — Multi-point**: three high-confidence foreground points plus two background negative points, providing fine-grained guidance with explicit exclusion zones.

### 5b. Composite Mask Scoring

Each prompt produces three candidate masks from SAM2 (a standard SAM2 behaviour). The best mask per prompt is selected using a composite score:

$$\text{score} = 0.50 \cdot d_{\text{prior}} + 0.30 \cdot s_{\text{sam}} + 0.20 \cdot c_{\text{peak}}$$

The three terms balance complementary criteria:

- $d_{\text{prior}}$ measures the Dice overlap between the candidate mask and the prior's core region. This anchors the mask to the prior's spatial localisation.
- $s_{\text{sam}}$ is SAM2's internal IoU confidence estimate. This leverages SAM2's own assessment of mask quality.
- $c_{\text{peak}}$ is a binary bonus (0 or 1) for whether the candidate mask covers the prior's single highest-confidence pixel. This prevents the selected mask from drifting away from the prior's most certain location.

The 0.50 / 0.30 / 0.20 weights prioritise prior alignment while still leveraging SAM2's confidence and peak-coverage signal.

### 5c. Frequency Aggregation and Uncertainty Estimation

The seven selected masks are aggregated into a pixel-wise frequency map by averaging:

$$f(x) = \frac{1}{N}\sum_{i=1}^{N} M_i(x), \quad N = 7$$

A pixel included in all seven masks receives a frequency of 1.0; a pixel included in only three receives 0.43; a pixel included in none receives 0.0. The frequency map therefore represents the level of agreement among diverse prompt variants.

The uncertainty at each pixel is computed as:

$$U(x) = f(x) \cdot (1 - f(x))$$

This quantity has a useful geometric property: it peaks at 0.25 when exactly half the prompts agree ($f = 0.5$) and is zero at both extremes (full agreement or full disagreement). High uncertainty therefore marks regions where prompt variants disagree — typically along ambiguous boundaries or in textured background areas — and signals where the segmentation is least reliable.

### 5d. Self-Referring Prior Update

If the mean uncertainty within the prior's region of interest exceeds a convergence threshold of $\text{conv\\_tol} = 0.10$, the prior is updated by blending it with the consensus prediction:

$$p_{t+1} = \alpha \cdot p_t + (1 - \alpha) \cdot H_t$$

where $H_t = (f > 0.40)$ is the consensus mask and $\alpha = 0.70$ controls update momentum. The consensus threshold of 0.40 means that pixels agreed upon by at least three of the seven prompts contribute to the updated prior.

This **self-referring** mechanism allows the query image's own segmentation results to refine the prior, progressively reducing dependence on the initial support-derived localisation. The high momentum ($\alpha = 0.70$) ensures stability — the prior cannot be overwhelmed by a single noisy iteration. The loop runs for at most three iterations, terminating early if the mean uncertainty drops below the convergence threshold.

The term "self-referring" reflects that the query image gradually corrects its own prior rather than depending entirely on the support image.

---

## 6. Adaptive Fallback Recovery

In certain cases, the primary pipeline may produce an empty or near-empty mask (fewer than 100 pixels). Rather than accepting a catastrophic failure, a three-tier fallback strategy activates progressively:

**Tier 1.** The frequency consensus threshold is lowered from 0.50 to 0.30, capturing regions where prompt agreement is weaker but still present. This recovers cases where the prompts mostly agreed but did not reach the strict half-majority needed for the default threshold.

**Tier 2.** If Tier 1 still produces an empty mask, the bounding box of the prior's largest connected component is extracted and sent as a single box prompt to SAM2. The candidate mask with the highest IoU confidence is selected. This bypasses the multi-prompt voting entirely and trusts SAM2's box-prompted segmentation directly.

**Tier 3.** If both prior tiers fail, the prior itself is thresholded at progressively lower levels (0.30, 0.20, 0.10) and used as the segmentation mask. This is a last-resort fallback that returns the raw prior as the prediction.

This graduated approach ensures that the pipeline always produces output, with quality degrading gracefully rather than failing entirely.

---

## 7. Grid-Proposal Discovery Fallback

For approximately 2–5% of query images across the evaluated datasets, the cross-correlation features fail to produce a prior that accurately localises the polyp. This occurs when the query polyp looks fundamentally different from the support, or when the UGMPE iterative loop causes the prior to drift away from the correct region.

### 7a. Drift Detection

The system compares the initial prior (before UGMPE modification) against the final segmentation mask. Drift is flagged when either of two conditions holds:

1. The final mask does not cover the initial prior's peak pixel.
2. The overlap between the final mask and the initial prior's core region (defined as pixels with prior value > 0.50) falls below 20%.

If either condition fires, the system concludes that the UGMPE loop has wandered away from where the prior originally said the polyp should be, and switches to discovery mode.

### 7b. Grid Proposals and Feature Scoring

When drift is detected, SAM2 is prompted with a dense 8×8 grid of point prompts, generating approximately 192 candidate masks (three per grid point). Each candidate is scored by DINOv2 feature similarity: the mean feature vector of the masked region is compared to the support's polyp and background prototypes, and the residual similarity score is computed using the same formula as the CPG residual gate.

This reuses the discriminative power of the prototype-residual mechanism (Section 3, Step 2) but applies it to SAM-generated candidate masks rather than to individual patches.

### 7c. Top-K Frequency Voting

The top 10 scoring proposals are aggregated into a frequency map. Pixels appearing in more than 30% of the top proposals form a consensus region. This approach is robust because incorrect proposals — those covering non-polyp objects — tend to scatter spatially and cancel out in the frequency map, whereas correct proposals concentrate their signal in the polyp region.

The consensus region is then refined with a final box prompt to SAM2, producing the discovery-mode segmentation.

The grid fallback fires only when drift is confirmed, so it does not interfere with cases where the primary pipeline succeeds.

---

## 8. Output

The final output is a binary segmentation mask at 1,024×1,024 resolution, indicating polyp versus non-polyp for every pixel. The mask is produced entirely without training or fine-tuning — both DINOv2 and SAM2 operate in their pre-trained, frozen states.

Base inference time is approximately 2.5 seconds per query image on dual T4 GPUs, increasing to a maximum of 4.0 seconds in the 2–5% of cases where the grid-proposal discovery fallback is triggered.

---

## Summary

The full pipeline can be summarised as five sequential stages:

| Stage | Module | Purpose |
|---|---|---|
| 1 | Feature Extraction (DINOv2) | Convert images to semantic patch features |
| 2 | Hybrid CPG | Produce a discriminative prior heat map |
| 3 | Scale-Cascaded Prior Fusion | Handle size variation across query polyps |
| 4 | UGMPE | Refine segmentation via uncertainty-guided multi-prompt evolution |
| 5 | Adaptive Fallback + Grid Discovery | Recover from primary-pipeline failures |

The key insight is that **prompt disagreement provides a practical uncertainty signal**, and that **self-referring updates allow the query image to refine its own prior** without requiring additional annotations. Combined with adaptive recovery mechanisms, this enables robust one-shot polyp segmentation across diverse imaging conditions with fixed hyperparameters.

For complete experimental results and comparison with prior work, see the [paper](./README.md#citation).
