# Asset Usability Taxonomy — v0.3.1 (General-Purpose Video Asset Screener)

**Source of truth for training a general-purpose usability screening model.**  
This taxonomy is designed to be **model-agnostic and project-agnostic**. It works across different video generation backends (Kling, Luma Dream Machine, Runway Gen-3, Pika, CogVideoX, Stable Video Diffusion, HunyuanVideo, Wan2.2, custom fine-tunes, etc.), prompt styles, durations, and use cases (social shorts, narrative clips, abstract, educational, action, etc.).

**Goal**: Train a multi-task model that predicts structured usability labels (verdict + dimension scores + hard-fail flags) aligned with human preference for "usable as-is in a production pipeline".

**Key principles**:
- Dimensions and flags are **technical + coherence-based**, not tied to any fixed IP, character bible, or specific aesthetic.
- "Style / prompt match" is generalized to **prompt/reference fidelity + internal consistency**.
- Anchor examples must be **diverse** across models, styles (photorealistic, stylized, anime, 3D, illustrative, cyberpunk, fantasy, documentary...), content types, and motion complexity.
- Every label must be explainable and actionable for post-processing or regeneration decisions.

---

## 1. Verdict (top-level label)

| Label | Meaning | Threshold rule |
|-------|---------|----------------|
| **PASS** | Usable as-is in downstream pipeline (editing, compositing, distribution) | All scored dimensions ≥ their gate min **AND** zero hard-fail flags |
| **FIX** | Usable after minor, automated or semi-automated post-production work | Minor technical issues only (color/exposure, crop, stabilize, light inpaint, deflicker). Core content, subject coherence, and prompt fidelity remain intact. "Minor" = fixable without regenerating or heavy manual intervention |
| **REJECT** | Not usable; requires full regeneration or heavy rework | Any hard-fail flag present, or multiple critical dimensions significantly below gate |

**FIX examples (general)**:
- Slight underexposure or flat contrast → FFmpeg color grading / curves
- Minor framing issues → intelligent crop / reframing
- Light temporal flicker or camera shake → deflicker + stabilization
- Isolated small artifacts (not affecting main subject) → targeted inpainting

If the issue requires changing subject identity, major motion re-timing, or fixing severe prompt misalignment → **REJECT**.

---

## 2. Hard-fail flags (closed vocabulary — any one triggers REJECT)

`hard_fail_flags` **MUST** only contain values from this canonical closed set. This fixed vocabulary enables consistent multi-label classification during model training and inference.

| ID                        | Section | Meaning & Trigger Conditions |
|---------------------------|---------|------------------------------|
| `watermark_contamination` | 2.1     | Identifiable watermarks, platform logos, burned-in UI elements, or generation service signatures visible in the asset. |
| `severe_ai_artifact`      | 2.2     | Severe, obvious generation failures: extra/missing limbs or fingers, melted/deformed geometry, floating disconnected elements, severe anatomical distortion, garbled or semantically incorrect on-screen text, major structural collapse. |
| `brief_non_compliance`    | 2.3     | Short but critical deviation from the input prompt or reference that renders the asset unusable (e.g., wrong key object appears even briefly, important action missing or altered in a way that breaks the intended narrative beat). |
| `reference_inconsistency` | 2.4     | Main subject(s) exhibit identity, appearance, clothing, or core visual attributes that drift or conflict with provided reference image(s) or within the clip itself (generalized subject inconsistency; applies to people, animals, objects, vehicles, environments). |
| `legal_copyright_risk`    | 2.5     | Content that poses legal copyright risk: recognizable real persons (celebrity likeness), branded IP, trademarked logos, or protected character designs that are not sufficiently transformative. |
| `privacy_data_risk`       | 2.6     | Content that poses privacy or data protection risk: unconsented real faces in identifiable contexts, personal data leakage, or sensitive biometric information. |
| `safety_policy_risk`      | 2.7     | Content that violates applicable safety policies (e.g., graphic violence, hate symbols, explicit harmful material, self-harm, or other disallowed categories per deployment policy). |
| `delivery_failure`        | 2.8     | Technical delivery or playback failure: corrupted frames, unplayable segments, severe encoding artifacts, missing critical streams, or asset that cannot be properly ingested by downstream systems. |
| `dataset_contamination`   | 2.9     | Clear signs of training data leakage or memorization (exact replication of real copyrighted clips, obvious training-set watermarks, or verbatim reproduction of protected content). |

**Notes**:
- These 9 IDs are the **only allowed values** for the `hard_fail_flags` field in all labeled data and model outputs.
- Borderline or subjective cases should be handled via scored dimensions (especially Prompt Fidelity & Internal Coherence or Temporal Stability) rather than forcing a hard-fail.
- For stylized or abstract content, apply `reference_inconsistency` and physics-related judgments more leniently when documented via `aesthetic_family` tags.
- Every triggered flag must be recorded with the exact ID (for model training) plus a short free-text justification (for human audit and error analysis).

---

## 3. Scored dimensions (0–4 scale)

**Anchor strategy (critical for consistency)**:
Create and maintain a living `taxonomy_anchors/` folder with **2–4 diverse real examples per bin per dimension**. Examples must span:
- Multiple video gen models
- Photorealistic ↔ heavily stylized
- Simple vs complex motion
- Different content categories (character, environment, action, abstract, text-heavy)
- Short social clips ↔ longer narrative takes

This diversity makes the trained screener model truly general-purpose.

### Sharpness / Focus
- **0 (unusable)**: Heavy blur, severe motion blur, or out-of-focus main subject; details unreadable
- **2 (borderline)**: Soft focus overall; main subject readable but lacks crisp detail; acceptable for background or low-priority assets
- **4 (excellent)**: Tack-sharp details across subject and relevant background; fine textures, edges, and small elements clearly resolved
- **Gate min**: 2 (raise to 3 for hero/main-subject assets)

### Exposure / Dynamic Range
- **0 (unusable)**: Severe clipping (pure white highlights or crushed blacks) with critical detail loss; or extremely flat/unusable contrast
- **2 (borderline)**: Recoverable clipping or somewhat flat lighting; details mostly present but lack pop or cinematic quality
- **4 (excellent)**: Full, well-distributed histogram; pleasing contrast with visible detail in both shadows and highlights; appropriate for the intended mood/aesthetic
- **Gate min**: 2

### Composition / Framing
- **0 (unusable)**: Main subject severely cut off, cluttered competing elements, no clear focal point, or fundamentally broken framing (e.g., horizon through head, extreme imbalance)
- **2 (borderline)**: Basic rule-of-thirds or leading lines followed but with distracting elements or imperfect balance
- **4 (excellent)**: Intentional, storytelling-driven composition; effective use of negative space, depth layers, and visual hierarchy that guides attention naturally
- **Gate min**: 2 (raise to 3 for hero assets)

### Prompt Fidelity & Internal Coherence (generalized "style match")
This dimension replaces project-bible-specific matching. It measures:
1. How well the asset follows the **input prompt / reference image(s)** provided to the generator.
2. **Internal consistency** of style, lighting, color palette, and aesthetic treatment throughout the asset (no unwanted drift).

- **0 (unusable)**: Largely ignores prompt or reference; hallucinates major wrong elements; severe internal style/lighting drift within the clip
- **2 (borderline)**: Partially follows prompt but misses key requested elements or shows noticeable internal inconsistencies (e.g., lighting changes mid-clip without motivation, color palette shifts)
- **4 (excellent)**: Strong, faithful adherence to prompt intent and any provided references; fully coherent internal style, lighting, and aesthetic treatment with no unexplained drift
- **Gate min**: 2 (raise to 3 for hero / high-priority assets)

**Anchor guidance**: Include examples from many different aesthetics and prompt complexities. A "4" in cyberpunk neon should look as coherent as a "4" in photorealistic nature documentary.

### Temporal Stability (video only)
- **0 (unusable)**: Severe flickering, morphing, identity drift, or lighting popping across frames; subject or environment changes appearance unpredictably
- **2 (borderline)**: Minor jitter or occasional small morphing during fast/complex motion; mostly holds together
- **4 (excellent)**: Rock-solid frame-to-frame consistency in subject identity, appearance, lighting, shadows, and reflections; no perceptible temporal artifacts
- **Gate min**: 2

### Motion Quality (video only)
- **0 (unusable)**: Unnatural or broken physics (sliding feet, floating objects, impossible interactions), stuttering, lack of weight/momentum, or completely static when motion was expected
- **2 (borderline)**: Mostly believable motion but with awkward timing, insufficient overlapping action, or minor physics issues in complex interactions
- **4 (excellent)**: Natural, physically plausible motion with good weight, arcs, anticipation, and follow-through; motivated and cinematic camera work where applicable
- **Gate min**: 2 (raise to 3 for action-heavy or physics-critical content)

**Note on abstract/stylized content**: For highly abstract or artistic generations, motion_quality and physics expectations can be relaxed via context tags (e.g., `aesthetic_family=abstract_art`). Document this clearly in anchors.

---

## 4. Context tags (multi-label, informational — used for routing, stratified evaluation, and per-category gate logic)

Core universal tags:

- **shot_type**: WS (wide/establishing) / MS (medium) / CU (close-up) / ECU (extreme close-up) / insert (detail) / OTS / POV / tracking
- **asset_role**: hero (primary focus) / supporting / background / plate (clean plate for VFX) / reference / establishing / detail / transition
- **source**: generated (optionally note model family: diffusion, transformer, etc.) / stock / real-shot / hybrid
- **content_category**: character_focus / full_body_action / environment_establishing / object_detail / abstract_motion / text_overlay / multi_subject_interaction / infographic / talking_head
- **aesthetic_family**: photorealistic / stylized_cinematic / anime_2d / 3d_render / illustration / painterly / cyberpunk_noir / fantasy / documentary / abstract_experimental (expand as needed; keep broad buckets for model training)
- **motion_complexity**: static_or_minimal / moderate_natural / high_complexity_physics / chaotic_or_stylized
- **duration_bucket**: very_short (<3s) / short (3-8s) / medium (8-20s) / long (>20s)
- **text_presence**: none / minor / prominent / heavy (affects whether garbled_text flag is relevant)

**Usage**:
- Enable **stratified gates** (e.g., hero + character_focus assets require higher prompt_fidelity and sharpness gates).
- Enable **stratified evaluation** of your trained model (does it perform equally on photorealistic vs anime? On simple vs complex motion?).
- Help downstream pipeline routing (different post-processing LUTs or stabilization strength per aesthetic_family).

---

## 5. Video-specific rules

- **Frame sampling for scoring**:
  - Uniform 1 fps + all scene-change / cut frames (detect via ffmpeg or scenedetect).
  - For very short clips (<4s): sample every 0.5s or all frames.
  - For longer clips: 1 fps is usually sufficient; prioritize scene changes and high-motion segments.

- **Clip-level verdict aggregation** (must be deterministic and documented):
  1. Any sampled frame triggers a hard-fail flag → entire clip = **REJECT**.
  2. Temporal stability and motion_quality → use **worst-frame** score (most conservative).
  3. All other dimensions → use **mean** score across sampled frames.
  4. If any mean or worst-frame score < gate min → consider **FIX** (if minor/technical) or **REJECT** (if severe or prompt_fidelity related).
  5. Record aggregation method and per-frame evidence for auditability.

- **Minimum usable duration**:
  - General recommendation: **≥ 3 seconds** for most social/educational/short-form use cases.
  - Hero narrative or emotional beats: **≥ 5–6 seconds**.
  - Inserts / transitions / abstract: **≥ 1–2 seconds**.
  - Clips below minimum are usually **REJECT** unless they serve a very specific purpose (e.g., perfect looping transition plate).

---

## 6. Ambiguity protocol (for human annotators and future model)

**Priority order when dimensions conflict** (highest first):
1. Any hard-fail flag (exact canonical ID)
2. Prompt Fidelity & Internal Coherence (or `reference_inconsistency` flag)
3. Temporal stability / motion quality
4. Technical dimensions (sharpness, exposure, composition) — most fixable via post-production

**Examples**:
- Excellent prompt fidelity + poor composition → **FIX** (crop/reframe)
- Strong composition + poor prompt fidelity or `reference_inconsistency` → **REJECT** or major rework (usually not "minor FIX")
- Good technicals + severe temporal instability → **REJECT** (hard to fix without regeneration)

**Unsure between FIX and REJECT**:
- Default to **FIX + "needs human review"** flag, **or**
- Require consensus from two independent annotators; disagreement escalates to senior review.
- Never reject on "vibe" alone — must cite specific flag or dimension ≤ 0/1 with evidence.

**Documentation requirement**:
Every **REJECT** must record at least one concrete reason (specific hard-fail flag or dimension score + short justification). This is essential for:
- Training data quality
- Model debugging / error analysis
- Continuous taxonomy improvement

---

## Changelog & Versioning

- **v0.1** — Initial template (project-specific placeholders)
- **v0.2** — Converted to **general-purpose** design (removed IP-specific references, generalized Prompt Fidelity dimension, added diverse anchor strategy and stratified context tags)
- **v0.3** — Canonical closed vocabulary for hard-fail flags (fixed 9-ID set for consistent multi-label classification)
- **v0.3.1** (current) — Added §7 Output schemas (annotation + minimal inference) as the enforceable data contract between labeling, training, and screening stages. Synced `scores` keys to §3 dimension headings.

**Next iteration triggers**:
- After labeling first 500–1000 diverse clips, analyze inter-annotator agreement and model confusion matrix → refine anchors, edge cases, and flag boundary definitions.
- Add or deprecate flag IDs only through formal taxonomy version bumps (closed vocabulary discipline).
- Consider adding a lightweight "overall human preference" scalar (0–100) alongside the structured labels for RLHF-style fine-tuning of generators.

---

## Recommended Implementation Path for General-Purpose Screener Model

1. **Data collection**: Curate or generate diverse clips from multiple video models + real footage. Use the taxonomy + diverse anchors for labeling (human or hybrid MLLM-assisted).
2. **Dataset structure**: Store video + structured JSON label (verdict, per-dim scores, flags, context tags, free-text justification for rejects).
3. **Model architecture ideas**:
   - Video encoder (VideoMAE, InternVideo, or CLIP4Clip style) + multi-task heads (classification for verdict/flags + regression for scores).
   - Or fine-tune a strong video MLLM with structured output for explainability.
4. **Evaluation**: Report per-dimension accuracy/F1, overall verdict accuracy, and stratified performance across aesthetic_family / content_category / motion_complexity.
5. **Production use**: New generated clip → model inference → auto-route to PASS / FIX queue (with suggested fixes) / REJECT (with reasons) + confidence scores.

This taxonomy gives you a **human-aligned, structured, explainable, and scalable** foundation for training a truly general-purpose video asset usability screener — exactly what production pipelines need when working with many different generators and evolving use cases.

---

**References & Further Reading** (from community):
- VBench (comprehensive hierarchical dimensions + human alignment)
- "A Survey of AI-Generated Video Evaluation" (2024/2025) — excellent breakdown of six common error types
- LGVQ / UGVQ datasets and papers (multi-dimensional spatial/temporal/alignment assessment)
- Awesome-Evaluation-of-Visual-Generation and Awesome-Video-Diffusion lists

---

## 7. Output schemas (contract — pipeline must validate against these)

### 7.1 Annotation schema (human labeling + training data)

```json
{
  "asset_id": "",
  "file_path": "",
  "duration_sec": null,
  "verdict": "PASS | FIX | REJECT",
  "hard_fail_flags": [],
  "scores": {
    "sharpness_focus": null,
    "exposure_dynamic_range": null,
    "composition_framing": null,
    "prompt_fidelity_coherence": null,
    "temporal_stability": null,
    "motion_quality": null
  },
  "context_tags": {
    "shot_type": [],
    "asset_role": [],
    "source": [],
    "content_category": [],
    "aesthetic_family": [],
    "motion_complexity": "",
    "duration_bucket": "",
    "text_presence": ""
  },
  "fix_actions": [],
  "needs_human_review": false,
  "reject_reason": "",
  "frame_evidence": [
    { "frame_time_sec": null, "issue": "", "flag_or_dimension": "" }
  ],
  "reviewer": "",
  "review_date": ""
}
```

**Rules**:
- `hard_fail_flags`: only the 9 canonical IDs from §2. Empty array = none.
- `scores`: integers 0–4 or null (null = N/A per dimension applicability). Keys are synced to §3 dimension headings (snake_case).
- If `verdict` = "REJECT": `reject_reason` non-empty **AND** (≥1 flag OR ≥1 score of 0). Enforced by the annotate stage.
- If `verdict` = "FIX": `fix_actions` non-empty.
- `frame_evidence` required for every triggered flag (per §5 auditability rule).
- `context_tags` arrays accept the allowed values defined in §4 (or free-form for new tags during exploration).

### 7.2 Minimal inference output schema (screen stage)

```json
{
  "asset_id": "",
  "verdict": "PASS | FIX | REJECT",
  "confidence": 0.0,
  "hard_fail_flags": [],
  "scores": {},
  "fix_actions": [],
  "primary_reasons": [],
  "needs_human_review": false
}
```

**Rules**:
- `confidence`: 0.0–1.0, calibrated on validation set.
- `primary_reasons`: 1–3 short strings citing canonical flag IDs or dimension names (e.g., "reference_inconsistency", "prompt_fidelity_coherence=1").
- Low-confidence FIX/REJECT boundary cases set `needs_human_review: true` (threshold tuned on validation set and documented in eval report).
- `scores` object may be sparse or contain only the dimensions the model predicts.

---

**Pipeline contract summary**:
- Annotation stage produces full 7.1 records → stored as training/eval dataset.
- Training produces model that outputs 7.2 records.
- Screening stage validates 7.2 output against this schema before routing to PASS / FIX queue / REJECT.
- Any drift between annotation schema and inference schema must be explicitly versioned and documented.
