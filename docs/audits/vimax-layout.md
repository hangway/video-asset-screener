# ViMax on-disk layout — source audit

Verified read-only against a shallow clone of
[HKUDS/ViMax](https://github.com/HKUDS/ViMax) (MIT), commit
`0253853cd0a8218b121e4297b243b11d4a454bff` (2026-06-30). All citations are
`file:line` in that commit. The clone was not modified.

## 1. Shot directory structure (script2video)

`Script2VideoPipeline` addresses everything relative to its `working_dir`:

```
{working_dir}/
  shots/
    {idx}/                                  # idx = ShotDescription.idx, 0-based int
      video.mp4                             # the generated clip
      shot_description.json                 # ShotDescription dump (prompt text)
      first_frame.png                       # i2v conditioning frame (always)
      last_frame.png                        # only when variation_type in {medium, large}
      first_frame_selector_output.json      # frame-selector diagnostics
      transition_video_from_shot_{p}.mp4    # optional camera-transition clip
      new_camera_{c}.png                    # optional new-camera keyframe
  character_portraits_registry.json         # portrait registry (see §3)
  character_portraits/
    {char_idx}_{identifier}/front|side|back.png
  final_video.mp4                           # concat of shots/*/video.mp4
```

Citations (`pipelines/script2video_pipeline.py`):

- `shots/{idx}/video.mp4`: L311 (concat read), L496 (write target).
- `shots/{idx}/shot_description.json`: L792 (written/read as a
  `ShotDescription` model dump — `model_validate(json.load(f))` at L797).
- `shots/{idx}/first_frame.png`: L334, L507; `last_frame.png` only for
  `variation_type in ["medium", "large"]`: L508-509 (also the conditioning
  inputs passed to the video generator, L506-509 → L514-518).
- `transition_video_from_shot_{parent}.mp4`: L359; `new_camera_{c}.png`:
  L377; `{frame_type}_selector_output.json`: L551.
- `final_video.mp4`: L303-315 (MoviePy concat of every shot's `video.mp4`).

## 2. Shot descriptions / prompts

`interfaces/shot_description.py` defines the persisted `ShotDescription`
model (the whole model is dumped to `shot_description.json`):

- `idx: int` — 0-based shot index (also the directory name).
- `visual_desc: str` — the vivid shot prompt; character identifiers appear
  in angle brackets (`<Alice>`).
- `motion_desc: str` — motion/dialogue prompt; **this + `audio_desc` is the
  actual video-generation prompt** (`script2video_pipeline.py` L514:
  `prompt=shot_description.motion_desc + "\n" + shot_description.audio_desc`).
- `ff_desc` / `lf_desc: str` — first/last-frame descriptions;
  `ff_vis_char_idxs` / `lf_vis_char_idxs: List[int]` — visible characters.
- `variation_type: Literal["large","medium","small"]` — drives whether a
  `last_frame.png` exists at all (see §1).
- `cam_idx`, `is_last`, `variation_reason`, `audio_desc` — metadata.

## 3. Character-portrait registry — VERIFIED

`{working_dir}/character_portraits_registry.json`
(`script2video_pipeline.py` L228-231 read, L648-666 write; declared type
`Dict[str, Dict[str, Dict[str, str]]]`, L201/L644). Value shape, from the
return of `generate_portraits_for_single_character` (L719-736):

```json
{
  "<identifier_in_scene>": {
    "front": {"path": ".../character_portraits/{idx}_{identifier}/front.png",
              "description": "A front view portrait of <identifier>."},
    "side":  {"path": "...side.png",  "description": "..."},
    "back":  {"path": "...back.png",  "description": "..."}
  }
}
```

Portrait files live at
`{working_dir}/character_portraits/{char.idx}_{char.identifier_in_scene}/{front,side,back}.png`
(L682, L686, L696, L705). The registry stores **absolute-or-run-relative
paths as generated** — a copied/moved working_dir can carry stale `path`
strings, so an adapter must fall back to the `character_portraits/`
directory structure when registry paths do not resolve.

Caveat: `idea2video_pipeline.py` L160 builds the portrait dir with
`safe_path_component(identifier)` (`utils/text.py` L4) while script2video
L682 uses the raw identifier — directory names may be sanitized differently
between the two pipelines; the registry `path` fields are authoritative when
they resolve.

## 4. idea2video nesting

`Idea2VideoPipeline` composes per-scene script2video runs:
`{working_dir}/scene_{idx}/` becomes each scene's `working_dir`
(`pipelines/idea2video_pipeline.py` L227-233), so shots live at
`{working_dir}/scene_{s}/shots/{idx}/video.mp4`. The portrait registry stays
at the TOP-LEVEL working_dir (L87) and is shared across scenes. Top level
also holds `characters.json` (L63), `story.txt` (L121), `script.json`
(L141), `final_video.mp4` (L245).

Adapter consequence: a ViMax dir may contain `shots/` directly (script2video)
or one level down under `scene_{n}/` (idea2video); both must be discovered.

## 5. Clip characteristics (for calibration)

- Duration: the Doubao/Seedance generator accepts `duration: Literal[5, 10]`
  default **5 s** (`tools/video_generator_doubao_seedance_yunwu_api.py`
  L36/L193); the Veo generator defaults to **8 s**
  (`tools/video_generator_veo_google_api.py` L38). So shots are typically
  5-8 s — short but ABOVE our §5 `<4 s` dense-sampling threshold in the
  default configs.
- Conditioning: every shot is i2v conditioned on `first_frame.png`, and
  medium/large-variation shots also on `last_frame.png` (L506-509) — the
  first/last ~second carries the transition between generated content and
  its conditioning targets, making head/tail edge stability the right lens.
- Shots are later concatenated as-is into `final_video.mp4` (L311), so a
  bad head/tail second on one shot lands directly in the final cut.

## 6. Calibration assessment (batch item 3)

Assessed against the characteristics in §5. Conclusion: **no global default
changes; one preset file** (`configs/vimax.yaml`).

1. **§5 sampling for 5-8 s clips.** At the default 1 fps a 5 s shot yields
   only ~5 sampled frames — thin for worst-frame dimension pooling and for
   head/tail baselines (body ≈ 3 frames). The §5 rule already prescribes
   0.5 s sampling below 4 s; ViMax shots sit just above that threshold. The
   preset raises `ingest.short_clip_threshold_sec` to 9.0 so 5-8 s ViMax
   shots get the dense 0.5 s sampling (§5's own short-clip rule, applied at
   a preset-scoped boundary — the taxonomy's "<4 s" is a floor for generic
   footage, and densifying sampling never loosens any gate).
2. **Edge-stability windows.** Defaults (1.0 s window, 3.0 sigma) target
   generic footage. For first/last-frame-conditioned i2v shots the edges are
   exactly where conditioning artifacts land, and with 0.5 s sampling a 1 s
   window holds ~2 frames — workable. The preset keeps the 1.0 s window but
   lowers `edge_outlier_sigma` to 2.5: edge anomalies are more probable a
   priori in this population, so slightly higher sensitivity is justified;
   trims route FIX (never REJECT), so the failure mode of a false positive
   is a cheap human-reviewable trim suggestion.
3. **Min-duration handling.** `ABSOLUTE_MIN_DURATION_SEC` guards sub-0.5 s
   deliveries; ViMax's 5/10 s (Seedance) and 8 s (Veo) presets sit far above
   it. No change — the rule is already correct for this population.
4. **Freeze detection.** i2v generations occasionally emit a near-still
   shot; the freezedetect defaults (noise -60 dB, min 1 s) plus the 90%
   coverage rule already catch the fully-frozen case. Unchanged.
5. **Reference consistency.** Portrait registry (§3) gives per-character
   reference views (front/side/back) — exactly the per-subject layout our
   `ReferenceIndex` models. The preset does not change the similarity
   threshold: portraits are white-background full-body renders while shots
   are styled scenes, so absolute cosine levels depend on the encoder; the
   `min_reference_similarity` default stays until measured on real ViMax
   output (structural verification only in this repo — see the
   DeterministicEncoder caveat in notes.md).
