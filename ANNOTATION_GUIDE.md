# Annotation guide

This guide is a practical companion to taxonomy.md. The taxonomy remains the
normative source of truth, including the exact verdicts, six scored dimensions,
nine hard-fail flags, and schema rules.

## Verdicts

- PASS: usable as-is in editing, compositing, or distribution.
- FIX: usable after minor post-production that does not require regenerating the
  core subject, action, identity, or narrative beat.
- REJECT: requires regeneration or major manual reconstruction.

REJECT must cite concrete evidence. FIX must contain actionable fix_actions.
PASS cannot contain hard-fail flags. Do not reject on vibe alone. If FIX versus
REJECT is ambiguous, mark the record for human review and explain uncertainty.

## Common FIX examples

- crop or reframe while preserving shot intent;
- correct exposure or mild white balance;
- stabilize a small amount of camera shake;
- deflicker a short brightness oscillation;
- trim a bad head or tail;
- perform minor inpainting that does not change the core subject or action.

Each FIX action should say what to change and where, such as “trim the first
0.4 seconds to remove the unstable entrance.”

## Common REJECT evidence

- identity drift that changes the subject across the clip;
- severe anatomy failure that cannot be repaired locally;
- object melting or persistent geometry collapse;
- broken physics that undermines the requested action;
- a prompt-critical action is missing;
- a watermark or delivery failure that prevents intended use.

Cite frame numbers or time ranges and name the existing taxonomy flag when one
applies. Do not invent a new hard-fail ID.

## Scored dimensions

Score the six taxonomy dimensions independently using the 0-4 rubric in
taxonomy.md. A low score is evidence, not a substitute for a concrete reason.
Use context tags for shot type, role, aesthetic family, and motion complexity
when known. Leave genuinely unavailable dimensions as the schema permits rather
than guessing.

## Evidence checklist

Before saving an annotation, confirm the verdict follows the PASS gate and
hard-fail rules; cite frames or time ranges; use only canonical flags; make FIX
actions reproducible; record prompt/reference availability honestly; flag
ambiguous cases for a second human review; and check privacy, safety, copyright,
and delivery risks.

Human review remains required for subjective quality, identity, and prompt
fidelity. The deterministic encoder is a technical baseline, not an identity
verifier. See taxonomy.md for the complete rubric.
