# Common shortcomings in AI-generated images

A reference guide to typical failure modes in current image diffusion and generative models—useful for training goals, caption design, and evaluation. For SDX-specific mitigations and config hooks, see [MODEL_WEAKNESSES.md](MODEL_WEAKNESSES.md) and [QUALITY_AND_ISSUES.md](QUALITY_AND_ISSUES.md).

---

## Photorealism and general image quality

### Surface detail, skin, and tangents

Models often excel at basic color theory and broad object recognition (shapes, designs, characteristics of nouns) but miss finer cues: realistic skin texture (pores, subtle freckles, veins, micro-imperfections, natural translucency), which can read as plastic or overly smooth.

**Tangents** are a persistent composition problem: insufficient overlap or separation between objects hurts clarity. Elements may meet awkwardly at edges, flattening depth and causing confusing mergers instead of deliberate overlaps or clean separations that guide the eye and reinforce 3D space.

### Spatial relationships, support, and interaction

Models lack a reliable sense of **weight, contact, and support**. Figures can appear to hover rather than sit or lean; chairs show no compression under load; clothing may not bunch, crease, or drape believably against surfaces. Hands often “float” near objects instead of gripping with plausible tension. Feet may not plant on ground planes; object–object relationships can feel weightless or disconnected.

### Lighting, global illumination, and environmental cohesion

Beyond simple shading, models struggle with **bounce light and color bleeding** (e.g. a red shirt subtly tinting a nearby white wall). Elements are often treated as isolated assets rather than parts of one lit volume. **Shadows** may disagree with a single dominant light direction, breaking immersion in complex scenes. In 3D-leaning renders, weak or missing **ambient occlusion** (contact shadows in crevices) can make objects feel ungrounded.

### Line work, brushwork, and edges (where visible)

Where line or edge structure matters, outputs can show **uniform line weight** and over-defined contours instead of dynamic thickness, tapering, and “lost and found” edges that dissolve into shadow—contributing to a hyper-processed or “deep-fried” look compared to intentional traditional work.

### Anatomy beyond the “hand problem”

The classic hand issue extends to **deeper structure**: impossible limb bends (“noodle limbs”), dubious elbows and knees, and weak attachment logic (e.g. neck–shoulder–trapezius transitions). Missing skin detail further undermines realism even when pose is roughly correct.

### Composition, visual flow, and design intent

Strong **center bias** and **horror vacui** (filling every corner) are common. Models underuse negative space, rule-of-thirds balance, and background **leading lines** that steer narrative focus. The result can lack deliberate rest and guided flow.

### Materials, texture logic, and aging

Surface labels (metal, fabric, skin) may be recognized while **placement logic** fails: rust or grime in random patches rather than moisture traps and wear patterns; weathering without story. **Subsurface scattering** (light through skin, wax, marble) is often weak or absent, so subjects read as painted plastic or clay. Folds, hair strands, and liquids may ignore gravity, wind, and interaction.

### Narrative consistency and functional logic

Scenes can look polished but **ahistorical**: generic scratches on armor instead of plausible battle damage; hems without mud or wear from terrain. Architecture and machinery may look impressive but **non-functional** (pipes to nowhere, impossible stairs). That weakens the sense of a lived-in world.

### Perspective, foreshortening, and depth

Standard views often work better than **strong foreshortening**: limbs or objects thrust toward the camera may lose scale or warp; deep stacks of overlap can merge instead of keeping readable silhouettes and the “coming at you” clarity artists build with overlap and scale.

### Facial nuance and emotion

Broad categories (happy, sad, angry) are easier than **micro-expressions** (a sarcastic lip quirk, Duchenne eye crinkle, coordinated muscle groups). Eyes may default to a flat stare; faces can be **over-symmetrical**, pushing uncanny valley. Natural facial asymmetry is often missing.

### Color grading, value, and discipline

Defaults often skew **high-contrast, high-saturation** (“rainbow” palettes). **Limited palettes** (e.g. Zorn-style restraint) are harder to hold; stray hues or noise can break mood. In grayscale, **value structure** may be chaotic if color is treated as a substitute for designed light/shadow rather than a separate compositional tool.

### Additional gaps (often overlooked)

- **Legible text and typography** — distorted letters, wrong spelling, unstable layout.
- **Small repeating structure** — buttons, patterns, stitches inconsistent across the image.
- **Fluids and particles** — splashes, smoke, motion fabric that feels stiff.
- **Multi-subject scenes** — background figures with inconsistent faces or proportions.
- **Complex prompts** — secondary elements drifting from intent.

---

## Digital art, screen-native, and hybrid workflows

**Raster painting (Photoshop, Procreate, Clip Studio, etc.)** — Models often default to overly smooth, airbrushed, or “AI-blended” surfaces instead of believable **brush economy** (hard vs soft edges, stroke direction, intentional texture). Midtones turn to mud; edges lose intention.

**Concept art, matte painting, photobash** — Weak **perspective and lighting unity** across collaged or painted regions; cutout look, scale drift, or conflicting color cast. Design reads as generic “grey sculpt” rather than a clear focal idea.

**Pixel art and retro game graphics** — **Subpixel blur**, inappropriate gradients, and inconsistent **pixel scale** break the medium. Good pixel work needs crisp tiling, palette discipline, and deliberate dither or AA.

**Vector, flat design, icons, UI illustration** — Wobbly paths, inconsistent stroke rules, and accidental photoreal or 3D leakage. Icons need clean silhouettes and consistent geometric discipline.

**Hand-painted game assets / stylized 3D** — **Albedo vs lighting** confusion, muddy texture paint, and **texel density** that jumps across UV islands. Reads as noise mud instead of directed hand-painted direction.

These map to mitigation ids `digital_painting`, `concept_matte_digital`, `pixel_digital`, `vector_flat_digital`, `stylized_game_digital` in `config/defaults/ai_image_shortcomings.py` (included in `auto` when keywords match, and in `all` with other non–2D-anime packs).

---

## Stylized 2D (anime, manga, cartoons, comics, cel-shade, watercolor, graphic novel)

Many failures trace back to the same roots—**no true physics**, **pattern completion over intent**, **data bias**—but stylized work breaks in ways tuned to **convention and exaggeration**, not photographic accuracy.

### Style consistency and drift

**Style drift** within one image or across a set: e.g. “anime” mixing cel-shading with painterly patches, or Western proportions with anime eyes. **Character consistency** suffers (hair tips, eye highlights, folds, facial structure varying between generations)—painful for sheets, comics, and animation keys. Popular aesthetics may dominate while rarer strip or indie looks stay hard to hit.

### Line art and edges in 2D

Intentional **variable weight**, tapering, silhouette clarity, and selective lost edges are often replaced by uniform, crisp, or extraneous outlines—mechanical rather than hand-drawn. In anime/manga this shows up as messy hair strands, awkward garment contours, or lines that lack energy.

### Shading, light, and cel logic

Stylized 2D uses **simplified, often hard-edged** shadow design. AI may contradict a single light, mix soft gradients into flat-color expectations, or place “SSS-like” plastic sheen where the style calls for flat reads. Stylized bounce/balance, when present, may be inconsistent; **palette discipline** is easy to break with sneaked extra hues or noise.

### Anatomy, proportion, and exaggeration

Even with non-real proportions, **rules must stay consistent**: eye size/highlight conventions, head–body ratios, limb taper. Errors feel louder because stylization amplifies them. Noodle limbs, floating hands, and muddy overlap in dynamic or foreshortened poses remain common.

### Composition, negative space, and storytelling

Center bias and horror vacui hurt **panel-like** or **illustration-first** layouts that rely on breathing room and leading lines. Faces may fall back to exaggerated emotion masks without subtle asymmetry or coordinated cues that sell “alive” cartoon acting.

### Texture, medium, and material in 2D

Medium simulation (watercolor bleed, paper tooth, ink wash, flat vectors, halftones) often becomes **uniform smoothness** or **random busy texture**. Wear and dirt lack narrative placement; motion effects (hair, cartoon water) can feel stiff.

### Extra challenges specific to 2D

- **Prompt adherence** — requests for children’s book or classic comic looks may veer toward photorealism or glossy CGI.
- **Multi-character / sequential work** — inconsistency compounds.
- **Text** — still weak for bubbles, titles, and signage.
- **Over-polished kitsch** — airbrushed, generically vibrant outputs vs. raw hand-drawn energy.

---

## Technical / 3D-render aesthetics (when the target looks like CG)

When the desired look is **high-end 3D**, models may invent **melting or fused geometry**, bad **UV logic** (stretched or swimming patterns), and inconsistent **focal length / camera grammar** within one frame—fine at thumbnail scale, weak under scrutiny.

---

## Root causes (summary)

Limitations cluster around: **no grounded physical simulation**; **correlational pixel/statistical modeling** rather than explicit artistic decisions; **training distribution and bias**; and **weak long-horizon consistency** for text, counts, and spatial relations. Mitigation in practice combines **better data and captions**, **prompt and negative design**, **inference tooling** (where the codebase provides it), and **post-production**—not a single architectural switch.

---

## In this repo

- **Registry and keyword detection:** `config/defaults/ai_image_shortcomings.py` (`config.ai_image_shortcomings`).
- **Artist-first medium guidance:** `config/defaults/art_mediums.py` (`config.art_mediums`) for traditional, digital, photography, and anatomy/proportion packs.
- **Style-domain + artist/game guidance:** `config/defaults/style_guidance.py` (`config.style_guidance`) for anime/comic/editorial/concept/game/photo language + artist/game-name stabilization cues.
- **Sampling:** `python sample.py ... --shortcomings-mitigation auto` (match prompts to categories) or `all` (full photoreal pack; add `--shortcomings-2d` for stylized 2D packs).
- **Sampling (medium/anatomy):** `python sample.py ... --art-guidance-mode auto|all --anatomy-guidance lite|strong` (optional `--no-art-guidance-photography`).
- **Sampling (style domains):** `python sample.py ... --style-guidance-mode auto|all` (optional `--no-style-guidance-artists`).
- **Inference framing:** `python sample.py ... --resize-mode center_crop|saliency_crop` (optional `--resize-saliency-face-bias`) to reduce stretched/non-semantic framing when target aspect differs.
- **Training:** `python train.py ... --train-shortcomings-mitigation auto|all --train-art-guidance-mode auto|all` with optional `--train-shortcomings-2d`, `--train-anatomy-guidance`, `--no-train-art-guidance-photography`.
- **Training (style domains):** `python train.py ... --train-style-guidance-mode auto|all` (optional `--no-train-style-guidance-artists`).
- **Offline manifests:** `python -m scripts.tools normalize_captions ... --shortcomings-mitigation auto --art-guidance-mode auto --style-guidance-mode auto` (plus optional 2D/anatomy toggles).
