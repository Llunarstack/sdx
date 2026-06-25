# SDX Website Design Brief

Image-generation platform design reference for SDX (not UI code). Each page includes a **detailed image-gen prompt** for mockups, concept art, or marketing visuals.

---

## Brand & Logo — SDX

**Name:** SDX — short, technical, memorable (diffusion / studio / eXperiment).

**Logo concept:** A minimal mark combining a **diffusion spiral** and a **layout grid**.

- Primary form: rounded-square app icon, 1024×1024 export, soft radial gradient deep indigo `#1a1a2e` → violet `#6c5ce7` with electric cyan `#00d4ff` bloom at 15% opacity in lower-right.
- Center glyph: three concentric arcs (grainy/noisy outer ring → crisp inner ring) that resolve into a **2×2 box grid** occupying the lower-right quadrant of the mark — represents regional box prompting and spatial control.
- Wordmark: **SDX** in geometric sans (Satoshi / General Sans / similar to Ideogram). Letter-spacing −2%. The **X** has a 12° diagonal cut through the right stroke (cross-attention metaphor).
- Favicon: grid-only variant at 32px; arcs dropped for legibility.
- Motion: on hover, arcs denoise from film grain to sharp over 400ms ease-out; grid boxes fade from 0% → 20% opacity staggered 50ms each.
- Dark-mode first. Light mode: white `#fafafa` field, indigo `#4c3aff` glyph, no gradient.

**Taglines:** “Layout-aware diffusion.” · “Prompt with precision.” · “Generate with structure.”

### Logo image-gen prompt

> App icon and wordmark design for “SDX”, an AI image generation platform. Rounded square icon on dark indigo-violet gradient background. Center symbol: three concentric semicircular arcs suggesting noise transforming into a clean signal, merging into a 2×2 grid of four small squares in the lower-right quadrant, like a layout composition tool. Minimal flat vector style, subtle cyan glow accent, no 3D bevel. Beside the icon, wordmark “SDX” in modern geometric sans-serif, tight kerning, the letter X has a subtle diagonal slash cut. Premium tech startup aesthetic, Figma/Dribbble quality, dark mode, 4K, crisp edges, purple and cyan color palette only, no photographs, no clutter.

---

## Global Design Language

| Element | Specification |
|--------|----------------|
| **Viewport** | Desktop primary 1440×900; content max-width 1280px centered in canvas area |
| **Chrome** | Left sidebar 240px (64px collapsed), right inspector 320px, center fluid |
| **Surfaces** | Base `#0d0d12`, elevated `#16161f`, hover `#1e1e2a`, border `#2a2a38` |
| **Accent** | Primary violet `#6c5ce7`, secondary cyan `#00d4ff`, success `#2ecc71`, warning `#f39c12` |
| **Type** | UI 13–14px/1.4, prompts 15–16px/1.5, labels 11px uppercase tracking 0.06em muted `#9aa0b5` |
| **Radius** | Cards 12px, buttons 8px, inputs 8px, modals 16px |
| **Shadow** | `0 8px 32px rgba(0,0,0,0.45)` on modals only |
| **Icons** | Lucide-style 1.5px stroke, 20px default |
| **Imagery** | Masonry thumbs 1:1 default; blur-up skeleton `#1a1a24` → fade image |
| **Inspiration** | PixAI density for feeds; Ideogram clarity for create; Linear.app polish for settings |

**Sidebar nav items (icon + label):** Home · Create · Edit · Explore · Styles · Characters · Benchmarks · Taste · Workflows · Docs · Settings (bottom-pinned).

---

## Page 1 — Home / Landing

**Purpose:** Convert cold visitors; communicate differentiation (layout control, character lock, quality stack) without login.

**Anatomy:**
1. **Sticky header (64px):** SDX logo left; nav links Docs, Explore, GitHub; right: “Sign in” ghost button + “Start creating” filled violet CTA.
2. **Hero (100vh min):** Full-bleed background = slow crossfade of 5 curated generations (parallax 8%). Foreground left-aligned copy block max 560px:
   - Eyebrow pill: `v11 · Layout-aware diffusion`
   - H1: “Structure your imagination.”
   - Subhead (18px muted): “Draw regions. Lock characters. Refine automatically. SDX is diffusion built for composers—not gamblers.”
   - CTAs: primary “Start creating →”, secondary “Browse gallery”
   - Micro trust row: GitHub stars, open-source badge, “No credit card”
3. **Hero visual right:** Floating UI card mockup showing box layout on a landscape image (cyan/violet boxes labeled “sky”, “figure”, “foreground”).
4. **Feature strip (5 cols):** Icon + title + 2-line desc each — Box layout · Character sessions · Auto-refine · Style genomes · Regional inpaint.
5. **Comparison section:** Split panel “Plain txt2img” (muddy composition, wrong object placement) vs “SDX box layout” (clean regions, readable scene). Caption: “Same prompt. Different control.”
6. **Model cards row:** 3 cards — SDXL preset, Anime, Photoreal — each with sample thumb, bullet features, “Use in Create” link.
7. **Workflow diagram:** 4 steps — Prompt → Layout → Generate → Refine — horizontal timeline with icons.
8. **Footer (4 col):** Product, Resources, Community, Legal. Discord + GitHub social icons.

**States:** Hero carousel loading skeleton; CTA hover glow `0 0 24px rgba(108,92,231,0.4)`.

### Page 1 image-gen prompt

> Full landing page UI mockup for “SDX” AI image generator website, desktop 1440px, dark theme. Sticky top navigation with SDX logo (diffusion spiral + grid icon), links, violet “Start creating” button. Hero section: left side large headline “Structure your imagination”, subtext about regional prompts and character lock, two call-to-action buttons. Right side floating product screenshot showing image generation canvas with colored bounding boxes labeled sky, character, foreground on a cinematic landscape. Below hero: five feature cards in a row with icons. Then split comparison section: left messy AI image badly composed, right clean well-composed AI image with visible layout boxes overlay. Modern SaaS marketing page, PixAI meets Ideogram aesthetic, indigo and cyan accents on charcoal background, subtle glassmorphism, professional typography, no lorem ipsum, high fidelity Figma export style, 4K.

---

## Page 2 — Create (main generator)

**Purpose:** Primary txt2img / img2img workspace. Highest traffic page.

**Anatomy:**
1. **Top bar (56px):** Breadcrumb `Create / New`; model dropdown “SDXL · best.pt”; aspect pills `1:1` `4:5` `3:2` `16:9` `Custom`; seed field with lock icon; steps `50` and CFG `7.5` as compact numeric steppers; queue indicator `0 jobs`.
2. **Center column:**
   - **Preview stage (min 480px tall):** Empty state = dotted border, icon, “Your image appears here”, 3 rotating tips. Populated = image with optional box overlay toggle. Bottom toolbar: download, upscale, send to Edit, copy seed, fullscreen.
   - **Prompt composer:** Tab `Prompt` | `Negative`. Large textarea 4 rows, placeholder “Describe the scene… Use (emphasis) or [de-emphasis].” Below: token count, lint chips (“⚠ negation detected: no glasses”, “✓ count: 3 people”).
   - **Generate bar:** Batch `1–4` stepper, est. time `~18s`, primary **Generate** button (violet, 120px wide), keyboard hint `⌘↵`.
3. **Right inspector (320px, tabs):**
   - **Layout:** Mini canvas 1:1 mirroring aspect ratio; tools: draw box, select, delete; region list with color swatch, name field, per-region prompt textarea, sketch upload thumb; “Import JSON layout” link; “Add global prompt” field.
   - **Style:** Holy-grail preset cards (balanced, photoreal, anime, aggressive); style pack dropdown; style genome slider `chaos 0.35`; LoRA slots ×3.
   - **Advanced:** Frontier toggle + serendipity dial; per-region CADS toggle; guidance schedule dropdown; scheduler `ddim` / `euler`.
   - **Character:** Session dropdown “None · Yuki · Marcus”; ref image strip (3 slots); “Lock prompt additions” textarea; negative lock field.
4. **Bottom drawer (collapsed default):** Negative prompt, dissect refs, control image upload.

**Interactions:** Drag box corners with snap to thirds; double-click region to edit prompt inline; live wireframe overlay on preview when Layout tab active; Generate pulses while running with step progress `Step 24/50`.

### Page 2 image-gen prompt

> Detailed UI mockup of AI image generation “Create” workspace, app name SDX, dark mode desktop 1440px. Three-column layout: narrow left sidebar navigation with icons, large center canvas showing a generated fantasy portrait with three colored rectangular bounding boxes overlaid (cyan, violet, amber) each with small text labels, below canvas a prompt text area with sample text about a woman in a forest, emphasis syntax hints. Top bar with model selector, aspect ratio pills 1:1 4:5 16:9, seed and steps controls. Right panel 320px with tabs “Layout”, “Style”, “Advanced”, “Character” — Layout tab active showing list of regions with mini prompt fields and color swatches, draw tools. Bottom purple Generate button. Ideogram-style regional prompting interface combined with PixAI generator density, charcoal UI, violet and cyan accents, crisp Figma-quality components, subtle borders, professional product design, 4K.

---

## Page 3 — Edit / Inpaint

**Purpose:** Region fix, img2img, mask painting, adherence debugging.

**Anatomy:**
1. **Top bar:** Back to Create; mode pills `Inpaint` `Img2img` `Fix region`; undo/redo; zoom 50–200%.
2. **Center canvas:** Source image full width; mask layer in red 40% overlay where painted; optional box region outlines from layout JSON in cyan. Brush cursor circle. Layer toggles: Image / Mask / Boxes / Heatmap.
3. **Left tool rail (48px):** Brush, eraser, box select, pan; brush size slider 4–128px; hardness slider.
4. **Right panel:**
   - **Fix region dropdown:** lists region names from layout file (`subject`, `background`, `sky`).
   - **Regional prompt override** textarea (pre-filled from box).
   - **Strength** slider 0–1 (img2img), **Inpaint mode** `MDM` | `Legacy`.
   - **Adherence heatmap** toggle — when on, preview shows inferno colormap overlay where cross-attention landed.
   - **Generate fix** CTA (amber accent to distinguish from Create).
5. **Filmstrip bottom:** Original | Mask | Result v1 | Result v2 — click to compare.

**States:** Empty = “Upload image or send from Create”; heatmap legend gradient bar.

### Page 3 image-gen prompt

> AI image editing and inpainting UI mockup, SDX app, dark theme. Center large image of a photorealistic street scene with semi-transparent red mask painted over a storefront sign area. Left vertical toolbar with brush, eraser, selection tools and brush size slider. Top mode tabs: Inpaint, Img2img, Fix region. Right sidebar: dropdown “Fix region: storefront_sign”, prompt override text field “neon sign reading OPEN”, strength slider, MDM inpaint mode toggle, “Show adherence heatmap” toggle enabled with orange-red heatmap overlay visible on part of the image. Bottom filmstrip comparing original, mask, and result thumbnails. Charcoal interface, cyan box outlines, amber generate button, professional creative tool aesthetic like Photoshop meets Ideogram editor, 1440px desktop, 4K Figma mockup.

---

## Page 4 — Explore / Gallery

**Purpose:** Discovery, remix culture, community feed (PixAI-style).

**Anatomy:**
1. **Header:** Search bar full-width “Search prompts, styles, users…”; sort `Trending` `New` `Top`.
2. **Filter chips (horizontal scroll):** All · Photoreal · Anime · Illustration · Layout-heavy · Character · Text-in-image · Frontier.
3. **Masonry grid:** 4 columns @ 1440px; card = image (hover scale 1.02), gradient scrim bottom, truncated prompt 2 lines, username `@pixelnova`, seed copy icon. Hover overlay buttons: Remix · Upscale · Variation · Save ♡.
4. **Modal on click:** Large image left; right meta panel — full prompt (copy button), negative, model, steps, seed, layout JSON download, “Open in Create”.
5. **Infinite scroll** loader at bottom = 3 pulsing skeleton cards.

**Card aspect mix:** 1:1, 4:5, 16:9 varied for visual rhythm.

### Page 4 image-gen prompt

> Social gallery feed page for AI art platform SDX, dark mode, Pinterest/PixAI style masonry grid of diverse AI generated images — anime character, photoreal portrait, typography poster with text, fantasy landscape. Each card shows image thumbnail, two lines of prompt text, username, small icons. Top search bar and horizontal filter chips: Photoreal, Anime, Illustration, Layout-heavy. Hover state on one card showing overlay buttons Remix, Upscale, Save. Right side optional detail drawer open on one image with full prompt text and seed metadata. Charcoal background #0d0d12, subtle card borders, violet accent on active filter, clean sans typography, infinite scroll aesthetic, 1440px wide desktop UI mockup, 4K, high detail.

---

## Page 5 — Styles & Genomes

**Purpose:** Browse, invent, and apply style genomes; export galleries.

**Anatomy:**
1. **Header:** “Style Genomes” + “Invent new” violet button; search; view toggle grid/list.
2. **Featured carousel:** 3 large hero cards — auto-scrolling style packs with 3 sample images each.
3. **Grid cards (280px):** Genome name “Neon Noir 07”; DNA string preview monospace `palette:#0ff,#f0f;grain:0.2;line:ink`; 2×2 thumb quad; palette swatches row (5 circles); tags `cinematic` `high-contrast`; actions: **Apply** · **Compare** · **Export HTML**.
4. **Compare mode (split view):** Same prompt, slider A|B between two genome outputs; genome selectors below.
5. **Invent drawer:** Chaos level slider, mode `normal` `wild`; “Generate 3 candidates” — shows loading helix animation.

### Page 5 image-gen prompt

> Style genome library UI for SDX AI art platform, dark theme. Grid of style cards each showing 2x2 grid of sample images, title “Neon Noir 07”, monospace DNA code string, row of five color palette swatches, tags cinematic and high-contrast, Apply button. Top featured carousel with three large style packs. Split-screen compare mode visible on right: same portrait prompt with before/after slider between two different color grades. “Invent new” button violet. Charcoal surfaces, cyan and purple accents, creative director tool vibe, 1440px desktop, Figma-quality component library, 4K mockup.

---

## Page 6 — Characters

**Purpose:** Persistent character identity across sessions (character lock).

**Anatomy:**
1. **List view (left 360px):** Character cards — avatar circle, name “Yuki”, lock badge, last used `2h ago`; + New character card dashed border.
2. **Editor (right):**
   - Name, description
   - **Prompt additions** (locked text): `silver hair, amber eyes, mole under left eye, red scarf`
   - **Negative lock:** `different hair color, aged up, extra fingers`
   - **Reference images:** 4-up upload grid with dissect preview
   - **Last outputs timeline:** horizontal scroll of 8 thumbs with seed under each
   - Sticky footer: **Generate with Yuki** (opens Create preloaded) · Save session JSON
3. **Empty state:** Illustration + “Create a character session to keep identity consistent across generations.”

### Page 6 image-gen prompt

> Character management UI for SDX AI generator, dark mode. Two-panel layout: left list of character cards with circular avatars, names Yuki and Marcus, selected state violet border. Right editor panel for character “Yuki”: text fields for prompt additions describing silver hair amber eyes red scarf, negative prompt lock field, grid of four reference photo thumbnails of anime-style character, horizontal timeline of recent generated images. Large violet button “Generate with Yuki”. Session lock iconography, consistency-focused character sheet tool, charcoal UI, clean typography, visual novel / game asset pipeline aesthetic, 1440px desktop mockup, 4K.

---

## Page 7 — Benchmarks & Quality

**Purpose:** Power users — checkpoint comparison, regression tracking.

**Anatomy:**
1. **Header:** “Benchmark Suite” · Run config dropdown `default suite` · **Run benchmark** (shows queue).
2. **Leaderboard table:** Columns — Rank, Checkpoint, Composite score, Text OCR, Count accuracy, Anatomy, Sparkline 7d trend. Sortable. Row expand → per-case breakdown.
3. **Charts row:** Line chart composite over time (from benchmark_history); bar chart per prompt case.
4. **Prompt cases panel:** List `text_sign`, `people_count`, `anatomy_fullbody` with expected constraints badges.
5. **Actions footer:** Export CSV · Export JSON · Mine DPO pairs · Append to history.

**Colors:** Score ≥0.8 green, 0.6–0.8 amber, <0.6 red.

### Page 7 image-gen prompt

> Data dashboard UI for AI model benchmarking, SDX platform, dark theme. Large table leaderboard comparing three model checkpoints with columns Rank, Composite score, OCR accuracy, Count accuracy, sparkline trend graphs in purple. Above table line chart showing score over time. Side panel listing prompt test cases: text_sign, people_count, anatomy_fullbody with badges. Top bar with Run benchmark button and suite selector. Developer analytics aesthetic like Grafana meets Linear, charcoal background, green amber red score colors, monospace numbers, 1440px desktop, 4K Figma mockup, no photographs.

---

## Page 8 — Taste / Preferences

**Purpose:** Personal preference learning — likes/dislikes → DPO training pairs.

**Anatomy:**
1. **Header:** “Taste Profile” · progress `47 ratings` · **Export DPO pairs** CTA.
2. **Split pane 50/50:**
   - **Liked** (green header): masonry of liked images, prompt snippet, score slider 0–1 on hover.
   - **Disliked** (muted red header): same layout, thumbs down icon.
3. **Rating mode toggle:** Grid select vs swipe (Tinder card stack with Like/Dislike buttons).
4. **Pair preview drawer:** Shows proposed win/lose pair from same prompt; confirm/export.
5. **Profile JSON preview** collapsible monospace block.

### Page 8 image-gen prompt

> Personal taste training UI for SDX AI platform, dark mode. Split screen: left column “Liked” with grid of beautiful AI portraits and landscapes green accent header, right column “Disliked” with lower quality images red accent header. Center floating card swipe interface like Tinder for rating images. Top bar “Taste Profile” with Export DPO pairs button and rating count. Bottom drawer showing win vs lose image pair comparison with same prompt label. RLHF training data curation tool aesthetic, charcoal UI, green and muted red section colors, violet primary buttons, 1440px desktop, 4K mockup.

---

## Page 9 — Workflows (Comfy bridge)

**Purpose:** Bridge power users to ComfyUI node graphs; export CLI settings.

**Anatomy:**
1. **Header:** Import JSON · Export from last run · Paste CLI flags.
2. **Center:** Read-only node graph canvas (pan/zoom) — nodes `SDXLoadCheckpoint`, `CLIPTextEncode` ×2, `SDXSampler`, `SaveImage` connected with colored wires (Comfy-style but SDX branded node headers).
3. **Right JSON editor:** Syntax-highlighted workflow JSON with copy button.
4. **Bottom strip:** Last export timestamp; mapping note “Map SDXLoadCheckpoint to your Comfy DiT bridge node.”
5. **CLI paste area:** `sample.py` flags mirrored as form fields with sync toggle.

### Page 9 image-gen prompt

> ComfyUI-style node workflow editor UI branded for SDX, dark theme. Center canvas with connected nodes: Load Checkpoint, CLIP Text Encode, SDX Sampler, Save Image, violet and cyan node headers, colored connection wires on dark grid background. Right panel syntax-highlighted JSON workflow. Top import export buttons. Bottom CLI flags text area showing sample.py command. Power user automation tool aesthetic, node-based programming interface, charcoal background, 1440px desktop, 4K Figma mockup.

---

## Page 10 — Docs & Learn

**Purpose:** In-app onboarding; reduce support burden.

**Anatomy:**
1. **Left doc nav (tree):** Getting started · Box layout JSON · Regional prompting · Frontier · Holy grail · Character sessions · CLI reference.
2. **Content area:** MDX-style prose, code blocks with copy, inline screenshots at 80% width.
3. **Interactive embed — Prompt diff:** Two textareas side by side; diff highlights `only_in_a` cyan, `only_in_b` violet, shared gray.
4. **“Try it” chips:** Open Create with this example preloaded.
5. **Search** `⌘K` command palette overlay for doc search.

### Page 10 image-gen prompt

> Documentation page UI inside SDX AI app, dark theme. Left sidebar documentation tree navigation: Getting started, Box layout, Frontier, Character sessions. Main content area with heading “Regional Box Prompting”, prose text, code block showing JSON layout example with syntax highlighting. Below interactive two-column prompt diff tool with highlighted differing words in cyan and purple. Top search command palette overlay. Developer docs aesthetic like Vite or Stripe docs, charcoal background, clean typography, embedded screenshots of generator UI, 1440px desktop, 4K mockup.

---

## Page 11 — Account / Settings

**Purpose:** Defaults, API (future), cache, notifications.

**Anatomy:**
1. **Sections (vertical nav):** General · Models · Generation defaults · Notifications · API keys (locked) · Appearance.
2. **General:** Display name, email, theme `Dark` `Light` `System`.
3. **Models:** Checkpoint path browser, default preset `sdxl`, VRAM hint badge.
4. **Generation defaults:** steps, CFG, holy-grail preset, auto-refine count, negative prompt template.
5. **Notifications:** Email on queue complete toggle; browser push toggle.
6. **Danger zone:** Clear cache, reset taste profile — red outline buttons.

### Page 11 image-gen prompt

> Settings page UI for SDX AI application, dark mode, Linear app style. Left section navigation: General, Models, Generation defaults, Notifications. Main panel showing form fields: default checkpoint path, preset dropdown SDXL, theme selector dark light system, sliders for default steps and CFG, toggle switches for email notifications. Clean minimal forms, charcoal surfaces, violet toggle active state, subtle dividers, 1440px desktop settings aesthetic, 4K Figma mockup.

---

## Mobile (375×812)

**Bottom nav (56px + safe area):** Create · Explore · Characters · Profile — center Create icon larger.

**Create mobile:** Full-screen prompt first; FAB **Generate**; layout via bottom sheet “Add region” simplified 2-box mode; preview swipe up fullscreen.

**Explore mobile:** 2-column masonry; pull-to-refresh.

**Characters mobile:** Stacked list → tap opens full-screen editor sheet.

### Mobile image-gen prompt

> Mobile iPhone UI mockup for SDX AI image generator app, dark mode, bottom navigation bar with Create Explore Characters Profile, center screen showing prompt input and generated square image with two colored bounding boxes, floating purple Generate button, iOS status bar, 375px width, modern native app aesthetic, indigo and cyan accents, 4K.

---

## Master prompt — full product suite (all pages)

> Comprehensive UI design system presentation for “SDX” AI image generation platform, dark mode, showing six screens arranged on charcoal canvas: (1) marketing landing page with hero and feature grid, (2) main Create workspace with bounding box regional prompts, (3) inpaint editor with brush mask and heatmap, (4) PixAI-style explore masonry gallery, (5) character session manager with reference images, (6) style genome library grid. Consistent violet #6c5ce7 and cyan #00d4ff accents, geometric SDX logo with diffusion spiral and grid, Ideogram clarity plus PixAI density, Figma-quality components, 1440px desktop screens, subtle glassmorphism, professional SaaS product design, 8K presentation board, no watermarks.

---

*This document describes product/design intent only. Implementation is separate from the SDX training/sampling codebase.*
