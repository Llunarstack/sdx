# 100 inventions to make SDX fucking good at images

Research-backed failure themes (FineGRAIN, compositional fidelity, anatomy preference papers, 2025–2026 landscape) plus **original SDX inventions**.

**Status legend:** `coded` = runnable API; `partial` = uses pre-existing SDX path; `scaffold` = trainable stub / wiring spec (weights not pretrained).

Registry: `utils/generation/inventions/registry.py` (`python -m scripts.tools invention_lab --inventory`).

---

## A. Composition, counting, binding, negation (1–20)

1. **BINDLOCK** — **coded** `bindlock.py`
2. **COUNTGATE** — **coded** `countgate.py`
3. **NEGATRON** — **coded** `negatron.py`
4. **HYDRA-SLOTS** — **coded** `hydra_slots.py`
5. **SAT-LAYOUT** — **coded** `composition_ext.check_sat_layout`
6. **Relation Transformer head** — **coded** `models.inventions_arch.RelationTransformerHead`
7. **Concept algebra loss** — **coded** `concept_algebra_loss`
8. **Negation token bank** — **coded** `NegationTokenBank`
9. **Count-as-class** — **coded** `CountClassToken`
10. **Attribute firewall** — **coded** `AttributeFirewallMask`
11. **Scene graph conditioner** — **coded** `compile_scene_graph`
12. **Anti-centering prior** — **coded** `anti_centering_addon`
13. **Z-order tokens** — **coded** `zorder_addon`
14. **Occlusion grammar** — **coded** `occlusion_grammar_expand`
15. **Multiplicity curriculum** — **coded** `multiplicity_curriculum_weight`
16. **Binding contrastive pairs** — **coded** `binding_contrastive_pair`
17. **Layout diffusion first** — **coded** `LayoutDiffusionPrior`
18. **Neuro-symbolic CFG** — **coded** `neuro_symbolic_cfg_scale`
19. **FailureAtlas prompts** — **coded** `load_failure_atlas` / `--atlas-out`
20. **ORACLE** — **coded** `failure_oracle`

## B. Anatomy (21–35)

21. **ANATOMON** — **coded**
22. **Hand expert LoRA bank** — **coded** `plan_hand_expert`
23. **Mesh-guided denoise** — **scaffold** (pose soft + future SMPL)
24. **Finger count critic** — **coded** `plan_finger_critic`
25. **Limb monotonicity loss** — **coded** `limb_monotonicity_loss`
26. **Pose skeleton soft** — **coded** `pose_skeleton_soft_plan`
27. **Face ID + expression** — **coded** `plan_face_id_expression`
28. **Eye symmetry breaker** — **coded**
29. **Teeth/mouth specialist** — **coded**
30. **Contact shadow** — **coded**
31. **Anatomy RLHF** — **scaffold** (preference_flywheel)
32. **Part-aware VAE** — **scaffold**
33. **Anatomy token vocab** — **coded**
34. **Two-stage body→detail** — **coded**
35. **CONSENSUS** — **coded**

## C. Texture / anti-AI-look (36–48)

36–48 all **coded** in `texture_ext.py` (FRICTION/SPECTRA + lens/EXIF, demosaic, sensor noise, BRDF, SSS, asymmetry lottery, JPEG spice, histogram hint, filmic highlights, grime, anti-beauty).

## D. Text / Ideogram (49–58)

49–58 **coded** in `glyph_ext.py` + `edit_skills` (OCR loop plan, Bezier spec, Font ID, Brand kit, multilingual, UI screenshot mode, vector export stub). Kerning (#55) scaffold note.

## E. Multi-character (59–70)

59–70 **coded** in `multi_char_cast` + `identity_ext` (per-char refs, conditioner wiring spec, clothing mask spec, interaction physics, permanence, auto sheet, voice stub, slot dropout, contrastive loss, cast memory).

## F. Data / boards (71–80)

71 **coded** corpus retrieve; 72–79 **coded** in `data_ext`; 73/80 **partial** existing training/flywheel scripts.

## G. Samplers / systems (81–90)

81 APEX **coded**; 82 CONSENSUS **coded**; 83–90 **coded** in `systems_ext` (oracle rungs, adaptive NFE, error redo, energy reject, flow/VP + distill + expert merge specs).

## H. Architectures (91–100)

91–98 **coded** modules in `models/inventions_arch.py`; 95/99 **partial** frontier multiview / block-AR; **100 Self-healing** **coded** (`plan_self_healing`, `--self-heal`).

---

## Quick start

```bash
python -m scripts.tools invention_lab --inventory
python -m scripts.tools invention_lab --atlas-out outputs/failure_atlas.jsonl
python -m scripts.tools invention_lab --prompt "four red cubes left of blue spheres, no text, hands" --enable auto --print-argv
python -m scripts.tools invention_lab --prompt "2girls" --self-heal --print-argv

python sample.py --ckpt ... --prompt "..." --invention-stack auto --invention-spectra \
  --invention-auto-spectra --invention-adaptive-steps \
  --scheduler apex --solver apex_adaptive
```

### Wired into the live stack

| Surface | How |
|---------|-----|
| `sample.py` | `--invention-stack`, `--invention-spectra`, `--invention-auto-spectra`, `--invention-adaptive-steps`, `--invention-self-heal` |
| OCR repair passthrough | forwards invention + corpus + cast flags |
| `pipelines.image_gen` | `ImageGenerateConfig.invention_*` |
| `perfect_gen` argv | defaults `--invention-stack auto` (+ spectra / adaptive) |
| VP + flow denoise | `invention_spectra` CFG phasing |
| `models` | `inventions_arch` scaffolds exported |

Central glue: `utils.generation.inventions.wire`.

### Honest note

“All 100” now exist as **code surfaces** (inference planners, losses, nn.Modules, eval atlas, self-heal plan). Items marked **scaffold/partial** still need training data, detectors (OCR/hands), or FAISS builds before they beat production systems end-to-end.
