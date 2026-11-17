---
title: Storyboard-to-GIF Animator
emoji: 🎞️
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# P44 · Storyboard-to-GIF Animator
Idea -> 6-shot storyboard (Groq LLM) -> fixed-seed frames (diffusion) -> BLIP + MiniLM consistency check -> cross-fade GIF/MP4 with music -> "AI-generated" label.

## Run / deploy
1. Create a Hugging Face Space (Gradio SDK) and upload `app.py`, `requirements.txt`, `README.md`.
2. Space Settings -> Secrets: add `GROQ_API_KEY` and `HF_TOKEN`. Never commit keys.
3. Optional variables: `LLM_MODEL`, `IMG_MODEL`, `USE_LOCAL=1` (local sd-turbo, slow on CPU).

## Syllabus mapping
| Unit | Where |
|---|---|
| U1 | 3 prompt techniques (dropdown), seed-grid latent view tab |
| U2 | LLM storyboard writer (transformer), diffusion latent-space note in report |
| U3 | GPT-style LLM + MiniLM (BERT family) + BLIP (image-text) |
| U4 | PyTorch/diffusers frames, cross-fade GIF/MP4, synthesised music |
| U5 | Keyword + LLM guardrails, similarity/hallucination metrics, disclosure label, MODEL_CARD.md |

## Results
Fill in after running the Evaluation tab (fixed vs random seed, 5 ideas), and add one failure-case screenshot.
