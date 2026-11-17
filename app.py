"""P44 - Storyboard-to-GIF Animator
U1 prompt techniques | U2 LLM storyboard | U3 fixed-seed frames + BLIP check
U4 cross-fade GIF/MP4 + music | U5 safety guardrail + AI-generated disclosure
Secrets (Space settings, never in code): GROQ_API_KEY, HF_TOKEN
"""
import os, re, json, random, wave, subprocess, tempfile
from functools import lru_cache

import numpy as np
import requests
import gradio as gr
import imageio
import imageio_ffmpeg
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient

# ----------------------------------------------------------------- config
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
IMG_MODEL = os.environ.get("IMG_MODEL", "stabilityai/sdxl-turbo")
USE_LOCAL = os.environ.get("USE_LOCAL", "0") == "1"  # local sd-turbo on CPU (slow)
SIZE, FPS, HOLD, FADE = 512, 12, 12, 10
DISCLOSURE = "AI-generated"

BLOCKLIST = re.compile(
    r"\b(kill|murder|gore|suicide|self[- ]harm|nude|naked|porn|sexual|rape|"
    r"terror|bomb[- ]making|child abuse|racial slur)\b", re.I)

# ----------------------------------------------------------------- U1/U2: prompts + LLM
SYSTEM = (
    "You are a storyboard artist. Reply with ONLY valid JSON, no prose, in this schema: "
    '{"safe": true, "style": "one shared art style phrase", '
    '"character": "one fixed visual description of the main character/subject", '
    '"shots": [{"shot": 1, "prompt": "visual description of the shot", "caption": "one line"}]} '
    "Exactly 6 shots. Set safe=false if the idea is violent, sexual, hateful or illegal."
)
FEWSHOT = (
    'Example idea: "a fox finds a lantern"\n'
    '{"safe": true, "style": "soft watercolor", "character": "small orange fox with white chest", '
    '"shots": [{"shot": 1, "prompt": "fox walking through misty forest at dawn", "caption": "A quiet morning."}, '
    '{"shot": 2, "prompt": "fox notices a glowing lantern under a tree", "caption": "Something glows."}]} '
    "(your answer must contain 6 shots)\n\n"
)
MODES = {
    "zero-shot": lambda idea: f"Idea: {idea}",
    "few-shot": lambda idea: FEWSHOT + f"Idea: {idea}",
    "role+constraints": lambda idea: (
        f"Idea: {idea}\nConstraints: keep the same character in every shot, one location "
        "change at most, do not add objects the idea did not mention, no text in the image."),
}


def call_llm(user_msg):
    if not GROQ_KEY:
        raise gr.Error("GROQ_API_KEY secret is not set.")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": LLM_MODEL, "temperature": 0.7,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user_msg}]},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def make_storyboard(idea, mode):
    if BLOCKLIST.search(idea):  # U5 guardrail, layer 1: keyword filter
        raise gr.Error("Blocked by safety filter: please try a different idea.")
    for _ in range(2):  # retry once on bad JSON
        try:
            sb = parse_json(call_llm(MODES[mode](idea)))
            break
        except Exception:
            sb = None
    if not sb:
        raise gr.Error("LLM did not return valid JSON. Try again.")
    if not sb.get("safe", True):  # U5 guardrail, layer 2: LLM self-check
        raise gr.Error("The model flagged this idea as unsafe.")
    sb["shots"] = sb["shots"][:6]
    if len(sb["shots"]) < 6:
        raise gr.Error("Storyboard had fewer than 6 shots. Try again.")
    if any(BLOCKLIST.search(s["prompt"]) for s in sb["shots"]):
        raise gr.Error("Generated prompts failed the safety filter.")
    return sb


# ----------------------------------------------------------------- U3: frames
@lru_cache(maxsize=1)
def local_pipe():
    from diffusers import AutoPipelineForText2Image
    return AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")


@lru_cache(maxsize=1)
def hosted_client():
    return InferenceClient(token=HF_TOKEN or None)


def gen_image(prompt, seed):
    if USE_LOCAL:
        g = torch.Generator().manual_seed(seed)
        return local_pipe()(prompt, num_inference_steps=2, guidance_scale=0.0,
                            height=SIZE, width=SIZE, generator=g).images[0]
    c = hosted_client()
    try:
        img = c.text_to_image(prompt, model=IMG_MODEL, seed=seed, width=SIZE, height=SIZE,
                              num_inference_steps=4, guidance_scale=0.0)
    except Exception:  # some providers reject extra params
        img = c.text_to_image(prompt, model=IMG_MODEL)
    return img.convert("RGB").resize((SIZE, SIZE))


def make_frames(sb, seed, fixed=True, progress=None):
    prefix = f"{sb.get('style', '')}, {sb.get('character', '')}, "
    frames, prompts = [], []
    for i, s in enumerate(sb["shots"]):
        if progress:
            progress((i + 1) / 7, desc=f"Frame {i + 1}/6")
        p = prefix + s["prompt"]
        frames.append(gen_image(p, seed if fixed else random.randint(0, 2**31 - 1)))
        prompts.append(p)
    return frames, prompts


@lru_cache(maxsize=1)
def blip():
    n = "Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(n), BlipForConditionalGeneration.from_pretrained(n).eval()


@lru_cache(maxsize=1)
def embedder():  # MiniLM = BERT-family encoder
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def caption(img):
    proc, model = blip()
    with torch.no_grad():
        out = model.generate(**proc(img, return_tensors="pt"), max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True)


STOP = set("with that this from have into over under near while their there about".split())


def score(frames, prompts, idea):
    caps = [caption(f) for f in frames]
    e = embedder()
    cap_e, pr_e = e.encode(caps, convert_to_tensor=True), e.encode(prompts, convert_to_tensor=True)
    sim = [float(util.cos_sim(cap_e[i], pr_e[i])) for i in range(len(caps))]
    cons = [float(util.cos_sim(cap_e[i], cap_e[i + 1])) for i in range(len(caps) - 1)]
    idea_words = set(re.findall(r"[a-z]+", idea.lower()))
    toks = [w for p in prompts for w in re.findall(r"[a-z]{4,}", p.lower()) if w not in STOP]
    added = sum(w not in idea_words for w in toks) / max(len(toks), 1)  # crude hallucination proxy
    return {"captions": caps, "prompt_sim": float(np.mean(sim)),
            "consistency": float(np.mean(cons)), "added_detail_rate": added}


# ----------------------------------------------------------------- U4/U5: video, music, label
def label(img, text):
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    f = ImageFont.load_default()
    d.rectangle([0, SIZE - 22, SIZE, SIZE], fill=(0, 0, 0, 150))
    d.text((6, SIZE - 17), text[:70], fill="white", font=f)
    w = d.textlength(DISCLOSURE, font=f)
    d.rectangle([SIZE - w - 14, 6, SIZE - 6, 24], fill=(0, 0, 0, 170))
    d.text((SIZE - w - 10, 11), DISCLOSURE, fill="white", font=f)
    return img


def crossfade(frames):
    seq = []
    for i, f in enumerate(frames):
        seq += [f] * HOLD
        if i < len(frames) - 1:
            seq += [Image.blend(f, frames[i + 1], (k + 1) / (FADE + 1)) for k in range(FADE)]
    return seq


def synth_music(path, seconds, n_chords=6):
    """Original ambient pad (no third-party licence). Swap in MusicGen if you wish."""
    sr = 22050
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    roots = [220.0, 246.94, 261.63, 293.66, 261.63, 220.0]
    seg = len(t) // n_chords
    audio = np.zeros_like(t)
    for i in range(n_chords):
        s = slice(i * seg, (i + 1) * seg if i < n_chords - 1 else len(t))
        tt = t[s]
        r = roots[i % len(roots)]
        chord = sum(np.sin(2 * np.pi * r * m * tt) for m in (1, 1.25, 1.5))
        env = np.minimum(1, np.minimum(np.arange(len(tt)), len(tt) - np.arange(len(tt))) / (0.4 * sr))
        audio[s] = chord * env
    audio = (audio / np.abs(audio).max() * 0.3 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())


def render(frames, captions, music_file):
    out = tempfile.mkdtemp()
    labelled = [label(f, c) for f, c in zip(frames, captions)]
    seq = crossfade(labelled)
    gif = os.path.join(out, "clip.gif")
    seq[0].save(gif, save_all=True, append_images=seq[1:], duration=int(1000 / FPS), loop=0)
    silent = os.path.join(out, "silent.mp4")
    with imageio.get_writer(silent, fps=FPS, codec="libx264", macro_block_size=1,
                            pixelformat="yuv420p") as w:
        for im in seq:
            w.append_data(np.array(im))
    wav = music_file or os.path.join(out, "music.wav")
    if not music_file:
        synth_music(wav, len(seq) / FPS)
    mp4 = os.path.join(out, "clip.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", silent, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    "-metadata", f"comment={DISCLOSURE} content (storyboard: LLM, frames: diffusion)",
                    mp4], check=True, capture_output=True)
    return gif, mp4


# ----------------------------------------------------------------- Gradio handlers
def run(idea, mode, seed, fixed, music_file, progress=gr.Progress()):
    seed = int(seed)
    progress(0.02, desc="Writing storyboard")
    sb = make_storyboard(idea, mode)
    frames, prompts = make_frames(sb, seed, fixed, progress)
    progress(0.85, desc="Scoring with BLIP + MiniLM")
    m = score(frames, prompts, idea)
    progress(0.93, desc="Rendering GIF/MP4")
    gif, mp4 = render(frames, [s["caption"] for s in sb["shots"]], music_file)
    metrics = (f"**Prompt-caption similarity:** {m['prompt_sim']:.3f}  \n"
               f"**Frame-to-frame consistency:** {m['consistency']:.3f}  \n"
               f"**Added-detail rate (hallucination proxy):** {m['added_detail_rate']:.2%}  \n"
               f"**Seed:** {seed if fixed else 'random per shot'} | **Mode:** {mode}")
    log = {"idea": idea, "mode": mode, "seed": seed if fixed else "random",
           "storyboard": sb, "full_prompts": prompts, "blip_captions": m["captions"],
           "disclosure": DISCLOSURE}
    gallery = [(f, f"{i + 1}. {c}") for i, (f, c) in enumerate(zip(frames, m["captions"]))]
    return gif, mp4, gallery, metrics, json.dumps(log, indent=2)


def evaluate(ideas_text, mode, seed, progress=gr.Progress()):
    ideas = [x.strip() for x in ideas_text.splitlines() if x.strip()][:5]
    rows, tot = [], {"fixed": [], "random": []}
    for n, idea in enumerate(ideas):
        sb = make_storyboard(idea, mode)
        row = [idea]
        for name, fx in (("fixed", True), ("random", False)):
            progress((n + (0.5 if fx else 1)) / len(ideas), desc=f"{idea[:25]} ({name})")
            fr, pr = make_frames(sb, int(seed), fx)
            m = score(fr, pr, idea)
            tot[name].append((m["prompt_sim"], m["consistency"]))
            row += [f"{m['prompt_sim']:.3f}", f"{m['consistency']:.3f}"]
        rows.append(row)
    avg = lambda k, j: np.mean([x[j] for x in tot[k]])
    rows.append(["**MEAN**", f"{avg('fixed', 0):.3f}", f"{avg('fixed', 1):.3f}",
                 f"{avg('random', 0):.3f}", f"{avg('random', 1):.3f}"])
    head = ("| Idea | Fixed: prompt-sim | Fixed: consistency | Random: prompt-sim | Random: consistency |\n"
            "|---|---|---|---|---|\n")
    return head + "\n".join("| " + " | ".join(r) + " |" for r in rows)


def seed_grid(prompt):
    """U1 latent-space view: same prompt, different initial noise."""
    return [(gen_image(prompt, s), f"seed {s}") for s in (1, 7, 42, 123)]


ABOUT = f"""
### Lifecycle
Idea -> prompt design (U1) -> LLM storyboard (U2) -> diffusion frames, fixed seed (U3) -> BLIP + MiniLM check -> cross-fade GIF/MP4 + music (U4) -> disclosure label (U5).

### Responsible AI
- **Disclosure:** every frame is stamped "{DISCLOSURE}"; MP4 metadata carries the same note.
- **Guardrails:** keyword filter, LLM `safe` flag, and a filter on generated prompts.
- **Licences:** check the licence of the image model you use (SDXL-Turbo is non-commercial research licence; sd-turbo likewise). Music is synthesised in-app, or your own upload; confirm you hold rights to it.
- **Limits:** characters can drift between shots, small text renders badly, BLIP captions are approximate.
"""

with gr.Blocks(title="P44 Storyboard-to-GIF Animator") as demo:
    gr.Markdown("# Storyboard-to-GIF Animator\nTurn an idea into a 6-shot animated clip.")
    with gr.Tab("Animator"):
        idea = gr.Textbox(label="Idea", value="a robot learns to paint a sunset")
        with gr.Row():
            mode = gr.Dropdown(list(MODES), value="role+constraints", label="Prompt technique")
            seed = gr.Number(value=42, label="Seed", precision=0)
            fixed = gr.Checkbox(value=True, label="Fixed seed for all frames")
        music = gr.Audio(type="filepath", label="Optional music (royalty-free only)")
        btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            gif_o, mp4_o = gr.Image(label="GIF"), gr.Video(label="MP4 with music")
        gal = gr.Gallery(label="Frames + BLIP captions", columns=6)
        met = gr.Markdown()
        log = gr.Code(label="Storyboard, prompts, captions (JSON)", language="json")
        btn.click(run, [idea, mode, seed, fixed, music], [gif_o, mp4_o, gal, met, log])
    with gr.Tab("Evaluation: fixed vs random seed"):
        gr.Markdown("One idea per line (max 5). Slow: 12 images per idea.")
        ev_ideas = gr.Textbox(lines=5, value="a fox finds a lantern\na robot learns to paint a sunset\n"
                              "a paper boat crosses a storm puddle\na cat explores a space station\n"
                              "a snowman visits the beach")
        with gr.Row():
            ev_mode = gr.Dropdown(list(MODES), value="role+constraints", label="Prompt technique")
            ev_seed = gr.Number(value=42, precision=0, label="Seed")
        ev_btn = gr.Button("Run evaluation")
        ev_out = gr.Markdown()
        ev_btn.click(evaluate, [ev_ideas, ev_mode, ev_seed], ev_out)
    with gr.Tab("Latent view (seeds)"):
        sp = gr.Textbox(value="a small orange fox in a misty forest, watercolor", label="Prompt")
        sg = gr.Gallery(columns=4)
        gr.Button("Show 4 seeds").click(seed_grid, sp, sg)
    with gr.Tab("About / Responsible AI"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue().launch()
