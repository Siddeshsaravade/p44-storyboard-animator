"""P44 core logic (no UI)
U1 prompt techniques | U2 LLM storyboard | U3 fixed-seed frames + BLIP check
U4 cross-fade GIF/MP4 + music | U5 safety guardrail + AI-generated disclosure
Secrets (Space settings, never in code): GROQ_API_KEY, HF_TOKEN
"""
import os, re, json, random, wave, subprocess, tempfile
from functools import lru_cache

import numpy as np
import requests
import imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient

class AppError(Exception):
    pass


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
        raise AppError("GROQ_API_KEY secret is not set.")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": LLM_MODEL, "temperature": 0.7,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user_msg}]},
        timeout=60)
    if r.status_code != 200:
        raise AppError(f"Groq API error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def make_storyboard(idea, mode):
    if BLOCKLIST.search(idea):  # U5 guardrail, layer 1: keyword filter
        raise AppError("Blocked by safety filter: please try a different idea.")
    sb, last_err = None, None
    for _ in range(2):  # retry once on bad JSON
        try:
            sb = parse_json(call_llm(MODES[mode](idea)))
            break
        except AppError:
            raise
        except Exception as e:
            last_err = e
    if not sb:
        raise AppError(f"Storyboard failed: {last_err}")
    if not sb.get("safe", True):  # U5 guardrail, layer 2: LLM self-check
        raise AppError("The model flagged this idea as unsafe.")
    sb["shots"] = sb["shots"][:6]
    if len(sb["shots"]) < 6:
        raise AppError("Storyboard had fewer than 6 shots. Try again.")
    if any(BLOCKLIST.search(s["prompt"]) for s in sb["shots"]):
        raise AppError("Generated prompts failed the safety filter.")
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
        import torch
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
    from transformers import BlipProcessor, BlipForConditionalGeneration
    n = "Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(n), BlipForConditionalGeneration.from_pretrained(n).eval()


@lru_cache(maxsize=1)
def embedder():  # MiniLM = BERT-family encoder
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def caption(img):
    import torch
    proc, model = blip()
    with torch.no_grad():
        out = model.generate(**proc(img, return_tensors="pt"), max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True)


STOP = set("with that this from have into over under near while their there about".split())


def score(frames, prompts, idea):
    from sentence_transformers import util
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


