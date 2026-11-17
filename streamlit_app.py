"""P44 Storyboard-to-GIF Animator - Streamlit UI. Secrets: GROQ_API_KEY, HF_TOKEN."""
import os, json
import streamlit as st

for k in ("GROQ_API_KEY", "HF_TOKEN"):  # copy Streamlit secrets into env before importing core
    try:
        if k in st.secrets:
            os.environ[k] = st.secrets[k]
    except Exception:
        pass

import numpy as np
import core
from core import AppError, MODES, DISCLOSURE

st.set_page_config(page_title="P44 Storyboard-to-GIF Animator", layout="wide")
st.title("Storyboard-to-GIF Animator")
st.caption("Turn an idea into a 6-shot animated clip.")

tab1, tab2, tab3, tab4 = st.tabs(["Animator", "Evaluation: fixed vs random", "Latent view (seeds)", "About / Responsible AI"])

with tab1:
    idea = st.text_input("Idea", "a robot learns to paint a sunset")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Prompt technique", list(MODES), index=2)
    seed = c2.number_input("Seed", value=42, step=1)
    fixed = c3.checkbox("Fixed seed for all frames", True)
    music = st.file_uploader("Optional music (royalty-free only)", type=["mp3", "wav"])
    if st.button("Generate", type="primary"):
        try:
            with st.status("Working...", expanded=True) as s:
                st.write("Writing storyboard")
                sb = core.make_storyboard(idea, mode)
                st.write("Generating 6 frames")
                frames, prompts = core.make_frames(sb, int(seed), fixed)
                st.write("Scoring with BLIP + MiniLM")
                m = core.score(frames, prompts, idea)
                st.write("Rendering GIF/MP4")
                mpath = None
                if music:
                    mpath = os.path.join(os.getcwd(), "uploaded_music" + os.path.splitext(music.name)[1])
                    open(mpath, "wb").write(music.getbuffer())
                gif, mp4 = core.render(frames, [x["caption"] for x in sb["shots"]], mpath)
                s.update(label="Done", state="complete")
            a, b = st.columns(2)
            a.image(gif, caption="GIF")
            b.video(mp4)
            st.image(frames, caption=[f"{i+1}. {c}" for i, c in enumerate(m["captions"])], width=160)
            st.markdown(f"**Prompt-caption similarity:** {m['prompt_sim']:.3f}  \n"
                        f"**Frame-to-frame consistency:** {m['consistency']:.3f}  \n"
                        f"**Added-detail rate (hallucination proxy):** {m['added_detail_rate']:.2%}")
            st.code(json.dumps({"idea": idea, "mode": mode, "seed": int(seed) if fixed else "random",
                                "storyboard": sb, "full_prompts": prompts,
                                "blip_captions": m["captions"], "disclosure": DISCLOSURE}, indent=2), "json")
        except AppError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.write("One idea per line (max 5). Slow: 12 images per idea.")
    ideas = st.text_area("Ideas", "a fox finds a lantern\na robot learns to paint a sunset\na paper boat crosses a storm puddle\na cat explores a space station\na snowman visits the beach")
    emode = st.selectbox("Prompt technique ", list(MODES), index=2)
    eseed = st.number_input("Seed ", value=42, step=1)
    if st.button("Run evaluation"):
        try:
            rows, tot = [], {"fixed": [], "random": []}
            lst = [x.strip() for x in ideas.splitlines() if x.strip()][:5]
            bar = st.progress(0.0)
            for n, i in enumerate(lst):
                sb = core.make_storyboard(i, emode)
                row = {"Idea": i}
                for name, fx in (("fixed", True), ("random", False)):
                    fr, pr = core.make_frames(sb, int(eseed), fx)
                    m = core.score(fr, pr, i)
                    tot[name].append((m["prompt_sim"], m["consistency"]))
                    row[f"{name}: prompt-sim"] = round(m["prompt_sim"], 3)
                    row[f"{name}: consistency"] = round(m["consistency"], 3)
                rows.append(row)
                bar.progress((n + 1) / len(lst))
            mean = {"Idea": "MEAN"}
            for name in tot:
                mean[f"{name}: prompt-sim"] = round(float(np.mean([x[0] for x in tot[name]])), 3)
                mean[f"{name}: consistency"] = round(float(np.mean([x[1] for x in tot[name]])), 3)
            st.table(rows + [mean])
        except AppError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    p = st.text_input("Prompt", "a small orange fox in a misty forest, watercolor")
    if st.button("Show 4 seeds"):
        try:
            seeds = (1, 7, 42, 123)
            st.image([core.gen_image(p, s) for s in seeds], caption=[f"seed {s}" for s in seeds], width=220)
        except Exception as e:
            st.error(f"Error: {e}")

with tab4:
    st.markdown(f"""
### Lifecycle
Idea -> prompt design (U1) -> LLM storyboard (U2) -> diffusion frames, fixed seed (U3) -> BLIP + MiniLM check -> cross-fade GIF/MP4 + music (U4) -> disclosure label (U5).

### Responsible AI
- **Disclosure:** every frame is stamped "{DISCLOSURE}"; MP4 metadata carries the same note.
- **Guardrails:** keyword filter, LLM `safe` flag, filter on generated prompts.
- **Licences:** check the licence of the image model you use. Music is synthesised in-app or your own upload.
- **Limits:** character drift, poor small text, approximate BLIP captions.
""")
