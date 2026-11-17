# Model Card · P44 Storyboard-to-GIF Animator

**Components:** Groq-hosted LLM (storyboard JSON) · image model set by `IMG_MODEL` (default stabilityai/sdxl-turbo; local option stabilityai/sd-turbo) · Salesforce/blip-image-captioning-base · sentence-transformers/all-MiniLM-L6-v2 · in-app synthesised music.

**Intended use:** coursework demo of a generative pipeline; short illustrative clips.
**Out of scope:** realistic depictions of real people, news/evidence, commercial use without checking each model's licence.

**Evaluation:** prompt-caption cosine similarity, frame-to-frame caption consistency, added-detail rate (proxy). Results: _insert table_.

**Known limitations:** character drift, garbled text in images, BLIP captions are coarse, added-detail rate is a crude proxy, LLM may add details not in the idea.

**Safety:** keyword filter, LLM safe flag, generated-prompt filter, "AI-generated" label on frames and MP4 metadata.

**Licences / IP:** verify licences for LLM, image model, BLIP (BSD-3), MiniLM (Apache-2.0). Music is generated in-app or must be royalty-free with attribution recorded here: _insert_.

**Data / privacy:** no user data stored; ideas are sent to the LLM and image APIs.
