
import streamlit as st
from PIL import Image
import requests
import base64
import io
import pandas as pd
import re

# NVIDIA Vision API settings
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"  # 🔒 Replace with your secure key

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def encode_image_url_to_base64(image_url):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), image

def query_nvidia_vision_api(base64_image):
    prompt_text = """
You are an experienced judge for international art competitions, with expertise in evaluating both emerging and professional artists. Your task is to analyze and critique artwork based on the following parameters, assigning each a skill category and providing justification or description.

Skill Categories (with examples):
Beginner - Just beginning to explore fundamentals like color mixing and brush techniques. Requires guidance at every step.
Intermediate - Has a basic grasp of art principles and is experimenting with complexity.
Advanced - Shows strong understanding of artistic principles and a personal voice.
Professional/Expert - Consistently delivers polished, distinctive work with technical mastery and creative depth.

Evaluation Parameters:
Originality - Does the piece offer a unique or inventive take on the subject?
Composition - Is the layout well-balanced, structured, and visually appealing?
Color and Tone - Are color choices and tonal contrasts used effectively to convey depth and mood?
Technical Proficiency - Does the work demonstrate mastery of brushwork, perspective, anatomy, etc.?
Overall Impression - What lasting impact does the artwork leave on the viewer?

Important Notes:
Avoid defaulting to "Intermediate" for all artworks. Use the full range of skill categories.
Do not hesitate to assign "Professional/Expert" when the execution clearly warrants it.
Do not hesitate to assign "Beginner" when the painting has poor design and uneven brushstrokes.

Response Format (one line per parameter):
[Description] - Describe the painting
[Originality] - [Skill Category] - [Comment]
[Composition] - [Skill Category] - [Comment]
[Color] - [Skill Category] - [Comment]
[Technique] - [Skill Category] - [Comment]
[Overall] - [Skill Category] - [Comment]
"""

    payload = {
        "model": "meta/llama-3.2-90b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "No description found.")

def query_nvidia_score(analysis_text):
    prompt = f"""
Based on the analysis of a painting, assign it a score:

- "Beginner: 1 to 2.5"
- "Intermediate: 2.6 to 5"
- "Advanced: 5.1 to 7.5"
- "Professional/Expert: 7.6 to 10"

Use your judgment to select a score within the appropriate range, considering the strength of the comments provided.

Analysis:
\"\"\"
{analysis_text}
\"\"\"
Response Format:
Score: [number]
"""

    payload = {
        "model": "meta/llama-3.2-90b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "No score found.")

def parse_response_to_dict(text_response):
    field_variants = {
        "Description": ["description"],
        "Originality": ["originality"],
        "Composition": ["composition"],
        "Color": ["Color", "color"],
        "Technique": ["technical proficiency", "technique", "technical"],
        "Overall": ["overall", "overall impression"]
    }

    results = {field: "" for field in field_variants}

    pattern = re.compile(r"^(.*?)[\s\-–—:]+(.*)$")

    for line in text_response.strip().split("\n"):
        line = line.strip()
        match = pattern.match(line)
        if match:
            key_raw, value = match.group(1).strip().lower(), match.group(2).lstrip("*- ").strip()
            for field, variants in field_variants.items():
                if any(variant in key_raw for variant in variants):
                    results[field] = value
                    break

    return results

def main():
    st.set_page_config(page_title="Art Judge", layout="centered")
    st.title("🎨 AI Art Judge")
    st.subheader("Powered by LLaMA 3.2 Vision - A Multimodal Large Language Model")
    st.markdown("""
    Hi there! 👋 Welcome to my app — an AI-powered art judge built using the LLaMA Vision model (https://www.llama.com/docs/how-to-guides/vision-capabilities/).
    This app analyzes artwork and gives detailed critiques based on different aspects of art. 
    Just paste in one or more image URLs (of your paintings, sketches, digital art, etc.) and let the model 
    do its magic! ✨

    Here's what it does:
    1. Describes your artwork briefly
    2. Evaluates it across key areas like Originality, Composition, Color, and Technique
    3. Categorizes each aspect as Beginner, Intermediate, Advanced, or Professional/Expert
    4. Provides an overall score from 1 to 10, where 1 is beginner level and 10 is expert-level work

    This is my first attempt at building something that could assist human judges in art competitions by adding a layer of objective, AI-driven analysis.
    Have fun experimenting — and I would love to hear what you think in the feedback box below! 💬

    Go ahead and paste your artwork URLs (comma-separated) to get started. 🖼️
     """)
    st.subheader("Sample Output:")
    st.markdown(
        "[Click here to view a sample evaluation of 3 artworks](http://localhost:8501/)"
    )
    st.markdown("---")
    st.markdown("Paste artwork image URLs (comma-separated) to receive detailed critiques and overall score.")
    input_urls = st.text_area("Enter image URLs", height=100)
    image_urls = [url.strip() for url in input_urls.split(",") if url.strip()]

    if image_urls:
        st.markdown("### Provided Artworks")
        num_columns = 3
        for row_start in range(0, len(image_urls), num_columns):
            cols = st.columns(num_columns)
            for col, url in zip(cols, image_urls[row_start:row_start + num_columns]):
                try:
                    response = requests.get(url)
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    resized_image = image.resize((256, 256))
                    col.image(resized_image, use_container_width=False)
                except Exception as e:
                    col.warning(f"Could not load image from URL: {url}")

        if st.button("Analyze"):
            all_results = []
            for i, url in enumerate(image_urls):
                with st.spinner(f"Analyzing Artwork {i+1}..."):
                    try:
                        base64_image, image_or_error = encode_image_url_to_base64(url)
                        if not base64_image:
                            st.error(f"Failed to load Artwork {i+1}: {image_or_error}")
                            continue
                        result_text = query_nvidia_vision_api(base64_image)
                        result_dict = parse_response_to_dict(result_text)
                        result_dict["Artwork"] = f"Artwork {i+1}"

                        # Step 2: Score based on the result
                        score_response = query_nvidia_score(result_text)
                        score_match = re.search(r"Score:\s*([\d.]+)", score_response)
                        if score_match:
                            result_dict["Score"] = float(score_match.group(1))
                        else:
                            result_dict["Score"] = None

                        all_results.append(result_dict)
                    except Exception as e:
                        st.error(f"API Error for Artwork {i+1}: {e}")

            if all_results:
                df = pd.DataFrame(all_results)[["Artwork", "Description", "Originality", "Composition", "Color", "Technique", "Overall", "Score"]]
                st.markdown("### Evaluation Summary Table")
                st.dataframe(df.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("### 💬 Feedback:")
    st.markdown("""
    If you have any suggestions or thoughts, please share them below. I'd love to hear from you!

    <form action="https://formsubmit.co/agrawal.sonali22@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <textarea name="feedback" rows="4" cols="50" placeholder="Write your feedback here..." required></textarea><br>
        <button type="submit">Send Feedback</button>
    </form>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### By: [Sonali Agrawal](https://agrawalsonali22.wixstudio.com/my-portfolio)")

if __name__ == "__main__":
    main()
