import streamlit as st
import base64
import asyncio
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Prompt Builder ---
def create_visual_prompt(story_text, style_choice, mood_setting, art_styles, color_palette):
    color_desc = ", ".join(color_palette).lower() + " color scheme"
    return f"{story_text}, {art_styles[style_choice]}, {mood_setting}, {color_desc}, masterpiece quality"

# --- Image Generator using OpenAI ---
async def generate_story_images(prompt_text, num_images=1, size="1024x1024"):
    """Generate multiple images from OpenAI Image API"""
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt_text,
        n=num_images,
        size=size
    )

    # Decode base64 images
    image_urls = []
    for i, img_data in enumerate(result.data, start=1):
        if hasattr(img_data, "b64_json") and img_data.b64_json:
            img_bytes = base64.b64decode(img_data.b64_json)
            filename = f"generated_image_{i}.png"
            with open(filename, "wb") as f:
                f.write(img_bytes)
            image_urls.append(filename)
        elif hasattr(img_data, "url") and img_data.url:
            image_urls.append(img_data.url)

    return image_urls

# --- Main App (shortened to focus on Create Art tab) ---
def setup_dreamcanvas_app():
    st.set_page_config(page_title="DreamCanvas", page_icon="🎨", layout="wide")
    st.title("🎨 DreamCanvas — Where Stories Become Art")

    # Example styles (your existing dict can stay)
    art_styles = {
        "Dreamscape": "ethereal, soft lighting, pastel colors, surreal atmosphere",
        "Comic Book": "bold outlines, vibrant colors, dynamic poses, speech bubbles",
        "Fantasy Art": "magical elements, rich colors, detailed textures, epic scale"
    }

    tab1, _, _ = st.tabs(["🖼️ Create Art", "🎭 Story Gallery", "⚙️ Advanced Studio"])
    with tab1:
        user_story = st.text_area("📝 Write your story", placeholder="Once upon a time...")
        chosen_style = st.selectbox("🎨 Visual Style:", list(art_styles.keys()))
        mood_slider = st.select_slider("Mood:", options=[
            "Dark & Mysterious", "Calm & Peaceful", "Bright & Energetic", "Epic & Dramatic"
        ])
        color_palette = st.multiselect("Colors:", ["Blues", "Reds", "Purples", "Golds"], default=["Blues"])
        image_count = st.slider("Number of Images:", 1, 5, 2)

        if st.button("✨ Create Magic"):
            if user_story.strip():
                full_prompt = create_visual_prompt(user_story, chosen_style, mood_slider, art_styles, color_palette)
                st.info(f"Generating {image_count} images...")

                # Run async generator
                images = asyncio.run(generate_story_images(full_prompt, num_images=image_count))

                for idx, img_path in enumerate(images, start=1):
                    st.image(img_path, caption=f"Variation {idx}")
                    with open(img_path, "rb") as f:
                        btn = st.download_button(
                            label=f"💾 Download Variation {idx}",
                            data=f,
                            file_name=f"story_art_{idx}.png",
                            mime="image/png"
                        )
            else:
                st.warning("Please write a story first!")

if __name__ == "__main__":
    setup_dreamcanvas_app()
