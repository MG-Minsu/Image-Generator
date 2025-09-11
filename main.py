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
def generate_story_images(story_text, num_images):
    # Split story into num_images parts (scenes)
    story_lines = story_text.split("\n\n")  # simple scene splitting
    scenes = story_lines[:num_images] if len(story_lines) >= num_images else story_lines
    
    images = []
    for i, scene in enumerate(scenes):
        prompt = f"Illustration for a children's story: {scene}. Bright, friendly, colorful."
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1  # one image per scene
        )
        images.append(result.data[0].b64_json)
    return images


# --- Main App ---
def setup_dreamcanvas_app():
    st.set_page_config(page_title="DreamCanvas", page_icon="🎨", layout="wide")
    st.title("🎨 DreamCanvas — Where Stories Become Art")

    # Example styles
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

        # Number of images directly (no paragraph logic)
        num_images = st.number_input("📸 How many images to generate?", min_value=1, max_value=10, value=3, step=1)

        if st.button("✨ Create Magic"):
            if user_story.strip():
                st.info(f"Generating {num_images} images based on your story...")

                # Build one main prompt
                full_prompt = create_visual_prompt(user_story, chosen_style, mood_slider, art_styles, color_palette)

                # Run async image generator
                images = asyncio.run(generate_story_images(full_prompt, num_images=num_images, model="dall-e-3"))

                for img_idx, img_path in enumerate(images, start=1):
                    st.image(img_path, caption=f"Story Image {img_idx}")
                    if isinstance(img_path, str) and img_path.endswith(".png"):
                        with open(img_path, "rb") as f:
                            st.download_button(
                                label=f"💾 Download Image {img_idx}",
                                data=f,
                                file_name=f"story_image_{img_idx}.png",
                                mime="image/png"
                            )

            else:
                st.warning("Please write a story first!")

if __name__ == "__main__":
    setup_dreamcanvas_app()
