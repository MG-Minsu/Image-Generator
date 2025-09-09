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
    """Generate images using DALL·E"""
    result = client.images.generate(
        model="dall-e-3",  # or "dall-e-2"
        prompt=prompt_text,
        size=size,
        quality="standard",   # or "hd"
        n=1 if "dall-e-3" else num_images
    )

    image_paths = []
    for i, img_data in enumerate(result.data, start=1):
        if hasattr(img_data, "b64_json") and img_data.b64_json:
            img_bytes = base64.b64decode(img_data.b64_json)
            filename = f"dalle_image_{i}.png"
            with open(filename, "wb") as f:
                f.write(img_bytes)
            image_paths.append(filename)
        elif hasattr(img_data, "url") and img_data.url:
            image_paths.append(img_data.url)

   


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
        user_story = st.text_area("📝 Write your story", placeholder="Once upon a time...\n\nSecond paragraph here...")
        chosen_style = st.selectbox("🎨 Visual Style:", list(art_styles.keys()))
        mood_slider = st.select_slider("Mood:", options=[
            "Dark & Mysterious", "Calm & Peaceful", "Bright & Energetic", "Epic & Dramatic"
        ])
        color_palette = st.multiselect("Colors:", ["Blues", "Reds", "Purples", "Golds"], default=["Blues"])
        image_count = st.multiselect("Number of Images per Paragraph: (Select 1 for now)", ["1", "2"], default=["1"])

        if st.button("✨ Create Magic"):
            if user_story.strip():
                # Split story into paragraphs (based on blank line)
                paragraphs = [p.strip() for p in user_story.split("\n\n") if p.strip()]

                st.info(f"Generating {len(paragraphs) * image_count} images ({len(paragraphs)} paragraphs × {image_count} each)...")

                for para_idx, paragraph in enumerate(paragraphs, start=1):
                    st.subheader(f"📖 Paragraph {para_idx}")
                    st.write(paragraph)

                    # Build prompt for this paragraph
                    full_prompt = create_visual_prompt(paragraph, chosen_style, mood_slider, art_styles, color_palette)

                    # Run async image generator
                    images = asyncio.run(generate_story_images(full_prompt, num_images=image_count))

                    for img_idx, img_path in enumerate(images, start=1):
                        st.image(img_path, caption=f"Paragraph {para_idx} - Variation {img_idx}")
                        if isinstance(img_path, str) and img_path.endswith(".png"):
                            with open(img_path, "rb") as f:
                                st.download_button(
                                    label=f"💾 Download Paragraph {para_idx} - Variation {img_idx}",
                                    data=f,
                                    file_name=f"story_para{para_idx}_var{img_idx}.png",
                                    mime="image/png"
                                )

            else:
                st.warning("Please write a story first!")

if __name__ == "__main__":
    setup_dreamcanvas_app()
