import streamlit as st
import base64
import asyncio
import aiohttp
import io
from PIL import Image
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Prompt Builder ---
def create_visual_prompt(story_text, style_choice, mood_setting, art_styles, color_palette, user_add=""):
    color_desc = ", ".join(color_palette).lower() + " color scheme" if color_palette else ""
    additional = f", {user_add}" if user_add.strip() else ""
    return f"{story_text}, {art_styles[style_choice]}, {mood_setting}, {color_desc}{additional}, masterpiece quality"

async def generate_story_images(story_text, num_images, user_add=""):
    """Generate images based on story scenes"""
    # Split story into scenes for multiple images
    story_lines = story_text.split("\n\n")
    
    # If we have fewer paragraphs than requested images, duplicate content
    if len(story_lines) < num_images:
        scenes = story_lines * (num_images // len(story_lines) + 1)
        scenes = scenes[:num_images]
    else:
        scenes = story_lines[:num_images]
    
    images = []
    
    for i, scene in enumerate(scenes):
        try:
            # Create a more specific prompt for each scene
            prompt = f"Friendly children's book illustration: {scene.strip()}. {user_add}. Bright, colorful, happy, storybook style, digital art, high quality."
            
            # Generate image
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt[:1000],  # Ensure prompt isn't too long
                size="256x256",  # Better quality than 256x256
                n=1,
                response_format="url"  # Get URL instead of base64 for easier handling
            )
            
            image_url = response.data[0].url
            
            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as img_response:
                    if img_response.status == 200:
                        image_data = await img_response.read()
                        images.append(image_data)
                    else:
                        st.error(f"Failed to download image {i+1}")
                        
        except Exception as e:
            st.error(f"Error generating image {i+1}: {str(e)}")
            continue
    
    return images

def download_image_button(image_data, filename, label):
    """Create a download button for image data"""
    st.download_button(
        label=label,
        data=image_data,
        file_name=filename,
        mime="image/png"
    )

# --- Main App ---
def setup_dreamcanvas_app():
    st.set_page_config(page_title="DreamCanvas", page_icon="🎨", layout="wide")
    st.title("🎨 DreamCanvas — Where Stories Become Art")
    
    # Example styles
    art_styles = {
        "Dreamscape": "ethereal, soft lighting, pastel colors, surreal atmosphere",
        "Comic Book": "bold outlines, vibrant colors, dynamic poses, speech bubbles",
        "Fantasy Art": "magical elements, rich colors, detailed textures, epic scale",
        "Watercolor": "soft brushstrokes, flowing colors, artistic texture",
        "Cartoon": "simple shapes, bright colors, playful style"
    }
    
    # Fixed the tab creation syntax
    tab1, tab2, tab3 = st.tabs(["🖼️ Create Art", "🎭 Story Gallery", "⚙️ Advanced Studio"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_story = st.text_area(
                "📝 Write your story", 
                placeholder="Once upon a time, in a magical forest...",
                height=300
            )

            user_add = st.text_area(
                "📝 Additional Instructions", 
                placeholder="The character is Red Riding Hood, wearing bright red cape...",
                height=200
            )
        
        with col2:
            chosen_style = st.selectbox("🎨 Visual Style:", list(art_styles.keys()))
            
            mood_slider = st.select_slider("🌟 Mood:", options=[
                "Dark & Mysterious", 
                "Calm & Peaceful", 
                "Bright & Energetic", 
                "Epic & Dramatic"
            ], value="Bright & Energetic")
            
            color_palette = st.multiselect(
                "🎨 Colors:", 
                ["Blues", "Reds", "Purples", "Golds", "Greens", "Oranges"], 
                default=["Blues", "Golds"]
            )
            
            # Number of images
            num_images = st.number_input(
                "📸 How many images?", 
                min_value=1, 
                max_value=15,  # Reduced max to prevent API overuse
                value=3, 
                step=1
            )
        
        if st.button("✨ Create Magic", type="primary"):
            if user_story.strip():
                with st.spinner(f"Generating {num_images} magical images..."):
                    try:
                        # Generate images
                        images = asyncio.run(generate_story_images(user_story, num_images, user_add))
                        
                        if images:
                            st.success(f"✅ Generated {len(images)} images!")
                            
                            # Display images in columns
                            cols = st.columns(min(len(images), 3))
                            
                            for img_idx, image_data in enumerate(images):
                                col_idx = img_idx % len(cols)
                                
                                with cols[col_idx]:
                                    # Convert bytes to PIL Image for display
                                    image = Image.open(io.BytesIO(image_data))
                                    st.image(image, caption=f"Story Scene {img_idx + 1}")
                                    
                                    # Download button
                                    download_image_button(
                                        image_data,
                                        f"story_image_{img_idx + 1}.png",
                                        f"💾 Download Image {img_idx + 1}"
                                    )
                        else:
                            st.error("Failed to generate any images. Please try again.")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("📝 Please write a story first!")
    
    with tab2:
        st.header("🎭 Story Gallery")
        st.info("This section could showcase previously generated stories and images!")
        
        # Placeholder for gallery functionality
        st.write("Coming soon: Browse and share your created stories!")
    
    with tab3:
        st.header("⚙️ Advanced Studio")
        st.info("Fine-tune your image generation with advanced settings")

if __name__ == "__main__":
    setup_dreamcanvas_app()
