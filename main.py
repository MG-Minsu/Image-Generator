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
def create_visual_prompt(story_text, style_choice, mood_setting, art_styles, color_palette):
    color_desc = ", ".join(color_palette).lower() + " color scheme"
    return f"{story_text}, {art_styles[style_choice]}, {mood_setting}, {color_desc}, masterpiece quality"

async def generate_story_images(story_text, num_images):
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
            prompt = f"Friendly children's book illustration: {scene.strip()}. Bright, colorful, happy, storybook style, digital art, high quality."
            
            # Generate image
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt[:1000],  # Ensure prompt isn't too long
                size="512x512",  # Better quality than 256x256
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
                max_value=6,  # Reduced max to prevent API overuse
                value=3, 
                step=1
            )
        
        if st.button("✨ Create Magic", type="primary"):
            if user_story.strip():
                with st.spinner(f"Generating {num_images} magical images..."):
                    try:
                        # Build the enhanced prompt
                        full_prompt = create_visual_prompt(
                            user_story, chosen_style, mood_slider, art_styles, color_palette
                        )
                        
                        # Generate images
                        images = asyncio.run(generate_story_images(user_story, num_images))
                        
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
        
        # Advanced settings
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Image Quality")
            resolution = st.selectbox("Resolution:", ["512x512", "1024x1024"])
            model_choice = st.selectbox("AI Model:", ["dall-e-2", "dall-e-3"])
            
            st.subheader("🎭 Character Templates")
            st.write("Quick character templates:")
            character_templates = {
                "Fantasy Adventure": "Hero: brave knight with silver armor, Princess: elegant with flowing blue dress, Dragon: majestic red dragon with golden eyes",
                "Animal Friends": "Bear: friendly brown bear with a red scarf, Rabbit: small white rabbit with pink nose, Fox: clever orange fox with bright eyes",
                "Space Adventure": "Astronaut: young explorer in white space suit, Alien: friendly green alien with big eyes, Robot: helpful silver robot with blue lights",
                "Fairy Tale": "Fairy: tiny fairy with sparkly wings, Wizard: old wise wizard with long beard and pointed hat, Unicorn: white unicorn with rainbow mane"
            }
            
            template_choice = st.selectbox("Choose template:", ["Custom"] + list(character_templates.keys()))
            if template_choice != "Custom":
                st.text_area("Template characters:", character_templates[template_choice], disabled=True)
                if st.button("Use This Template"):
                    st.session_state.template_characters = character_templates[template_choice]
        
        with col2:
            st.subheader("🎨 Style Options")
            artistic_influence = st.slider("Artistic Style Intensity", 0.1, 1.0, 0.7)
            
            st.subheader("🎯 Common Requests")
            st.write("Click to add to your specific requests:")
            
            common_requests = [
                "magical sparkles and glitter effects",
                "sunset/sunrise lighting",
                "Disney/Pixar animation style",
                "watercolor painting effect",
                "include a rainbow in background",
                "show characters as silhouettes",
                "add speech bubbles with dialogue",
                "vintage storybook illustration style"
            ]
            
            for request in common_requests:
                if st.button(f"+ {request}", key=f"req_{request}"):
                    if 'specific_requests' not in st.session_state:
                        st.session_state.specific_requests = ""
                    if request not in st.session_state.specific_requests:
                        st.session_state.specific_requests += f", {request}" if st.session_state.specific_requests else request
        
        st.subheader("💡 Pro Tips")
        tips = [
            "**Character Consistency**: Describe characters in detail (age, hair color, clothing) for consistent appearance across scenes",
            "**Scene Variety**: Use paragraph breaks in your story to create natural scene transitions",
            "**Visual Details**: Include environmental details (time of day, weather, location) in your story",
            "**Style Mixing**: Combine different art styles in specific requests for unique looks",
            "**Color Harmony**: Choose 2-3 colors that work well together for better visual appeal"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
        
        st.info("💡 Higher resolutions and DALL-E 3 cost more API credits but produce better quality images!")

if __name__ == "__main__":
    setup_dreamcanvas_app()
