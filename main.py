import streamlit as st
import base64
import asyncio
import aiohttp
import io
import zipfile
from PIL import Image
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Prompt Builder ---
def create_visual_prompt(story_text, style_choice, mood_setting, art_styles, color_palette):
    color_desc = ", ".join(color_palette).lower() + " color scheme" if color_palette else ""
    return f"{story_text}, {art_styles[style_choice]}, {mood_setting}, {color_desc}, masterpiece quality"

def generate_enhanced_scene_prompt(scene_text, style_choice, mood_setting, art_styles, color_palette):
    """Use Gemini to enhance scene descriptions for better image generation"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create base visual prompt
        base_prompt = create_visual_prompt(scene_text, style_choice, mood_setting, art_styles, color_palette)
        
        enhancement_prompt = f"""
        Transform this story scene into a detailed visual description perfect for image generation:
        
        Scene: {base_prompt}
        
        Create a vivid, detailed description that includes:
        - Main characters and their expressions
        - Setting and environment details
        - Lighting and atmosphere
        - Visual composition
        - Specific artistic elements
        
        Keep it concise but visually rich, suitable for children's book illustration style.
        Format as a single paragraph description perfect for the Imagen AI model.
        """
        
        response = model.generate_content(enhancement_prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Scene enhancement failed: {str(e)}")
        return create_visual_prompt(scene_text, style_choice, mood_setting, art_styles, color_palette)

async def generate_story_images(story_text, num_images, style_choice, mood_setting, art_styles, color_palette):
    """Generate images using Google's Imagen 4.0 model"""
    
    # Split story into scenes
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
            # Enhance the scene prompt using Gemini
            enhanced_prompt = generate_enhanced_scene_prompt(
                scene.strip(), style_choice, mood_setting, art_styles, color_palette
            )
            
            # Display the enhanced prompt to user
            st.write(f"**Scene {i+1} Enhanced Prompt:** {enhanced_prompt}")
            
            # Generate image using Imagen 4.0
            imagen_model = genai.GenerativeModel('imagen-4.0-generate-001')
            
            response = imagen_model.generate_content([enhanced_prompt])
            
            # Extract image data
            if response.parts and len(response.parts) > 0:
                image_part = response.parts[0]
                if hasattr(image_part, 'data'):
                    image_data = image_part.data
                    images.append(image_data)
                else:
                    st.error(f"No image data received for scene {i+1}")
            else:
                st.error(f"No response parts received for scene {i+1}")
                
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

def create_zip_file(images):
    """Create a ZIP file containing all images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, image_data in enumerate(images):
            filename = f"story_image_{idx + 1}.png"
            zip_file.writestr(filename, image_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def download_all_images_button(images):
    """Create a download button for all images as a ZIP file"""
    if images:
        zip_data = create_zip_file(images)
        st.download_button(
            label="📦 Download All Pictures (ZIP)",
            data=zip_data,
            file_name="story_images.zip",
            mime="application/zip"
        )

# --- Main App ---
def setup_dreamcanvas_app():
    st.set_page_config(page_title="DreamCanvas - Imagen 4.0", page_icon="🎨", layout="wide")
    st.title("🎨 DreamCanvas — AI Image Generation with Imagen 4.0")
    st.write("Transform your stories into beautiful images using Google's Imagen 4.0 and Gemini AI")
    
    # Example styles
    art_styles = {
        "Dreamscape": "ethereal, soft lighting, pastel colors, surreal atmosphere",
        "Comic Book": "bold outlines, vibrant colors, dynamic poses, speech bubbles",
        "Fantasy Art": "magical elements, rich colors, detailed textures, epic scale",
        "Watercolor": "soft brushstrokes, flowing colors, artistic texture",
        "Cartoon": "simple shapes, bright colors, playful style",
        "Realistic": "photorealistic, detailed textures, natural lighting",
        "Anime": "anime style, expressive eyes, dynamic poses, vibrant colors",
        "Children's Book": "friendly children's book illustration, warm colors, inviting characters"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_story = st.text_area(
            "📝 Write your story", 
            placeholder="Once upon a time, in a magical forest, a little fox discovered a glowing crystal...\n\nThe fox touched the crystal and suddenly could understand all the forest animals...\n\nTogether, they embarked on an adventure to save their home from an evil wizard...",
            height=300,
            help="Write your story with each paragraph as a separate scene. Each paragraph will become one image."
        )
    
    with col2:
        chosen_style = st.selectbox("🎨 Visual Style:", list(art_styles.keys()), index=7)  # Default to Children's Book
        
        mood_slider = st.select_slider("🌟 Mood:", options=[
            "Dark & Mysterious", 
            "Calm & Peaceful", 
            "Bright & Energetic", 
            "Epic & Dramatic"
        ], value="Bright & Energetic")
        
        color_palette = st.multiselect(
            "🎨 Colors:", 
            ["Blues", "Reds", "Purples", "Golds", "Greens", "Oranges", "Pinks", "Silvers"], 
            default=["Blues", "Golds"]
        )
        
        num_images = st.number_input(
            "📸 How many images?", 
            min_value=1, 
            max_value=10,
            value=3, 
            step=1,
            help="Number of scenes/images to generate"
        )
    
    if st.button("✨ Generate Images with Imagen 4.0", type="primary"):
        if user_story.strip():
            with st.spinner(f"🎨 Generating {num_images} images with Imagen 4.0..."):
                try:
                    # Generate images using Imagen 4.0
                    images = asyncio.run(generate_story_images(
                        user_story, num_images, chosen_style, mood_slider, art_styles, color_palette
                    ))
                    
                    if images:
                        st.success(f"✅ Generated {len(images)} images!")
                        
                        # Add "Download All Pictures" button at the top
                        download_all_images_button(images)
                        st.divider()
                        
                        # Display images in columns
                        cols = st.columns(min(len(images), 3))
                        
                        for img_idx, image_data in enumerate(images):
                            col_idx = img_idx % len(cols)
                            
                            with cols[col_idx]:
                                # Convert bytes to PIL Image for display
                                try:
                                    image = Image.open(io.BytesIO(image_data))
                                    st.image(image, caption=f"Story Scene {img_idx + 1}")
                                    
                                    # Download button
                                    download_image_button(
                                        image_data,
                                        f"story_image_{img_idx + 1}.png",
                                        f"💾 Download Image {img_idx + 1}"
                                    )
                                except Exception as e:
                                    st.error(f"Error displaying image {img_idx + 1}: {str(e)}")
                        
                    else:
                        st.error("Failed to generate any images. Please try again.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Make sure you have access to the Imagen 4.0 model and your API key is properly configured.")
        else:
            st.warning("📝 Please write a story first!")
    
    # Instructions section
    with st.expander("📖 How to Use DreamCanvas"):
        st.write("""
        **Step 1:** Write your story in the text area, with each paragraph representing a different scene.
        
        **Step 2:** Choose your visual style, mood, and color preferences.
        
        **Step 3:** Set how many images you want to generate.
        
        **Step 4:** Click "Generate Images with Imagen 4.0" to create your illustrations.
        
        **Step 5:** Download individual images or all images as a ZIP file.
        
        **Tips for better results:**
        - Write clear, descriptive scenes
        - Each paragraph should describe one specific moment
        - Include character descriptions and emotions
        - Describe the setting and atmosphere
        - Be specific about visual details you want to see
        """)
    
    # Technical info
    with st.expander("🔧 Technical Information"):
        st.write("""
        **Powered by:**
        - **Imagen 4.0**: Google's latest image generation model for high-quality, creative images
        - **Gemini 1.5 Flash**: For enhancing story prompts with detailed visual descriptions
        
        **Requirements:**
        - Google AI API key with access to Imagen 4.0 model
        - Gemini API access for prompt enhancement
        
        **Features:**
        - AI-enhanced prompts for better image quality
        - Multiple art styles and mood options
        - Batch image generation and download
        - High-quality outputs suitable for children's books
        """)

if __name__ == "__main__":
    # Check if Gemini API key is configured
    try:
        setup_dreamcanvas_app()
    except Exception as e:
        st.error("Please configure your GEMINI_API_KEY in Streamlit secrets")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        st.stop()
