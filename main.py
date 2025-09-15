import streamlit as st
import base64
import asyncio
import aiohttp
import io
import zipfile
import json
import time
from PIL import Image

# --- Flux API Configuration ---
FLUX_API_BASE = "https://api.bfl.ml"  # Black Forest Labs Flux API
# You can also use other free Flux API endpoints like:
# - "https://api.together.xyz" (Together AI)
# - "https://api.replicate.com" (Replicate)

async def generate_image_with_flux(prompt, width=512, height=512):
    """Generate image using Flux API"""
    headers = {
        "Content-Type": "application/json",
        # Add your API key if required
        # "Authorization": f"Bearer {st.secrets.get('FLUX_API_KEY', '')}"
    }
    
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": None  # Random seed
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Using Hugging Face Inference API (free tier available)
            hf_api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            hf_headers = {
                "Authorization": f"Bearer {st.secrets.get('HUGGINGFACE_API_KEY', '')}"
            }
            
            async with session.post(
                hf_api_url, 
                headers=hf_headers,
                json={"inputs": prompt}
            ) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return image_data
                else:
                    st.error(f"API Error: {response.status}")
                    return None
                    
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

async def generate_image_with_together_ai(prompt):
    """Alternative: Generate image using Together AI (free tier available)"""
    headers = {
        "Authorization": f"Bearer {st.secrets.get('TOGETHER_API_KEY', '')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "steps": 4,
        "n": 1,
        "response_format": "b64_json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.together.xyz/v1/images/generations",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Decode base64 image
                    image_data = base64.b64decode(result['data'][0]['b64_json'])
                    return image_data
                else:
                    error_text = await response.text()
                    st.error(f"Together AI API Error: {response.status} - {error_text}")
                    return None
    except Exception as e:
        st.error(f"Error with Together AI: {str(e)}")
        return None

# --- Prompt Builder ---
def create_visual_prompt(story_text, style_choice, mood_setting, art_styles, color_palette):
    color_desc = ", ".join(color_palette).lower() + " color scheme" if color_palette else ""
    
    base_prompt = f"Children's book illustration: {story_text}"
    style_prompt = f"{art_styles[style_choice]}, {mood_setting}"
    color_prompt = color_desc
    quality_prompt = "high quality, detailed, storybook art, friendly and engaging"
    
    full_prompt = f"{base_prompt}, {style_prompt}, {color_prompt}, {quality_prompt}"
    
    # Flux works better with concise prompts
    return full_prompt[:500]  # Limit prompt length

async def generate_story_images(story_text, num_images, style_choice, mood_setting, art_styles, color_palette):
    """Generate images based on story scenes using Flux API"""
    # Split story into scenes for multiple images
    story_lines = [line.strip() for line in story_text.split("\n\n") if line.strip()]
    
    # If we have fewer paragraphs than requested images, duplicate content
    if len(story_lines) < num_images:
        scenes = story_lines * (num_images // len(story_lines) + 1)
        scenes = scenes[:num_images]
    else:
        scenes = story_lines[:num_images]
    
    images = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, scene in enumerate(scenes):
        try:
            status_text.text(f"Generating image {i+1} of {num_images}...")
            progress_bar.progress((i + 1) / num_images)
            
            # Create a more specific prompt for each scene
            prompt = create_visual_prompt(
                scene, style_choice, mood_setting, art_styles, color_palette
            )
            
            # Try Hugging Face first (free tier)
            image_data = await generate_image_with_flux(prompt)
            
            # If Hugging Face fails, try Together AI
            if not image_data and st.secrets.get('TOGETHER_API_KEY'):
                image_data = await generate_image_with_together_ai(prompt)
            
            if image_data:
                images.append(image_data)
            else:
                st.warning(f"Failed to generate image {i+1}")
                
            # Add delay to avoid rate limiting
            if i < len(scenes) - 1:  # Don't wait after the last image
                await asyncio.sleep(1)
                
        except Exception as e:
            st.error(f"Error generating image {i+1}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
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
    st.set_page_config(page_title="DreamCanvas", page_icon="🎨", layout="wide")
    st.title("🎨 DreamCanvas — Where Stories Become Art")
    st.caption("Powered by Flux AI - Fast and Free Image Generation")
    
    # Check API configuration
    if not st.secrets.get('HUGGINGFACE_API_KEY') and not st.secrets.get('TOGETHER_API_KEY'):
        st.warning("⚠️ No API keys configured. Add HUGGINGFACE_API_KEY or TOGETHER_API_KEY to your secrets.")
        st.info("""
        **Free Options:**
        1. **Hugging Face** (Free tier): Get API key at https://huggingface.co/settings/tokens
        2. **Together AI** (Free credits): Get API key at https://api.together.xyz/settings/api-keys
        """)
    
    # Example styles optimized for Flux
    art_styles = {
        "Storybook": "colorful storybook illustration, children's book art",
        "Cartoon": "cartoon style, animated, bright colors, simple shapes",
        "Watercolor": "watercolor painting, soft brushstrokes, artistic texture",
        "Digital Art": "digital illustration, clean lines, vibrant colors",
        "Fantasy": "fantasy art, magical elements, enchanted atmosphere",
        "Comic": "comic book style, bold outlines, dynamic composition"
    }
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Create Art", "🎭 Story Gallery", "⚙️ Advanced Studio"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_story = st.text_area(
                "📝 Write your story", 
                placeholder="Once upon a time, in a magical forest, a young girl named Luna discovered a glowing crystal that could make wishes come true...",
                height=300
            )
        
        with col2:
            chosen_style = st.selectbox("🎨 Visual Style:", list(art_styles.keys()))
            
            mood_slider = st.select_slider("🌟 Mood:", options=[
                "Dark & Mysterious", 
                "Calm & Peaceful", 
                "Bright & Cheerful", 
                "Epic & Dramatic"
            ], value="Bright & Cheerful")
            
            color_palette = st.multiselect(
                "🎨 Color Palette:", 
                ["Blues", "Reds", "Purples", "Golds", "Greens", "Oranges", "Pastels"], 
                default=["Blues", "Golds"]
            )
            
            # Number of images
            num_images = st.number_input(
                "📸 Number of images:", 
                min_value=1, 
                max_value=10,  # Reduced for free tier limits
                value=3, 
                step=1
            )
            
            st.info("💡 Tip: Flux works best with 1-5 images at a time")
        
        if st.button("✨ Create Magic", type="primary"):
            if user_story.strip():
                with st.spinner(f"Generating {num_images} magical images with Flux AI..."):
                    try:
                        # Generate images
                        images = await generate_story_images(
                            user_story, num_images, chosen_style, 
                            mood_slider, art_styles, color_palette
                        )
                        
                        if images:
                            st.success(f"✅ Generated {len(images)} images with Flux!")
                            
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
                            st.error("Failed to generate any images. Please check your API keys and try again.")
                            
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
        st.info("Fine-tune your Flux image generation with advanced settings")
        
        st.subheader("API Settings")
        
        if st.secrets.get('HUGGINGFACE_API_KEY'):
            st.success("✅ Hugging Face API configured")
        else:
            st.warning("❌ Hugging Face API not configured")
            
        if st.secrets.get('TOGETHER_API_KEY'):
            st.success("✅ Together AI API configured")
        else:
            st.warning("❌ Together AI API not configured")
        
        st.subheader("Flux Model Options")
        st.write("""
        **Available Models:**
        - FLUX.1-schnell: Fast generation (4 steps)
        - FLUX.1-dev: Higher quality (28 steps)
        
        **Free Tier Limits:**
        - Hugging Face: Rate limited, may have queues
        - Together AI: Free credits available
        """)

if __name__ == "__main__":
    # Run the async app
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy() if hasattr(asyncio, 'WindowsProactorEventLoopPolicy') else asyncio.DefaultEventLoopPolicy())
    setup_dreamcanvas_app()
