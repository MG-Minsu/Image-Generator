import streamlit as st
import base64
import asyncio
import aiohttp
import io
import zipfile
from PIL import Image
import google.generativeai as genai
import requests

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Prompt Builder ---
def create_visual_prompt(story_text, style_choice, mood_setting, art_styles, color_palette):
    color_desc = ", ".join(color_palette).lower() + " color scheme" if color_palette else ""
 
    return f"{story_text}, {art_styles[style_choice]}, {mood_setting}, {color_desc}, masterpiece quality"

def generate_enhanced_scene_prompt(scene_text):
    """Use Gemini to enhance scene descriptions for better image generation"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        enhancement_prompt = f"""
        Transform this story scene into a detailed visual description suitable for image generation:
        
        Scene: {scene_text}
        
        Create a vivid, detailed description that includes:
        - Main characters and their expressions
        - Setting and environment details
        - Lighting and atmosphere
        - Visual style elements
        - Composition suggestions
        
        Keep it concise but visually rich, suitable for children's book illustration style.
        """
        
        response = model.generate_content(enhancement_prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Scene enhancement failed: {str(e)}")
        return scene_text

async def generate_story_images_with_dalle(story_text, num_images):
    """Generate images using DALL-E via external API (since Gemini doesn't generate images directly)"""
    # Note: You'll still need an image generation service
    # This is a placeholder for image generation logic
    
    story_lines = story_text.split("\n\n")
    
    if len(story_lines) < num_images:
        scenes = story_lines * (num_images // len(story_lines) + 1)
        scenes = scenes[:num_images]
    else:
        scenes = story_lines[:num_images]
    
    images = []
    
    # Use Gemini to enhance each scene description
    enhanced_scenes = []
    for scene in scenes:
        enhanced_scene = generate_enhanced_scene_prompt(scene)
        enhanced_scenes.append(enhanced_scene)
    
    # For actual image generation, you would need to use:
    # 1. A different image generation API (DALL-E, Midjourney, Stable Diffusion)
    # 2. Or integrate with Google's Imagen (if available)
    # 3. Or use a service like Replicate
    
    # Placeholder: Generate mock images (you'll replace this with actual image generation)
    for i, enhanced_scene in enumerate(enhanced_scenes):
        try:
            # This is where you'd call your image generation API
            # For now, creating a placeholder
            st.info(f"Enhanced scene {i+1}: {enhanced_scene[:100]}...")
            
            # Placeholder for actual image generation
            # You could use services like:
            # - Replicate API with Stable Diffusion
            # - Hugging Face Inference API
            # - Any other image generation service
            
        except Exception as e:
            st.error(f"Error processing scene {i+1}: {str(e)}")
            continue
    
    return images

def generate_story_with_gemini(user_input, style_preferences):
    """Generate or enhance story using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        story_prompt = f"""
        Based on this input: {user_input}
        
        Create or enhance this into a beautiful children's story with the following characteristics:
        - Style: {style_preferences}
        - Suitable for children
        - Vivid, descriptive scenes
        - Engaging narrative
        - 3-5 paragraphs
        - Each paragraph should be a distinct scene that could be illustrated
        
        Make sure each paragraph describes a specific moment or scene that would work well as an illustration.
        """
        
        response = model.generate_content(story_prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Story generation failed: {str(e)}")
        return user_input

def analyze_story_with_gemini(story_text):
    """Analyze story to provide insights and suggestions"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        analysis_prompt = f"""
        Analyze this children's story and provide:
        1. Main themes
        2. Key characters
        3. Visual elements that would work well in illustrations
        4. Suggested art style
        5. Color palette recommendations
        
        Story: {story_text}
        
        Provide a brief, helpful analysis.
        """
        
        response = model.generate_content(analysis_prompt)
        return response.text.strip()
    except Exception as e:
        return "Analysis unavailable"

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
    st.set_page_config(page_title="DreamCanvas with Gemini", page_icon="🎨", layout="wide")
    st.title("🎨 DreamCanvas — AI-Powered Stories with Gemini")
    
    # Example styles
    art_styles = {
        "Dreamscape": "ethereal, soft lighting, pastel colors, surreal atmosphere",
        "Comic Book": "bold outlines, vibrant colors, dynamic poses, speech bubbles",
        "Fantasy Art": "magical elements, rich colors, detailed textures, epic scale",
        "Watercolor": "soft brushstrokes, flowing colors, artistic texture",
        "Cartoon": "simple shapes, bright colors, playful style"
    }
    
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Create Art", "📚 Story Generator", "🔍 Story Analyzer", "⚙️ Advanced Studio"])
    
    with tab1:
        st.header("🖼️ Create Illustrated Stories")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_story = st.text_area(
                "📝 Write your story", 
                placeholder="Once upon a time, in a magical forest...",
                height=300
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
            
            num_images = st.number_input(
                "📸 How many images?", 
                min_value=1, 
                max_value=10,
                value=3, 
                step=1
            )
        
        if st.button("✨ Enhance Story & Create Scenes", type="primary"):
            if user_story.strip():
                with st.spinner("🤖 Gemini is analyzing your story..."):
                    # Use Gemini to enhance the story
                    style_desc = f"{art_styles[chosen_style]}, {mood_slider}"
                    enhanced_story = generate_story_with_gemini(user_story, style_desc)
                    
                    st.subheader("📖 Enhanced Story")
                    st.write(enhanced_story)
                    
                    # Analyze the story
                    analysis = analyze_story_with_gemini(enhanced_story)
                    
                    with st.expander("🔍 Story Analysis"):
                        st.write(analysis)
                    
                    # Generate scene descriptions for illustration
                    story_scenes = enhanced_story.split("\n\n")
                    
                    st.subheader("🎨 Scene Descriptions for Illustration")
                    for i, scene in enumerate(story_scenes[:num_images], 1):
                        enhanced_scene = generate_enhanced_scene_prompt(scene)
                        
                        with st.expander(f"Scene {i}"):
                            st.write("**Original Scene:**")
                            st.write(scene)
                            st.write("**Enhanced for Illustration:**")
                            st.write(enhanced_scene)
                    
                    st.info("💡 **Note:** To generate actual images, you'll need to integrate with an image generation service like DALL-E, Stable Diffusion, or Midjourney API, as Gemini focuses on text generation.")
            else:
                st.warning("📝 Please write a story first!")
    
    with tab2:
        st.header("📚 AI Story Generator")
        st.write("Let Gemini create a story based on your ideas!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            story_prompt = st.text_area(
                "💭 Describe your story idea:",
                placeholder="A brave little mouse who wants to become a knight...",
                height=150
            )
            
            story_elements = st.multiselect(
                "📋 Include these elements:",
                ["Magic", "Friendship", "Adventure", "Animals", "Castle", "Forest", "Ocean", "Dragons", "Treasure", "Mystery"],
                default=["Friendship", "Adventure"]
            )
        
        with col2:
            target_age = st.selectbox("👶 Target Age:", ["3-5 years", "6-8 years", "9-12 years"])
            story_length = st.select_slider("📏 Story Length:", ["Short", "Medium", "Long"], value="Medium")
            story_theme = st.selectbox("🎭 Theme:", ["Courage", "Kindness", "Learning", "Family", "Nature", "Magic"])
        
        if st.button("📝 Generate Story", type="primary"):
            if story_prompt.strip():
                with st.spinner("🤖 Gemini is creating your story..."):
                    elements_text = ", ".join(story_elements) if story_elements else ""
                    full_prompt = f"{story_prompt}. Include: {elements_text}. Theme: {story_theme}. For {target_age}."
                    
                    generated_story = generate_story_with_gemini(full_prompt, f"{story_length} length, {story_theme} theme")
                    
                    st.subheader("📖 Your Generated Story")
                    st.write(generated_story)
                    
                    # Offer to analyze the generated story
                    if st.button("🔍 Analyze This Story"):
                        analysis = analyze_story_with_gemini(generated_story)
                        st.subheader("📊 Story Analysis")
                        st.write(analysis)
    
    with tab3:
        st.header("🔍 Story Analyzer")
        st.write("Get AI insights about your story!")
        
        analysis_story = st.text_area(
            "📝 Paste your story here for analysis:",
            height=300
        )
        
        if st.button("🔍 Analyze Story", type="primary"):
            if analysis_story.strip():
                with st.spinner("🤖 Analyzing your story..."):
                    analysis = analyze_story_with_gemini(analysis_story)
                    
                    st.subheader("📊 Analysis Results")
                    st.write(analysis)
    
    with tab4:
        st.header("⚙️ Advanced Studio")
        st.info("Advanced features powered by Gemini AI")
        
        st.subheader("🎨 Custom Art Direction")
        art_direction = st.text_area(
            "Describe your artistic vision:",
            placeholder="I want a whimsical, hand-drawn style with soft watercolors...",
            height=100
        )
        
        st.subheader("🎭 Character Development")
        character_description = st.text_area(
            "Describe your main character:",
            placeholder="A small dragon with purple scales and kind eyes...",
            height=100
        )
        
        st.subheader("🌍 World Building")
        world_description = st.text_area(
            "Describe the story world:",
            placeholder="A floating city in the clouds with rainbow bridges...",
            height=100
        )
        
        if st.button("🚀 Generate Advanced Story Concept"):
            if any([art_direction.strip(), character_description.strip(), world_description.strip()]):
                with st.spinner("🤖 Creating advanced story concept..."):
                    advanced_prompt = f"""
                    Create a detailed children's story concept with:
                    Art Direction: {art_direction}
                    Main Character: {character_description}
                    World: {world_description}
                    
                    Provide a complete story outline with detailed scene descriptions.
                    """
                    
                    concept = generate_story_with_gemini(advanced_prompt, "detailed concept development")
                    
                    st.subheader("🎨 Advanced Story Concept")
                    st.write(concept)

if __name__ == "__main__":
    # Check if Gemini API key is configured
    try:
        setup_dreamcanvas_app()
    except Exception as e:
        st.error("Please configure your GEMINI_API_KEY in Streamlit secrets")
        st.stop()
