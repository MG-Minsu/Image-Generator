import streamlit as st
import requests
import asyncio
from datetime import datetime
import json

# Initialize image generation client (using Stability AI as example)
STABILITY_API_KEY = st.secrets["STABILITY_API_KEY"]
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# Creative story themes for image generation
visual_themes = [
    "Mystical forest creatures having tea parties",
    "Cyberpunk cities floating in the clouds", 
    "Ancient libraries with books that glow",
    "Time travelers meeting at cosmic cafes",
    "Dragons learning to paint watercolors",
    "Underwater kingdoms with coral castles",
    "Steampunk inventors and their flying machines",
    "Magical gardens where flowers sing",
    "Space pirates discovering new planets",
    "Victorian ghosts hosting elegant balls"
]

# Art style presets with unique characteristics
art_styles = {
    "Dreamscape": "ethereal, soft lighting, pastel colors, surreal atmosphere",
    "Comic Book": "bold outlines, vibrant colors, dynamic poses, speech bubbles",
    "Vintage Poster": "retro design, muted tones, minimalist composition",
    "Fantasy Art": "magical elements, rich colors, detailed textures, epic scale",
    "Minimalist": "clean lines, negative space, simple forms, monochromatic",
    "Impressionist": "loose brushstrokes, natural lighting, outdoor scenes",
    "Neon Noir": "dark atmosphere, neon lighting, rain-soaked streets, shadows"
}

def create_visual_prompt(story_text, style_choice, mood_setting):
    """Transform story into optimized image generation prompt"""
    base_prompt = f"{story_text}, {art_styles[style_choice]}, {mood_setting}, masterpiece quality"
    return base_prompt

async def generate_story_image(prompt_text, style_settings):
    """Generate image using AI API (placeholder for actual implementation)"""
    # This would connect to your chosen image generation API
    # Examples: DALL-E, Midjourney, Stability AI, or Replicate
    
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text_prompts": [{"text": prompt_text, "weight": 1}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "style_preset": style_settings.get("preset", "enhance")
    }
    
    # Simulate API call (replace with actual implementation)
    await asyncio.sleep(2)  # Simulate processing time
    return "https://example-generated-image-url.com"

def setup_dreamcanvas_app():
    
    # App header and branding
    st.set_page_config(page_title="DreamCanvas", page_icon="🎨", layout="wide")
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 3rem; margin: 0;'>🎨 DreamCanvas</h1>
        <p style='color: white; font-size: 1.2rem; margin: 0;'>Where Stories Become Art</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("**Transform your wildest stories into stunning visual masterpieces!** Whether you're a writer seeking inspiration, an educator creating learning materials, or someone who just loves to dream - DreamCanvas turns your imagination into reality.")
    
    # Main interface with tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Create Art", "🎭 Story Gallery", "⚙️ Advanced Studio"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Your Story Canvas")
            
            # Story input methods
            input_method = st.radio(
                "How would you like to create?",
                ["Free Writing", "Story Prompts", "Theme Mixer"]
            )
            
            user_story = ""
            if input_method == "Free Writing":
                user_story = st.text_area(
                    "Write your story here...",
                    height=150,
                    placeholder="Once upon a time, in a world where colors had personalities..."
                )
            
            elif input_method == "Story Prompts":
                selected_theme = st.selectbox("Choose a magical theme:", visual_themes)
                story_twist = st.text_input("Add your unique twist:", placeholder="but with a modern smartphone...")
                user_story = f"{selected_theme} {story_twist}"
            
            else:  # Theme Mixer
                character = st.selectbox("Main Character:", ["A brave knight", "A curious scientist", "A magical cat", "A time traveler"])
                setting = st.selectbox("Setting:", ["in an enchanted forest", "on a distant planet", "in a steampunk city", "inside a computer"])
                conflict = st.selectbox("Challenge:", ["must solve a riddle", "searches for a lost artifact", "breaks an ancient curse", "saves their world"])
                user_story = f"{character} {setting} {conflict}"
        
        with col2:
            st.subheader("🎨 Art Direction")
            
            chosen_style = st.selectbox("Visual Style:", list(art_styles.keys()))
            
            mood_slider = st.select_slider(
                "Mood Spectrum:",
                options=["Dark & Mysterious", "Calm & Peaceful", "Bright & Energetic", "Warm & Cozy", "Epic & Dramatic"]
            )
            
            color_palette = st.multiselect(
                "Color Focus:",
                ["Blues", "Reds", "Greens", "Purples", "Golds", "Pastels", "Monochrome"],
                default=["Blues", "Purples"]
            )
            
            image_count = st.slider("Number of Variations:", 1, 4, 2)
        
        # Generate button
        if st.button("✨ Create Magic", type="primary", use_container_width=True):
            if user_story.strip():
                st.subheader("🎭 Your Visual Story")
                
                for i in range(image_count):
                    with st.container():
                        col_img, col_details = st.columns([1, 1])
                        
                        with col_img:
                            with st.spinner(f"Painting variation {i+1}..."):
                                # Create the enhanced prompt
                                color_desc = ", ".join(color_palette).lower() + " color scheme"
                                full_prompt = create_visual_prompt(user_story, chosen_style, f"{mood_slider}, {color_desc}")
                                
                                # Generate image (placeholder)
                                image_url = asyncio.run(generate_story_image(full_prompt, {"preset": chosen_style.lower()}))
                                
                                # Display placeholder image
                                st.image("https://via.placeholder.com/512x512/667eea/white?text=Generated+Art", 
                                        caption=f"Variation {i+1}", use_column_width=True)
                        
                        with col_details:
                            st.write("**Story Source:**")
                            st.write(user_story[:100] + "...")
                            st.write("**Art Prompt:**")
                            st.code(full_prompt[:150] + "...")
                            st.write("**Style:** " + chosen_style)
                            st.write("**Mood:** " + mood_slider)
                            
                            if st.button(f"💾 Save Art {i+1}", key=f"save_{i}"):
                                st.success("Added to your gallery!")
                        
                        st.divider()
            else:
                st.warning("Please write a story first!")
    
    with tab2:
        st.subheader("🖼️ Community Gallery")
        st.write("Discover amazing creations from other dreamers...")
        
        # Mock gallery
        gallery_cols = st.columns(3)
        sample_titles = ["Dragon's Tea Party", "Cyberpunk Garden", "Time Traveler's Dilemma"]
        
        for i, col in enumerate(gallery_cols):
            with col:
                st.image(f"https://via.placeholder.com/300x300/764ba2/white?text=Art+{i+1}", 
                        caption=sample_titles[i])
                st.write(f"⭐ 4.{8+i}/5 • 👁️ {120+i*30} views")
                if st.button(f"View Details", key=f"gallery_{i}"):
                    st.info("Gallery feature coming soon!")
    
    with tab3:
        st.subheader("⚙️ Advanced Creation Studio")
        st.write("Fine-tune every aspect of your visual story...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Composition Controls**")
            composition = st.selectbox("Frame Type:", ["Wide Landscape", "Portrait", "Square", "Cinematic"])
            perspective = st.selectbox("Viewpoint:", ["Eye Level", "Bird's Eye", "Low Angle", "Close-up"])
            lighting = st.selectbox("Lighting:", ["Natural", "Dramatic", "Soft", "Neon", "Candlelit"])
        
        with col2:
            st.write("**Technical Settings**")
            quality = st.slider("Quality Level:", 1, 10, 8)
            creativity = st.slider("AI Creativity:", 0.1, 2.0, 1.0, 0.1)
            style_strength = st.slider("Style Influence:", 0.1, 1.0, 0.7, 0.1)
        
        st.write("**Negative Prompts** (What to avoid)")
        avoid_elements = st.text_input("Exclude from image:", placeholder="blurry, text, watermark...")
    
    # Footer
    with st.expander("💡 Pro Tips for Better Results"):
        st.write("""
        **Writing Effective Stories for AI Art:**
        • Be specific about visual details (colors, textures, lighting)
        • Include emotional context (mysterious, cheerful, ancient)
        • Describe the scene composition (foreground, background)
        • Use sensory language (glowing, misty, crystalline)
        
        **Style Combinations:**
        • Mix genres: "Cyberpunk + Impressionist"
        • Add time periods: "Victorian Steampunk"
        • Include artistic movements: "Art Nouveau fantasy"
        
        **Advanced Techniques:**
        • Layer multiple scenes for complex narratives
        • Use consistent character descriptions across images
        • Experiment with unusual perspectives and angles
        """)
    
    st.markdown("---")
    st.write("🎨 **DreamCanvas AI** - Bringing Stories to Life • Created with ❤️ for Creative Minds")

if __name__ == "__main__":
    setup_dreamcanvas_app()
