import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO
import re
from typing import List, Tuple
import zipfile
import google.generativeai as genai

# Set page config
st.set_page_config(
    page_title="SRT Image Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .config-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    
    .info-box {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        padding: 1rem;
        border-radius: 5px;
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API clients
@st.cache_resource
def init_apis():
    try:
        replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        return replicate_client, gemini_model
    except Exception as e:
        st.error("❌ Please configure your API keys in Streamlit secrets")
        st.stop()

def parse_srt(srt_content: str) -> List[Tuple[str, str, str, str]]:
    """Parse SRT content and return list of (timestamp, start_time, end_time, text) tuples"""
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    subtitles = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            seq_num = lines[0].strip()
            timestamp = lines[1].strip()
            text = ' '.join(lines[2:]).strip()
            
            if ' --> ' in timestamp:
                start_time, end_time = timestamp.split(' --> ')
                subtitles.append((timestamp.strip(), start_time.strip(), end_time.strip(), text))
    
    return subtitles

def group_subtitles(subtitles: List[Tuple[str, str, str, str]], group_size: int = 2) -> List[Tuple[str, str]]:
    """Group subtitles by specified size"""
    grouped_entries = []
    
    for i in range(0, len(subtitles), group_size):
        group = subtitles[i:i+group_size]
        timestamp = group[0][1]
        clean_timestamp = re.sub(r'[^\w:,\-_]', '_', timestamp)
        
        combined_text = " ".join([subtitle[3].strip() for subtitle in group if subtitle[3].strip()])
        
        if combined_text:
            grouped_entries.append((clean_timestamp, combined_text))
    
    return grouped_entries

def process_individual_subtitles(subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Process each subtitle individually"""
    entries = []
    for timestamp, start_time, end_time, text in subtitles:
        text = text.strip()
        if text:
            clean_timestamp = re.sub(r'[^\w:,\-_]', '_', start_time)
            entries.append((clean_timestamp, text))
    return entries

def describe_scene_with_gemini(text: str, style_prompt: str, model) -> str:
    """Use Gemini to describe scenes for image generation"""
    prompt = f"""Create a visual scene description optimized for AI image generation from this subtitle:

SUBTITLE: "{text}"

Create a 25-40 word description focusing on:
- Visual actions and settings
- Character positioning and expressions  
- Lighting and atmosphere
- Cinematic composition

Style requirements: {style_prompt if style_prompt else 'cinematic realism'}

Return only the optimized scene description."""

    try:
        response = model.generate_content(prompt)
        description = response.text.strip().strip('"\'')
        
        if style_prompt and style_prompt.lower() not in description.lower():
            description = f"{description}, {style_prompt}"
        
        return description
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return create_fallback_description(text, style_prompt)
        else:
            st.warning(f"Gemini API error: {str(e)}. Using fallback.")
            return create_fallback_description(text, style_prompt)

def create_fallback_description(text: str, style_prompt: str = "") -> str:
    """Create fallback descriptions when Gemini is unavailable"""
    text_lower = text.lower()
    
    # Detect scene type
    if any(word in text_lower for word in ['running', 'walking', 'moving']):
        base = "Person in dynamic motion"
    elif any(word in text_lower for word in ['talking', 'speaking', 'conversation']):
        base = "Characters in conversation"
    elif any(word in text_lower for word in ['looking', 'watching', 'staring']):
        base = "Focused observation scene"
    else:
        base = "Cinematic scene"
    
    # Detect setting
    if any(word in text_lower for word in ['house', 'home', 'room']):
        setting = "indoor domestic environment"
    elif any(word in text_lower for word in ['office', 'work']):
        setting = "professional office setting"
    elif any(word in text_lower for word in ['outside', 'street', 'car']):
        setting = "urban outdoor location"
    else:
        setting = "atmospheric setting"
    
    # Include the subtitle text in the description for context
    description = f"{base} in {setting}, depicting: {text}, cinematic lighting, professional composition"
    
    if style_prompt:
        description += f", {style_prompt}"
    
    return description

def generate_image(prompt: str, client) -> Image.Image:
    """Generate image using Flux model"""
    try:
        output = client.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "width": 1024,
                "height": 574,
                "num_outputs": 1,
                "num_inference_steps": 4
            }
        )
        
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# Initialize APIs
replicate_client, gemini_model = init_apis()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 SRT Image Generator</h1>
    <p>Transform subtitle files into stunning visual scenes</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    processing_mode = st.radio(
        "Subtitle Processing",
        ["Individual subtitles", "Group subtitles"],
        help="Choose how to process subtitle entries"
    )
    
    if processing_mode == "Group subtitles":
        group_size = st.number_input(
            "Group Size",
            min_value=2,
            max_value=50,
            value=2,
            help="Number of subtitles to group together"
        )
    else:
        group_size = 1
    
    description_mode = st.radio(
        "Scene Description",
        ["Enhanced AI descriptions", "Basic descriptions"],
        help="AI descriptions provide better visual prompts"
    )
    
    st.markdown("---")
    
    style_prompt = st.text_area(
        "🎨 Visual Style (Optional)",
        placeholder="e.g., 'cyberpunk neon', 'watercolor painting', 'film noir'",
        height=80,
        help="Add consistent visual style to all images"
    )
    
    st.markdown("---")
    st.info("📐 Images: 1024×574px (16:9)")
    
    force_fallback = st.checkbox(
        "Skip AI descriptions",
        help="Use basic descriptions only"
    )

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # File Upload
    st.subheader("📁 Upload SRT File")
    uploaded_file = st.file_uploader(
        "Choose your subtitle file",
        type=['srt'],
        help="Upload a valid .srt subtitle file"
    )

with col2:
    if uploaded_file:
        st.subheader("📊 Quick Stats")
        # We'll populate this after parsing

# Process uploaded file
if uploaded_file is not None:
    # Parse SRT
    try:
        try:
            srt_content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            srt_content = uploaded_file.read().decode('latin-1')
        
        subtitles = parse_srt(srt_content)
        
        if not subtitles:
            st.error("❌ No subtitles found. Please check your SRT file format.")
            st.stop()
        
        # Update stats
        with col2:
            st.metric("Total Subtitles", len(subtitles))
            
            # Estimate duration
            if len(subtitles) > 0:
                try:
                    last_time = subtitles[-1][2]  # end_time of last subtitle
                    time_parts = last_time.split(':')
                    total_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
                    st.metric("Duration", f"~{total_minutes} min")
                except:
                    st.metric("Duration", "Unknown")
        
        # Process subtitles based on mode
        if processing_mode == "Individual subtitles":
            processed_entries = process_individual_subtitles(subtitles)
        else:  # Group subtitles
            processed_entries = group_subtitles(subtitles, group_size)
        
        st.markdown(f"""
        <div class="success-box">
            ✅ Successfully processed <strong>{len(processed_entries)}</strong> entries from <strong>{len(subtitles)}</strong> subtitles
        </div>
        """, unsafe_allow_html=True)
        
        # Preview processed entries
        with st.expander("👀 Preview Processed Entries"):
            show_all_entries = st.checkbox("Show all entries", key="show_all_processed")
            
            entries_to_show = processed_entries if show_all_entries else processed_entries[:5]
            
            for i, (timestamp, text) in enumerate(entries_to_show):
                st.markdown(f"**{i+1}.** `{timestamp}`")
                st.write(text)
                st.markdown("---")
            
            if not show_all_entries and len(processed_entries) > 5:
                st.info(f"Showing first 5 of {len(processed_entries)} entries. Check 'Show all entries' to see more.")
        
        # Generate scene descriptions
        st.subheader("🎭 Scene Descriptions")
        
        if description_mode == "Enhanced AI descriptions" and not force_fallback:
            st.info("🤖 Creating AI-optimized scene descriptions...")
            
            scene_descriptions = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (timestamp, text) in enumerate(processed_entries):
                status_text.text(f"Processing {i+1}/{len(processed_entries)}: {text[:50]}...")
                progress_bar.progress(i / len(processed_entries))
                
                description = describe_scene_with_gemini(text, style_prompt, gemini_model)
                scene_descriptions.append((timestamp, text, description))
            
            progress_bar.progress(1.0)
            status_text.success("✅ All descriptions created!")
        
        else:
            st.info("📝 Using basic scene descriptions...")
            scene_descriptions = []
            for timestamp, text in processed_entries:
                if style_prompt.strip():
                    description = f"A cinematic scene depicting: {text}, {style_prompt.strip()}"
                else:
                    description = f"A cinematic scene depicting: {text}"
                scene_descriptions.append((timestamp, text, description))
        
        # Preview scene descriptions
        with st.expander("🎬 Preview Scene Descriptions"):
            show_all_descriptions = st.checkbox("Show all descriptions", key="show_all_descriptions")
            
            descriptions_to_show = scene_descriptions if show_all_descriptions else scene_descriptions[:3]
            
            for i, (timestamp, original, description) in enumerate(descriptions_to_show):
                st.markdown(f"**{i+1}.** `{timestamp}`")
                st.markdown(f"*Original:* {original}")
                st.markdown(f"*Description:* {description}")
                st.markdown("---")
            
            if not show_all_descriptions and len(scene_descriptions) > 3:
                st.info(f"Showing first 3 of {len(scene_descriptions)} descriptions. Check 'Show all descriptions' to see more.")
        
        # Cost estimate
        estimated_cost = len(scene_descriptions) * 0.003
        st.markdown(f"""
        <div class="info-box">
            💰 <strong>Estimated cost:</strong> ~${estimated_cost:.3f} USD for {len(scene_descriptions)} images
        </div>
        """, unsafe_allow_html=True)
        
        # Generate images
        st.subheader("🎨 Generate Images")
        
        col_gen1, col_gen2 = st.columns(2)
        
        with col_gen1:
            generate_all = st.button("🚀 Generate All Images", type="primary", use_container_width=True)
        
        with col_gen2:
            if len(scene_descriptions) > 5:
                generate_sample = st.button("🎯 Generate 5 Sample Images", use_container_width=True)
            else:
                generate_sample = False
        
        if generate_all or generate_sample:
            descriptions_to_use = scene_descriptions[:5] if generate_sample else scene_descriptions
            
            # Initialize session state for storing images if not exists
            if 'generated_images' not in st.session_state:
                st.session_state.generated_images = []
            if 'image_data_for_download' not in st.session_state:
                st.session_state.image_data_for_download = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_images = []
            image_data_for_download = []
            
            for i, (timestamp, original_text, description) in enumerate(descriptions_to_use):
                status_text.text(f"Generating image {i+1}/{len(descriptions_to_use)}...")
                progress_bar.progress(i / len(descriptions_to_use))
                
                image = generate_image(description, replicate_client)
                
                if image:
                    generated_images.append((image, description, original_text, timestamp))
                    
                    # Prepare for download
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    buf.seek(0)
                    image_data_for_download.append((buf.getvalue(), f"{timestamp}.png"))
            
            progress_bar.progress(1.0)
            status_text.success(f"🎉 Generated {len(generated_images)} images!")
            
            # Store in session state to prevent disappearing on download
            st.session_state.generated_images = generated_images
            st.session_state.image_data_for_download = image_data_for_download
            
            # Display results
            if st.session_state.generated_images:
                st.subheader("🖼️ Generated Images")
                
                # Bulk download button
                if len(st.session_state.image_data_for_download) > 1:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for img_data, filename in st.session_state.image_data_for_download:
                            zip_file.writestr(filename, img_data)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="📦 Download All Images (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="srt_generated_images.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="download_all_zip"
                    )
                    st.markdown("---")
                
                # Display images
                for i, (image, description, original_text, timestamp) in enumerate(st.session_state.generated_images):
                    with st.container():
                        st.markdown(f"### 🎬 Scene {i+1}: `{timestamp}`")
                        
                        img_col, info_col = st.columns([3, 2])
                        
                        with img_col:
                            st.image(image, use_container_width=True)
                        
                        with info_col:
                            st.markdown("**📝 Original Subtitle:**")
                            st.write(original_text)
                            
                            st.markdown("**🎭 Scene Description:**")
                            st.write(description)
                            
                            # Individual download
                            buf = BytesIO()
                            image.save(buf, format='PNG')
                            buf.seek(0)
                            st.download_button(
                                label="💾 Download",
                                data=buf.getvalue(),
                                file_name=f"scene_{i+1}_{timestamp}.png",
                                mime="image/png",
                                key=f"download_individual_{i}_{timestamp}",
                                use_container_width=True
                            )
                        
                        st.markdown("---")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Display previously generated images if they exist in session state
elif 'generated_images' in st.session_state and st.session_state.generated_images:
    st.subheader("🖼️ Previously Generated Images")
    
    # Bulk download button for previous images
    if len(st.session_state.image_data_for_download) > 1:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_data, filename in st.session_state.image_data_for_download:
                zip_file.writestr(filename, img_data)
        
        zip_buffer.seek(0)
        st.download_button(
            label="📦 Download All Previous Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="srt_generated_images.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_previous_all_zip"
        )
        st.markdown("---")
    
    # Display previous images
    for i, (image, description, original_text, timestamp) in enumerate(st.session_state.generated_images):
        with st.expander(f"🎬 Scene {i+1}: {timestamp}"):
            img_col, info_col = st.columns([2, 1])
            with img_col:
                st.image(image, use_container_width=True)
            with info_col:
                st.markdown("**📝 Original:**")
                st.write(original_text)
                st.markdown("**🎭 Description:**")
                st.write(description)
                
                # Individual download for previous images
                buf = BytesIO()
                image.save(buf, format='PNG')
                buf.seek(0)
                st.download_button(
                    label="💾 Download",
                    data=buf.getvalue(),
                    file_name=f"scene_{i+1}_{timestamp}.png",
                    mime="image/png",
                    key=f"download_previous_{i}_{timestamp}",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🚀 Built with Streamlit | 🤖 Powered by Flux AI & Gemini</div>", 
    unsafe_allow_html=True
)
