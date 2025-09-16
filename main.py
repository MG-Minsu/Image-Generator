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
    layout="wide"
)

# Get API keys from secrets
try:
    api_key = st.secrets["REPLICATE_API_TOKEN"]
    client = replicate.Client(api_token=api_key)
except:
    st.error("Please add REPLICATE_API_TOKEN to your Streamlit secrets")
    st.stop()

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    client1 = replicate.Client(api_key=gemini_api_key)
    gemini_model = client.run('gemini-1.5-flash')
except:
    st.error("Please add GEMINI_API_KEY to your Streamlit secrets")
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
                start_time = start_time.strip()
                end_time = end_time.strip()
            else:
                start_time = timestamp
                end_time = timestamp
            
            subtitles.append((timestamp, start_time, end_time, text))
    
    return subtitles

def group_subtitles_by_two(subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Group subtitles in pairs of 2 entries"""
    grouped_entries = []
    
    for i in range(0, len(subtitles), 2):
        current = subtitles[i]
        next_subtitle = subtitles[i + 1] if i + 1 < len(subtitles) else None
        
        timestamp = current[1]
        clean_timestamp = re.sub(r'[^\w:,\-_]', '_', timestamp)
        
        combined_text = current[3].strip()
        if next_subtitle:
            combined_text += " " + next_subtitle[3].strip()
        
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

def describe_scene_with_gemini(text: str) -> str:
    """Use Gemini to describe what's happening in the sentence for visual representation"""
    prompt = f"""Analyze this subtitle text and create a visual scene description optimized for AI image generation.

SUBTITLE TEXT: "{text}"

TASK: Describe the visual scene in a way that will work perfectly with AI image generation models like Flux. Focus on:
- What actions are taking place
- Setting/environment details
- Characters and their positioning
- Visual elements, objects, colors
- Mood, lighting, atmosphere
- Camera perspective

RULES:
- Write 25-40 words that work well for AI image generation
- Use specific, visual keywords
- Include cinematic/photographic terms when appropriate
- Be concrete about visual elements
- Focus on what would make a compelling image
- Don't just repeat dialogue - interpret the visual scene

EXAMPLE:
Input: "Hello, how are you today?"
Output: Two people facing each other in friendly greeting gesture, warm indoor lighting, modern casual setting, medium shot composition, natural expressions, contemporary clothing

OUTPUT: Return only the optimized scene description for image generation."""

    try:
        response = gemini_model.generate_content(prompt)
        description = response.text.strip()
        description = description.strip('"\'')
        return description
    except Exception as e:
        # Fallback to original text if Gemini fails
        return f"A cinematic scene depicting: {text}"

def generate_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """Generate image using Flux model"""
    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
                "num_inference_steps": 4
            }
        )
        
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# App title and description
st.title("🎬 SRT Image Generator")
st.write("Upload an SRT subtitle file and generate images with Gemini scene descriptions")

# Configuration
st.subheader("⚙️ Configuration")

col1, col2 = st.columns(2)
with col1:
    grouping_mode = st.selectbox(
        "Subtitle Grouping",
        ["Group by 2 subtitles", "Individual subtitles"],
        help="How to group your subtitle entries"
    )

with col2:
    use_gemini_description = st.selectbox(
        "Scene Description",
        ["Use Gemini to describe scenes", "Use original subtitle text"],
        help="How to create image prompts"
    )

# Sidebar for image settings
with st.sidebar:
    st.header("🖼️ Image Settings")
    
    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        ["16:9 (Widescreen)", "1:1 (Square)", "4:3 (Classic)", "3:4 (Portrait)", "9:16 (Vertical)"],
        help="Choose the aspect ratio for generated images"
    )
    
    size_option = st.selectbox(
        "Image Size",
        ["Small (512px)", "Medium (768px)", "Large (1024px)"],
        index=1,
        help="Choose the base size for images"
    )
    
    def get_dimensions(ratio_text, size_text):
        if "512" in size_text:
            base_size = 512
        elif "768" in size_text:
            base_size = 768
        else:
            base_size = 1024
        
        if "1:1" in ratio_text:
            return base_size, base_size
        elif "16:9" in ratio_text:
            width = base_size
            height = int(base_size * 9 / 16)
            return width, height
        elif "4:3" in ratio_text:
            width = base_size
            height = int(base_size * 3 / 4)
            return width, height
        elif "3:4" in ratio_text:
            width = int(base_size * 3 / 4)
            height = base_size
            return width, height
        elif "9:16" in ratio_text:
            width = int(base_size * 9 / 16)
            height = base_size
            return width, height
        else:
            return base_size, base_size
    
    width, height = get_dimensions(aspect_ratio, size_option)
    st.caption(f"📐 Image dimensions: {width} × {height} pixels")

st.divider()

# File upload
st.subheader("📁 Upload SRT File")
uploaded_file = st.file_uploader(
    "Choose your SRT file",
    type=['srt'],
    help="Upload a subtitle file (.srt format)"
)

if uploaded_file is not None:
    # Parse SRT file
    srt_content = uploaded_file.read().decode('utf-8')
    subtitles = parse_srt(srt_content)
    
    if not subtitles:
        st.error("❌ Could not parse any subtitles from the uploaded file. Please check the SRT format.")
    else:
        st.success(f"✅ Successfully parsed {len(subtitles)} subtitle entries")
        
        # Process subtitles based on grouping mode
        st.subheader("📝 Processing Subtitles")
        
        if grouping_mode.startswith("Group by 2"):
            st.info("👥 Grouping subtitles by pairs of 2...")
            processed_entries = group_subtitles_by_two(subtitles)
        else:
            st.info("📄 Processing each subtitle individually...")
            processed_entries = process_individual_subtitles(subtitles)
        
        st.write(f"**Processed entries:** {len(processed_entries)}")
        
        # Show preview of processed entries
        with st.expander("👀 Preview Processed Entries"):
            for i, (timestamp, text) in enumerate(processed_entries[:10]):  # Show first 10
                st.text(f"{i+1}. [{timestamp}] {text}")
            if len(processed_entries) > 10:
                st.text(f"... and {len(processed_entries) - 10} more entries")
        
        # Create scene descriptions
        st.subheader("🎭 Scene Descriptions")
        
        if use_gemini_description.startswith("Use Gemini"):
            st.info("🤖 Using Gemini to create optimized scene descriptions for image generation...")
            
            # Create descriptions with progress tracking
            scene_descriptions = []
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (timestamp, text) in enumerate(processed_entries):
                    status_text.text(f"Creating optimized scene description {i+1} of {len(processed_entries)}...")
                    progress_bar.progress(i / len(processed_entries))
                    
                    description = describe_scene_with_gemini(text)
                    scene_descriptions.append((timestamp, text, description))
                
                progress_bar.progress(1.0)
                status_text.text("✅ All optimized scene descriptions created!")
            
        else:
            st.info("📝 Using original subtitle text...")
            scene_descriptions = [(timestamp, text, text) for timestamp, text in processed_entries]
        
        # Show preview of scene descriptions
        with st.expander("🎬 Preview Scene Descriptions"):
            for i, (timestamp, original, description) in enumerate(scene_descriptions[:10]):
                st.write(f"**{i+1}. [{timestamp}]**")
                st.write(f"*Original:* {original}")
                st.write(f"*Scene Description:* {description}")
                st.write("---")
            if len(scene_descriptions) > 10:
                st.text(f"... and {len(scene_descriptions) - 10} more descriptions")
        
        # Generate images
        st.subheader("🎨 Generate Images")
        
        if st.button("🚀 Generate All Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_images = []
            image_data_for_download = []
            
            for i, (timestamp, original_text, description) in enumerate(scene_descriptions):
                status_text.text(f"Generating image {i+1} of {len(scene_descriptions)}...")
                progress_bar.progress(i / len(scene_descriptions))
                
                # Use the scene description as the prompt for Flux AI
                image = generate_image(description, width, height)
                
                if image:
                    generated_images.append((image, description, original_text, timestamp))
                    
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    filename = f"{timestamp}.png"
                    image_data_for_download.append((buf.getvalue(), filename))
            
            progress_bar.progress(1.0)
            status_text.text("🎉 All images generated!")
            
            # Display generated images
            if generated_images:
                st.subheader("🖼️ Generated Images")
                
                for i, (image, description, original_text, timestamp) in enumerate(generated_images):
                    st.markdown(f"### 🎬 Image {i+1}: `{timestamp}`")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.write("**📝 Original Subtitle:**")
                        st.write(original_text)
                        st.write("**🎭 Scene Description (Used for Image):**")
                        st.write(description)
                        
                        # Download button
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        st.download_button(
                            label="💾 Download Image",
                            data=buf.getvalue(),
                            file_name=f"{timestamp}.png",
                            mime="image/png",
                            key=f"download_{i}_{timestamp}",
                            use_container_width=True
                        )
                    
                    st.divider()
                
                # Bulk download
                if len(image_data_for_download) > 1:
                    st.subheader("📦 Bulk Download")
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for img_data, filename in image_data_for_download:
                            zip_file.writestr(filename, img_data)
                    
                    st.download_button(
                        label="📦 Download All Images (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="srt_generated_images.zip",
                        mime="application/zip",
                        key="download_all_images",
                        use_container_width=True
                    )

st.markdown("---")
st.markdown("🚀 **Built with Streamlit** | 🤖 **Powered by Flux AI & Gemini**")
