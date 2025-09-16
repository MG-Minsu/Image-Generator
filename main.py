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
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
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

def enhance_prompt_with_gemini(text: str) -> str:
    """Use Gemini to create an enhanced visual prompt from subtitle text"""
    prompt = f"""Transform this subtitle text into a cinematic visual description for AI image generation.

SUBTITLE TEXT: "{text}"

REQUIREMENTS:
- Create a visual scene description (10-20 words)
- Focus on what can be SEEN, not dialogue
- Include setting, mood, actions, lighting if relevant
- Make it cinematic and descriptive
- Avoid quotes or dialogue text

OUTPUT: Just return the enhanced visual description, nothing else.

Example input: "Hello there. How are you doing today?"
Example output: Two people having a friendly conversation in a bright, comfortable living room"""

    try:
        response = gemini_model.generate_content(prompt)
        enhanced = response.text.strip()
        enhanced = enhanced.strip('"\'')
        return enhanced
    except Exception as e:
        return text

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
st.write("Upload an SRT subtitle file and generate images")

# Main configuration
col1, col2 = st.columns([1, 1])
with col1:
    processing_mode = st.selectbox(
        "Processing Mode",
        ["Gemini Enhanced (Group by 2)", "Individual Subtitles"],
        help="Choose how to process your subtitles"
    )

# Sidebar configuration
with st.sidebar:
    st.header("Image Settings")
    
    aspect_ratio = st.selectbox(
        "Select Aspect Ratio *",
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
    st.caption(f"Image dimensions: {width} × {height} pixels")

# File upload
uploaded_file = st.file_uploader(
    "Upload SRT file",
    type=['srt'],
    help="Upload a subtitle file (.srt format)"
)

if uploaded_file is not None:
    srt_content = uploaded_file.read().decode('utf-8')
    subtitles = parse_srt(srt_content)
    
    if not subtitles:
        st.error("Could not parse any subtitles from the uploaded file. Please check the SRT format.")
    else:
        st.success(f"Successfully parsed {len(subtitles)} subtitle entries")
        
        # Process based on selected mode
        use_gemini = processing_mode.startswith("Gemini Enhanced")
        
        if use_gemini:
            st.info("🤖 Grouping subtitles by 2 and using Gemini AI to create enhanced prompts...")
            grouped_entries = group_subtitles_by_two(subtitles)
            
            entries = []
            progress_bar = st.progress(0)
            for i, (timestamp, combined_text) in enumerate(grouped_entries):
                progress_bar.progress(i / len(grouped_entries))
                enhanced_prompt = enhance_prompt_with_gemini(combined_text)
                entries.append((timestamp, enhanced_prompt))
            progress_bar.progress(1.0)
            st.success("✅ Gemini enhancement complete!")
            
        else:
            entries = process_individual_subtitles(subtitles)
        
        st.success(f"Found {len(entries)} entries to process")
        
        # Show preview
        preview_title = "Preview Enhanced Entries" if use_gemini else "Preview Subtitle Entries"
        with st.expander(preview_title):
            for i, (timestamp, text) in enumerate(entries):
                prefix = "Enhanced:" if use_gemini else "Text:"
                st.text(f"{i+1}. [{timestamp}] {prefix} {text}")
        
        if entries:
            if st.button("🎨 Generate All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generated_images = []
                image_data_for_download = []
                
                for i, (timestamp, text) in enumerate(entries):
                    status_text.text(f"Generating image {i+1} of {len(entries)}...")
                    progress_bar.progress((i) / len(entries))
                    
                    image = generate_image(text, width, height)
                    
                    if image:
                        generated_images.append((image, text, timestamp))
                        
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        filename = f"{timestamp}.png"
                        image_data_for_download.append((buf.getvalue(), filename))
                
                progress_bar.progress(1.0)
                status_text.text("All images generated!")
                
                if generated_images:
                    st.subheader("Generated Images")
                    
                    for i, (image, prompt, timestamp) in enumerate(generated_images):
                        st.markdown(f"### Image {i+1}: `{timestamp}`")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.image(image, use_container_width=True)
                        
                        with col2:
                            label = "Enhanced prompt:" if use_gemini else "Original text:"
                            st.write(f"**{label}** {prompt}")
                            
                            buf = BytesIO()
                            image.save(buf, format='PNG')
                            st.download_button(
                                label=f"Download",
                                data=buf.getvalue(),
                                file_name=f"{timestamp}.png",
                                mime="image/png",
                                key=f"download_{i}_{timestamp}"
                            )
                        
                        st.divider()
                    
                    if len(image_data_for_download) > 1:
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for img_data, filename in image_data_for_download:
                                zip_file.writestr(filename, img_data)
                        
                        st.download_button(
                            label="📦 Download All Images (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="srt_generated_images.zip",
                            mime="application/zip",
                            key="download_all_images"
                        )

st.markdown("---")
st.markdown("Built with Streamlit 🚀 | Powered by Flux AI")
