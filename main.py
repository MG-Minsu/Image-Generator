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
    gemini_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Please add GEMINI_API_KEY to your Streamlit secrets")
    st.stop()

def parse_srt(srt_content: str) -> List[Tuple[str, str, str, str]]:
    """Parse SRT content and return list of (timestamp, start_time, end_time, text) tuples"""
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    subtitles = []
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # First line is the sequence number (we'll skip this)
            seq_num = lines[0].strip()
            # Second line is the timestamp
            timestamp = lines[1].strip()
            # Remaining lines are the subtitle text
            text = ' '.join(lines[2:]).strip()
            
            # Extract start and end times
            if ' --> ' in timestamp:
                start_time, end_time = timestamp.split(' --> ')
                start_time = start_time.strip()
                end_time = end_time.strip()
            else:
                start_time = timestamp
                end_time = timestamp
            
            subtitles.append((timestamp, start_time, end_time, text))
    
    return subtitles

def group_subtitles_with_gemini(subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Use Gemini to group subtitles into complete sentences and create enhanced prompts"""
    if not subtitles:
        return []
    
    # Prepare subtitle text for Gemini
    subtitle_text = ""
    for i, (timestamp, start_time, end_time, text) in enumerate(subtitles):
        subtitle_text += f"{i+1}. [{start_time}] {text}\n"
    
    prompt = f"""
    Please analyze these subtitles and group them into complete, meaningful sentences. Then create enhanced visual prompts suitable for AI image generation.

    Subtitles:
    {subtitle_text}

    Instructions:
    1. Group related subtitle fragments into complete sentences
    2. Each group should represent a coherent visual scene or moment
    3. For each group, provide:
       - The timestamp of the first subtitle in that group (format: HH:MM:SS)
       - An enhanced visual prompt that describes what should be shown in an image
    4. The enhanced prompts should be cinematic and descriptive, focusing on visual elements, settings, actions, and mood
    5. Avoid dialogue-heavy prompts; instead describe the scene visually

    Format your response exactly like this:
    TIMESTAMP: HH:MM:SS
    PROMPT: [Enhanced visual description]

    TIMESTAMP: HH:MM:SS  
    PROMPT: [Enhanced visual description]

    (Continue for all groups...)
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return parse_gemini_response(response.text)
    except Exception as e:
        st.error(f"Error using Gemini API: {str(e)}")
        # Fallback to original processing
        return process_srt_entries(subtitles)

def parse_gemini_response(response_text: str) -> List[Tuple[str, str]]:
    """Parse Gemini's response to extract timestamps and prompts"""
    entries = []
    lines = response_text.strip().split('\n')
    
    current_timestamp = None
    current_prompt = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('TIMESTAMP:'):
            if current_timestamp and current_prompt:
                # Clean timestamp for filename
                clean_timestamp = re.sub(r'[^\w:,\-_]', '_', current_timestamp)
                entries.append((clean_timestamp, current_prompt))
            
            current_timestamp = line.replace('TIMESTAMP:', '').strip()
            current_prompt = None
            
        elif line.startswith('PROMPT:'):
            current_prompt = line.replace('PROMPT:', '').strip()
    
    # Don't forget the last entry
    if current_timestamp and current_prompt:
        clean_timestamp = re.sub(r'[^\w:,\-_]', '_', current_timestamp)
        entries.append((clean_timestamp, current_prompt))
    
    return entries

def process_srt_entries(subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Fallback function: Process SRT entries to use one entry per image with timestamps"""
    entries = []
    
    for timestamp, start_time, end_time, text in subtitles:
        text = text.strip()
        if text:  # Only add non-empty entries
            # Clean timestamp for filename and display (keep colon format for readability)
            clean_timestamp = re.sub(r'[^\w:,\-_]', '_', start_time)
            entries.append((clean_timestamp, text))
    
    return entries
    """Enhance subtitle text to be more suitable for image generation"""
    # Remove common dialogue markers and clean up text
    text = re.sub(r'^\s*-\s*', '', text)  # Remove dialogue dashes
    text = re.sub(r'\[.*?\]', '', text)   # Remove sound effects in brackets
    text = re.sub(r'\(.*?\)', '', text)   # Remove parenthetical notes
    
    # Add visual descriptive context if the text seems too dialogue-heavy
    if len(text.split()) < 5 or any(word in text.lower() for word in ['said', 'says', 'told', 'asked']):
        text = f"A cinematic scene depicting: {text}"
    
    return text.strip()

def enhance_prompt_for_image_generation(text: str) -> str:
    """Generate image using Flux model"""
    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": enhance_prompt_for_image_generation(prompt),
                "width": width,
                "height": height,
                "num_outputs": 1,
                "num_inference_steps": 4
            }
        )
        
        # Get the image URL
        image_url = output[0] if isinstance(output, list) else output
        
        # Download image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# App title and description
st.title("🎬 SRT Image Generator")
st.write("Upload an SRT subtitle file and generate images for each subtitle entry")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Image Settings")
    
    # Aspect ratio selection (user must choose)
    aspect_ratio = st.selectbox(
        "Select Aspect Ratio *",
        ["16:9 (Widescreen)", "1:1 (Square)", "4:3 (Classic)", "3:4 (Portrait)", "9:16 (Vertical)"],
        help="Choose the aspect ratio for generated images"
    )
    
    # Size selection
    size_option = st.selectbox(
        "Image Size",
        ["Small (512px)", "Medium (768px)", "Large (1024px)"],
        index=1,
        help="Choose the base size for images"
    )
    
    # Calculate actual dimensions based on ratio and size
    def get_dimensions(ratio_text, size_text):
        # Extract base size
        if "512" in size_text:
            base_size = 512
        elif "768" in size_text:
            base_size = 768
        else:  # 1024
            base_size = 1024
        
        # Calculate width and height based on ratio
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
    
    # Display calculated dimensions
    st.caption(f"Image dimensions: {width} × {height} pixels")
    
def generate_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
        enable_prompt_enhancement = st.checkbox(
            "Enhance prompts for better images",
            value=True,
            help="Automatically improve subtitle text for image generation"
        )

# File upload
uploaded_file = st.file_uploader(
    "Upload SRT file",
    type=['srt'],
    help="Upload a subtitle file (.srt format)"
)

if uploaded_file is not None:
    # Read and parse SRT file
    srt_content = uploaded_file.read().decode('utf-8')
    subtitles = parse_srt(srt_content)
    
    if not subtitles:
        st.error("Could not parse any subtitles from the uploaded file. Please check the SRT format.")
    else:
        st.success(f"Successfully parsed {len(subtitles)} subtitle entries")
        
        # Process SRT entries
        if use_gemini_processing:
            st.info("🤖 Using Gemini AI to group subtitles and create enhanced prompts...")
            entries = group_subtitles_with_gemini(subtitles)
        else:
            entries = process_srt_entries(subtitles)
        
        st.success(f"Found {len(entries)} {'AI-processed groups' if use_gemini_processing else 'subtitle entries'} to process")
        
        # Show preview of all entries
        preview_title = "Preview AI-Processed Groups" if use_gemini_processing else "Preview All Subtitle Entries"
        with st.expander(preview_title):
            for i, (timestamp, text) in enumerate(entries):
                if use_gemini_processing:
                    st.text(f"{i+1}. [{timestamp}] Enhanced prompt: {text}")
                else:
                    st.text(f"{i+1}. [{timestamp}] {text}")
        
        if entries:
            # Generate images button
            if st.button("🎨 Generate All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generated_images = []
                image_data_for_download = []
                
                for i, (timestamp, text) in enumerate(entries):
                    status_text.text(f"Generating image {i+1} of {len(entries)}...")
                    progress_bar.progress((i) / len(entries))
                    
                    # Generate image
                    if use_gemini_processing:
                        # Gemini already provided enhanced prompts
                        prompt = text
                    else:
                        # Use traditional enhancement if needed
                        prompt = enhance_prompt_for_image_generation(text) if enable_prompt_enhancement else text
                    
                    image = generate_image(prompt, width, height)
                    
                    if image:
                        generated_images.append((image, prompt, timestamp, text))
                        
                        # Save image data for download with timestamp as filename
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        filename = f"{timestamp}.png"
                        image_data_for_download.append((buf.getvalue(), filename))
                
                progress_bar.progress(1.0)
                status_text.text("All images generated!")
                
                # Display generated images
                if generated_images:
                    st.subheader("Generated Images")
                    
                    # Display images in a grid
                    for i, (image, prompt, timestamp, original_text) in enumerate(generated_images):
                        st.markdown(f"### Image {i+1}: `{timestamp}`")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.image(image, use_container_width=True)
                        
                        with col2:
                            st.write(f"**{'Enhanced prompt' if use_gemini_processing else 'Original text'}:** {original_text}")
                            if not use_gemini_processing and enable_prompt_enhancement:
                                with st.expander("Enhanced prompt"):
                                    st.text(prompt)
                            
                            # Individual download button
                            buf = BytesIO()
                            image.save(buf, format='PNG')
                            st.download_button(
                                label=f"Download",
                                data=buf.getvalue(),
                                file_name=f"{timestamp}.png",
                                mime="image/png",
                                key=f"download_{timestamp}"
                            )
                        
                        st.divider()
                    
                    # Create ZIP file for bulk download
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
                            key="download_all"
                        )

st.markdown("---")
st.markdown("Built with Streamlit 🚀 | Powered by Flux AI")
