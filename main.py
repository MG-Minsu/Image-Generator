import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO
import re
from typing import List, Tuple
import zipfile
import time

# Set page config
st.set_page_config(
    page_title="SRT Image Generator",
    page_icon="🎬",
    layout="wide"
)

# Get API key from secrets
try:
    api_key = st.secrets["REPLICATE_API_TOKEN"]
    client = replicate.Client(api_token=api_key)
except:
    st.error("Please add REPLICATE_API_TOKEN to your Streamlit secrets")
    st.stop()

def parse_srt(srt_content: str) -> List[Tuple[str, str, str]]:
    """Parse SRT content and return list of (timestamp, start_time, text) tuples"""
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    subtitles = []
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # First line is the sequence number
            seq_num = lines[0].strip()
            # Second line is the timestamp
            timestamp = lines[1].strip()
            # Remaining lines are the subtitle text
            text = ' '.join(lines[2:]).strip()
            
            # Extract start time for sorting/reference
            start_time = timestamp.split(' --> ')[0] if ' --> ' in timestamp else timestamp
            
            subtitles.append((seq_num, start_time, text))
    
    return subtitles

def group_subtitles(subtitles: List[Tuple[str, str, str]], num_groups: int) -> List[str]:
    """Group subtitles into specified number of groups and combine their text"""
    if not subtitles or num_groups <= 0:
        return []
    
    # Calculate group size
    total_subtitles = len(subtitles)
    group_size = max(1, total_subtitles // num_groups)
    
    groups = []
    for i in range(0, total_subtitles, group_size):
        group_texts = []
        group_end = min(i + group_size, total_subtitles)
        
        for j in range(i, group_end):
            group_texts.append(subtitles[j][2])  # Get the text part
        
        combined_text = ' '.join(group_texts)
        groups.append(combined_text)
        
        # If we've reached the desired number of groups, break
        if len(groups) >= num_groups:
            break
    
    return groups

def enhance_prompt_for_image_generation(text: str) -> str:
    """Enhance subtitle text to be more suitable for image generation"""
    # Remove common dialogue markers and clean up text
    text = re.sub(r'^\s*-\s*', '', text)  # Remove dialogue dashes
    text = re.sub(r'\[.*?\]', '', text)   # Remove sound effects in brackets
    text = re.sub(r'\(.*?\)', '', text)   # Remove parenthetical notes
    
    # Add visual descriptive context if the text seems too dialogue-heavy
    if len(text.split()) < 5 or any(word in text.lower() for word in ['said', 'says', 'told', 'asked']):
        text = f"A cinematic scene depicting: {text}"
    
    return text.strip()

def generate_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
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
st.write("Upload an SRT subtitle file and generate images based on subtitle content")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    num_images = st.slider(
        "Number of images to generate",
        min_value=1,
        max_value=20,
        value=5,
        help="This will group your subtitles into the specified number of segments"
    )
    
    st.subheader("Image Settings")
    col1, col2 = st.columns(2)
    with col1:
        width = st.selectbox("Width", [512, 768, 1024], index=0)
    with col2:
        height = st.selectbox("Height", [512, 768, 1024], index=0)
    
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
        st.success(f"Successfully parsed {len(subtitles)} subtitles")
        
        # Show preview of subtitles
        with st.expander("Preview Subtitles"):
            for i, (seq, time, text) in enumerate(subtitles[:10]):  # Show first 10
                st.text(f"{seq}. [{time}] {text}")
            if len(subtitles) > 10:
                st.text(f"... and {len(subtitles) - 10} more")
        
        # Group subtitles
        grouped_texts = group_subtitles(subtitles, num_images)
        
        if grouped_texts:
            st.subheader(f"Generated {len(grouped_texts)} text groups:")
            
            # Show grouped texts
            for i, text in enumerate(grouped_texts):
                with st.expander(f"Group {i+1} - Preview"):
                    enhanced_text = enhance_prompt_for_image_generation(text) if enable_prompt_enhancement else text
                    st.write(f"**Original text:** {text[:200]}{'...' if len(text) > 200 else ''}")
                    if enable_prompt_enhancement:
                        st.write(f"**Enhanced prompt:** {enhanced_text}")
            
            # Generate images button
            if st.button("🎨 Generate All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generated_images = []
                image_data_for_download = []
                
                for i, text in enumerate(grouped_texts):
                    status_text.text(f"Generating image {i+1} of {len(grouped_texts)}...")
                    progress_bar.progress((i) / len(grouped_texts))
                    
                    # Generate image
                    prompt = enhance_prompt_for_image_generation(text) if enable_prompt_enhancement else text
                    image = generate_image(prompt, width, height)
                    
                    if image:
                        generated_images.append((image, prompt, i+1))
                        
                        # Save image data for download
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        image_data_for_download.append((buf.getvalue(), f"srt_image_{i+1}.png"))
                    
                    # Small delay to avoid rate limiting
                    time.sleep(1)
                
                progress_bar.progress(1.0)
                status_text.text("All images generated!")
                
                # Display generated images
                if generated_images:
                    st.subheader("Generated Images")
                    
                    # Create columns for image display
                    cols = st.columns(min(3, len(generated_images)))
                    
                    for i, (image, prompt, img_num) in enumerate(generated_images):
                        with cols[i % len(cols)]:
                            st.image(image, caption=f"Image {img_num}", use_container_width=True)
                            with st.expander(f"Prompt for Image {img_num}"):
                                st.text(prompt)
                            
                            # Individual download button
                            buf = BytesIO()
                            image.save(buf, format='PNG')
                            st.download_button(
                                label=f"Download Image {img_num}",
                                data=buf.getvalue(),
                                file_name=f"srt_image_{img_num}.png",
                                mime="image/png",
                                key=f"download_{img_num}"
                            )
                    
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

# Information section
with st.expander("ℹ️ How to use"):
    st.markdown("""
    **Steps:**
    1. Upload an SRT subtitle file using the file uploader
    2. Configure the number of images you want to generate (this will group your subtitles)
    3. Adjust image dimensions and other settings in the sidebar
    4. Click "Generate All Images" to create images based on subtitle content
    
    **How it works:**
    - Your SRT file is parsed to extract subtitle text
    - Subtitles are grouped into the number of segments you specify
    - Each group's text is combined and used as a prompt for image generation
    - Images are generated using the Flux AI model
    
    **Tips:**
    - Enable "Enhance prompts" for better image quality
    - Use fewer images for longer, more detailed prompts per image
    - Use more images for more granular scene representation
    """)

# Setup instructions
with st.expander("⚙️ Setup Instructions"):
    st.markdown("""
    **To configure API access:**
    
    Create `.streamlit/secrets.toml` in your project:
    ```toml
    REPLICATE_API_TOKEN = "your_api_key_here"
    ```
    
    Get your API key from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
    """)

st.markdown("---")
st.markdown("Built with Streamlit 🚀 | Powered by Flux AI")
