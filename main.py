import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO
import re
from typing import List, Tuple
import zipfile

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

def split_into_sentences(subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Split subtitle text into individual sentences with their timestamps"""
    sentences = []
    
    for timestamp, start_time, end_time, text in subtitles:
        # Split text into sentences using multiple sentence delimiters
        sentence_parts = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentence_parts):
            sentence = sentence.strip()
            if sentence:  # Only add non-empty sentences
                # Create a unique timestamp for each sentence
                sentence_timestamp = f"{start_time}-{i+1}" if len(sentence_parts) > 2 else start_time
                # Clean timestamp for filename (remove problematic characters)
                clean_timestamp = re.sub(r'[^\w\-_]', '_', sentence_timestamp)
                sentences.append((clean_timestamp, sentence))
    
    return sentences

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
st.write("Upload an SRT subtitle file and generate images for each sentence")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
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
    
    # Option to limit number of sentences for processing
    max_sentences = st.number_input(
        "Max sentences to process (0 = all)",
        min_value=0,
        max_value=100,
        value=10,
        help="Limit processing to avoid long generation times"
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
        
        # Split into sentences
        sentences = split_into_sentences(subtitles)
        
        # Apply sentence limit if specified
        if max_sentences > 0:
            sentences = sentences[:max_sentences]
            st.info(f"Processing first {len(sentences)} sentences (limited by max setting)")
        
        st.success(f"Found {len(sentences)} sentences to process")
        
        # Show preview of all sentences
        with st.expander("Preview All Sentences"):
            for i, (timestamp, sentence) in enumerate(sentences):
                st.text(f"{i+1}. [{timestamp}] {sentence}")
        
        if sentences:
            # Generate images button
            if st.button("🎨 Generate All Images", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generated_images = []
                image_data_for_download = []
                
                for i, (timestamp, sentence) in enumerate(sentences):
                    status_text.text(f"Generating image {i+1} of {len(sentences)}...")
                    progress_bar.progress((i) / len(sentences))
                    
                    # Generate image
                    prompt = enhance_prompt_for_image_generation(sentence) if enable_prompt_enhancement else sentence
                    image = generate_image(prompt, width, height)
                    
                    if image:
                        generated_images.append((image, prompt, timestamp, sentence))
                        
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
                            st.write(f"**Original text:** {original_text}")
                            if enable_prompt_enhancement:
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

# Information section
with st.expander("ℹ️ How to use"):
    st.markdown("""
    **Steps:**
    1. Upload an SRT subtitle file using the file uploader
    2. Adjust image dimensions and other settings in the sidebar
    3. Set a maximum number of sentences to process (to avoid long generation times)
    4. Click "Generate All Images" to create images for each sentence
    
    **How it works:**
    - Your SRT file is parsed to extract subtitle text and timestamps
    - Each subtitle is split into individual sentences
    - Each sentence is used as a prompt for image generation
    - Images are saved with timestamps as filenames
    - Images are generated using the Flux AI model
    
    **Tips:**
    - Enable "Enhance prompts" for better image quality
    - Use the sentence limit to test with a smaller batch first
    - Timestamps are cleaned for use as filenames (special characters replaced with underscores)
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
