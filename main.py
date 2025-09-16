import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO
import re
from typing import List, Tuple
import zipfile
import google.generativeai as genai
import time

# Set page config
st.set_page_config(
    page_title="SRT Image Generator",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'image_data_for_download' not in st.session_state:
    st.session_state.image_data_for_download = []

def initialize_apis():
    """Initialize API clients with better error handling"""
    replicate_client = None
    gemini_model = None
    
    # Initialize Replicate
    try:
        if "REPLICATE_API_TOKEN" not in st.secrets:
            st.error("❌ REPLICATE_API_TOKEN not found in secrets")
            return None, None
            
        api_key = st.secrets["REPLICATE_API_TOKEN"]
        if not api_key or api_key.strip() == "":
            st.error("❌ REPLICATE_API_TOKEN is empty")
            return None, None
            
        replicate_client = replicate.Client(api_token=api_key.strip())
        st.success("✅ Replicate API initialized")
    except Exception as e:
        st.error(f"❌ Replicate API initialization failed: {str(e)}")
        return None, None

    # Initialize Gemini (optional)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
            if gemini_api_key and gemini_api_key.strip() != "":
                genai.configure(api_key=gemini_api_key.strip())
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                st.success("✅ Gemini API initialized")
    except Exception as e:
        st.warning(f"⚠️ Gemini API not available: {str(e)}")
    
    return replicate_client, gemini_model

def parse_srt(srt_content: str) -> List[Tuple[str, str, str, str]]:
    """Parse SRT content with better error handling"""
    try:
        # Handle different line endings
        srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        subtitles = []
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                seq_num = lines[0].strip()
                timestamp = lines[1].strip()
                text = ' '.join(lines[2:]).strip()
                
                # Clean up text
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                text = re.sub(r'\{[^}]+\}', '', text)  # Remove formatting tags
                text = text.strip()
                
                if ' --> ' in timestamp:
                    start_time, end_time = timestamp.split(' --> ')
                    start_time = start_time.strip()
                    end_time = end_time.strip()
                else:
                    start_time = timestamp
                    end_time = timestamp
                
                if text:  # Only add if there's actual text content
                    subtitles.append((timestamp, start_time, end_time, text))
        
        return subtitles
    except Exception as e:
        st.error(f"Error parsing SRT: {str(e)}")
        return []

def create_enhanced_fallback_description(text: str) -> str:
    """Create enhanced fallback descriptions"""
    text_lower = text.lower()
    scene_elements = []
    
    # Detect dialogue vs action
    if any(word in text_lower for word in ['said', 'says', 'asked', 'replied', 'whispered', 'shouted']):
        scene_elements.append("people in conversation")
    elif any(word in text_lower for word in ['running', 'walking', 'moving', 'going', 'coming']):
        scene_elements.append("person in motion")
    elif any(word in text_lower for word in ['looking', 'watching', 'seeing', 'staring']):
        scene_elements.append("focused gaze and observation")
    else:
        scene_elements.append("cinematic scene")
    
    # Detect setting
    if any(word in text_lower for word in ['house', 'home', 'room', 'kitchen', 'bedroom']):
        scene_elements.append("indoor domestic setting")
    elif any(word in text_lower for word in ['office', 'work', 'desk', 'meeting']):
        scene_elements.append("professional office environment")
    elif any(word in text_lower for word in ['outside', 'street', 'park', 'car', 'road']):
        scene_elements.append("outdoor urban environment")
    else:
        scene_elements.append("atmospheric lighting")
    
    # Add cinematic quality
    scene_elements.extend(["cinematic composition", "professional photography"])
    
    description = f"A {', '.join(scene_elements[:3])}, depicting: {text}"
    return description

def describe_scene_with_gemini(text: str, gemini_model) -> str:
    """Use Gemini to describe scenes with better error handling"""
    if not gemini_model:
        return create_enhanced_fallback_description(text)
        
    prompt = f"""Create a visual scene description for AI image generation from this subtitle:

"{text}"

Requirements:
- 25-40 words optimized for AI image generation
- Focus on visual elements, actions, setting, mood
- Use specific, concrete details
- Include lighting and composition terms
- Make it cinematic and engaging

Output only the scene description:"""

    try:
        response = gemini_model.generate_content(prompt)
        description = response.text.strip().strip('"\'')
        return description if description else create_enhanced_fallback_description(text)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ Gemini API quota exceeded. Using fallback descriptions.")
        return create_enhanced_fallback_description(text)

def generate_image(prompt: str, replicate_client, width: int = 1024, height: int = 576) -> Image.Image:
    """Generate image with comprehensive error handling and debugging"""
    
    if not replicate_client:
        st.error("❌ Replicate client not initialized")
        return None
    
    # Debug info
    st.write(f"🔍 Generating: {width}×{height} | Prompt: {prompt[:100]}...")
    
    try:
        # Use validated parameters
        input_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": 1,
            "num_inference_steps": 4
        }
        
        # Try the API call
        with st.spinner("🎨 Generating image..."):
            output = replicate_client.run(
                "black-forest-labs/flux-schnell",
                input=input_params
            )
        
        # Handle the response
        if not output:
            st.error("❌ No output received from API")
            return None
            
        image_url = output[0] if isinstance(output, list) else output
        
        if not image_url:
            st.error("❌ No image URL in response")
            return None
        
        # Download the image
        st.write(f"📥 Downloading from: {image_url}")
        
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        if len(response.content) == 0:
            st.error("❌ Empty image response")
            return None
        
        # Open the image
        image = Image.open(BytesIO(response.content))
        
        # Verify image
        actual_width, actual_height = image.size
        st.write(f"✅ Success! Image: {actual_width}×{actual_height}")
        
        if actual_width != width or actual_height != height:
            st.warning(f"⚠️ Size mismatch: requested {width}×{height}, got {actual_width}×{actual_height}")
        
        return image
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error downloading image: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Image generation failed: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None

# Main app
st.title("🎬 SRT Image Generator")
st.write("Upload an SRT subtitle file and generate images with AI")

# Initialize APIs
replicate_client, gemini_model = initialize_apis()

if not replicate_client:
    st.stop()

# Configuration
st.subheader("⚙️ Configuration")

col1, col2 = st.columns(2)
with col1:
    grouping_mode = st.selectbox(
        "Subtitle Grouping",
        ["Individual subtitles", "Group by 2 subtitles"],
    )

with col2:
    use_gemini = st.selectbox(
        "Scene Description",
        ["Use enhanced fallback", "Use Gemini descriptions"] if gemini_model else ["Use enhanced fallback"],
    )

# Image settings
with st.sidebar:
    st.header("🖼️ Image Settings")
    
    dimension_options = {
        "Landscape 16:9 (1024×576)": (1024, 576),
        "Square (1024×1024)": (1024, 1024),
        "Cinema (1280×720)": (1280, 720),
        "Standard (960×540)": (960, 540),
    }
    
    selected_size = st.selectbox("Image Size:", list(dimension_options.keys()))
    width, height = dimension_options[selected_size]
    
    st.info(f"📐 Dimensions: {width} × {height} pixels")
    
    max_images = st.slider("Max Images", 1, 20, 5)

# File upload
st.subheader("📁 Upload SRT File")
uploaded_file = st.file_uploader("Choose SRT file", type=['srt'])

if uploaded_file is not None:
    # Parse SRT
    try:
        srt_content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        srt_content = uploaded_file.read().decode('latin-1')
    
    subtitles = parse_srt(srt_content)
    
    if not subtitles:
        st.error("❌ Could not parse subtitles")
    else:
        st.success(f"✅ Parsed {len(subtitles)} subtitles")
        
        # Process subtitles
        if grouping_mode.startswith("Group"):
            processed_entries = []
            for i in range(0, len(subtitles), 2):
                current = subtitles[i]
                next_sub = subtitles[i + 1] if i + 1 < len(subtitles) else None
                
                timestamp = re.sub(r'[^\w\-]', '_', current[1])
                text = current[3]
                if next_sub:
                    text += " " + next_sub[3]
                
                processed_entries.append((timestamp, text))
        else:
            processed_entries = []
            for timestamp, start_time, end_time, text in subtitles:
                clean_timestamp = re.sub(r'[^\w\-]', '_', start_time)
                processed_entries.append((clean_timestamp, text))
        
        # Limit entries
        if len(processed_entries) > max_images:
            processed_entries = processed_entries[:max_images]
            st.warning(f"⚠️ Limited to {max_images} entries")
        
        st.write(f"**Processing {len(processed_entries)} entries**")
        
        # Create descriptions
        scene_descriptions = []
        if use_gemini.startswith("Use Gemini") and gemini_model:
            st.info("🤖 Creating Gemini descriptions...")
            progress_bar = st.progress(0)
            
            for i, (timestamp, text) in enumerate(processed_entries):
                description = describe_scene_with_gemini(text, gemini_model)
                scene_descriptions.append((timestamp, text, description))
                progress_bar.progress((i + 1) / len(processed_entries))
        else:
            st.info("🔧 Using enhanced fallback descriptions...")
            for timestamp, text in processed_entries:
                description = create_enhanced_fallback_description(text)
                scene_descriptions.append((timestamp, text, description))
        
        # Preview descriptions
        with st.expander("🎬 Preview Descriptions"):
            for i, (timestamp, original, description) in enumerate(scene_descriptions[:3]):
                st.write(f"**{i+1}. [{timestamp}]**")
                st.write(f"*Original:* {original}")
                st.write(f"*Description:* {description}")
                st.divider()
        
        # Generate images
        st.subheader("🎨 Generate Images")
        
        if st.button("🚀 Generate All Images", type="primary"):
            st.write("🚀 Starting image generation...")
            
            generated_images = []
            image_data_for_download = []
            
            # Create containers for progress
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (timestamp, original_text, description) in enumerate(scene_descriptions):
                    status_text.text(f"Generating image {i+1} of {len(scene_descriptions)}: {timestamp}")
                    progress_bar.progress(i / len(scene_descriptions))
                    
                    # Generate image
                    image = generate_image(description, replicate_client, width, height)
                    
                    if image:
                        generated_images.append((image, description, original_text, timestamp))
                        
                        # Prepare for download
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        buf.seek(0)
                        filename = f"{timestamp}.png"
                        image_data_for_download.append((buf.getvalue(), filename))
                        
                        st.success(f"✅ Generated {i+1}/{len(scene_descriptions)}")
                    else:
                        st.error(f"❌ Failed: {timestamp}")
                    
                    # Small delay to prevent rate limiting
                    time.sleep(1)
                
                progress_bar.progress(1.0)
                status_text.text(f"🎉 Completed! Generated {len(generated_images)} images")
            
            # Store in session state
            st.session_state.generated_images = generated_images
            st.session_state.image_data_for_download = image_data_for_download
            
            # Display results
            if generated_images:
                st.subheader("🖼️ Generated Images")
                
                for i, (image, description, original_text, timestamp) in enumerate(generated_images):
                    with st.expander(f"🎬 Image {i+1}: {timestamp}", expanded=True):
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.image(image, use_container_width=True)
                        
                        with col2:
                            st.write("**Original:**", original_text)
                            st.write("**Description:**", description)
                            
                            buf = BytesIO()
                            image.save(buf, format='PNG')
                            buf.seek(0)
                            st.download_button(
                                "💾 Download",
                                buf.getvalue(),
                                f"{timestamp}.png",
                                "image/png",
                                key=f"dl_{i}",
                                use_container_width=True
                            )
                
                # Bulk download
                if len(image_data_for_download) > 1:
                    st.subheader("📦 Bulk Download")
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for img_data, filename in image_data_for_download:
                            zip_file.writestr(filename, img_data)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        "📦 Download All (ZIP)",
                        zip_buffer.getvalue(),
                        "srt_images.zip",
                        "application/zip",
                        use_container_width=True
                    )

st.markdown("---")
st.markdown("🚀 **Built with Streamlit** | 🤖 **Powered by Flux AI**")
