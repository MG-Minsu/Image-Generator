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
    replicate_client = replicate.Client(api_token=api_key)
except:
    st.error("Please add REPLICATE_API_TOKEN to your Streamlit secrets")
    st.stop()

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
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

def describe_scene_with_gemini(text: str, additional_instructions: str = "") -> str:
    """Use Gemini to describe what's happening in the sentence for visual representation"""
    additional_context = f"\n\nADDITIONAL INSTRUCTIONS: {additional_instructions}" if additional_instructions.strip() else ""
    
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
- Don't just repeat dialogue - interpret the visual scene{additional_context}

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
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            st.warning("⚠️ Gemini API quota exceeded. Using enhanced fallback descriptions.")
            return create_enhanced_fallback_description(text)
        else:
            st.warning(f"Gemini API error: {error_msg}. Using fallback description.")
            return create_enhanced_fallback_description(text)

def create_enhanced_fallback_description(text: str) -> str:
    """Create better fallback descriptions when Gemini API is unavailable"""
    text_lower = text.lower()
    
    # Simple keyword-based scene enhancement
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
    
    # Detect setting clues
    if any(word in text_lower for word in ['house', 'home', 'room', 'kitchen', 'bedroom']):
        scene_elements.append("indoor domestic setting")
    elif any(word in text_lower for word in ['office', 'work', 'desk', 'meeting']):
        scene_elements.append("professional office environment")
    elif any(word in text_lower for word in ['outside', 'street', 'park', 'car', 'road']):
        scene_elements.append("outdoor urban environment")
    elif any(word in text_lower for word in ['forest', 'tree', 'nature', 'mountain']):
        scene_elements.append("natural outdoor setting")
    else:
        scene_elements.append("atmospheric lighting")
    
    # Detect emotional tone
    if any(word in text_lower for word in ['happy', 'laugh', 'smile', 'joy']):
        scene_elements.append("warm positive mood")
    elif any(word in text_lower for word in ['sad', 'cry', 'tear', 'worried']):
        scene_elements.append("somber emotional atmosphere")
    elif any(word in text_lower for word in ['angry', 'mad', 'fight', 'argue']):
        scene_elements.append("tense dramatic lighting")
    else:
        scene_elements.append("natural realistic lighting")
    
    # Add cinematic quality
    scene_elements.append("cinematic composition")
    scene_elements.append("professional photography")
    
    # Combine elements
    description = f"A {', '.join(scene_elements[:4])}, depicting: {text}"
    
    return description

def generate_image(prompt: str, width: int = 1024, height: int = 574) -> Image.Image:
    """Generate image using Flux model"""
    try:
        output = replicate_client.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "width": width,
                "height": height,
                "num_outputs": 1,
                "num_inference_steps": 4
            }
        )
        
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        response.raise_for_status()  # Raise exception for bad status codes
        image = Image.open(BytesIO(response.content))
        
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def regenerate_single_image(index: int, description: str, timestamp: str, additional_instructions: str = ""):
    """Regenerate a single image and update session state"""
    with st.spinner(f"Regenerating image {index + 1}..."):
        # If using Gemini descriptions, regenerate the description too
        if 'use_gemini_description' in st.session_state and st.session_state.use_gemini_description.startswith("Use Gemini"):
            original_text = st.session_state.generated_images[index][2]  # Get original text
            new_description = describe_scene_with_gemini(original_text, additional_instructions)
        else:
            new_description = description
        
        new_image = generate_image(new_description)
        
        if new_image:
            # Update the image in session state
            st.session_state.generated_images[index] = (
                new_image, 
                new_description, 
                st.session_state.generated_images[index][2],  # Keep original text
                timestamp
            )
            
            # Update download data too
            buf = BytesIO()
            new_image.save(buf, format='PNG')
            buf.seek(0)
            filename = f"{timestamp}.png"
            st.session_state.image_data_for_download[index] = (buf.getvalue(), filename)
            
            st.success(f"✅ Image {index + 1} regenerated successfully!")
            st.rerun()
        else:
            st.error(f"❌ Failed to regenerate image {index + 1}")

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'image_data_for_download' not in st.session_state:
    st.session_state.image_data_for_download = []

# App title and description
st.title("🎬 SRT Image Generator")
st.write("Upload an SRT subtitle file and generate images with Gemini scene descriptions")

# Download All Button at the Top
if 'image_data_for_download' in st.session_state and len(st.session_state.image_data_for_download) > 1:
    st.subheader("📦 Download All Images")
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
        type="primary"
    )
    st.divider()

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
    # Store in session state for regeneration
    st.session_state.use_gemini_description = use_gemini_description

# Additional Instructions for Gemini Prompts
st.subheader("📝 Additional Instructions for All Gemini Prompts")
additional_instructions = st.text_area(
    "Add extra context or style instructions that will be applied to all Gemini scene descriptions:",
    placeholder="e.g., 'Make all scenes photorealistic with dramatic lighting' or 'Focus on close-up character emotions' or 'Use a vintage film aesthetic'",
    height=100,
    help="These instructions will be added to every Gemini prompt to customize the style and content of your scene descriptions."
)

# Sidebar for image settings
with st.sidebar:
    st.header("🖼️ Image Settings")
    
    # Fixed dimensions - closest to qHD but divisible by 16
    width, height = 1024, 574  # 960÷16=60, 544÷16=34
    st.info(f"📐 Image dimensions: {width} × {height} pixels (16:9 ratio)")
    
    # Option to disable Gemini if quota exceeded
    st.markdown("---")
    force_fallback = st.checkbox(
        "🔧 Use Enhanced Fallback Descriptions Only",
        help="Skip Gemini API calls and use smart keyword-based descriptions instead"
    )

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
    try:
        srt_content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Try different encoding if UTF-8 fails
            uploaded_file.seek(0)
            srt_content = uploaded_file.read().decode('latin-1')
        except Exception as e:
            st.error(f"Could not decode file. Please ensure it's a valid SRT file. Error: {str(e)}")
            st.stop()
    
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
        
        st.write(f"**Processed entries:** {len(processed_entries)} (All entries will be processed)")
        
        # Show preview of processed entries
        with st.expander("👀 Preview Processed Entries"):
            show_all_entries = st.checkbox("Show all entries", key="show_all_processed")
            entries_to_show = processed_entries if show_all_entries else processed_entries[:10]
            
            for i, (timestamp, text) in enumerate(entries_to_show):
                st.text(f"{i+1}. [{timestamp}] {text}")
            
            if not show_all_entries and len(processed_entries) > 10:
                st.text(f"... and {len(processed_entries) - 10} more entries")
        
        # Create scene descriptions
        st.subheader("🎭 Scene Descriptions")
        
        if use_gemini_description.startswith("Use Gemini") and not force_fallback:
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
                    
                    description = describe_scene_with_gemini(text, additional_instructions)
                    scene_descriptions.append((timestamp, text, description))
                
                progress_bar.progress(1.0)
                status_text.text("✅ All optimized scene descriptions created!")
        
        elif force_fallback:
            st.info("🔧 Using enhanced fallback descriptions (Gemini API disabled)...")
            scene_descriptions = []
            for timestamp, text in processed_entries:
                description = create_enhanced_fallback_description(text)
                scene_descriptions.append((timestamp, text, description))
            
        else:
            st.info("📝 Using original subtitle text...")
            scene_descriptions = [(timestamp, text, f"A cinematic scene: {text}") for timestamp, text in processed_entries]
        
        # Show preview of scene descriptions
        with st.expander("🎬 Preview Scene Descriptions"):
            show_all_descriptions = st.checkbox("Show all descriptions", key="show_all_descriptions")
            descriptions_to_show = scene_descriptions if show_all_descriptions else scene_descriptions[:5]
            
            for i, (timestamp, original, description) in enumerate(descriptions_to_show):
                st.write(f"**{i+1}. [{timestamp}]**")
                st.write(f"*Original:* {original}")
                st.write(f"*Scene Description:* {description}")
                st.write("---")
            
            if not show_all_descriptions and len(scene_descriptions) > 5:
                st.text(f"... and {len(scene_descriptions) - 5} more descriptions")
        
        # Generate images
        st.subheader("🎨 Generate Images")
        
        # Show cost estimate
        estimated_cost = len(scene_descriptions) * 0.003  # Rough estimate for Flux Schnell
        st.info(f"💰 Estimated cost: ~${estimated_cost:.3f} USD for {len(scene_descriptions)} images")
        
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
                    
                    # Convert image to bytes for download
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    buf.seek(0)
                    filename = f"{timestamp}.png"
                    image_data_for_download.append((buf.getvalue(), filename))
                else:
                    st.error(f"Failed to generate image for timestamp: {timestamp}")
            
            progress_bar.progress(1.0)
            status_text.text(f"🎉 Generated {len(generated_images)} out of {len(scene_descriptions)} images!")
            
            # Store in session state
            st.session_state.generated_images = generated_images
            st.session_state.image_data_for_download = image_data_for_download
            
            # Rerun to show the download button at the top
            st.rerun()

# Display generated images (both new and previously generated)
if st.session_state.generated_images:
    st.subheader("🖼️ Generated Images")
    
    for i, (image, description, original_text, timestamp) in enumerate(st.session_state.generated_images):
        st.markdown(f"### 🎬 Image {i+1}: `{timestamp}`")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(image, use_container_width=True)
        
        with col2:
            st.write("**📝 Original Subtitle:**")
            st.write(original_text)
            st.write("**🎭 Scene Description (Used for Image):**")
            st.write(description)
            
            # Individual controls
            col2a, col2b = st.columns(2)
            
            with col2a:
                # Download button for individual image
                buf = BytesIO()
                image.save(buf, format='PNG')
                buf.seek(0)
                st.download_button(
                    label="💾 Download",
                    data=buf.getvalue(),
                    file_name=f"{timestamp}.png",
                    mime="image/png",
                    key=f"download_{i}_{timestamp}",
                    use_container_width=True
                )
            
            with col2b:
                # Regenerate button for individual image
                if st.button(
                    "🔄 Regenerate",
                    key=f"regen_{i}_{timestamp}",
                    use_container_width=True,
                    help="Generate a new version of this specific image"
                ):
                    regenerate_single_image(i, description, timestamp, additional_instructions)
        
        st.divider()

st.markdown("---")
st.markdown("🚀 **Built with Streamlit** | 🤖 **Powered by Flux AI & Gemini**")
