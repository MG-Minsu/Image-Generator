import streamlit as st
import io
import zipfile
import re
import requests
from PIL import Image
import replicate
import os

# Initialize Replicate API
if "REPLICATE_API_TOKEN" in st.secrets:
    replicate.api_token = st.secrets["REPLICATE_API_TOKEN"]
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

def parse_srt_file(content):
    """Parse SRT file and extract subtitle blocks"""
    blocks = content.strip().split('\n\n')
    subtitles = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # Extract subtitle info
                number = int(lines[0])
                time_range = lines[1]
                text = ' '.join(lines[2:]).strip()
                
                # Parse timestamps
                start, end = time_range.split(' --> ')
                
                subtitles.append({
                    'number': number,
                    'start': start.strip(),
                    'end': end.strip(), 
                    'text': text
                })
            except:
                continue
    
    return subtitles

def group_subtitles_into_scenes(subtitles, num_scenes):
    """Group subtitles into the specified number of scenes"""
    if not subtitles or num_scenes <= 0:
        return []
    
    total_subs = len(subtitles)
    subs_per_scene = max(1, total_subs // num_scenes)
    
    scenes = []
    for i in range(num_scenes):
        start_idx = i * subs_per_scene
        
        # For the last scene, include all remaining subtitles
        if i == num_scenes - 1:
            end_idx = total_subs
        else:
            end_idx = min(start_idx + subs_per_scene, total_subs)
        
        if start_idx < total_subs:
            scene_subs = subtitles[start_idx:end_idx]
            
            # Combine all text from this scene group
            combined_text = ' '.join([sub['text'] for sub in scene_subs])
            
            scenes.append({
                'scene_number': i + 1,
                'start_time': scene_subs[0]['start'],
                'end_time': scene_subs[-1]['end'],
                'text': combined_text,
                'subtitle_count': len(scene_subs)
            })
    
    return scenes

def create_image_prompt(scene_text, style="cinematic"):
    """Create optimized prompt for image generation"""
    # Clean the text
    clean_text = re.sub(r'[^\w\s.,!?-]', '', scene_text)
    clean_text = ' '.join(clean_text.split())  # Remove extra spaces
    
    # Limit text length for better prompts
    if len(clean_text) > 150:
        clean_text = clean_text[:150] + "..."
    
    # Style-based prompt templates
    style_templates = {
        "cinematic": f"{clean_text}, cinematic scene, dramatic lighting, professional cinematography, high quality",
        "realistic": f"{clean_text}, photorealistic, natural lighting, detailed, high resolution",
        "artistic": f"{clean_text}, digital art, concept art, detailed illustration, vibrant colors",
        "documentary": f"{clean_text}, documentary photography, natural moment, authentic, candid shot",
        "dramatic": f"{clean_text}, dramatic scene, intense lighting, emotional atmosphere, high contrast"
    }
    
    return style_templates.get(style, f"{clean_text}, {style}, high quality, detailed")

def generate_flux_image(prompt):
    """Generate image using Flux Schnell with cleaner API format"""
    try:
        input_params = {
            "prompt": prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "16:9",
            "output_format": "webp",
            "output_quality": 85,
            "num_inference_steps": 4
        }
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input=input_params
        )
        
        # Access the file and download it
        if output and len(output) > 0:
            # Get the first (and only) output
            image_file = output[0]
            
            # Read the image data directly
            image_data = image_file.read()
            return image_data
        
        return None
        
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return None

def create_filename_from_timestamp(timestamp, scene_num):
    """Create clean filename from SRT timestamp"""
    # Convert 00:01:23,456 to 00-01-23-456
    clean_time = timestamp.replace(':', '-').replace(',', '-')
    return f"scene_{scene_num:02d}_{clean_time}.webp"

def create_download_zip(scenes_data):
    """Create ZIP file with all generated images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for scene in scenes_data:
            if scene['image_data']:
                filename = create_filename_from_timestamp(scene['start_time'], scene['scene_number'])
                zip_file.writestr(filename, scene['image_data'])
                
                # Add scene info text file
                info_filename = f"scene_{scene['scene_number']:02d}_info.txt"
                info_content = f"""Scene {scene['scene_number']}
Start Time: {scene['start_time']}
End Time: {scene['end_time']}
Duration: {scene['start_time']} → {scene['end_time']}
Subtitle Count: {scene['subtitle_count']}
Text: {scene['text']}
Prompt: {scene['prompt']}
"""
                zip_file.writestr(info_filename, info_content.encode('utf-8'))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="SRT Scene Generator", 
        page_icon="🎬", 
        layout="wide"
    )
    
    st.title("🎬 SRT Scene Generator")
    st.markdown("Generate images from subtitle files using Flux AI")
    
    # Check API configuration
    if "REPLICATE_API_TOKEN" not in st.secrets:
        st.error("⚠️ Replicate API token not configured")
        st.code('Add to secrets.toml: REPLICATE_API_TOKEN = "r8_your_token"')
        st.stop()
    else:
        st.success("✅ Replicate API configured")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload SRT File", 
        type=['srt'],
        help="Upload your subtitle file to generate scene images"
    )
    
    if not uploaded_file:
        st.info("👆 Upload an SRT file to begin")
        
        with st.expander("📝 SRT Format Example"):
            st.code("""1
00:00:01,000 --> 00:00:05,000
A person walks through a beautiful forest

2
00:00:06,000 --> 00:00:10,000
The sunlight filters through the green leaves

3
00:00:11,000 --> 00:00:15,000
Birds are singing in the distance""")
        return
    
    # Parse uploaded SRT file
    try:
        srt_content = uploaded_file.read().decode('utf-8')
        subtitles = parse_srt_file(srt_content)
        
        if not subtitles:
            st.error("❌ No valid subtitles found in file")
            return
            
        st.success(f"✅ Loaded {len(subtitles)} subtitles successfully")
        
    except Exception as e:
        st.error(f"❌ Error reading SRT file: {str(e)}")
        return
    
    # Configuration options
    st.subheader("⚙️ Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_scenes = st.slider(
            "📸 Number of Images to Generate", 
            min_value=1, 
            max_value=min(20, len(subtitles)), 
            value=min(5, len(subtitles)),
            help="How many images to create from your SRT file"
        )
    
    with col2:
        style = st.selectbox(
            "🎨 Visual Style",
            ["cinematic", "realistic", "artistic", "documentary", "dramatic"],
            help="Choose the style for generated images"
        )
    
    # Show how subtitles will be grouped
    scenes = group_subtitles_into_scenes(subtitles, num_scenes)
    
    st.subheader("📋 Scene Preview")
    st.info(f"Your {len(subtitles)} subtitles will be grouped into {len(scenes)} scenes")
    
    for scene in scenes:
        with st.expander(f"Scene {scene['scene_number']}: {scene['start_time']} → {scene['end_time']}"):
            st.write(f"**Timespan:** {scene['start_time']} to {scene['end_time']}")
            st.write(f"**Includes:** {scene['subtitle_count']} subtitle(s)")
            st.write(f"**Text:** {scene['text'][:200]}{'...' if len(scene['text']) > 200 else ''}")
    
    # Generate images
    if st.button("🎨 Generate Images", type="primary"):
        if not scenes:
            st.error("No scenes to generate")
            return
        
        st.subheader("🔄 Generating Images...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generated_scenes = []
        
        for i, scene in enumerate(scenes):
            # Update progress
            progress = (i + 1) / len(scenes)
            progress_bar.progress(progress)
            status_text.text(f"Generating scene {i + 1}/{len(scenes)}: {scene['start_time']}")
            
            # Create prompt and generate image
            prompt = create_image_prompt(scene['text'], style)
            image_data = generate_flux_image(prompt)
            
            # Store results
            scene_result = {
                **scene,
                'prompt': prompt,
                'image_data': image_data,
                'success': image_data is not None
            }
            generated_scenes.append(scene_result)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show generation summary
        successful_count = sum(1 for scene in generated_scenes if scene['success'])
        st.success(f"🎉 Generated {successful_count}/{len(scenes)} images successfully!")
        
        if successful_count == 0:
            st.error("No images were generated. Please check your API token and try again.")
            return
        
        # Download all button
        if successful_count > 0:
            zip_data = create_download_zip(generated_scenes)
            st.download_button(
                label="📦 Download All Images",
                data=zip_data,
                file_name="srt_scenes_with_timestamps.zip",
                mime="application/zip",
                help="Download all generated images with timestamp-based filenames"
            )
        
        st.divider()
        
        # Display results
        st.subheader("🖼️ Generated Scenes")
        
        for scene in generated_scenes:
            st.write(f"### Scene {scene['scene_number']}: {scene['start_time']} → {scene['end_time']}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**⏰ Time Range:** `{scene['start_time']}` to `{scene['end_time']}`")
                st.write(f"**📝 Subtitles:** {scene['subtitle_count']} combined")
                st.write(f"**📄 Text:** {scene['text'][:100]}...")
                
                if scene['success']:
                    filename = create_filename_from_timestamp(scene['start_time'], scene['scene_number'])
                    st.download_button(
                        label="💾 Download Image",
                        data=scene['image_data'],
                        file_name=filename,
                        mime="image/webp",
                        key=f"download_scene_{scene['scene_number']}"
                    )
                    
                    # Show file info
                    file_size = len(scene['image_data']) / 1024
                    st.caption(f"📊 Size: {file_size:.1f} KB")
                else:
                    st.error("❌ Generation failed")
            
            with col2:
                if scene['success']:
                    try:
                        image = Image.open(io.BytesIO(scene['image_data']))
                        st.image(
                            image, 
                            caption=f"Scene {scene['scene_number']} at {scene['start_time']}",
                            use_column_width=True
                        )
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                else:
                    st.info("No image to display")
            
            # Show prompt in expandable section
            with st.expander(f"🔍 View Prompt for Scene {scene['scene_number']}"):
                st.code(scene['prompt'], language="text")
            
            st.divider()

if __name__ == "__main__":
    main()
