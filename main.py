import streamlit as st
import io
import zipfile
import re
import requests
from PIL import Image
import replicate

def parse_srt(srt_content):
    """Parse SRT content into subtitle data"""
    subtitles = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                subtitle_id = int(lines[0])
                time_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parse time range: 00:00:20,000 --> 00:00:24,400
                times = time_line.split(' --> ')
                start_time = times[0]
                end_time = times[1]
                
                subtitles.append({
                    'id': subtitle_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
            except:
                continue
    
    return subtitles

def select_scenes(subtitles, num_images, method="even"):
    """Select key scenes for image generation"""
    if not subtitles or num_images <= 0:
        return []
    
    if method == "even":
        # Even distribution across timeline
        if num_images >= len(subtitles):
            return subtitles
        step = len(subtitles) / num_images
        indices = [int(i * step) for i in range(num_images)]
        return [subtitles[i] for i in indices]
    
    elif method == "longest":
        # Select longest scenes (assuming longer = more content)
        return subtitles[:num_images]
    
    elif method == "keywords":
        # Select scenes with visual keywords
        keywords = ['look', 'see', 'show', 'appear', 'walk', 'run', 'fight', 'beautiful', 'dramatic']
        scored = []
        for sub in subtitles:
            score = sum(1 for word in keywords if word.lower() in sub['text'].lower())
            scored.append((sub, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:num_images]]
    
    return subtitles[:num_images]

def enhance_prompt(text, style="cinematic"):
    """Create better prompts from subtitle text"""
    # Clean subtitle text
    clean_text = re.sub(r'\[.*?\]|\(.*?\)|<.*?>', '', text).strip()
    
    # Add style and quality modifiers
    if style == "cinematic":
        prompt = f"{clean_text}, cinematic photography, dramatic lighting, high quality, detailed"
    elif style == "realistic":
        prompt = f"{clean_text}, photorealistic, natural lighting, sharp focus, professional photography"
    elif style == "artistic":
        prompt = f"{clean_text}, digital art, concept art, detailed illustration, masterpiece"
    else:
        prompt = f"{clean_text}, {style}, high quality, detailed"
    
    return prompt

def generate_image(prompt, api_token):
    """Generate image using Flux Schnell"""
    try:
        replicate.api_token = api_token
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "16:9",
                "output_format": "webp",
                "output_quality": 90,
                "num_inference_steps": 4
            }
        )
        
        # Download image
        if output and len(output) > 0:
            response = requests.get(output[0], timeout=60)
            if response.status_code == 200:
                return response.content
        
        return None
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None

def create_zip(results):
    """Create ZIP file with all images"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for i, result in enumerate(results):
            if result['image_data']:
                filename = f"scene_{i+1}_{result['start_time'].replace(':', '-')}.webp"
                zip_file.writestr(filename, result['image_data'])
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.set_page_config(page_title="SRT Scene Generator", page_icon="🎬", layout="wide")
    st.title("🎬 SRT Scene Generator with Flux")
    
    # API Token
    if "REPLICATE_API_TOKEN" in st.secrets:
        api_token = st.secrets["REPLICATE_API_TOKEN"]
        st.success("✅ API token configured")
    else:
        api_token = st.text_input("Replicate API Token:", type="password")
        if not api_token:
            st.error("Please add REPLICATE_API_TOKEN to secrets or enter it above")
            st.info("Get token from: https://replicate.com/account/api-tokens")
            return
    
    # File Upload
    uploaded_file = st.file_uploader("Upload SRT File", type=['srt'])
    if not uploaded_file:
        st.info("Upload an SRT file to start")
        return
    
    # Parse SRT
    try:
        srt_content = uploaded_file.read().decode('utf-8')
        subtitles = parse_srt(srt_content)
        if not subtitles:
            st.error("No valid subtitles found")
            return
        st.success(f"Parsed {len(subtitles)} subtitles")
    except Exception as e:
        st.error(f"Failed to parse SRT: {str(e)}")
        return
    
    # Settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_images = st.number_input("Number of images", 1, 10, 3)
    
    with col2:
        method = st.selectbox("Selection method", [
            ("even", "Even distribution"),
            ("longest", "Longest scenes"), 
            ("keywords", "Visual keywords")
        ], format_func=lambda x: x[1])
    
    with col3:
        style = st.selectbox("Style", [
            "cinematic", "realistic", "artistic", "dramatic", "vintage"
        ])
    
    # Generate Images
    if st.button("🎨 Generate Images", type="primary"):
        # Select scenes
        selected_scenes = select_scenes(subtitles, num_images, method[0])
        if not selected_scenes:
            st.error("No scenes selected")
            return
        
        st.info(f"Generating {len(selected_scenes)} images...")
        
        # Generate images
        results = []
        progress_bar = st.progress(0)
        
        for i, scene in enumerate(selected_scenes):
            progress_bar.progress((i + 1) / len(selected_scenes))
            
            # Create prompt
            prompt = enhance_prompt(scene['text'], style)
            st.write(f"**Scene {i+1}:** {scene['start_time']} - {prompt[:100]}...")
            
            # Generate image
            image_data = generate_image(prompt)
            
            results.append({
                **scene,
                'prompt': prompt,
                'image_data': image_data
            })
        
        progress_bar.empty()
        
        # Show results
        successful = sum(1 for r in results if r['image_data'])
        st.success(f"Generated {successful}/{len(results)} images")
        
        if successful > 1:
            zip_data = create_zip(results)
            st.download_button(
                "📦 Download All",
                data=zip_data,
                file_name="scenes.zip",
                mime="application/zip"
            )
        
        # Display images
        for i, result in enumerate(results):
            st.subheader(f"Scene {i+1}: {result['start_time']} → {result['end_time']}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Text:** {result['text']}")
                if result['image_data']:
                    st.download_button(
                        f"💾 Download",
                        data=result['image_data'],
                        file_name=f"scene_{i+1}.webp",
                        mime="image/webp"
                    )
            
            with col2:
                if result['image_data']:
                    image = Image.open(io.BytesIO(result['image_data']))
                    st.image(image, use_column_width=True)
                else:
                    st.error("Failed to generate")
            
            with st.expander("View prompt"):
                st.code(result['prompt'])

if __name__ == "__main__":
    main()
