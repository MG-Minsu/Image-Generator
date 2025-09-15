import streamlit as st
import io
import zipfile
from datetime import datetime, timedelta
import re
import requests
import base64
from PIL import Image
import json

class SRTParser:
    """Handle SRT file parsing and subtitle extraction"""
    
    @staticmethod
    def parse_srt(srt_content):
        """Parse SRT content into structured subtitle data"""
        subtitles = []
        blocks = srt_content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    # Extract subtitle number
                    subtitle_id = int(lines[0])
                    
                    # Extract time range
                    time_line = lines[1]
                    start_time, end_time = SRTParser._parse_time_range(time_line)
                    
                    # Extract text (can be multiple lines)
                    text = '\n'.join(lines[2:])
                    
                    subtitles.append({
                        'id': subtitle_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text,
                        'duration': SRTParser._calculate_duration(start_time, end_time)
                    })
                except (ValueError, IndexError) as e:
                    st.warning(f"Skipped malformed subtitle block: {block[:50]}...")
                    continue
        
        return subtitles
    
    @staticmethod
    def _parse_time_range(time_line):
        """Parse SRT time format: 00:00:20,000 --> 00:00:24,400"""
        pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
        match = re.match(pattern, time_line)
        if not match:
            raise ValueError(f"Invalid time format: {time_line}")
        
        start_str, end_str = match.groups()
        return start_str, end_str
    
    @staticmethod
    def _calculate_duration(start_time, end_time):
        """Calculate duration between two SRT timestamps"""
        def time_to_seconds(time_str):
            h, m, s_ms = time_str.split(':')
            s, ms = s_ms.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        return end_seconds - start_seconds

class SceneSelector:
    """Handle scene selection logic for image generation"""
    
    @staticmethod
    def identify_key_scenes(subtitles, num_images, selection_method="even_distribution"):
        """Identify key scenes from subtitles for image generation"""
        if not subtitles:
            return []
        
        if selection_method == "even_distribution":
            return SceneSelector._even_distribution(subtitles, num_images)
        elif selection_method == "longest_duration":
            return SceneSelector._longest_duration(subtitles, num_images)
        elif selection_method == "keyword_based":
            return SceneSelector._keyword_based(subtitles, num_images)
        else:
            return SceneSelector._even_distribution(subtitles, num_images)
    
    @staticmethod
    def _even_distribution(subtitles, num_images):
        """Select scenes evenly distributed across the video timeline"""
        if num_images >= len(subtitles):
            return subtitles
        
        step = len(subtitles) / num_images
        selected_indices = [int(i * step) for i in range(num_images)]
        return [subtitles[i] for i in selected_indices]
    
    @staticmethod
    def _longest_duration(subtitles, num_images):
        """Select scenes with the longest duration"""
        sorted_subtitles = sorted(subtitles, key=lambda x: x['duration'], reverse=True)
        return sorted_subtitles[:num_images]
    
    @staticmethod
    def _keyword_based(subtitles, num_images):
        """Select scenes containing visual keywords"""
        visual_keywords = [
            'look', 'see', 'watch', 'show', 'appear', 'scene', 'view',
            'action', 'move', 'walk', 'run', 'jump', 'dance', 'fight',
            'beautiful', 'amazing', 'stunning', 'dramatic', 'exciting',
            'enters', 'exits', 'opens', 'closes', 'light', 'dark', 'bright'
        ]
        
        scored_subtitles = []
        for subtitle in subtitles:
            score = sum(1 for keyword in visual_keywords 
                       if keyword.lower() in subtitle['text'].lower())
            scored_subtitles.append((subtitle, score))
        
        # Sort by score, then by duration
        scored_subtitles.sort(key=lambda x: (x[1], x[0]['duration']), reverse=True)
        return [item[0] for item in scored_subtitles[:num_images]]

class PromptEnhancer:
    """Handle prompt enhancement without external AI services"""
    
    @staticmethod
    def enhance_scene_prompt(subtitle_text, style_settings):
        """Create detailed visual prompts from subtitle text using rule-based enhancement"""
        
        # Clean and prepare the base text
        clean_text = PromptEnhancer._clean_subtitle_text(subtitle_text)
        
        # Add visual context based on keywords
        enhanced_text = PromptEnhancer._add_visual_context(clean_text)
        
        # Build style components
        style_prompt = PromptEnhancer._build_style_prompt(style_settings)
        
        # Combine everything
        final_prompt = f"{enhanced_text}, {style_prompt}, high quality, detailed, professional, sharp focus, good lighting"
        
        # Ensure prompt length is optimal for Flux
        final_prompt = PromptEnhancer._optimize_prompt_length(final_prompt)
        
        return final_prompt
    
    @staticmethod
    def _clean_subtitle_text(text):
        """Clean subtitle text and prepare for visual description"""
        # Remove common subtitle artifacts
        clean = re.sub(r'\[.*?\]', '', text)  # Remove [sound effects]
        clean = re.sub(r'\(.*?\)', '', clean)  # Remove (speaker names)
        clean = re.sub(r'<.*?>', '', clean)   # Remove <formatting>
        clean = re.sub(r'♪.*?♪', '', clean)   # Remove music notation
        clean = clean.strip()
        
        return clean
    
    @staticmethod
    def _add_visual_context(text):
        """Add visual context based on text content"""
        visual_enhancements = {
            # Actions
            'walk': 'person walking',
            'run': 'person running dynamically',
            'fight': 'intense action scene',
            'dance': 'graceful dancing movement',
            'talk': 'people in conversation',
            'shout': 'dramatic emotional expression',
            
            # Emotions
            'happy': 'joyful expression, bright atmosphere',
            'sad': 'melancholic mood, soft lighting',
            'angry': 'intense expression, dramatic lighting',
            'scared': 'fearful expression, dark atmosphere',
            'excited': 'energetic expression, vibrant scene',
            
            # Settings
            'house': 'residential interior or exterior',
            'car': 'automotive scene',
            'office': 'professional workplace environment',
            'park': 'outdoor natural setting',
            'restaurant': 'dining establishment interior',
            'school': 'educational facility',
            'hospital': 'medical facility interior',
            
            # Time of day
            'morning': 'bright morning light, golden hour',
            'night': 'evening atmosphere, artificial lighting',
            'sunset': 'warm sunset lighting, golden hour',
            'dawn': 'early morning light, soft atmosphere'
        }
        
        enhanced = text
        for keyword, enhancement in visual_enhancements.items():
            if keyword.lower() in text.lower():
                enhanced = f"{enhancement}, {enhanced}"
                break
        
        return enhanced
    
    @staticmethod
    def _build_style_prompt(style_settings):
        """Build style-specific prompt components"""
        components = []
        
        # Add style
        if style_settings['style']:
            components.append(style_settings['style'])
        
        # Add mood
        if style_settings['mood']:
            components.append(style_settings['mood'])
        
        # Add colors
        if style_settings['colors']:
            color_text = ', '.join(style_settings['colors'])
            components.append(color_text)
        
        return ', '.join(components)
    
    @staticmethod
    def _optimize_prompt_length(prompt):
        """Optimize prompt length for Flux model"""
        # Flux works well with prompts under 200 words
        words = prompt.split()
        if len(words) > 150:
            # Keep the most important parts
            words = words[:150]
            prompt = ' '.join(words)
        
        return prompt

class FluxImageGenerator:
    """Handle image generation using Flux Schnell API"""
    
    def __init__(self, flux_api_url, api_key=None):
        self.api_url = flux_api_url
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.headers["Content-Type"] = "application/json"
    
    def generate_scene_image(self, prompt, max_retries=3):
        """Generate image using Flux API"""
        for attempt in range(max_retries):
            try:
                # Prepare payload for Flux
                payload = {
                    "prompt": prompt,
                    "width": 1024,
                    "height": 1024,
                    "steps": 4,  # Flux Schnell uses 4 steps
                    "guidance": 3.5,
                    "seed": -1,  # Random seed
                    "sampler": "euler"
                }
                
                # Make API request
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    # Handle different response formats
                    try:
                        result = response.json()
                        if 'image' in result:
                            # Base64 encoded image
                            image_data = base64.b64decode(result['image'])
                            return image_data
                        elif 'images' in result and result['images']:
                            # Array of base64 images
                            image_data = base64.b64decode(result['images'][0])
                            return image_data
                    except json.JSONDecodeError:
                        # Direct binary response
                        if response.headers.get('content-type', '').startswith('image/'):
                            return response.content
                
                elif response.status_code == 503:
                    # Service unavailable, retry
                    if attempt < max_retries - 1:
                        st.info(f"Service busy... Retrying in 15 seconds (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(15)
                        continue
                    else:
                        st.error("Service is currently unavailable. Please try again later.")
                        return None
                
                else:
                    error_msg = f"API error: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail.get('error', error_detail.get('message', 'Unknown error'))}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                    
                    st.error(error_msg)
                    return None
            
            except requests.exceptions.Timeout:
                st.error(f"Request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(10)
                    continue
                return None
            except Exception as e:
                st.error(f"Image generation failed: {str(e)}")
                return None
        
        return None

class UIComponents:
    """Handle Streamlit UI components and layout"""
    
    @staticmethod
    def setup_page():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="SRT Scene Generator - Flux", 
            page_icon="🎬", 
            layout="wide"
        )
        
        st.title("🎬 SRT Scene Generator with Flux")
        st.markdown("Upload SRT files and generate high-quality images using Flux AI")
    
    @staticmethod
    def render_api_config():
        """Render API configuration section"""
        st.subheader("🔧 Flux API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input(
                "Flux API URL",
                value="http://localhost:8000/generate",
                help="URL to your Flux API endpoint"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key (optional)",
                type="password",
                help="API key if required by your endpoint"
            )
        
        return api_url, api_key
    
    @staticmethod
    def render_style_controls():
        """Render style configuration controls"""
        st.subheader("🎨 Visual Style Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            style = st.selectbox("📸 Photography Style", [
                "cinematic photography",
                "documentary style",
                "portrait photography",
                "landscape photography",
                "street photography",
                "studio lighting",
                "natural lighting",
                "dramatic lighting",
                "film noir style",
                "vintage photography"
            ])
        
        with col2:
            mood = st.selectbox("🌟 Mood & Atmosphere", [
                "bright and cheerful",
                "dark and moody", 
                "dramatic and intense",
                "peaceful and serene",
                "energetic and dynamic",
                "mysterious atmosphere",
                "romantic and intimate",
                "epic and grand",
                "suspenseful tension",
                "warm and cozy"
            ])
        
        with col3:
            colors = st.multiselect("🎨 Color Palette", [
                "warm golden tones",
                "cool blue tones", 
                "vibrant saturated colors",
                "muted earth tones",
                "black and white",
                "pastel soft colors",
                "high contrast",
                "natural color grading",
                "neon accent colors",
                "sunset warm colors"
            ], default=["natural color grading"])
        
        return {"style": style, "mood": mood, "colors": colors}
    
    @staticmethod
    def render_scene_selection():
        """Render scene selection controls"""
        st.subheader("🎯 Scene Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_images = st.number_input(
                "📸 Number of Images to Generate", 
                min_value=1, 
                max_value=20,
                value=5,
                help="How many scenes to extract and generate images for"
            )
        
        with col2:
            method = st.selectbox("📋 Scene Selection Method", [
                "even_distribution",
                "longest_duration", 
                "keyword_based"
            ], format_func=lambda x: {
                "even_distribution": "📊 Even Distribution (Timeline)",
                "longest_duration": "⏱️ Longest Duration Scenes",
                "keyword_based": "🔍 Visual Keyword Based"
            }[x])
        
        return num_images, method
    
    @staticmethod
    def display_results(results):
        """Display generated images with timing information"""
        if not results:
            st.warning("No images were generated.")
            return
        
        # Download all button
        if len(results) > 1:
            zip_data = UIComponents._create_zip(results)
            st.download_button(
                "📦 Download All Images (ZIP)",
                data=zip_data,
                file_name="flux_scene_images.zip",
                mime="application/zip"
            )
        
        st.divider()
        
        # Display individual results
        for i, result in enumerate(results):
            with st.container():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader(f"🎬 Scene {i + 1}")
                    st.write(f"**⏰ Time:** `{result['start_time']}` → `{result['end_time']}`")
                    st.write(f"**⏱️ Duration:** {result['duration']:.1f} seconds")
                    st.write(f"**📝 Subtitle:** {result['text'][:150]}...")
                    
                    if result['image_data']:
                        st.download_button(
                            f"💾 Download Scene {i + 1}",
                            data=result['image_data'],
                            file_name=f"flux_scene_{i + 1}_{result['start_time'].replace(':', '-').replace(',', '-')}.png",
                            mime="image/png"
                        )
                    
                    st.metric("File Size", f"{len(result['image_data']) / 1024:.1f} KB" if result['image_data'] else "Failed")
                
                with col2:
                    if result['image_data']:
                        try:
                            image = Image.open(io.BytesIO(result['image_data']))
                            st.image(
                                image, 
                                caption=f"Generated at {result['start_time']} - {result['end_time']}",
                                use_column_width=True
                            )
                        except Exception as e:
                            st.error(f"Could not display image: {str(e)}")
                    else:
                        st.error("❌ Image generation failed for this scene")
                
                if result.get('enhanced_prompt'):
                    with st.expander(f"🔍 View Enhanced Prompt for Scene {i + 1}"):
                        st.code(result['enhanced_prompt'], language="text")
                
                st.divider()
    
    @staticmethod
    def _create_zip(results):
        """Create ZIP file with all generated images"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(results):
                if result['image_data']:
                    filename = f"flux_scene_{i + 1}_{result['start_time'].replace(':', '-').replace(',', '-')}.png"
                    zip_file.writestr(filename, result['image_data'])
                    
                    # Also add a text file with scene info
                    info_content = f"""Scene {i + 1} Information:
Time: {result['start_time']} → {result['end_time']}
Duration: {result['duration']:.1f} seconds
Subtitle: {result['text']}
Enhanced Prompt: {result.get('enhanced_prompt', 'N/A')}
"""
                    info_filename = f"scene_{i + 1}_info.txt"
                    zip_file.writestr(info_filename, info_content.encode('utf-8'))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

def main():
    """Main application function"""
    UIComponents.setup_page()
    
    # API Configuration
    api_url, api_key = UIComponents.render_api_config()
    
    # File upload
    st.subheader("📁 Upload SRT File")
    uploaded_file = st.file_uploader(
        "Choose your SRT subtitle file", 
        type=['srt'],
        help="Upload your subtitle file to generate scene images"
    )
    
    if not uploaded_file:
        st.info("👆 Upload an SRT file to get started")
        
        # Show example
        with st.expander("📝 SRT File Format Example"):
            st.code("""1
00:00:01,000 --> 00:00:04,000
A hero stands on a mountain peak overlooking the valley

2
00:00:05,000 --> 00:00:08,000
Suddenly a dragon appears in the distance, breathing fire

3
00:00:09,000 --> 00:00:12,000
The epic battle begins with magical spells flying everywhere""")
        
        with st.expander("🚀 API Setup Instructions"):
            st.markdown("""
            **Local Setup (Recommended):**
            1. Install ComfyUI or Automatic1111 with Flux Schnell
            2. Set up API endpoint (default: http://localhost:8000/generate)
            3. Configure the API URL above
            
            **Cloud Services:**
            - RunPod, Vast.ai, or similar GPU cloud providers
            - Replicate API with Flux models
            - Custom Flux deployment
            
            **API Response Format:**
            Your API should return JSON with 'image' field containing base64 encoded image data.
            """)
        return
    
    # Parse SRT file
    try:
        srt_content = uploaded_file.read().decode('utf-8')
        subtitles = SRTParser.parse_srt(srt_content)
        
        if not subtitles:
            st.error("❌ No valid subtitles found in the uploaded file")
            return
        
        st.success(f"✅ Successfully parsed {len(subtitles)} subtitles")
        
        # Show preview
        with st.expander("👀 Preview Parsed Subtitles"):
            for i, sub in enumerate(subtitles[:5]):
                st.write(f"**{i+1}.** `{sub['start_time']}` → `{sub['end_time']}` ({sub['duration']:.1f}s)")
                st.write(f"   *{sub['text']}*")
            if len(subtitles) > 5:
                st.write(f"... and {len(subtitles) - 5} more subtitles")
        
    except Exception as e:
        st.error(f"❌ Failed to parse SRT file: {str(e)}")
        return
    
    # Configuration
    style_settings = UIComponents.render_style_controls()
    num_images, selection_method = UIComponents.render_scene_selection()
    
    # Generate images
    if st.button("🎨 Generate Scene Images with Flux", type="primary"):
        # Validate API configuration
        if not api_url:
            st.error("⚠️ Flux API URL is not configured. Please check your secrets.toml file.")
            st.code("""
# Add to .streamlit/secrets.toml:
FLUX_API_URL = "http://localhost:8000/generate"
FLUX_API_KEY = "your-api-key"  # optional
            """)
            return
        
        with st.spinner("🔄 Analyzing scenes and generating images..."):
            try:
                # Select key scenes
                selected_scenes = SceneSelector.identify_key_scenes(
                    subtitles, num_images, selection_method
                )
                
                if not selected_scenes:
                    st.error("❌ No scenes selected for image generation")
                    return
                
                st.info(f"🎯 Selected {len(selected_scenes)} scenes for image generation")
                
                # Initialize components
                enhancer = PromptEnhancer()
                generator = FluxImageGenerator(api_url, api_key)
                
                # Generate images
                results = []
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                for i, scene in enumerate(selected_scenes):
                    # Update progress
                    progress = (i + 1) / len(selected_scenes)
                    progress_bar.progress(progress)
                    status_placeholder.write(f"🎬 Processing scene {i + 1}/{len(selected_scenes)}: `{scene['start_time']}`")
                    
                    # Enhance prompt
                    enhanced_prompt = enhancer.enhance_scene_prompt(
                        scene['text'], style_settings
                    )
                    
                    # Generate image
                    image_data = generator.generate_scene_image(enhanced_prompt)
                    
                    # Store result
                    results.append({
                        **scene,
                        'enhanced_prompt': enhanced_prompt,
                        'image_data': image_data
                    })
                
                progress_bar.empty()
                status_placeholder.empty()
                
                # Show summary
                successful = sum(1 for r in results if r['image_data'])
                st.success(f"🎉 Generated {successful}/{len(results)} images successfully!")
                
                # Display results
                st.subheader("🖼️ Generated Scenes")
                UIComponents.display_results(results)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
