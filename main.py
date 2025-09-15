import streamlit as st
import io
import zipfile
from datetime import datetime, timedelta
import re
from PIL import Image
import google.generativeai as genai

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
            'beautiful', 'amazing', 'stunning', 'dramatic', 'exciting'
        ]
        
        scored_subtitles = []
        for subtitle in subtitles:
            score = sum(1 for keyword in visual_keywords 
                       if keyword.lower() in subtitle['text'].lower())
            scored_subtitles.append((subtitle, score))
        
        # Sort by score, then by duration
        scored_subtitles.sort(key=lambda x: (x[1], x[0]['duration']), reverse=True)
        return [item[0] for item in scored_subtitles[:num_images]]

class ImageGenerator:
    """Handle AI image generation using Gemini and Imagen"""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.imagen_model = genai.GenerativeModel('imagen-4.0-generate-001')
    
    def enhance_scene_prompt(self, subtitle_text, style_settings):
        """Use Gemini to create detailed visual prompts from subtitle text"""
        try:
            enhancement_prompt = f"""
            Convert this subtitle/dialogue into a detailed visual scene description for image generation:
            
            Text: "{subtitle_text}"
            
            Create a vivid visual description that includes:
            - Characters and their expressions/actions
            - Setting and environment details
            - Lighting and mood
            - Visual composition and camera angle
            - Specific artistic elements
            
            Style: {style_settings['style']}
            Mood: {style_settings['mood']}
            Colors: {', '.join(style_settings['colors'])}
            
            Format as a single detailed paragraph suitable for AI image generation.
            Focus on visual elements that would make a compelling scene.
            """
            
            response = self.gemini_model.generate_content(enhancement_prompt)
            return response.text.strip()
        
        except Exception as e:
            st.error(f"Failed to enhance prompt: {str(e)}")
            return f"{subtitle_text}, {style_settings['style']}, {style_settings['mood']}"
    
    def generate_scene_image(self, enhanced_prompt):
        """Generate image using Imagen 4.0"""
        try:
            response = self.imagen_model.generate_content([enhanced_prompt])
            
            if response.parts and len(response.parts) > 0:
                image_part = response.parts[0]
                if hasattr(image_part, 'data'):
                    return image_part.data
            
            return None
        
        except Exception as e:
            st.error(f"Image generation failed: {str(e)}")
            return None

class UIComponents:
    """Handle Streamlit UI components and layout"""
    
    @staticmethod
    def setup_page():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="SRT Scene Generator", 
            page_icon="🎬", 
            layout="wide"
        )
        
        st.title("🎬 SRT Scene Generator")
        st.markdown("Upload SRT files and generate images for key video scenes")
    
    @staticmethod
    def render_style_controls():
        """Render style configuration controls"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            style = st.selectbox("🎨 Visual Style", [
                "Cinematic", "Documentary", "Anime", "Cartoon", 
                "Realistic", "Artistic", "Comic Book", "Fantasy"
            ])
        
        with col2:
            mood = st.selectbox("🌟 Mood", [
                "Dramatic", "Peaceful", "Energetic", "Dark", 
                "Bright", "Mysterious", "Epic", "Intimate"
            ])
        
        with col3:
            colors = st.multiselect("🎨 Color Palette", [
                "Warm tones", "Cool tones", "Vibrant", "Muted", 
                "Monochrome", "Pastel", "High contrast", "Natural"
            ], default=["Natural"])
        
        return {"style": style, "mood": mood, "colors": colors}
    
    @staticmethod
    def render_scene_selection():
        """Render scene selection controls"""
        col1, col2 = st.columns(2)
        
        with col1:
            num_images = st.number_input(
                "📸 Number of Images", 
                min_value=1, 
                max_value=20, 
                value=5
            )
        
        with col2:
            method = st.selectbox("📋 Scene Selection Method", [
                "even_distribution",
                "longest_duration", 
                "keyword_based"
            ], format_func=lambda x: {
                "even_distribution": "Even Distribution",
                "longest_duration": "Longest Scenes",
                "keyword_based": "Keyword Based"
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
                "📦 Download All Images",
                data=zip_data,
                file_name="scene_images.zip",
                mime="application/zip"
            )
        
        st.divider()
        
        # Display individual results
        for i, result in enumerate(results):
            with st.container():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader(f"Scene {i + 1}")
                    st.write(f"**Time:** {result['start_time']} → {result['end_time']}")
                    st.write(f"**Duration:** {result['duration']:.1f}s")
                    st.write(f"**Text:** {result['text'][:100]}...")
                    
                    if result['image_data']:
                        st.download_button(
                            f"💾 Download Scene {i + 1}",
                            data=result['image_data'],
                            file_name=f"scene_{i + 1}_{result['start_time'].replace(':', '-')}.png",
                            mime="image/png"
                        )
                
                with col2:
                    if result['image_data']:
                        try:
                            image = Image.open(io.BytesIO(result['image_data']))
                            st.image(image, caption=f"Generated scene at {result['start_time']}")
                        except Exception as e:
                            st.error(f"Could not display image: {str(e)}")
                    else:
                        st.error("Image generation failed for this scene")
                
                if result.get('enhanced_prompt'):
                    with st.expander(f"View Enhanced Prompt for Scene {i + 1}"):
                        st.write(result['enhanced_prompt'])
                
                st.divider()
    
    @staticmethod
    def _create_zip(results):
        """Create ZIP file with all generated images"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(results):
                if result['image_data']:
                    filename = f"scene_{i + 1}_{result['start_time'].replace(':', '-')}.png"
                    zip_file.writestr(filename, result['image_data'])
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

def main():
    """Main application function"""
    UIComponents.setup_page()
    
    # Check API key
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("Please configure GEMINI_API_KEY in Streamlit secrets")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload SRT File", 
        type=['srt'],
        help="Upload your subtitle file to generate scene images"
    )
    
    if not uploaded_file:
        st.info("👆 Upload an SRT file to get started")
        return
    
    # Parse SRT file
    try:
        srt_content = uploaded_file.read().decode('utf-8')
        subtitles = SRTParser.parse_srt(srt_content)
        
        if not subtitles:
            st.error("No valid subtitles found in the uploaded file")
            return
        
        st.success(f"✅ Parsed {len(subtitles)} subtitles")
        
    except Exception as e:
        st.error(f"Failed to parse SRT file: {str(e)}")
        return
    
    # Configuration controls
    st.subheader("⚙️ Configuration")
    style_settings = UIComponents.render_style_controls()
    num_images, selection_method = UIComponents.render_scene_selection()
    
    # Generate images
    if st.button("🎨 Generate Scene Images", type="primary"):
        with st.spinner("Generating images..."):
            try:
                # Select key scenes
                selected_scenes = SceneSelector.identify_key_scenes(
                    subtitles, num_images, selection_method
                )
                
                if not selected_scenes:
                    st.error("No scenes selected for image generation")
                    return
                
                st.info(f"Selected {len(selected_scenes)} scenes for image generation")
                
                # Initialize image generator
                generator = ImageGenerator(st.secrets["GEMINI_API_KEY"])
                
                # Generate images for each scene
                results = []
                progress_bar = st.progress(0)
                
                for i, scene in enumerate(selected_scenes):
                    # Update progress
                    progress_bar.progress((i + 1) / len(selected_scenes))
                    
                    # Enhance prompt
                    enhanced_prompt = generator.enhance_scene_prompt(
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
                
                # Display results
                st.subheader("🖼️ Generated Scenes")
                UIComponents.display_results(results)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Step 1:** Upload your SRT (subtitle) file
        
        **Step 2:** Configure visual style, mood, and colors
        
        **Step 3:** Choose number of images and scene selection method:
        - **Even Distribution**: Spreads scenes evenly across the timeline
        - **Longest Scenes**: Selects scenes with the most dialogue
        - **Keyword Based**: Prioritizes scenes with visual keywords
        
        **Step 4:** Click "Generate Scene Images" to create visuals
        
        **Step 5:** View results with timestamps and download individual or all images
        
        **Tips:**
        - Better SRT files with descriptive text produce better images
        - Use "Keyword Based" selection for more visually interesting scenes
        - Generated images include exact timestamps for video editing
        """)

if __name__ == "__main__":
    main()
