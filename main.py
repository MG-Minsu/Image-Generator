import streamlit as st
import torch
from diffusers import FluxPipeline
from PIL import Image
import io
import time

# Set page config
st.set_page_config(
    page_title="FLUX Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'pipe' not in st.session_state:
    st.session_state.pipe = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load the FLUX model with caching"""
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_image(pipe, prompt, guidance_scale, height, width, num_inference_steps, max_sequence_length):
    """Generate image using the FLUX pipeline"""
    try:
        with st.spinner("Generating image..."):
            start_time = time.time()
            
            result = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            return result.images[0], generation_time
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None, 0

# Main app
def main():
    st.title("🎨 FLUX Image Generator")
    st.markdown("Generate high-quality images using FLUX.1-schnell model")
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("Model Status")
        
        if not st.session_state.model_loaded:
            if st.button("Load FLUX Model", type="primary"):
                with st.spinner("Loading FLUX model... This may take a few minutes."):
                    st.session_state.pipe = load_model()
                    if st.session_state.pipe is not None:
                        st.session_state.model_loaded = True
                        st.success("Model loaded successfully!")
                        st.rerun()
        else:
            st.success("✅ Model loaded and ready!")
            if st.button("Unload Model"):
                st.session_state.pipe = None
                st.session_state.model_loaded = False
                st.rerun()
    
    # Main interface
    if st.session_state.model_loaded and st.session_state.pipe is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Generation Settings")
            
            # Prompt input
            prompt = st.text_area(
                "Prompt",
                value="A cat holding a sign that says hello world",
                height=100,
                help="Describe the image you want to generate"
            )
            
            # Advanced settings in expander
            with st.expander("Advanced Settings"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    guidance_scale = st.slider(
                        "Guidance Scale",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.1,
                        help="Higher values follow the prompt more closely"
                    )
                    
                    num_inference_steps = st.slider(
                        "Inference Steps",
                        min_value=1,
                        max_value=20,
                        value=4,
                        help="More steps = higher quality but slower generation"
                    )
                
                with col_b:
                    height = st.selectbox(
                        "Height",
                        options=[512, 768, 1024],
                        index=1,
                        help="Image height in pixels"
                    )
                    
                    width = st.selectbox(
                        "Width",
                        options=[512, 768, 1024, 1360],
                        index=3,
                        help="Image width in pixels"
                    )
                
                max_sequence_length = st.slider(
                    "Max Sequence Length",
                    min_value=128,
                    max_value=512,
                    value=256,
                    step=32,
                    help="Maximum length for text encoding"
                )
            
            # Generate button
            if st.button("Generate Image", type="primary", use_container_width=True):
                if prompt.strip():
                    image, gen_time = generate_image(
                        st.session_state.pipe,
                        prompt,
                        guidance_scale,
                        height,
                        width,
                        num_inference_steps,
                        max_sequence_length
                    )
                    
                    if image is not None:
                        st.session_state.generated_image = image
                        st.session_state.generation_time = gen_time
                        st.session_state.last_prompt = prompt
                else:
                    st.warning("Please enter a prompt!")
        
        with col2:
            st.header("Generated Image")
            
            # Display generated image
            if hasattr(st.session_state, 'generated_image'):
                st.image(
                    st.session_state.generated_image,
                    caption=f"Prompt: {st.session_state.last_prompt}",
                    use_column_width=True
                )
                
                # Show generation time
                st.info(f"⏱️ Generation time: {st.session_state.generation_time:.2f} seconds")
                
                # Download button
                img_buffer = io.BytesIO()
                st.session_state.generated_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="Download Image",
                    data=img_buffer,
                    file_name=f"flux_generated_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("👆 Generate an image using the settings on the left")
    
    else:
        # Instructions when model is not loaded
        st.info("""
        ### Welcome to FLUX Image Generator!
        
        **To get started:**
        1. Click "Load FLUX Model" in the sidebar
        2. Wait for the model to load (this may take a few minutes on first run)
        3. Enter your prompt and generate images!
        
        **System Requirements:**
        - GPU with sufficient VRAM (recommended)
        - At least 8GB RAM
        - Stable internet connection for model download
        """)
        
        with st.expander("About FLUX.1-schnell"):
            st.markdown("""
            FLUX.1-schnell is a fast, high-quality text-to-image model that can generate 
            detailed images in just a few inference steps. It's optimized for speed while 
            maintaining excellent image quality.
            
            **Features:**
            - Fast generation (typically 4 steps)
            - High resolution support
            - Excellent prompt adherence
            - Efficient memory usage with CPU offloading
            """)

if __name__ == "__main__":
    main
