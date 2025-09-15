import streamlit as st
import replicate
import requests
from PIL import Image
import io
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="FLUX SCHNELL Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 FLUX SCHNELL Image Generator")
st.markdown("Generate high-quality images using the FLUX SCHNELL model from Black Forest Labs via Replicate.")

# Sidebar for API key input
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Replicate API Token",
    type="password",
    help="Enter your Replicate API token. Get one at https://replicate.com/account/api-tokens"
)

if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    
    # Prompt input
    prompt = st.text_area(
        "Image Prompt",
        value='black forest gateau cake spelling out the words "FLUX SCHNELL", tasty, food photography, dynamic shot',
        height=100,
        help="Describe the image you want to generate"
    )
    
    # Additional settings (you can expand these based on FLUX SCHNELL capabilities)
    with st.expander("Advanced Settings"):
        st.info("FLUX SCHNELL is optimized for speed and uses default settings for most parameters.")
        save_to_disk = st.checkbox("Save images to disk", value=False)
    
    # Generate button
    generate_button = st.button("🚀 Generate Image", type="primary", use_container_width=True)

with col2:
    st.header("Output")
    
    if generate_button:
        if not api_key:
            st.error("Please enter your Replicate API token in the sidebar.")
        elif not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            try:
                with st.spinner("Generating image... This may take a few moments."):
                    # Set up the input for Replicate
                    input_data = {
                        "prompt": prompt
                    }
                    
                    # Run the model
                    output = replicate.run(
                        "black-forest-labs/flux-schnell",
                        input=input_data
                    )
                    
                    # Display the generated images
                    if output:
                        st.success(f"Generated {len(output)} image(s)!")
                        
                        for index, item in enumerate(output):
                            # Get the image URL
                            image_url = item.url() if hasattr(item, 'url') else str(item)
                            
                            # Display the image
                            st.image(image_url, caption=f"Generated Image {index + 1}")
                            
                            # Provide download link
                            st.markdown(f"**Image URL:** [{image_url}]({image_url})")
                            
                            # Save to disk if requested
                            if save_to_disk:
                                try:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"flux_output_{timestamp}_{index}.webp"
                                    
                                    with open(filename, "wb") as file:
                                        file.write(item.read())
                                    
                                    st.success(f"✅ Saved as {filename}")
                                    
                                except Exception as save_error:
                                    st.error(f"Error saving image: {str(save_error)}")
                    
                    else:
                        st.warning("No images were generated.")
                        
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                
                # Provide helpful error messages
                if "authentication" in str(e).lower():
                    st.error("Authentication failed. Please check your Replicate API token.")
                elif "quota" in str(e).lower() or "limit" in str(e).lower():
                    st.error("You may have reached your API quota or rate limit. Please try again later.")
                else:
                    st.error("Make sure you have the required packages installed: `pip install replicate streamlit pillow`")

# Information section
with st.expander("ℹ️ About FLUX SCHNELL"):
    st.markdown("""
    **FLUX SCHNELL** is a fast text-to-image model from Black Forest Labs that generates high-quality images quickly.
    
    **Features:**
    - Fast generation speed
    - High-quality outputs
    - Optimized for efficiency
    
    **Requirements:**
    - Replicate API token (free tier available)
    - Internet connection
    
    **Setup:**
    1. Get your API token from [Replicate](https://replicate.com/account/api-tokens)
    2. Enter it in the sidebar
    3. Enter your prompt and click Generate!
    """)

# Example prompts
with st.expander("💡 Example Prompts"):
    st.markdown("""
    Try these example prompts:
    
    - `a majestic lion in golden savanna, wildlife photography, sunset lighting`
    - `futuristic city skyline at night, neon lights, cyberpunk style`
    - `cozy coffee shop interior, warm lighting, books and plants`
    - `abstract digital art, flowing colors, modern art style`
    - `vintage car on mountain road, scenic landscape, golden hour`
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Replicate's FLUX SCHNELL model")
