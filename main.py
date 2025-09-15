import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Flux Photo Generator",
    page_icon="🎨",
    layout="centered"
)

# Get API key from secrets
try:
    api_key = st.secrets["REPLICATE_API_TOKEN"]
    client = replicate.Client(api_token=api_key)
except:
    st.error("Please add REPLICATE_API_TOKEN to your Streamlit secrets")
    st.stop()

# App title
st.title("🎨 Flux Photo Generator")
st.write("Generate photos using Flux AI")

# Text input for prompt
prompt = st.text_area(
    "Enter your photo prompt:",
    placeholder="A majestic mountain landscape at sunset with a crystal clear lake reflecting the golden sky",
    height=100
)

# Generate button
if st.button("Generate Photo", type="primary"):
    if not prompt:
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating photo..."):
            try:
                # Generate image using Flux Schnell (fastest model)
                output = replicate.run(
                    "black-forest-labs/flux-schnell",
                    input={
                        "prompt": prompt,
                        "width": 512,
                        "height": 512,
                        "num_outputs": 1,
                        "num_inference_steps": 4
                    }
                )
                
                # Get the image URL
                image_url = output[0] if isinstance(output, list) else output
                
                # Download and display image
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                
                st.image(image, caption=prompt, use_column_width=True)
                
                # Download button
                buf = BytesIO()
                image.save(buf, format='PNG')
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="flux_generated.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")

# Setup instructions
with st.expander("⚙️ Setup"):
    st.markdown("""
    **To configure secrets:**
    
    Create `.streamlit/secrets.toml` in your project:
    ```toml
    REPLICATE_API_TOKEN = "your_api_key_here"
    ```
    
    Get your API key from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
    """)

st.markdown("---")
st.markdown("Built with Streamlit 🚀")
