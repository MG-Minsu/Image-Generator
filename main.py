import streamlit as st
import replicate
import os

st.title("FLUX SCHNELL Image Generator")

# API key input
api_key = st.text_input("Replicate API Token", type="password")
if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key

# Input parameters
prompt = st.text_area("Prompt", 
    value='black forest gateau cake spelling out the words "FLUX SCHNELL", tasty, food photography, dynamic shot')

col1, col2 = st.columns(2)
with col1:
    go_fast = st.checkbox("Go Fast", value=True)
    megapixels = st.selectbox("Megapixels", ["0.25", "1", "2"], index=1)
    num_outputs = st.number_input("Number of outputs", min_value=1, max_value=4, value=1)

with col2:
    aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"], index=0)
    output_format = st.selectbox("Output format", ["webp", "jpg", "png"], index=0)
    output_quality = st.slider("Output quality", 1, 100, 80)

num_inference_steps = st.slider("Inference steps", 1, 10, 4)

if st.button("Generate Image"):
    try:
        with st.spinner("Generating..."):
            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": prompt,
                    "go_fast": go_fast,
                    "megapixels": megapixels,
                    "num_outputs": num_outputs,
                    "aspect_ratio": aspect_ratio,
                    "output_format": output_format,
                    "output_quality": output_quality,
                    "num_inference_steps": num_inference_steps
                }
            )
            
            for i, image in enumerate(output):
                st.image(image.url())
                
                # Download button
                if st.button(f"Save Image {i+1}", key=f"save_{i}"):
                    with open(f"flux_image_{i}.{output_format}", "wb") as file:
                        file.write(image.read())
                    st.success(f"Image saved as flux_image_{i}.{output_format}")
                    
    except Exception as e:
        st.error(f"Error: {e}")
