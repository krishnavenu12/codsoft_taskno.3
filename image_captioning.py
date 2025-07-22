# blip_captioning_app.py
import streamlit as st
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch

# Set Streamlit page config
st.set_page_config(page_title="🧠 BLIP Image Captioning", layout="centered")
st.title("📸 Image Captioning ")

# Load BLIP model only once
@st.cache_resource
def load_blip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="base_coco",
        is_eval=True,
        device=device
    )
    return model, vis_processors, device

model, vis_processors, device = load_blip_model()

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    raw_image = Image.open(uploaded).convert("RGB")
    st.image(raw_image, caption="Your Image", use_container_width=True)

    with st.spinner("Generating caption with BLIP... 🧠"):
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image})[0]

    st.success(f"📝 Caption: {caption}")
