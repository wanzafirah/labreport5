import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
import pandas as pd

#Step 1
st.set_page_config(page_title="AI Lab5", layout="centered")
st.title("Image Classifier (CPU)")

#Step 3
device = torch.device("cpu")

#Step 4
@st.cache_resource
def load_my_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    model.eval()
    return model, weights

model, weights = load_my_model()

#Step 5
preprocess = weights.transforms()

#Step 6
uploaded_file = st.file_uploader("Upload an Image (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300, caption="Your Image")

    #Step 7
    input_tensor = preprocess(image)
    
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    #Step 8
    probs = torch.nn.functional.softmax(output[0], dim=0)
    
    #Top5 highest
    top5_prob, top5_id = torch.topk(probs, 5)

    class_names = weights.meta["categories"]

    results = {}
    for i in range(5):
        label = class_names[top5_id[i]] 
        score = top5_prob[i].item() 
        results[label] = score

    #Step 9
    st.subheader("Top 5 Predictions")
    df = pd.DataFrame(list(results.items()), columns=["Object", "Confidence"])
    df.set_index("Object", inplace=True)
    
    st.bar_chart(df)
