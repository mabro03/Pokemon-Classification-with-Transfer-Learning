import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- [설정] ---
NUM_CLASSES = 150

st.set_page_config(page_title="포켓몬 분류기", layout="centered")
st.title("🐾 포켓몬 도감 분류기")

@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pth"):
        st.error("학습된 모델 파일(best_model.pth)을 찾을 수 없습니다!")
        return None, None

    # 저장된 데이터 로드 (CPU 환경 고려)
    checkpoint = torch.load("best_model.pth", map_location="cpu")
    model_name = checkpoint['model_name'].lower()
    
    # 1. 저장된 이름에 맞춰 모델 구조 생성
    if "resnet" in model_name:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    # 2. 가중치 로드
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint.get('class_names', [f"Class_{i}" for i in range(NUM_CLASSES)])

model, class_names = load_model()

# 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("포켓몬 사진을 업로드하세요!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)
    
    if st.button("분류 시작"):
        input_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
            
        result_name = class_names[top_idx.item()]
        
        st.divider()
        st.balloons()
        st.success(f"이 포켓몬은 **{result_name}**일 확률이 {top_prob.item()*100:.2f}%입니다!")