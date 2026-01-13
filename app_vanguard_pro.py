import torch
import streamlit as st
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import json
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis

# --- MARRTY LLC: IFR31-P1 IDENTITY PORTAL ---
st.set_page_config(page_title="Marrty IFR31-P1", page_icon="🛡️", layout="wide")

# Sidebar Branding and Project Metadata
st.sidebar.title("🛡️ Project IFR31-P1")
st.sidebar.markdown(f"""
**System Specifications**
- **Model Name:** IFR31
- **Version:** V1 (Master Final)
- **Phase:** P1 (Phase 1)
- **Developed by:** Marrty LLC
- **Co-developed by:** Rhaul PR

**Phase 1 Scope**
- **Registered Students:** 12 Identities
- **Validation Accuracy:** 98.41%

**Contact Information**
- 📧 [rahul@marrty.com](mailto:rahul@marrty.com)
- 📧 [info@marrty.com](mailto:info@marrty.com)
""")

st.sidebar.warning("⚠️ Internal Use Only: Usage restricted to Marrty LLC authorized personnel.")

# --- MODEL ARCHITECTURE (IFR31 V1) ---
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s

class IFR31_V1_Master(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet101()
        self.backbone.fc = nn.Identity()
        self.feature_layer = nn.Linear(2048, 512)
        self.arc_head = ArcMarginProduct(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.feature_layer(features)
        logits = F.linear(F.normalize(embeddings), F.normalize(self.arc_head.weight))
        return logits * self.arc_head.s

# --- CORE SYSTEM LOADING ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_ifr31_system():
    # Load Label Mapping
    with open("labels_v1.json", "r") as f:
        mapping = json.load(f)
    idx_to_folder = {v: k for k, v in mapping.items()}
    
    # Initialize IFR31 ResNet-101 Model
    model = IFR31_V1_Master(len(idx_to_folder))
    # Note: Ensure Marrty_VANGUARD_V1.pth is renamed to IFR31_V1_MASTER.pth locally
    model.load_state_dict(torch.load("IFR31_V1_MASTER.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    
    # Initialize Pro-Detector
    detector = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    
    return model, idx_to_folder, detector

# System Health Check
try:
    model, idx_to_folder, detector = load_ifr31_system()
except Exception:
    st.error("IFR31 Core Files Missing: Ensure IFR31_V1_MASTER.pth and labels_v1.json are in the directory.")

# --- IDENTITY PORTAL UI ---
st.title("🛡️ Marrty LLC: Project IFR31-P1")
st.markdown("### Secure Biometric Verification Engine | Version 1.0")
st.divider()

uploaded_file = st.file_uploader("Upload Image for IFR31 Scanning", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert('RGB')
    cv_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(raw_img, caption="Marrty LLC: Target Subject", use_container_width=True)

    if st.button("🚀 INITIATE IFR31 SCAN"):
        with st.spinner("Analyzing biometric data..."):
            faces = detector.get(cv_img)
            if not faces:
                st.error("Scan Failed: No face signatures detected.")
            else:
                # Primary Face Selection
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                
                # Pro-Alignment Geometry
                dst_pts = np.array([[30.2946, 51.6963], [71.7054, 51.6963], [51.0000, 71.7366], 
                                    [33.5493, 92.3655], [68.4507, 92.3655]], dtype=np.float32)
                tform = cv2.estimateAffinePartial2D(face.kps, dst_pts)[0]
                aligned_face = cv2.warpAffine(cv_img, tform, (112, 112))
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                
                # Neural Embedding Extraction
                t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
                img_t = t(Image.fromarray(aligned_rgb)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(img_t)
                    probs = F.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                name = idx_to_folder[pred.item()]
                confidence = conf.item() * 100
                
                with col2:
                    st.image(aligned_rgb, caption="Biometric Normalization", width=150)
                    if confidence > 95:
                        st.success(f"**IFR31 MATCH: {name.replace('_', ' ')}**")
                        st.metric("Identity Confidence", f"{confidence:.2f}%")
                    else:
                        st.warning("Low Confidence Detection")
                        st.write(f"Identity likely **{name}** ({confidence:.2f}%). Verification required.")
