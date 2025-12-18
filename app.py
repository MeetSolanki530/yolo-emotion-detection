import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model once
@st.cache_resource
def load_model():
    return YOLO(
        r"runs/detect/train2/weights/best.pt"
    )

model = load_model()
class_names = model.names

st.title("Facial Expression Detection")
st.markdown("Upload an image to detect facial expressions (happy, sad, angry, contempt, disgust, fear, neutral, surprised, sleepy).")

uploaded_file = st.file_uploader(
    "Upload image", type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold", 0.1, 1.0, 0.4
)
if st.button("Run Detection"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
    
        results = model(img_np, conf=confidence)[0]

        detections = results.boxes
        output_img = img_np.copy()

        detected_labels = []

        for box in detections:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output_img,
                f"{label} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            detected_labels.append(label)

        st.image(output_img, caption="Detection Result", width=800)
        
        if detected_labels:
            st.subheader("Detected Expressions")
            for lbl in set(detected_labels):
                st.write(f"- {lbl}")
        else:
            st.write("No expressions detected above confidence threshold.")
