import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageEnhance
import requests
import pandas as pd
import io

# Initialize the processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Streamlit app
st.title("Object Detection")

# Sidebar for image download
st.sidebar.header("Options")
st.sidebar.write("Download the processed image below:")

# Pull-down menu to explain the model
with st.expander("What does this model do?"):
    st.write("""
        DETR (DEtection TRansformers) is a model developed by Facebook AI Research for object detection.
        It uses a transformer architecture to directly predict the bounding boxes and class labels of objects
        in an image. DETR simplifies the object detection pipeline by eliminating the need for many hand-designed
        components typically used in other object detection models.
    """)

color_map = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
    4: "cyan",
    5: "magenta",
    6: "orange",
    7: "purple",
    8: "brown",
    9: "pink"
}

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        image = Image.open(uploaded_file)
        # Enhance the image (optional)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1)  # Increase sharpness

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5) 
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform object detection
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        table_data = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            color = color_map[label.item() % len(color_map)]
            draw.rectangle(box, outline=color, width=1)
            table_data.append([model.config.id2label[label.item()], color, round(score.item(), 2)])
    
        # Display the image with bounding boxes
        st.image(image, caption="Image with Bounding Boxes", use_column_width=True)
        
        # Display the table with class, color, and score
        df = pd.DataFrame(table_data, columns=["Class", "Color", "Score"])
        st.write("The table")
        st.table(df)

        # Save processed image to a buffer
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Add download button to sidebar
        st.sidebar.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )
