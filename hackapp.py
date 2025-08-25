import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from lime import lime_image
from skimage.segmentation import mark_boundaries

st.set_page_config(page_title="Pneumonia Detection with Explainable AI", layout="wide")

# Load your trained model once and cache it
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("hackthonfinal.keras")  # Update path if needed
    return model

model = load_model()

IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="block5_conv3"):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def integrated_gradients(model, img_input, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros(img_input.shape).astype(np.float32)
    img_input = tf.cast(img_input, tf.float32)
    baseline = tf.cast(baseline, tf.float32)
    alphas = tf.linspace(0., 1., steps+1)
    delta = img_input - baseline
    grads = []
    for alpha in alphas:
        interpolated = baseline + alpha * delta
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = model(interpolated)
            loss = pred[:, 0]
        grad = tape.gradient(loss, interpolated)
        grads.append(grad[0].numpy())
    avg_grads = np.average(grads[:-1] + grads[1:], axis=0) / 2.0
    integrated_grads = delta.numpy()[0] * avg_grads
    return integrated_grads

def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), colormap)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed

@st.cache_resource
def get_lime_explainer():
    return lime_image.LimeImageExplainer()

lime_explainer = get_lime_explainer()

def explain_with_lime(model, img_array):
    img = (img_array[0] * 255).astype(np.uint8)
    
    def predict_fn(images):
        images = images.astype(np.float32) / 255.0
        return model.predict(images)
    
    explanation = lime_explainer.explain_instance(
        img, 
        predict_fn, 
        top_labels=2, 
        hide_color=0, 
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0], 
        positive_only=True, 
        num_features=10, 
        hide_rest=False
    )
    
    lime_img = mark_boundaries(temp / 255.0, mask)
    lime_img = (lime_img * 255).astype(np.uint8)
    return lime_img

def generate_human_friendly_message(prediction):
    if prediction > 0.7:
        decision = "pneumonia"
        message = (
            "The AI model strongly suggests signs of pneumonia in this chest X-ray. "
            "The highlighted regions correspond to areas likely affected by infection. "
            "It is advisable to seek further clinical evaluation."
        )
    elif prediction < 0.3:
        decision = "normal"
        message = (
            "The AI model finds no evidence of pneumonia in this image. "
            "The lung fields appear clear and free of abnormalities typically associated with infection."
        )
    else:
        decision = "uncertain"
        message = (
            "The AI model is uncertain about this case. "
            "The findings are ambiguous, and clinical review is highly recommended for accurate diagnosis."
        )
    return decision, message

# App Title and Description
st.title("🩺 Pneumonia Detection with Explainable AI")
st.markdown("""
This app uses a deep learning model to detect pneumonia in chest X-ray images 
and provides visual explanations to help understand the AI’s decision-making process.
""")

st.markdown("### Upload Chest X-ray Image (JPEG/PNG)")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    decision, friendly_message = generate_human_friendly_message(prediction)

    st.markdown("## Diagnosis Summary")
    if decision == "pneumonia":
        st.success(friendly_message)
    elif decision == "normal":
        st.info(friendly_message)
    else:
        st.warning(friendly_message)

    st.markdown("---")
    st.markdown("## Visual Explanations")
    st.markdown("These highlight important areas influencing the AI's diagnosis.")

    img_cv = cv2.cvtColor(np.array(image.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_RGB2BGR)

    # Use columns for side-by-side display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Grad-CAM")
        heatmap = get_gradcam_heatmap(model, img_array)
        gradcam_img = overlay_heatmap(img_cv, heatmap)
        st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.expander("About Grad-CAM"):
            st.write("""
                Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the key regions of the chest X-ray 
                image that the AI model focused on to make its pneumonia diagnosis. This helps visualize which parts 
                of the lungs influenced the decision.
            """)

    with col2:
        st.subheader("Integrated Gradients")
        ig_attributions = integrated_gradients(model, tf.convert_to_tensor(img_array))
        ig_mask = np.sum(np.abs(ig_attributions), axis=-1)
        ig_mask = (ig_mask - ig_mask.min()) / (ig_mask.max() - ig_mask.min() + 1e-8)
        ig_img = overlay_heatmap(img_cv, ig_mask)
        st.image(cv2.cvtColor(ig_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with st.expander("About Integrated Gradients"):
            st.write("""
                Integrated Gradients quantify the contribution of each pixel to the model’s prediction by 
                comparing the input image to a baseline. This pixel-level attribution helps understand the detailed 
                importance of image features.
            """)

    with col3:
        st.subheader("LIME")
        with st.spinner("Generating LIME explanation..."):
            lime_img = explain_with_lime(model, img_array)
            st.image(lime_img, use_container_width=True)
        with st.expander("About LIME"):
            st.write("""
                LIME (Local Interpretable Model-agnostic Explanations) breaks the image into superpixels and identifies 
                which regions have the most influence on the classification decision. It provides a local, interpretable 
                explanation of why the model predicted pneumonia or normal.
            """)

    st.markdown("---")
    st.info("⚠️ Note: These explanations help interpret AI predictions but do not replace professional medical judgment.")
else:
    st.info("Please upload a chest X-ray image to get started.")
