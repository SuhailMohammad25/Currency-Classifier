import streamlit as st
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
import os
import base64

# TensorFlow Lite runtime (much lighter!)
import tflite_runtime.interpreter as tflite

# Page configuration
st.set_page_config(page_title="Currency Classifier", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for better styling
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .metric-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h3 {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load TensorFlow Lite model"""
    interpreter = tflite.Interpreter(model_path="best_currency_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def get_session_state():
    """Initialize session state"""
    return {
        'image_size': 224,
        'class_names': ['10', '20', '50', '100', '200', '500'],
        'hindi_numbers': {
            '10': 'दस',
            '20': 'बीस',
            '50': 'पचास',
            '100': 'सौ',
            '200': 'दो सौ',
            '500': 'पांच सौ'
        }
    }

def preprocess_image(img_array):
    """Apply histogram equalization preprocessing"""
    from skimage.exposure import equalize_adapthist
    
    # Ensure image is float32 in range [0, 1]
    if img_array.max() > 1:
        img_array = img_array / 255.0
    
    # Apply histogram equalization per channel
    processed = np.zeros_like(img_array)
    for channel in range(3):
        processed[:, :, channel] = equalize_adapthist(img_array[:, :, channel], clip_limit=0.03)
    
    # Clip to [0.02, 0.98] percentile and normalize
    p2, p98 = np.percentile(processed, (2, 98))
    processed = np.clip(processed, p2, p98)
    processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-6)
    
    return processed

def predict_currency(image_path):
    """Make prediction on the image using TFLite"""
    try:
        interpreter = load_model()
        session_state = get_session_state()
        
        # Load image using PIL
        img = Image.open(image_path).convert('RGB')
        img = img.resize((session_state['image_size'], session_state['image_size']))
        img_array = np.array(img, dtype=np.float32)
        
        # Apply preprocessing
        processed_img = preprocess_image(img_array)
        img_array = np.expand_dims(processed_img, axis=0).astype(np.float32)
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get predictions
        predictions_raw = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_class = np.argmax(predictions_raw)
        
        # Get sorted predictions
        sorted_indices = np.argsort(predictions_raw)[::-1]
        
        # Top 2 difference
        top_2_diff = predictions_raw[sorted_indices[0]] - predictions_raw[sorted_indices[1]]
        
        # Simple validation
        is_valid = top_2_diff >= 0.01
        
        return is_valid, predicted_class, predictions_raw, sorted_indices
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return False, None, None, None

def speak_result(denomination, language):
    """Generate audio using gTTS"""
    try:
        if language == "English":
            text = f"I've identified this as a {denomination}₹ note. For better results, please capture the picture in good lighting."
            lang = 'en'
        else:
            session_state = get_session_state()
            hindi_number = session_state['hindi_numbers'][str(denomination)]
            text = f"मैंने इसे {hindi_number}₹ का नोट पहचाना है। बेहतर परिणामों के लिए, कृपया अच्छी रोशनी में तस्वीर लें।"
            lang = 'hi'
        
        try:
            # Create audio using gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            tts.save(temp_path)
            
            # Read and play the audio
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Play audio in background using HTML audio
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg"></audio>'
            st.html(audio_html)
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            pass
        
    except Exception as e:
        pass

# Main app
st.title("🪙 Currency Classifier")
st.markdown("Classify Indian currency notes using AI")

session_state = get_session_state()

# Language selection
col1, col2 = st.columns(2)
with col1:
    language = st.radio("Language / भाषा:", ["English", "हिंदी"], horizontal=True)

# Image upload
st.markdown("---")
uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    if st.button("🔍 Classify Currency", use_container_width=True):
        with st.spinner("Analyzing image..."):
            is_valid, predicted_class, predictions, sorted_indices = predict_currency(uploaded_file)
        
        if is_valid and predicted_class is not None:
            denomination = session_state['class_names'][predicted_class]
            
            # Display result
            st.markdown("---")
            st.markdown("### ✅ Classification Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Denomination", f"₹{denomination}")
            with col2:
                # Get confidence from raw predictions array
                confidence_value = float(predictions[predicted_class])
                st.metric("Confidence", f"{confidence_value*100:.2f}%")
            
            # Show top 3 predictions
            st.markdown("### Top Predictions")
            top_3_indices = sorted_indices[:3]
            
            prediction_data = []
            for idx in top_3_indices:
                conf_value = float(predictions[idx])
                prediction_data.append({
                    "Denomination": f"₹{session_state['class_names'][idx]}",
                    "Confidence": f"{conf_value*100:.2f}%"
                })
            
            st.table(prediction_data)
            
            # Play audio automatically (silently)
            speak_result(denomination, language)
            
        else:
            st.markdown("---")
            st.error("❌ Unable to make a confident prediction. Please ensure you're using a clear image of an Indian currency note.")
else:
    st.info("👆 Upload an image to get started!")
