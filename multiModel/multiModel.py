import streamlit as st
from PIL import Image
import pytesseract
import whisper
import ollama
from pypdf import PdfReader


## Streamlit UI

st.title("Multimodal AI Demo")
st.write("This demo showcases the capabilities of a multimodal AI system that can process and understand various types of data, including text, images, audio, and PDFs.")


# Load whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")
speech_model = load_whisper_model()

## text Input
st.header("Text Input")
text_input = st.text_area("Enter your text here:")
if st.button("Process Text"):
    if text_input:
        st.write("Processing text...")
        response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": text_input}])
        st.success(response.choices[0].message.content)

    else:
        st.warning("Please enter some text to process.")

## Audio Input
st.header("Audio Input")
uploaded_audio = st.file_uploader("Upload an audio file (mp3, wav, etc.)", type=["mp3", "wav", "ogg"])
if uploaded_audio:
    st.write("Transcribing audio...")
    audio_bytes = uploaded_audio.read()
    temp_file_path = f"{uploaded_audio.name}.mp3"
    with open(temp_file_path, "wb") as f:
        f.write(audio_bytes)
    result = speech_model.transcribe(temp_file_path)
    st.write("Transcription:")
    st.write(result["text"])

    response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": result["text"]}])
    st.success(response.choices[0].message.content)

## Image Input
st.header("Image Input")
uploaded_image = st.file_uploader("Upload an image file (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
if uploaded_image:
    st.write("Extracting text from image...")
    image = Image.open(uploaded_image)
    text = pytesseract.image_to_string(image)
    st.write("Extracted Text:")
    st.write(text)

    response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": text}])
    st.success(response.choices[0].message.content)

## PDF Input
st.header("PDF Input")
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_pdf:
    st.write("Processing PDF...")
    pdf_reader = PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    st.write("Extracted Text:")
    st.write(text)

    response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": text}])
    st.success(response.choices[0].message.content)

# To run this Streamlit app, use the following command:
# py -m streamlit run .\multiModel.py