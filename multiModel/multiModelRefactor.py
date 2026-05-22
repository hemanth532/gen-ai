import os
import tempfile

import ollama
import pytesseract
import streamlit as st
import whisper
from PIL import Image
from pypdf import PdfReader


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multimodal AI Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multimodal AI Demo")
st.write(
    """
    Upload **text, audio, images, or PDFs**
    and let AI extract + summarize the content.
    """
)


# =========================
# CONFIG
# =========================
OLLAMA_MODEL = "llama3.2:1b"

# Uncomment if tesseract PATH issue exists
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_whisper_model():
    """Load Whisper model once."""
    return whisper.load_model("base")


speech_model = load_whisper_model()


# =========================
# HELPER FUNCTIONS
# =========================
def get_llm_response(prompt: str) -> str:
    """Send prompt to Ollama."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Error calling Ollama: {str(e)}"


def extract_text_from_pdf(uploaded_pdf) -> str:
    """Extract text from PDF."""
    text = ""

    pdf_reader = PdfReader(uploaded_pdf)

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text.strip()


def transcribe_audio(uploaded_audio) -> str:
    """Transcribe uploaded audio using Whisper."""

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    ) as temp_audio:

        temp_audio.write(uploaded_audio.read())
        temp_path = temp_audio.name

    try:
        result = speech_model.transcribe(temp_path)
        return result["text"]

    finally:
        os.remove(temp_path)


def extract_text_from_image(uploaded_image) -> str:
    """Extract text from image using OCR."""

    image = Image.open(uploaded_image)

    return pytesseract.image_to_string(image)


def process_content(text: str):
    """Display extracted text + AI response."""

    if not text.strip():
        st.warning("No text extracted.")
        return

    st.subheader("📄 Extracted Text")
    st.text_area(
        "Content",
        text,
        height=200
    )

    with st.spinner("Generating AI response..."):
        response = get_llm_response(text)

    st.subheader("🤖 AI Response")
    st.success(response)


# =========================
# TEXT INPUT
# =========================
st.divider()
st.header("📝 Text Input")

text_input = st.text_area(
    "Enter text",
    placeholder="Type something..."
)

if st.button("Process Text"):

    if text_input.strip():
        process_content(text_input)

    else:
        st.warning("Please enter text.")


# =========================
# AUDIO INPUT
# =========================
st.divider()
st.header("🎵 Audio Input")

uploaded_audio = st.file_uploader(
    "Upload audio",
    type=["mp3", "wav", "ogg"],
    key="audio_uploader"
)

if uploaded_audio:

    with st.spinner("Transcribing audio..."):
        text = transcribe_audio(uploaded_audio)

    process_content(text)


# =========================
# IMAGE INPUT
# =========================
st.divider()
st.header("🖼️ Image Input")

uploaded_image = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"],
    key="image_uploader"
)

if uploaded_image:

    with st.spinner("Extracting text from image..."):
        text = extract_text_from_image(uploaded_image)

    process_content(text)


# =========================
# PDF INPUT
# =========================
st.divider()
st.header("📄 PDF Input")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_pdf:

    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_pdf)

    process_content(text)


# To run this Streamlit app, use the following command:
# py -m streamlit run .\multiModelRefactor.py