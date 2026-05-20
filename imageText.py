import pytesseract
from PIL import Image
def extract_text_from_image(image_path):
    try:
        #open image 
        image = Image.open(image_path)
        # use pytesseract to extract text from image 
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Exmaple usage 
if __name__ == "__main__":
    image_path = r"data/healthcare.jpeg"
    extracted_text = extract_text_from_image(image_path)
    if extracted_text:
        print("extracted text")
        print(extracted_text)