import easyocr

reader = easyocr.Reader(['en'])

result = reader.readtext("data/healthcare.jpeg")

for item in result:
    print(item[1])