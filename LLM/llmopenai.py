import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def process_text(text):
    prompt = f"""give me the output summary in 1 sentence: {text}"""
    response = call_llm(prompt)
    return response

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


#Extracted text from the given file and processing it using the LLM
def main():
    file_name = input("Enter the file name: ")
    file = open(file_name, "r")
    content = file.read()   
    file.close()

    result = process_text(content)
    print("\n Processed Output :\n")
    print(result)

if __name__ == "__main__":    main()