from ollama import chat

def process_text(text):
    prompt = f"""Return only valid JSON with the following format
{{
    "summary": "",
    "Risk": "",
    "Recommendations": ""
}}
: {text}"""
    response = call_llm(prompt)
    return response

def call_llm(prompt):
    response = chat(
        model='phi4-mini', #gemma4, phi4-mini
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.message.content

def main():
    file_name = input("Enter the file name: ")
    file = open(file_name, "r")
    content = file.read()   
    file.close()

    result = process_text(content)
    print("\n Processed Output :\n")
    print(result)

if __name__ == "__main__":
    main() 