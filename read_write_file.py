def process_text(txt):
    if len(txt) > 5:
        return txt.upper()
    else :
        return txt.lower()

"""
file  = open("input.txt","w")
file.write("Hello, World!")
file.close()
"""
# file_name = input("Enter the file name: ") # reading input from console window
# file = open(file_name,"r")

file = open("input.txt","r")
content = file.read()
file.close()

result = process_text(content)
print(result)