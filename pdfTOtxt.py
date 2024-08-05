import fitz

# Open the PDF file
pdf_file = "C:/Users/Dell/Desktop/MADDE5.pdf"
pdf_document = fitz.open(pdf_file)

# Iterate through each page and extract text
full_text = ""
for page_num in range(pdf_document.page_count):
    page = pdf_document[page_num]
    full_text += page.get_text()

# Remove line breaks and extra whitespaces
full_text = " ".join(full_text.split())

# Save the content in a text file
with open("C:/Users/Dell/Desktop/MADDE5.txt", "w") as file:
    file.write(full_text)

# Close the PDF document
pdf_document.close()