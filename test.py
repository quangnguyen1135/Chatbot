from docx2python import docx2python

def extract_text_from_docx(docx_file_path):
    # Load the DOCX file
    doc = docx2python(docx_file_path)

    # Extract text from paragraphs
    text = ""
    for paragraph in doc.body:
        print(paragraph)  # Add this line to see the content of each paragraph
        if isinstance(paragraph, list):
            # If paragraph is a list, check the type of each element
            paragraph_text = ""
            for element in paragraph:
                if hasattr(element, 'text'):
                    # If element has a 'text' attribute, it's likely a sentence object
                    paragraph_text += element.text.strip() + " "
                elif isinstance(element, str):
                    # If element is a string, it's a standalone sentence
                    paragraph_text += element.strip() + " "
        else:
            # If paragraph is not a list, it's likely a single sentence object
            paragraph_text = paragraph.text.strip()

        text += paragraph_text

    return text

# Đường dẫn tới tệp DOCX
docx_file_path = "./data/Company_Policy.docx"

# Trích xuất văn bản từ tài liệu DOCX
text = extract_text_from_docx(docx_file_path)
print(text)
