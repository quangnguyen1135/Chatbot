import re
from docx import Document

def extract_google_drive_links(docx_path):
    """Extracts Google Drive sharable file links from a DOCX file.

    Args:
        docx_path (str): The path to the DOCX file.

    Returns:
        list: A list of extracted Google Drive sharable file links.
    """

    links = []
    document = Document(docx_path)
    for paragraph in document.paragraphs:
        # Check if the paragraph contains a Google Drive link pattern
        if re.search(r"https://drive\.google\.com/file/d/([^\/]+)/view", paragraph.text):
            # Extract the link from the paragraph text
            link_match = re.search(r"https://drive\.google\.com/file/d/([^\/]+)/view", paragraph.text)
            if link_match:
                links.append(link_match.group(0))  # Add the link to the list
    return links

# Assuming the DOCX file is located within your project directory
docx_path = "data/Company_Policy.docx"

# Extract Google Drive links from the DOCX file
google_drive_links = extract_google_drive_links(docx_path)

if google_drive_links:
    print("Extracted Google Drive Links:")
    for link in google_drive_links:
        print(link)
else:
    print("No Google Drive sharable file links found in the DOCX file.")
