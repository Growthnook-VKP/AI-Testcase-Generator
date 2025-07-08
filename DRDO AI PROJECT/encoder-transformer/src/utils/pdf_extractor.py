import fitz  # PyMuPDF: Library for reading and manipulating PDF files

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyMuPDF.

    Args:
        file_path (str): Path to the PDF file to be read.

    Returns:
        str: Combined text extracted from all pages of the PDF.
    """
    text = ""  # Initialize an empty string to store the extracted text
    try:
        pdf_document = fitz.open(file_path)  # Open the PDF file
        # Iterate through each page in the PDF
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)  # Load the current page
            text += page.get_text()  # Extract text from the page and append to the result
        pdf_document.close()  # Close the PDF file
    except Exception as e:
        # Print an error message if something goes wrong
        print(f"Error reading PDF: {e}")
    return text  # Return the combined text from all pages