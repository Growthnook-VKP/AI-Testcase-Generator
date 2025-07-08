import re  # Regular expressions library for text processing

def clean_text(raw_text):
    """
    Cleans extracted text by removing unwanted characters, excessive spaces, and page markers.

    Args:
        raw_text (str): Raw text from the PDF.

    Returns:
        str: Cleaned and normalized text.
    """
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', raw_text)
    # Remove page numbers like 'Page 1', 'page 2', etc.
    text = re.sub(r'Page \\d+|page \\d+', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()  # Remove leading/trailing whitespace


def split_into_chunks(cleaned_text, chunk_size=500):
    """
    Splits the cleaned text into chunks of specified word count.

    Args:
        cleaned_text (str): The cleaned document text.
        chunk_size (int): Number of words per chunk (default: 500).

    Returns:
        List[str]: List of text chunks.
    """
    words = cleaned_text.split()  # Split text into a list of words
    # Create chunks by joining 'chunk_size' words together
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks  # Return the list of text chunks