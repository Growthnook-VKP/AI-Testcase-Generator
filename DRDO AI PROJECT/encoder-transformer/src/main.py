import sys  # Provides access to system-specific parameters and functions
import os   # Provides functions for interacting with the operating system

# Add the project root directory to sys.path so that 'src' can be imported as a package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # PyTorch library for tensor operations and neural networks
from src.encoder.encoder import Encoder  # Import the custom Transformer Encoder implementation
from src.utils.pdf_extractor import extract_text_from_pdf  # Utility to extract text from PDF files
from src.utils.text_preprocessor import clean_text, split_into_chunks  # Utilities for text cleaning and chunking

def main():
    # ---------------- Model Hyperparameters ----------------
    d_model = 512        # Dimensionality of input/output features
    num_layers = 6       # Number of encoder layers (blocks)
    num_heads = 8        # Number of attention heads in each multi-head attention block
    d_ff = 2048          # Dimensionality of the feed-forward network inside each encoder block
    dropout = 0.1        # Dropout rate for regularization
    seq_len = 10         # Length of the input sequence
    batch_size = 2       # Number of samples in a batch

    # ---------------- PDF Extraction and Preprocessing ----------------
    pdf_path = r'C:\Users\ADMIN\Desktop\DRDO AI PROJECT\TKPV DRDO.pdf'  # Path to the PDF file to process

    # Extract raw text from the PDF file
    raw_text = extract_text_from_pdf(pdf_path)
    # Clean the extracted text (remove unwanted characters, normalize, etc.)
    cleaned_text = clean_text(raw_text)
    # Split the cleaned text into manageable chunks (e.g., for further processing or model input)
    chunks = split_into_chunks(cleaned_text, chunk_size=500)  # Each chunk has up to 500 units (words/characters)

    # Print the number of chunks obtained from the PDF
    print(f"Number of chunks: {len(chunks)}")

    # ---------------- Encoder Example (Random Data) ----------------
    # Create a random input tensor simulating a batch of sequences
    # Shape: (batch_size, seq_len, d_model)
    x = torch.rand(batch_size, seq_len, d_model)
    # Create a mask tensor (all ones means no masking is applied)
    # Shape: (batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    # Instantiate the Encoder model with the specified hyperparameters
    encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
    # Pass the input and mask through the encoder to get the output
    output = encoder(x, mask)
    # Print the shape of the encoder's output tensor
    print("Encoder output shape:", output.shape)

# Entry point: run main() if this script is executed directly
if __name__ == "__main__":
    main()
