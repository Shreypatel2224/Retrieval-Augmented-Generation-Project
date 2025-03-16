import re
import string
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import os

# Text cleaning functions
def remove_whitespace(text):
    """Remove extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_punctuation(text):
    """Remove punctuation."""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    """Remove stopwords."""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def no_white_punc(text):
    """Remove whitespace and punctuation."""
    text = remove_whitespace(text)
    text = remove_punctuation(text)
    return text

def no_white_punc_stop(text):
    """Remove whitespace, punctuation, and stopwords."""
    text = remove_whitespace(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text

# File handling functions
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file."""
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def save_processed_text(text, output_path):
    """Save processed text to a file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)