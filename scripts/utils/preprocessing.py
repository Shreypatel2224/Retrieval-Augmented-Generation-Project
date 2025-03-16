import os
from text_cleaning import (
    extract_text_from_pdf,
    extract_text_from_txt,
    no_white_punc,
    no_white_punc_stop,
    save_processed_text
)

def preprocess_files(input_folder, output_folder, strategy):
    """
    Preprocess all files in the input folder and save the results in the output folder.
    :param input_folder: Path to the raw files folder (e.g., 'data/raw/txt/' or 'data/raw/pdf/').
    :param output_folder: Path to the processed files folder (e.g., 'data/processed/txt/basic_cleaning/').
    :param strategy: The preprocessing strategy to apply (e.g., 'basic_cleaning').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            continue  # Skip non-PDF/TXT files

        # Apply the preprocessing strategy
        if strategy == 'no_white_punc':
            processed_text = no_white_punc(text)
        elif strategy == 'no_white_punc_stop':
            processed_text = no_white_punc_stop(text)

        # Save the processed text
        if (filename.endswith('.pdf')):
            filename = filename.replace('.pdf', '.txt')    
        output_path = os.path.join(output_folder, filename)
        save_processed_text(processed_text, output_path)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # Define paths and strategy
    # raw_txt_folder = "Data/Raw/Txt"
    # raw_pdf_folder = "Data/Raw/Pdf"
    # processed_txt_folder = "Data/Processed/no_white_or_punc"
    # processed_pdf_folder = "Data/Processed/no_white_or_punc"
    # strategy = 'no_white_punc'

    raw_txt_folder = "Data/Raw/Txt"
    raw_pdf_folder = "Data/Raw/Pdf"
    processed_txt_folder = "Data/Processed/no_white_punc_or_stop"
    processed_pdf_folder = "Data/Processed/no_white_punc_or_stop"
    strategy = 'no_white_punc_stop'

    # Preprocess TXT files
    preprocess_files(raw_txt_folder, processed_txt_folder, strategy)

    # Preprocess PDF files
    preprocess_files(raw_pdf_folder, processed_pdf_folder, strategy)