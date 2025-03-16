import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_text_from_folder(folder_path):
    """Load all text files from a folder and return a list of texts."""
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

def chunk_text(texts, chunk_size, chunk_overlap):
    """Chunk a list of texts into smaller segments."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

def save_chunks(chunks, output_folder):
    """Save chunks to a folder as individual text files."""
    os.makedirs(output_folder, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_folder, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as file:
            file.write(chunk)

def main():
    # Define input and output paths
    input_folders = [
        "Data/Processed/no_white_punc_or_stop",
        "Data/Processed/no_white_or_punc",
    ]
    output_base = "Data/Chunked"

    # Define chunking strategies
    chunk_sizes = [200, 500, 1000]
    chunk_overlaps = [0, 50, 100]

    # Process each input folder
    for input_folder in input_folders:
        folder_name = os.path.basename(input_folder)
        texts = load_text_from_folder(input_folder)

        # Apply chunking strategies
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                chunks = chunk_text(texts, chunk_size, chunk_overlap)
                output_folder = os.path.join(
                    output_base,
                    folder_name,
                    f"{chunk_size}_tokens",
                    f"{chunk_overlap}_overlap",
                )
                save_chunks(chunks, output_folder)
                print(f"Saved {len(chunks)} chunks to {output_folder}")

if __name__ == "__main__":
    main()