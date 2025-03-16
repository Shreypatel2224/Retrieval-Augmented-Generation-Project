import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_chunks_from_folder(folder_path):
    """Load all chunked text files from a folder and return a list of chunks."""
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                chunks.append(file.read())
    return chunks

def generate_embeddings(chunks, model_name):
    """Generate embeddings for a list of text chunks using a specified model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings, output_folder, prefix):
    """Save embeddings as .npy files."""
    os.makedirs(output_folder, exist_ok=True)
    for i, embedding in enumerate(embeddings):
        np.save(os.path.join(output_folder, f"{prefix}_embedding_{i+1}.npy"), embedding)

def main():
    # Define embedding models to compare
    models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    }

    # Define input and output paths
    input_base = "Data/Chunked"
    output_base = "Data/Embeddings"

    # Process each chunked folder
    for root, dirs, files in os.walk(input_base):
        if any(file.endswith(".txt") for file in files):
            chunks = load_chunks_from_folder(root)
            folder_name = os.path.relpath(root, input_base)

            # Generate embeddings for each model
            for model_name, model_path in models.items():
                embeddings = generate_embeddings(chunks, model_path)
                output_folder = os.path.join(output_base, folder_name, model_name)
                save_embeddings(embeddings, output_folder, prefix=model_name)
                print(f"Saved embeddings for {model_name} in {output_folder}")

if __name__ == "__main__":
    main()