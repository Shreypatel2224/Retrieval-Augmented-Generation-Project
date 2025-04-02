import os
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama
import time
import psutil
import csv

# Initialize Chroma client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Constants
VECTOR_DIM = 768  
COLLECTION_NAME = "embedding_collection"

# Initialize embedding models
embedding_models = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "paraphrase-MiniLM-L6-v2": SentenceTransformer("paraphrase-MiniLM-L6-v2"),
}

# Function to measure memory usage
def get_memory_usage():
    """
    Get the current memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Function to measure time and memory for a function
def measure_performance(func, *args, **kwargs):
    """
    Measure the time and memory usage of a function.

    Args:
        func: The function to measure.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        dict: A dictionary containing the result, time taken (seconds), and memory usage (MB).
    """
    start_time = time.time()
    start_memory = get_memory_usage()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = get_memory_usage()

    return {
        "result": result,
        "time_seconds": end_time - start_time,
        "memory_mb": end_memory - start_memory,
    }

# Clear Chroma collection
def clear_chroma_collection():
    """
    Clear the Chroma collection if it exists.
    """
    print("Clearing existing Chroma collection...")
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Chroma collection cleared.")
    except chromadb.errors.InvalidArgumentError:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Skipping deletion.")

# Create a collection in Chroma
def create_chroma_collection():
    collection = chroma_client.create_collection(COLLECTION_NAME)
    print("Collection created successfully.")
    return collection

# Load precomputed embeddings from chosen folder
def load_embeddings_from_folder(embedding_folder):
    embeddings = []
    metadata = []  # Store metadata (model, embedding_id) for each embedding

    for file_name in os.listdir(embedding_folder):
        if file_name.endswith(".npy"):
            file_path = os.path.join(embedding_folder, file_name)
            embedding = np.load(file_path)
            embeddings.append(embedding)

            # Extract metadata from the file name
            parts = file_name.split("_")
            model_name = parts[0]  
            embedding_id = parts[2].replace(".npy", "")  

            metadata.append({
                "model": model_name,  # Store the model name
                "embedding_id": embedding_id,  # Store the embedding ID
            })

    return embeddings, metadata

# Store embeddings in Chroma
def store_embeddings(embeddings, metadata):
    collection = chroma_client.get_collection(COLLECTION_NAME)
    ids = [f"{meta['model']}_embedding_{meta['embedding_id']}" for meta in metadata]
    documents = [f"Model: {meta['model']}, Embedding ID: {meta['embedding_id']}" for meta in metadata]
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadata, documents=documents)
    print(f"Stored {len(embeddings)} embeddings in Chroma.")

# Get embedding for a given text using SentenceTransformers
def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> list:
    """
    Generate an embedding for the given text using the specified SentenceTransformer model.

    Args:
        text (str): The input text to embed.
        model_name (str): The name of the embedding model to use.

    Returns:
        list: The embedding vector.
    """
    if model_name not in embedding_models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(embedding_models.keys())}")
    return embedding_models[model_name].encode(text).tolist()

# Search embeddings in Chroma
def search_embeddings(query, model_name="all-MiniLM-L6-v2", top_k=3):
    """
    Search for similar embeddings in Chroma.

    Args:
        query (str): The query text.
        model_name (str): The name of the embedding model to use.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of dictionaries containing the top results.
    """
    query_embedding = get_embedding(query, model_name=model_name)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extract top results
    top_results = []
    for i in range(len(results["ids"][0])):  # Iterate over the number of results
        top_results.append({
            "model": results["metadatas"][0][i]["model"],
            "embedding_id": results["metadatas"][0][i]["embedding_id"],
            "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
        })

    return top_results

# Generate RAG response using Llama 2 7B
def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From model {result.get('model', 'Unknown model')}, embedding ID {result.get('embedding_id', 'Unknown ID')} "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Use Llama 2 7B instead of Mistral
    response = ollama.chat(
        model="llama2:7b", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Log performance metrics to a CSV file
def log_performance_to_csv(embedding_folder, database, llm, model_name, query, search_time, response_time, memory_mb, response):
    """
    Log performance metrics to a CSV file.

    Args:
        embedding_folder (str): The folder containing the embeddings.
        database (str): The database used (e.g., "Chroma").
        llm (str): The LLM used (e.g., "Llama 2 7B").
        model_name (str): The embedding model used.
        query (str): The user's query.
        search_time (float): Time taken for the search query.
        response_time (float): Time taken for the response generation.
        memory_mb (float): Memory used for the query.
        response (str): The response generated by the LLM.
    """
    csv_file = "performance_metrics.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Embedding Folder", "Database", "LLM", "Embedding Model", "Query", 
                "Search Time (seconds)", "Response Time (seconds)", "Memory (MB)", "Response"
            ])
        writer.writerow([
            embedding_folder, database, llm, model_name, query, 
            search_time, response_time, memory_mb, response
        ])

# Interactive search interface
def interactive_search(embedding_folder):
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Prompt user to select an embedding model
        print("Available embedding models:")
        for i, model_name in enumerate(embedding_models.keys()):
            print(f"{i + 1}. {model_name}")
        model_choice = int(input("Select an embedding model (1, 2, or 3): ")) - 1
        model_name = list(embedding_models.keys())[model_choice]

        # Measure performance of search_embeddings
        search_metrics = measure_performance(search_embeddings, query, model_name=model_name)
        context_results = search_metrics["result"]

        print(f"Time taken for search: {search_metrics['time_seconds']:.2f} seconds")
        print(f"Memory used for search: {search_metrics['memory_mb']:.2f} MB")

        # Measure performance of generate_rag_response
        response_metrics = measure_performance(generate_rag_response, query, context_results)
        response = response_metrics["result"]

        print(f"Time taken for response generation: {response_metrics['time_seconds']:.2f} seconds")
        print(f"Memory used for response generation: {response_metrics['memory_mb']:.2f} MB")

        # Log performance metrics to CSV, including the response
        log_performance_to_csv(
            embedding_folder, "Chroma", "Llama 2 7B", model_name, query, 
            search_metrics['time_seconds'], response_metrics['time_seconds'], 
            search_metrics['memory_mb'], response
        )

        # Generate and display response
        print("\n--- Response ---")
        print(response)

# Main function
def main(embedding_folder):
    clear_chroma_collection()
    collection = create_chroma_collection()

    # Load embeddings and metadata
    embeddings, metadata = load_embeddings_from_folder(embedding_folder)

    # Store embeddings in Chroma
    store_embeddings(embeddings, metadata)

    # Start interactive search
    interactive_search(embedding_folder)

if __name__ == "__main__":
    embedding_folder = "Data/Embeddings/no_white_punc_or_stop/1000_tokens/100_overlap/all-mpnet-base-v2"  # Replace with folder path to compare with
    main(embedding_folder)