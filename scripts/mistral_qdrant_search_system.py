import os
import numpy as np
import qdrant_client
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import ollama
import time
import psutil
import csv

# Initialize Qdrant client
qdrant_client = qdrant_client.QdrantClient(host="localhost", port=6333)

# Constants
VECTOR_DIM = 768  # Dimension of embeddings
COLLECTION_NAME = "embedding_collection"

# Initialize multiple embedding models
embedding_models = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "paraphrase-MiniLM-L6-v2": SentenceTransformer("paraphrase-MiniLM-L6-v2"),
}

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Function to measure time and memory for a function
def measure_performance(func, *args, **kwargs):
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

# Clear Qdrant collection
def clear_qdrant_collection():
    print("Clearing existing Qdrant collection...")
    qdrant_client.delete_collection(COLLECTION_NAME)
    print("Qdrant collection cleared.")

# Create a collection in Qdrant
def create_qdrant_collection():
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print("Collection created successfully.")

# Load precomputed embeddings from a folder
def load_embeddings_from_folder(embedding_folder):
    embeddings = []
    metadata = []  # Store metadata (model, embedding_id) for each embedding

    for file_name in os.listdir(embedding_folder):
        if file_name.endswith(".npy"):
            file_path = os.path.join(embedding_folder, file_name)
            embedding = np.load(file_path).tolist()
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

# Store embeddings in Qdrant
def store_embeddings(embeddings, metadata):
    points = [
        {
            "id": i,
            "vector": embeddings[i],
            "payload": metadata[i],
        }
        for i in range(len(embeddings))
    ]
    qdrant_client.upsert(COLLECTION_NAME, points=points)
    print(f"Stored {len(embeddings)} embeddings in Qdrant.")

# Get embedding for a given text
def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> list:
    if model_name not in embedding_models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(embedding_models.keys())}")
    return embedding_models[model_name].encode(text).tolist()

# Search embeddings in Qdrant
def search_embeddings(query, model_name="all-MiniLM-L6-v2", top_k=3):
    query_embedding = get_embedding(query, model_name=model_name)
    results = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=top_k)
    return [
        {
            "model": r.payload["model"],
            "embedding_id": r.payload["embedding_id"],
            "similarity": r.score,
        }
        for r in results
    ]

# Generate RAG response using Mistral
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

    # Use Mistral
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Log performance metrics to a CSV file
def log_performance_to_csv(embedding_folder, database, llm, model_name, query, search_time, response_time, memory_mb, response):
    """
    Log performance metrics to a CSV file.

    Args:
        embedding_folder (str): The folder containing the embeddings.
        database (str): The database used (e.g., "Qdrant").
        llm (str): The LLM used (e.g., "Mistral").
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
    print("🔍 RAG Search Interface\nType 'exit' to quit")
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break
        print("Available embedding models:")
        for i, model_name in enumerate(embedding_models.keys()):
            print(f"{i + 1}. {model_name}")
        model_choice = int(input("Select an embedding model (1, 2, or 3): ")) - 1
        model_name = list(embedding_models.keys())[model_choice]
        search_metrics = measure_performance(search_embeddings, query, model_name=model_name)
        context_results = search_metrics["result"]
        response_metrics = measure_performance(generate_rag_response, query, context_results)
        response = response_metrics["result"]
        print("\n--- Response ---")
        print(response)

        # Log performance metrics to CSV, including the response
        log_performance_to_csv(
            embedding_folder, "Qdrant", "Mistral", model_name, query, 
            search_metrics['time_seconds'], response_metrics['time_seconds'], 
            search_metrics['memory_mb'], response
        )

# Main function
def main(embedding_folder):
    clear_qdrant_collection()
    create_qdrant_collection()
    embeddings, metadata = load_embeddings_from_folder(embedding_folder)
    store_embeddings(embeddings, metadata)
    interactive_search(embedding_folder)

if __name__ == "__main__":
    embedding_folder = "Data/Embeddings/no_white_or_punc/1000_tokens/100_overlap/all-MiniLM-L6-v2"
    main(embedding_folder)