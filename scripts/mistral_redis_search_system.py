import os
import numpy as np
import redis
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
from sentence_transformers import SentenceTransformer
import ollama
import time
import psutil
import csv

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Constants
VECTOR_DIM = 768  # Dimension of embeddings
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Initialize multiple embedding models
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

# Clear Redis store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA model TEXT
        embedding_id TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

# Load precomputed embeddings from a folder
def load_embeddings_from_folder(embedding_folder):
    embeddings = []
    metadata = []  # Store metadata (model, embedding_id) for each embedding

    for file_name in os.listdir(embedding_folder):
        if file_name.endswith(".npy"):
            file_path = os.path.join(embedding_folder, file_name)
            embedding = np.load(file_path)
            embeddings.append(embedding)

            # Extract metadata from file name
            parts = file_name.split("_")
            model_name = parts[0]  
            embedding_id = parts[2].replace(".npy", "")  

            metadata.append({
                "model": model_name,  # Store the model name
                "embedding_id": embedding_id,  # Store the embedding ID
            })

    return embeddings, metadata

# Store embeddings in Redis
def store_embeddings(embeddings, metadata):
    for embedding, meta in zip(embeddings, metadata):
        key = f"{DOC_PREFIX}:{meta['model']}_embedding_{meta['embedding_id']}"
        redis_client.hset(
            key,
            mapping={
                "model": meta["model"],
                "embedding_id": meta["embedding_id"],
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as byte array
            },
        )
    print(f"Stored {len(embeddings)} embeddings in Redis.")

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

# Search embeddings in Redis
def search_embeddings(query, model_name="all-MiniLM-L6-v2", top_k=3):
    query_embedding = get_embedding(query, model_name=model_name)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("model", "embedding_id", "vector_distance")
        .dialect(2)
    )

    results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})
    top_results = [
        {
            "model": result.model,
            "embedding_id": result.embedding_id,
            "similarity": result.vector_distance,
        }
        for result in results.docs
    ][:top_k]

    return top_results

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
        database (str): The database used (e.g., "Redis").
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
            embedding_folder, "Redis", "Mistral", model_name, query, 
            search_metrics['time_seconds'], response_metrics['time_seconds'], 
            search_metrics['memory_mb'], response
        )

        # Generate and display response
        print("\n--- Response ---")
        print(response)

# Main function
def main(embedding_folder):
    clear_redis_store()
    create_hnsw_index()

    # Load embeddings and metadata
    embeddings, metadata = load_embeddings_from_folder(embedding_folder)

    # Store embeddings in Redis
    store_embeddings(embeddings, metadata)

    # Start interactive search
    interactive_search(embedding_folder)

if __name__ == "__main__":
    embedding_folder = "Data/Embeddings/no_white_or_punc/1000_tokens/100_overlap/all-mpnet-base-v2"  # Replace with folder path to compare with
    main(embedding_folder)