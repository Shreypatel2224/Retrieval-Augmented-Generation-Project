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
qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)

# Initialize multiple embedding models
embedding_models = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("all-mpnet-base-v2"),
    "paraphrase-MiniLM-L6-v2": SentenceTransformer("paraphrase-MiniLM-L6-v2"),
}

# Get model-specific collection name
def get_collection_name(model_name):
    return f"{model_name}_collection"

# Measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Measure time and memory of a function
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
def clear_qdrant_collection(model_name):
    collection_name = get_collection_name(model_name)
    print(f"Clearing Qdrant collection '{collection_name}'...")
    qdrant.delete_collection(collection_name=collection_name)
    print("Qdrant collection cleared.")

# Create a collection in Qdrant
def create_qdrant_collection(model_name):
    vector_dim = len(embedding_models[model_name].encode("test"))
    collection_name = get_collection_name(model_name)
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created with dimension {vector_dim}.")

# Load embeddings from folder
def load_embeddings_from_folder(embedding_folder):
    embeddings = []
    metadata = []
    for file_name in os.listdir(embedding_folder):
        if file_name.endswith(".npy"):
            file_path = os.path.join(embedding_folder, file_name)
            embedding = np.load(file_path).tolist()
            embeddings.append(embedding)
            parts = file_name.split("_")
            model_name = parts[0]
            embedding_id = parts[2].replace(".npy", "")
            metadata.append({"model": model_name, "embedding_id": embedding_id})
    return embeddings, metadata

# Store embeddings in Qdrant
def store_embeddings(embeddings, metadata, model_name):
    collection_name = get_collection_name(model_name)
    points = [
        {"id": i, "vector": embeddings[i], "payload": metadata[i]}
        for i in range(len(embeddings))
    ]
    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(embeddings)} embeddings in Qdrant collection '{collection_name}'.")

# Get embedding for a given text
def get_embedding(text, model_name):
    return embedding_models[model_name].encode(text).tolist()

# Search embeddings
def search_embeddings(query, model_name, top_k=3):
    query_embedding = get_embedding(query, model_name)
    collection_name = get_collection_name(model_name)
    results = qdrant.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)
    return [
        {
            "model": r.payload["model"],
            "embedding_id": r.payload["embedding_id"],
            "similarity": r.score,
        }
        for r in results
    ]

# Generate RAG response
def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From model {r['model']}, embedding ID {r['embedding_id']} with similarity {r['similarity']:.2f}"
            for r in context_results
        ]
    )
    prompt = f"""You are a helpful AI assistant.\nUse the following context to answer the query as accurately as possible. If the context is not relevant to the query, say 'I don't know'.\n\nContext:\n{context_str}\n\nQuery: {query}\n\nAnswer:"""
    response = ollama.chat(
        model="llama2:7b", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# Log performance
def log_performance_to_csv(embedding_folder, database, llm, model_name, query, search_time, response_time, memory_mb, response):
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

# Interactive search
def interactive_search(embedding_base_folder):
    print("\U0001F50D RAG Search Interface\nType 'exit' to quit")
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
        log_performance_to_csv(
            embedding_base_folder, "Qdrant", "Llama 2 7B", model_name, query,
            search_metrics['time_seconds'], response_metrics['time_seconds'],
            search_metrics['memory_mb'], response
        )

# Main function
def main(embedding_base_folder):
    for model_name in embedding_models.keys():
        embedding_folder = os.path.join(embedding_base_folder, model_name)
        clear_qdrant_collection(model_name)
        create_qdrant_collection(model_name)
        embeddings, metadata = load_embeddings_from_folder(embedding_folder)
        store_embeddings(embeddings, metadata, model_name)

if __name__ == "__main__":
    embedding_base_folder = "Data/Embeddings/no_white_or_punc/1000_tokens/100_overlap"
    main(embedding_base_folder)
    interactive_search(embedding_base_folder)