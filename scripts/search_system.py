import os
import numpy as np
import redis
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
from sentence_transformers import SentenceTransformer  # For embedding queries
import ollama  # Import ollama module

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Constants
VECTOR_DIM = 768  # Dimension of your embeddings
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Initialize the SentenceTransformer model for query embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
        SCHEMA file TEXT
        page TEXT
        chunk TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

# Load precomputed embeddings from a folder
def load_embeddings_from_folder(embedding_folder):
    embeddings = []
    metadata = []  # Store metadata (file, page, chunk) for each embedding
    for file_name in os.listdir(embedding_folder):
        if file_name.endswith(".npy"):  # Assuming embeddings are stored as .npy files
            file_path = os.path.join(embedding_folder, file_name)
            embedding = np.load(file_path)
            embeddings.append(embedding)

            # Extract metadata from the file name (customize this based on your file naming convention)
            # Example: "file1_page2_chunk3.npy"
            parts = file_name.split("_")
            file = parts[0]
            page = parts[1].replace("page", "")
            chunk = parts[2].replace("chunk", "").replace(".npy", "")
            metadata.append({"file": file, "page": page, "chunk": chunk})

    return embeddings, metadata

# Store embeddings in Redis
def store_embeddings(embeddings, metadata):
    for embedding, meta in zip(embeddings, metadata):
        key = f"{DOC_PREFIX}:{meta['file']}_page_{meta['page']}_chunk_{meta['chunk']}"
        redis_client.hset(
            key,
            mapping={
                "file": meta["file"],
                "page": meta["page"],
                "chunk": meta["chunk"],
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as byte array
            },
        )
    print(f"Stored {len(embeddings)} embeddings in Redis.")

# Get embedding for a given text using SentenceTransformers
def get_embedding(text: str) -> list:
    """
    Generate an embedding for the given text using SentenceTransformers.

    Args:
        text (str): The input text to embed.

    Returns:
        list: The embedding vector.
    """
    return embedding_model.encode(text).tolist()

# Search embeddings in Redis
def search_embeddings(query, top_k=3):
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )

    results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})
    top_results = [
        {
            "file": result.file,
            "page": result.page,
            "chunk": result.chunk,
            "similarity": result.vector_distance,
        }
        for result in results.docs
    ][:top_k]

    return top_results

# Generate RAG response
def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
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

    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Interactive search interface
def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)

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
    interactive_search()

if __name__ == "__main__":
    embedding_folder = "Data/Embeddings/no_white_or_punc/200_tokens/0_overlap/all-MiniLM-L6-v2"  # Replace with your folder path
    main(embedding_folder)