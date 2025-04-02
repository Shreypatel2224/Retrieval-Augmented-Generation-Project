# DS4300 Practical 02 - Retrieval-Augmented Generation System

## Project Overview
A local RAG system that allows querying collective DS4300 course notes using different combinations of embedding models, vector databases, and LLM backends.

## Prerequisites
- Docker Desktop
- Python 3.8+
- Ollama

## Setup Instructions

### 1. Start Database Services
```bash
# Open Docker Desktop first, then run:
docker-compose up -d
```
This starts:

- Redis (port 6379)
- Qdrant (port 6333)
- Chroma (port 8000)

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download LLM Models
```bash
ollama pull mistral:latest
ollama pull llama2:7b
```

## Configuration

### Generating Embeddings
Since `.npy` embedding files are included in `.gitignore`, you must generate embeddings locally before running the system. Use the provided script:

```bash
python Retrieval-Augmented-Generation-Project/scripts/utils/generate_embeddings.py
```

This script will:
- Read chunked text files from `Data/Chunked`
- Generate embeddings using multiple models
- Save them in `Data/Embeddings`

Ensure your `Data/Chunked` directory contains the necessary text files before running the script.

### Query Scripts: Usage and Embedding Folder Setup 
To change the embedding data source:

1. Open the desired query script in a text editor.
2. Locate the `embedding_folder` variable at the bottom of the file.
3. Modify the path to point to your desired embeddings folder:

```python
if __name__ == "__main__":
    # Update this path to match your embeddings folder
    embedding_folder = "Data/Embeddings/no_white_or_punc/500_tokens/50_overlap/all-mpnet-base-v2"
    main(embedding_folder)
```
*Note for Qdrant files the embedding foler does not contain the embedding model it allows you to select a model after running the script by prompting the user in the command line, changing tokens or overlap will still need to be changed here.

How the file path appears: embedding_folder = "Data/Embeddings/no_white_or_punc/500_tokens/50_overlap

### Available Embedding Folder Structures:
```plaintext
Data/Embeddings/
├── no_white_or_punc/
│   ├── 200_tokens/
│   │   ├── 0_overlap/
│   │   ├── 50_overlap/
│   │   └── 100_overlap/
│   ├── 500_tokens/...    
│   └── 1000_tokens/...
└── no_white_punc_or_stop/...
```

## Running the System

### Available Scripts

| Vector DB | LLM         | Script Path                                  |
|-----------|------------|---------------------------------------------|
| Qdrant    | Llama2 7B  | `scripts/llama2_qdrant_search_system.py`    |
| Qdrant    | Mistral    | `scripts/mistral_qdrant_search_system.py`   |
| Redis     | Llama2 7B  | `scripts/llama2_redis_search_system.py`     |
| Redis     | Mistral    | `scripts/mistral_redis_search_system.py`    |
| Chroma    | Llama2 7B  | `scripts/llama2_chroma_search_system.py`    |
| Chroma    | Mistral    | `scripts/mistral_chroma_search_system.py`   |

### Execution
```bash
python scripts/[script_name].py
```
Example:
```bash
python scripts/mistral_chroma_search_system.py
```

## Interactive Interface Flow

### Model Selection:
```plaintext
Available embedding models:
1. all-MiniLM-L6-v2
2. all-mpnet-base-v2
3. paraphrase-MiniLM-L6-v2
Select an embedding model (1, 2, or 3):
```
Must match the embeddings in your configured folder.

Example: Choose `2` for `all-mpnet-base-v2` folders.

*Note: for Qdrant DB, there is no embedding model in the configured folder path, you may just select the desired model when prompted in the terminal and it will run.

### Query Input:
```plaintext
Enter your search query: [your question]
```

### View Response:
```plaintext
--- Response ---
[Generated answer with sources]
```

## Key Requirements

### Model-Folder Matching:
- Your selected model number must correspond to the embedding model used in your configured folder.
- Mismatches will result in poor performance.

### Performance Tracking:
- All queries are automatically logged to `performance_metrics.csv`.
- Metrics include:
  - Query processing time
  - Response generation time
  - Memory usage
  - Full responses

## Shutdown Procedures
```bash
# Stop database containers
docker-compose down

# Check running LLM instances
ollama ps
