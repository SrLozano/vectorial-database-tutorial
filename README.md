# Milvus RAG Tutorial 🤖

A hands-on tutorial for learning Retrieval-Augmented Generation (RAG) using vector databases. This tutorial uses completely local components—Milvus Lite for vector storage and sentence-transformers for embeddings—so you can learn RAG concepts without cloud services or API keys. Work through the interactive notebook to understand vector embeddings, semantic search, and hybrid retrieval strategies.

## 🚀 Installation

Follow these steps to set up the tutorial environment:

1. **Clone or download this repository**

2. **Create a virtual environment**

   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** On first run, sentence-transformers will download the embedding model (~500MB). This happens once and is cached locally.

5. **Configure the environment**

   Create a `.env` file in the project root with the following content:
   ```bash
   # Local Milvus database file path
   MILVUS_DB_PATH=./milvus_demo.db

   # Local embedding model (cached after first download)
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

6. **Run the notebook**
   ```bash
   jupyter notebook milvus_tutorial.ipynb
   ```

**Prerequisites:** Python 3.8+ and ~500MB disk space for the embedding model.

## 📖 What This Tutorial Covers

- **Vector Embeddings** - Dense embeddings using sentence-transformers and sparse embeddings with BM25
- **Milvus Lite** - Local file-based vector database for storing and searching embeddings
- **Semantic Search** - Finding similar content using dense vector similarity
- **Lexical Search** - Keyword-based search using BM25 sparse vectors
- **Hybrid Search** - Combining semantic and lexical search for optimal results
- **Metadata Filtering** - Using structured filters to narrow search results
- **Collection Management** - Creating schemas, building indexes, and inserting data
- **Best Practices** - Performance optimization and production deployment patterns
