from sentence_transformers import SentenceTransformer
from data_processing import prepare_rag_data
import pickle

# ------------------------------------------
# Load BGE-M3 Model
# ------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------
# Generate Embeddings
# ------------------------------------------
def generate_embeddings():

    documents = prepare_rag_data()
    
    if not documents:
        print("No documents found. Skipping embedding generation.")
        return []

    # ------------------------------------------
    # Batch Encoding for Better Performance
    # ------------------------------------------
    # Extract all texts for batch processing
    texts = [doc["text"] for doc in documents]
    
    # Encode all texts in batches
    # - batch_size=28: Smaller batch size for memory-constrained environments
    # - normalize_embeddings=True: L2 normalization for cosine similarity
    # - show_progress_bar=True: Visual progress during encoding
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=4,  # Reduced from 32 to 4 for better memory management
        show_progress_bar=True
    )
    
    # Combine documents with their embeddings
    embeddings = []
    for doc, vector in zip(documents, vectors):
        embeddings.append({
            "document": doc,
            "embedding": vector
        })

    # Save embeddings to disk
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\n✅ {len(embeddings)} embeddings saved successfully.")
    print(f"   Document chunks: {len(documents)}")

    return embeddings


# ------------------------------------------
# Testing
# ------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("Embedding Generation")
    print("=" * 60)
    
    embeddings = generate_embeddings()
    
    if embeddings:
        print(f"\n📊 Statistics:")
        print(f"   Total Embeddings: {len(embeddings)}")
        print(f"   Vector Dimension: {len(embeddings[0]['embedding'])}")
        print(f"   Model: BAAI/bge-m3")
        print(f"   Batch Size: 4")
    
    print("\n✅ Embedding generation completed.")