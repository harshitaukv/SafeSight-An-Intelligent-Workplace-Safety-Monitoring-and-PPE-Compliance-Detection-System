import faiss
import pickle
import numpy as np
import os


# ------------------------------------------
# Build FAISS Index
# ------------------------------------------
def build_faiss_index():

    print("=" * 60)
    print("🔨 Building FAISS Index")
    print("=" * 60)

    # Check if embeddings.pkl exists
    if not os.path.exists("embeddings.pkl"):
        print("❌ embeddings.pkl not found!")
        print("   Please run 'python embedding.py' first.")
        return

    # Load embeddings
    try:
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        print(f"✅ Loaded {len(embeddings)} embeddings from embeddings.pkl")
    except Exception as e:
        print(f"❌ Error loading embeddings.pkl: {e}")
        return

    if len(embeddings) == 0:
        print("❌ No embeddings found in embeddings.pkl")
        return

    # Get dimension from first embedding
    dimension = len(embeddings[0]["embedding"])
    print(f"📐 Vector dimension: {dimension}")

    # ------------------------------------------
    # Use Inner Product (IP) instead of L2
    # ------------------------------------------
    # Since embeddings are normalized (normalize_embeddings=True),
    # Inner Product is equivalent to Cosine Similarity.
    # This is the recommended approach for BGE-M3 and other
    # SentenceTransformer models.
    # ------------------------------------------
    print("🏗️  Creating FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(dimension)

    # Convert embeddings to numpy array
    vectors = np.array(
        [e["embedding"] for e in embeddings],
        dtype="float32"
    )
    print(f"📊 Vectors shape: {vectors.shape}")

    # Add vectors to index
    print("➕ Adding vectors to index...")
    index.add(vectors)
    print(f"✅ Added {index.ntotal} vectors to index")

    # Save FAISS index
    print("💾 Saving FAISS index to faiss_index.index...")
    faiss.write_index(index, "faiss_index.index")
    print("✅ FAISS index saved")

    # Save documents metadata
    print("💾 Saving documents metadata to documents.pkl...")
    documents = [e["document"] for e in embeddings]
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    print(f"✅ Saved {len(documents)} documents to documents.pkl")

    # Print summary
    print("=" * 60)
    print("✅ FAISS INDEX CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"📄 Indexed Documents : {len(embeddings)}")
    print(f"📐 Vector Dimension  : {dimension}")
    print(f"🔍 Index Type        : IndexFlatIP (Inner Product / Cosine Similarity)")
    print(f"📁 Index File        : faiss_index.index")
    print(f"📁 Metadata File     : documents.pkl")
    print("=" * 60)

    # Print sample documents
    print("\n📚 Sample documents in index:")
    for i, doc in enumerate(documents[:5]):
        filename = doc.get('filename') or doc.get('image') or 'Unknown'
        doc_type = doc.get('type', 'unknown')
        print(f"  {i+1}. {doc_type:12} | {filename}")
    if len(documents) > 5:
        print(f"  ... and {len(documents) - 5} more documents")

    return index


# ------------------------------------------
# Verify FAISS Index
# ------------------------------------------
def verify_faiss_index():
    """Verify that the FAISS index and documents are properly saved."""
    
    print("\n" + "=" * 60)
    print("🔍 Verifying FAISS Index")
    print("=" * 60)
    
    # Check if files exist
    files_to_check = ["faiss_index.index", "documents.pkl", "embeddings.pkl"]
    all_exist = True
    
    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename} exists ({size:,} bytes)")
        else:
            print(f"❌ {filename} does not exist!")
            all_exist = False
    
    if not all_exist:
        print("⚠️ Some files are missing. Please rebuild the knowledge base.")
        return False
    
    # Load and verify documents
    try:
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Loaded {len(documents)} documents from documents.pkl")
    except Exception as e:
        print(f"❌ Error loading documents.pkl: {e}")
        return False
    
    # Load and verify index
    try:
        index = faiss.read_index("faiss_index.index")
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
        print(f"   Dimension: {index.d}")
    except Exception as e:
        print(f"❌ Error loading faiss_index.index: {e}")
        return False
    
    # Verify counts match
    if len(documents) != index.ntotal:
        print(f"⚠️ Count mismatch: documents.pkl has {len(documents)}, index has {index.ntotal}")
        return False
    
    print("✅ Verification complete - Index is valid!")
    return True


# ------------------------------------------
# Testing
# ------------------------------------------
if __name__ == "__main__":

    # Build the index
    build_faiss_index()
    
    # Verify the index
    verify_faiss_index()