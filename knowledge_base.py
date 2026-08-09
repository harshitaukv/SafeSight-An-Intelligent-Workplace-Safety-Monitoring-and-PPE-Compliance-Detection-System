from embedding import generate_embeddings
from faiss_db import build_faiss_index
import traceback
import time  # 1. ADDED FOR UPDATE TIME
import os
import pickle


def update_knowledge_base():
    """
    Update the knowledge base by generating embeddings and building the FAISS index.
    
    Returns:
        dict: Status of the update operation with success flag, processing time, and statistics.
    """
    
    # 1. MEASURE UPDATE TIME
    start = time.perf_counter()
    
    try:
        print("=" * 60)
        print("🔄 Updating Knowledge Base...")
        print("=" * 60)

        # ----------------------------------------------------
        # Step 1: Generate embeddings from all documents
        # ----------------------------------------------------
        print("\n[1/2] Generating embeddings...")
        generate_embeddings()
        print("✓ Embeddings generated successfully.\n")

        # ----------------------------------------------------
        # Step 2: Build FAISS index
        # ----------------------------------------------------
        print("[2/2] Building FAISS index...")
        build_faiss_index()
        print("✓ FAISS index built successfully.\n")

        # 3. VERIFY GENERATED FILES
        required_files = [
            "faiss_index.index",
            "documents.pkl"
        ]
        
        missing = [
            f for f in required_files
            if not os.path.exists(f)
        ]
        
        if missing:
            raise FileNotFoundError(
                f"Missing generated files: {missing}"
            )
        
        # 4. SHOW DATASET STATISTICS
        # Load documents.pkl to gather statistics
        with open("documents.pkl", "rb") as f:
            docs = pickle.load(f)
        
        manuals = sum(
            d.get("type") == "manual"
            for d in docs
        )
        
        inspections = sum(
            d.get("type") == "inspection"
            for d in docs
        )
        
        # 1. CALCULATE ELAPSED TIME
        elapsed = round(time.perf_counter() - start, 2)
        
        # 5. BETTER CONSOLE OUTPUT
        print("=" * 60)
        print("📊 Knowledge Base Summary")
        print("=" * 60)
        print(f"Documents Indexed : {len(docs)}")
        print(f"Manual Chunks     : {manuals}")
        print(f"Inspection Reports: {inspections}")
        print(f"Processing Time   : {elapsed} sec")
        print("=" * 60)
        
        # 2. RETURN STATUS
        return {
            "success": True,
            "processing_time": elapsed,
            "documents_indexed": len(docs),
            "manual_chunks": manuals,
            "inspection_reports": inspections
        }

    except FileNotFoundError as e:
        print("=" * 60)
        print("❌ Knowledge Base Update Failed")
        print(f"File not found: {e}")
        print("=" * 60)
        print("\n💡 Hint: Make sure documents exist in MongoDB or the data directory.")
        
        # 2. RETURN STATUS FOR EXCEPTIONS
        return {
            "success": False,
            "error": str(e),
            "error_type": "FileNotFoundError"
        }

    except Exception as e:
        print("=" * 60)
        print("❌ Knowledge Base Update Failed")
        print(f"Error: {e}")
        print("=" * 60)
        print("\n💡 Debug Information:")
        traceback.print_exc()
        
        # 2. RETURN STATUS FOR EXCEPTIONS
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    result = update_knowledge_base()
    
    # Display result summary
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    if result["success"]:
        print(f"✅ Success: Knowledge Base Updated")
        print(f"   Processing Time: {result['processing_time']} sec")
        print(f"   Documents Indexed: {result['documents_indexed']}")
        print(f"   Manual Chunks: {result['manual_chunks']}")
        print(f"   Inspection Reports: {result['inspection_reports']}")
    else:
        print(f"❌ Failed: {result['error']}")
        print(f"   Error Type: {result.get('error_type', 'Unknown')}")
    print("=" * 60)