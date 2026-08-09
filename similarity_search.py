import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

# -------------------------------
# Load Embedding Model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Load FAISS Index
# -------------------------------
# Fixed: Changed from "safesight.index" to "faiss_index.index"
# This matches what faiss_db.py creates
index = faiss.read_index("faiss_index.index")

# -------------------------------
# DEBUG: Check Index Dimension
# -------------------------------
print("=" * 60)
print("DEBUG: Similarity Search Initialization")
print("=" * 60)
print(f"Index dimension: {index.d}")
print(f"Index contains: {index.ntotal} documents")
print("=" * 60)

# -------------------------------
# Load Documents
# -------------------------------
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

# ✅ DEBUG: Print _id for inspection documents
print("\n========== DEBUG _id ==========")
for doc in documents:
    if doc.get("type") == "inspection":
        print("Image:", doc.get("image"))
        print("_id :", doc.get("_id"))
        print("Keys:", doc.keys())
        print("-" * 40)
print("================================\n")

# ✅ DEBUG: Print document types
print("========== DOCUMENT TYPES ==========")
for doc in documents:
    print(
        "TYPE:", doc.get("type"),
        "| DOC_TYPE:", doc.get("document_type"),
        "| FILE:", doc.get("filename")
    )
print("====================================\n")

# Step 1: Print manuals in database
print("\n================ MANUALS IN DATABASE ================")

manual_docs = [d for d in documents if d.get("type") == "manual"]

print(f"Total Manual Chunks: {len(manual_docs)}")

for i, doc in enumerate(manual_docs[:10], 1):
    print(
        f"{i}. {doc.get('filename')} | "
        f"Chunk {doc.get('chunk')} | "
        f"{doc.get('document_type')}"
    )

print("=====================================================\n")

# Print all document filenames for debugging
print("📚 Documents in index:")
for i, doc in enumerate(documents):
    filename = doc.get('filename') or doc.get('image') or 'Unknown'
    doc_type = doc.get('type', 'unknown')
    print(f"  {i}: {doc_type:12} | {filename}")
print("=" * 60)

# -------------------------------
# Check for Duplicate Inspection Images
# -------------------------------
print("\n🔍 Checking for duplicate inspection images...")
print("-" * 60)

image_counts = Counter()
for doc in documents:
    if doc.get("type") == "inspection":
        image_counts[doc.get("image")] += 1

duplicate_found = False
for image, count in image_counts.items():
    if count > 1:
        duplicate_found = True
        print(f"⚠️  {image}: {count} occurrences")
        
        # Show details of duplicates
        for doc in documents:
            if doc.get("image") == image and doc.get("type") == "inspection":
                print(f"      Date: {doc.get('date', 'Unknown')}")
                print(f"      Status: {doc.get('status', 'Unknown')}")
                print(f"      Workers: {doc.get('total_workers', 'Unknown')}")
                print(f"      Score: {doc.get('compliance_rate', 'Unknown')}%")
                print("      ---")

if not duplicate_found:
    print("✅ No duplicate inspection images found.")
print("=" * 60)


# --------------------------------------------------
# FIXED: search() function - Retrieves more vectors before filtering
# --------------------------------------------------
def search(
    query,
    top_k=5,
    document_type=None,
    search_scope="all",
    selected_manuals=None,
):
    """
    Search for documents similar to the query.
    
    Args:
        query: The search query string
        top_k: Number of results to return
        document_type: Optional filter - "manual" or "inspection"
        search_scope: "all" or "selected" - determines which documents to search
        selected_manuals: List of manual filenames to search within (if search_scope is "selected")
    """

    if selected_manuals is None:
        selected_manuals = []

    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # -------------------------------
    # DEBUG: Check Query Dimension
    # -------------------------------
    print(f"\n🔍 Query: {query[:50]}...")
    print(f"Query dimension: {query_embedding.shape[1]}")
    print(f"Index dimension: {index.d}")
    
    if query_embedding.shape[1] != index.d:
        print(f"❌ DIMENSION MISMATCH! Query: {query_embedding.shape[1]}, Index: {index.d}")
        print("   This will cause FAISS search to fail!")
        return []

    # ============================================================
    # FIX: Retrieve a larger candidate pool before filtering
    # ============================================================
    # Preserve the original requested value
    requested_top_k = top_k
    
    # Retrieve more vectors to ensure we have enough after filtering
    # At least 100, or 20x the requested amount, but not more than total
    search_k = min(max(requested_top_k * 20, 100), index.ntotal)
    
    print(f"🔍 Requested: {requested_top_k}, Searching: {search_k} candidates")

    # Similarity Search with larger candidate pool
    scores, indices = index.search(query_embedding, search_k)

    # ============================================================
    # DEBUG: Print what FAISS is returning (DEBUG=True now)
    # ============================================================
    DEBUG = False  # Set to True for detailed debug

    if DEBUG:
        print("-" * 60)
        print("📊 FAISS RAW RESULTS:")
        print(f"   Scores shape: {scores.shape}")
        print(f"   Indices shape: {indices.shape}")
        print(f"   Scores: {scores}")
        print(f"   Indices: {indices}")
    
    # Check for invalid results
    if len(scores[0]) == 0:
        print("   ⚠️ No scores returned!")
        return []

    # Check for invalid indices
    valid_indices = [idx for idx in indices[0] if idx != -1]
    
    if DEBUG:
        print(f"   Best score: {scores[0][0]:.4f}")
        print(f"   Worst score: {scores[0][-1]:.4f}")
        print(f"   Average score: {np.mean(scores[0]):.4f}")
        print(f"   Valid indices: {valid_indices}")
        
        # Print the actual documents being returned (before deduplication)
        print("\n📄 Retrieved Documents (Before Deduplication):")
        for idx in valid_indices:
            if idx < len(documents):
                doc = documents[idx]
                filename = doc.get('filename') or doc.get('image') or 'Unknown'
                doc_type = doc.get('type', 'unknown')
                date = doc.get('date', 'N/A')
                print(f"   Index {idx}: {doc_type:12} | {filename} | Date: {date}")
            else:
                print(f"   Index {idx}: INVALID (out of range)")
        print("-" * 60)

    # --------------------------------------------------
    # Build raw results with confidence levels
    # --------------------------------------------------
    raw_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        
        # Confidence Level
        if score > 0.60:
            confidence = "High"
        elif score > 0.45:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        raw_results.append({
            "score": float(score),
            "confidence": confidence,
            "document": documents[idx]
        })

    # Filter by document type
    if document_type:
        raw_results = [
            r
            for r in raw_results
            if r["document"].get("type") == document_type
        ]
        print(f"📂 Document type filter ({document_type}): {len(raw_results)} results")

    # Selected manuals filter (only when document_type == "manual")
    if (
        search_scope == "selected"
        and document_type == "manual"
    ):
        selected_manuals = selected_manuals or []
        raw_results = [
            r
            for r in raw_results
            if r["document"].get("filename") in selected_manuals
        ]
        print(f"📋 Selected manuals filter applied: {len(raw_results)} results")

    # --------------------------------------------------
    # Better Deduplication - One result per manual
    # --------------------------------------------------
    results = []
    seen = {}

    for result in raw_results:
        doc = result["document"]

        if doc.get("type") == "inspection":
            key = (
                doc.get("image"),
                doc.get("date")
            )
        else:
            # One result per manual (not per chunk)
            key = doc.get("filename")

        # Keep highest scoring result
        if key not in seen or result["score"] > seen[key]["score"]:
            seen[key] = result

    results = list(seen.values())

    # Sort again after deduplication
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    duplicate_count = len(raw_results) - len(results)

    print(f"\n📊 Deduplication Results:")
    print(f"   Original results: {len(raw_results)}")
    print(f"   Duplicates removed: {duplicate_count}")
    print(f"   Unique results: {len(results)}")

    # Print the actual documents being returned (after deduplication)
    if DEBUG:
        print("\n📄 Retrieved Documents (After Deduplication):")
        for i, result in enumerate(results, 1):
            doc = result["document"]
            filename = doc.get('filename') or doc.get('image') or 'Unknown'
            doc_type = doc.get('type', 'unknown')
            date = doc.get('date', 'N/A')
            print(f"   {i}. {doc_type:12} | {filename} | Date: {date} | Score: {result['score']:.4f} | Confidence: {result['confidence']}")
        print("-" * 60)

    # --------------------------------------------------
    # ✅ SUMMARY MODE: Return ALL inspection reports for summary queries
    # --------------------------------------------------
    summary_keywords = [
        "summary",
        "summarize",
        "overall summary",
        "inspection summary",
        "site summary",
        "give me summary",
    ]

    is_summary = any(k in query.lower() for k in summary_keywords)

    if document_type == "inspection" and is_summary:
        print(f"📊 Summary mode: Returning all {len(results)} inspection reports")
        # Return all results for summary
        return results
    else:
        # Return the requested number of results (not hardcoded 5)
        results = results[:requested_top_k]
        print(f"Retrieved {len(raw_results)} → Deduplicated {len(results)} → Returning {len(results)}")
        return results


# -------------------------------
# Testing
# -------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("Testing Similarity Search")
    print("=" * 60)
    print(f"Index contains {index.ntotal} documents\n")

    # Test queries with different document types and search scopes
    test_queries = [
        ("What PPE is required for construction workers?", None, "all", []),
        ("What PPE is required for construction workers?", "manual", "all", []),
        ("What PPE is required for construction workers?", "manual", "selected", ["OSHA3151.pdf"]),
        ("Safety violations on site", "inspection", "all", []),
        ("Give me a summary of inspection reports", "inspection", "all", []),
    ]

    for query, doc_type, scope, manuals in test_queries:
        print(f"\n🔍 Query: {query}")
        print(f"   Document Type: {doc_type}")
        print(f"   Search Scope: {scope}")
        if manuals:
            print(f"   Selected Manuals: {', '.join(manuals)}")
        print("-" * 60)

        results = search(
            query,
            top_k=10,
            document_type=doc_type,
            search_scope=scope,
            selected_manuals=manuals
        )

        if not results:
            print("No results found.")
            continue

        print(f"✅ Returning {len(results)} results")
        
        for i, result in enumerate(results, 1):
            doc = result["document"]
            score = result["score"]
            confidence = result.get("confidence", "Unknown")

            # Determine document type for display
            doc_type_display = doc.get('document_type', 'Unknown')
            filename = doc.get('filename', 'Unknown')
            
            # Format based on document type
            if doc_type_display in ["PDF Manual", "DOCX Manual", "Text Document"]:
                print(f"\n📄 Result {i} (Score: {score:.4f}, Confidence: {confidence}):")
                print(f"  Document: {filename}")
                print(f"  Type: {doc_type_display}")
            else:
                print(f"\n🖼️  Result {i} (Score: {score:.4f}, Confidence: {confidence}):")
                print(f"  Image: {filename}")
                print(f"  Status: {doc.get('status', 'Unknown')}")
                site = doc.get('site_name', 'Unknown')
                if site != 'Unknown':
                    print(f"  Site: {site}")
                if doc.get('date'):
                    print(f"  Date: {doc.get('date')}")
                if doc.get('compliance_rate') is not None:
                    print(f"  Compliance: {doc.get('compliance_rate')}%")
                if doc.get('total_workers'):
                    print(f"  Workers: {doc.get('total_workers')}")

            # Preview the text (first 200 characters)
            text_preview = doc.get('text', '')[:200].replace('\n', ' ')
            print(f"  Preview: {text_preview}...")

    print("\n" + "=" * 60)