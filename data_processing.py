from database import detections, documents
from violation_checker import compute_compliance_rate


# ==========================================
# Split Long Documents into Chunks
# ==========================================
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks


def prepare_rag_data():

    rag_documents = []

    # ==========================================
    # Inspection Reports (One Document Per Image)
    # ==========================================
    records = detections.find()

    for record in records:

        date = record.get("timestamp", "Unknown")

        for image in record.get("images", []):

            workers = image.get("workers", [])
            site_name = image.get("site_name", "Construction Site")
            image_name = image.get("name", "Unknown")
            
            # Skip images with no workers
            if not workers:
                continue

            # ==========================================
            # Collect Worker Information
            # ==========================================
            worker_details = []
            safe_workers = 0
            unsafe_workers = 0
            violation_counts = {}
            
            for worker in workers:
                worker_id = worker.get("worker_id", "Unknown")
                raw_status = worker.get("status", "Unknown").strip()
                status = raw_status.lower()
                missing = worker.get("missing", [])
                
                # Count safe/unsafe
                if status == "safe":
                    safe_workers += 1
                else:
                    unsafe_workers += 1
                
                # Count violations
                for item in missing:
                    violation_counts[item] = violation_counts.get(item, 0) + 1
                
                # Build worker detail
                if missing:
                    missing_text = ", ".join(sorted(missing))
                else:
                    missing_text = "None"
                
                worker_details.append(f"""
Worker {worker_id}
Status: {raw_status}
Missing PPE: {missing_text}
""")
            
            # ==========================================
            # Calculate Statistics
            # ==========================================
            total_workers = len(workers)
            overall_status = "UNSAFE" if unsafe_workers > 0 else "SAFE"
            # % of required PPE items actually present — the same formula used
            # everywhere else in the app. This used to be safe_workers /
            # total_workers, which for a single-worker image can only ever
            # produce 0% or 100%: one worker missing just their goggles read
            # as "0% compliant" despite wearing four of five items. That
            # binary number then travelled into the FAISS index and out to
            # the AI Assistant's report cards.
            compliance_rate = compute_compliance_rate(workers)
            
            # ==========================================
            # Build Inspection Text
            # ==========================================
            inspection_text = f"""
Inspection Report

Inspection Status: {overall_status}

Workers Detected: {total_workers}
Workers Violating PPE: {unsafe_workers}
Safe Workers: {safe_workers}
Compliance Rate: {compliance_rate}%

Image: {image_name}
Site: {site_name}
Date: {date}

"Worker Details"
{''.join(worker_details)}

Inspection Summary
{unsafe_workers} worker(s) violated PPE requirements.
Overall inspection marked {overall_status}.
"""
            
            # Add violation details if any
            if violation_counts:
                violation_summary = "\nViolations Detected:\n"
                # Sort by frequency (most frequent first)
                for ppe_item, count in sorted(
                    violation_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    violation_summary += f"  - {ppe_item}: {count}\n"
                inspection_text += violation_summary
            
            # ==========================================
            # Create ONE Document Per Image with _id
            # ==========================================
            rag_documents.append({
                "_id": str(record["_id"]),  # ✅ ADD THIS LINE - MongoDB ObjectId

                "type": "inspection",
                "status": "Unsafe" if unsafe_workers > 0 else "Safe",
                "image": image_name,
                "site_name": site_name,
                "date": date,
                "total_workers": total_workers,
                "safe_workers": safe_workers,
                "unsafe_workers": unsafe_workers,
                "compliance_rate": compliance_rate,
                # Carried through so anything reading this document later can
                # recompute compliance itself rather than trusting whatever
                # formula was in use when the index was last built.
                "workers": workers,
                "missing": list(violation_counts.keys()),
                "text": inspection_text
            })

    # ==========================================
    # Safety Manuals (Chunked)
    # ==========================================
    manual_docs = documents.find()

    for doc in manual_docs:

        text = doc.get("text", "").strip()

        if not text:
            continue

        # Split long documents into chunks
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):

            rag_documents.append({
                "type": "manual",
                "filename": doc.get("filename"),
                "document_type": doc.get("document_type", "Manual"),
                "chunk": i + 1,
                "total_chunks": len(chunks),
                "text": f"""
Safety Manual

Filename: {doc.get('filename')}

Document Type: {doc.get('document_type', 'Manual')}

Section: {i + 1} of {len(chunks)}

{chunk}
"""
            })

    return rag_documents


# ==========================================
# Testing
# ==========================================
if __name__ == "__main__":

    print("=" * 60)
    print("Testing Data Processing")
    print("=" * 60)

    rag_docs = prepare_rag_data()

    print(f"\nTotal Documents Prepared: {len(rag_docs)}")

    # Count by type
    inspection_count = sum(1 for d in rag_docs if d["type"] == "inspection")
    manual_count = sum(1 for d in rag_docs if d["type"] == "manual")

    print(f"  - Inspection Reports: {inspection_count}")
    print(f"  - Manual Chunks: {manual_count}")

    # Show sample inspection report if available
    sample_inspection = next((d for d in rag_docs if d["type"] == "inspection"), None)
    if sample_inspection:
        print("\n" + "=" * 60)
        print("Sample Inspection Report:")
        print("=" * 60)
        print(f"Image: {sample_inspection['image']}")
        print(f"Site: {sample_inspection['site_name']}")
        print(f"Date: {sample_inspection['date']}")
        print(f"Status: {sample_inspection['status']}")
        print(f"Total Workers: {sample_inspection['total_workers']}")
        print(f"Safe Workers: {sample_inspection['safe_workers']}")
        print(f"Unsafe Workers: {sample_inspection['unsafe_workers']}")
        print(f"Compliance Rate: {sample_inspection['compliance_rate']}%")
        print(f"Missing PPE: {', '.join(sample_inspection['missing']) if sample_inspection['missing'] else 'None'}")
        print(f"ID: {sample_inspection.get('_id', 'NO ID')}")  # ✅ Verify _id is present
        print("\nFull Text Preview:")
        print(sample_inspection['text'][:500] + "...")
        
    # Show sample manual if available
    sample_manual = next((d for d in rag_docs if d["type"] == "manual"), None)
    if sample_manual:
        print("\n" + "=" * 60)
        print("Sample Manual Chunk:")
        print("=" * 60)
        print(f"Filename: {sample_manual['filename']}")
        print(f"Document Type: {sample_manual['document_type']}")
        print(f"Chunk: {sample_manual['chunk']} of {sample_manual['total_chunks']}")
        print(f"Text Preview: {sample_manual['text'][:200]}...")