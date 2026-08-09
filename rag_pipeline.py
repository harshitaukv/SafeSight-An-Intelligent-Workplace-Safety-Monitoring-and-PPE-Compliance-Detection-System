from similarity_search import search
from llm_provider import build_llm
import pickle
import os
import re
from risk_engine import assess_risk, get_recommendations
from report_generator import generate_report
from agents import (
    RetrievalAgent,
    RiskAssessmentAgent,
    ManualAgent,
    ReportAgent,
    ResponseAgent,
)

# --------------------------------------------------
# Load the configured chat model
# --------------------------------------------------
# This used to pin llama3.1 locally, independently of langgraph_pipeline.py
# — so setting SAFESIGHT_LLM_MODEL changed one path and silently not the
# other. Both now resolve through llm_provider.
llm = build_llm()

# --------------------------------------------------
# Load Documents at Startup (Avoid Disk Reads)
# --------------------------------------------------
def load_documents():
    """Load documents from disk once at startup."""
    try:
        with open("documents.pkl", "rb") as f:
            all_docs = pickle.load(f)
        
        inspection_reports = [
            doc for doc in all_docs
            if doc.get("type") == "inspection"
        ]
        
        print("=" * 60)
        print("📚 Documents Loaded at Startup:")
        print(f"   Total documents: {len(all_docs)}")
        print(f"   Inspection reports: {len(inspection_reports)}")
        print("=" * 60)
        
        return all_docs, inspection_reports
    except FileNotFoundError:
        print("❌ documents.pkl not found! Run knowledge_base.py first.")
        return [], []

# Load once at module import
ALL_DOCS, INSPECTION_REPORTS = load_documents()

# --------------------------------------------------
# Initialize Agents
# --------------------------------------------------

# Helper function for retrieval
def retrieve_documents(query, top_k=10, document_type=None, search_scope="all", selected_manuals=None):
    """Wrapper for the search function to match agent expectations."""
    return search(
        query,
        top_k=top_k,
        document_type=document_type,
        search_scope=search_scope,
        selected_manuals=selected_manuals
    )

# Initialize all agents
retrieval_agent = RetrievalAgent(retrieve_documents)

risk_agent = RiskAssessmentAgent(
    assess_risk,
    get_recommendations,
)

manual_agent = ManualAgent()

report_agent = ReportAgent(llm)

response_agent = ResponseAgent(llm)


# --------------------------------------------------
# Intent Detection (Python-based, no LLM)
# --------------------------------------------------
def detect_intent(question):
    """
    Detect the user's intent using Python pattern matching.
    No LLM call required.
    """
    question_lower = question.lower()
    
    # -------------------------------
    # Check for Analytics Intent
    # -------------------------------
    analytics_patterns = [
        "overall compliance",
        "overall statistics",
        "overall summary",
        "analytics",
        "dashboard",
        "total workers",
        "total inspections",
        "total violations",
        "compliance percentage",
        "average compliance",
        "total helmet",
        "total gloves",
        "total boots",
        "total goggles",
        "total vest",
        "compliance rate"
    ]
    
    if any(pattern in question_lower for pattern in analytics_patterns):
        return "analytics", {}
    
    # -------------------------------
    # Check for Filtered Reports Intent
    # -------------------------------
    filtered_patterns = [
        "all reports",
        "all inspection reports",
        "all unsafe reports",
        "all safe reports",
        "reports missing",
        "show reports",
        "list reports",
        "give reports",
        "unsafe reports",
        "safe reports",
        "reports with",
        "reports where"
    ]
    
    if any(pattern in question_lower for pattern in filtered_patterns):
        # Extract filters from the question
        filters = extract_filters(question_lower)
        return "filtered_reports", filters
    
    # -------------------------------
    # Default: Semantic Search
    # -------------------------------
    return "semantic_search", {}


# --------------------------------------------------
# Extract Filters from Question (Python-based)
# --------------------------------------------------
def extract_filters(question_lower):
    """
    Extract filters from the question using Python pattern matching.
    No LLM call required.
    """
    filters = {}
    
    # -------------------------------
    # Status Filters
    # -------------------------------
    if "unsafe" in question_lower:
        filters["status"] = "UNSAFE"
    elif "safe" in question_lower:
        filters["status"] = "SAFE"
    
    # -------------------------------
    # PPE Missing Filters
    # -------------------------------
    missing_ppe = []
    ppe_items = ["helmet", "vest", "gloves", "boots", "goggles"]
    
    for item in ppe_items:
        if item in question_lower and "missing" in question_lower:
            missing_ppe.append(item.capitalize())
    
    if missing_ppe:
        filters["missing_ppe"] = missing_ppe
    
    # -------------------------------
    # Site Filters
    # -------------------------------
    if "site" in question_lower:
        site_match = re.search(r'site\s+([a-zA-Z0-9\s]+)', question_lower)
        if site_match:
            filters["site"] = site_match.group(1).strip()
    
    # -------------------------------
    # Compliance Operators
    # -------------------------------
    compliance_pattern = r'compliance\s*(above|below|>|<|>=|<=|over|under)\s*(\d+)'
    match = re.search(compliance_pattern, question_lower)
    
    if match:
        operator_map = {
            "above": ">",
            "over": ">",
            "below": "<",
            "under": "<"
        }
        op = match.group(1)
        value = int(match.group(2))
        
        if op in operator_map:
            filters["compliance_rate"] = {
                "operator": operator_map[op],
                "value": value
            }
        else:
            filters["compliance_rate"] = {
                "operator": op,
                "value": value
            }
    
    # -------------------------------
    # Worker Count Filters
    # -------------------------------
    patterns = [
        r"(more than|less than|>|<|>=|<=)\s*(\d+)\s*workers?",
        r"workers?\s*(more than|less than|>|<|>=|<=)\s*(\d+)"
    ]
    
    match = None
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            break
    
    if match:
        op_map = {
            "more than": ">",
            "less than": "<"
        }
        op = match.group(1)
        value = int(match.group(2))
        
        if op in op_map:
            filters["total_workers"] = {
                "operator": op_map[op],
                "value": value
            }
        else:
            filters["total_workers"] = {
                "operator": op,
                "value": value
            }
    
    return filters


# --------------------------------------------------
# Compute Analytics from All Inspection Reports
# --------------------------------------------------
def compute_analytics(inspection_reports):
    """
    Compute statistics from all inspection reports.
    Returns a summary dictionary and a formatted text summary.
    """
    
    total_workers = 0
    safe_workers = 0
    unsafe_workers = 0
    total_inspections = len(inspection_reports)
    
    # PPE violation counts
    helmet = 0
    vest = 0
    gloves = 0
    boots = 0
    goggles = 0
    
    # Track compliance rates for average
    compliance_rates = []
    sites = set()
    
    for doc in inspection_reports:
        total_workers += doc.get("total_workers", 0)
        safe_workers += doc.get("safe_workers", 0)
        unsafe_workers += doc.get("unsafe_workers", 0)
        
        site = doc.get("site_name")
        if site:
            sites.add(site)
        
        compliance_rate = doc.get("compliance_rate")
        if compliance_rate is not None:
            compliance_rates.append(compliance_rate)
        
        # Count PPE violations
        for item in doc.get("missing", []):
            item_lower = item.lower()
            if "helmet" in item_lower:
                helmet += 1
            elif "vest" in item_lower:
                vest += 1
            elif "glove" in item_lower:
                gloves += 1
            elif "boot" in item_lower:
                boots += 1
            elif "goggle" in item_lower:
                goggles += 1
    
    # Calculate overall compliance
    overall_compliance = (
        round((safe_workers / total_workers) * 100, 1)
        if total_workers > 0 else 100.0
    )
    
    avg_compliance = (
        round(sum(compliance_rates) / len(compliance_rates), 1)
        if compliance_rates else overall_compliance
    )
    
    # Build summary
    summary = {
        "total_inspections": total_inspections,
        "total_workers": total_workers,
        "safe_workers": safe_workers,
        "unsafe_workers": unsafe_workers,
        "overall_compliance": overall_compliance,
        "avg_compliance": avg_compliance,
        "unique_sites": len(sites),
        "sites": sorted(list(sites)),
        "ppe_violations": {
            "Helmet": helmet,
            "Vest": vest,
            "Gloves": gloves,
            "Boots": boots,
            "Goggles": goggles
        }
    }
    
    # Format text
    text_parts = [
        "=" * 60,
        "📊 SAFESIGHT ANALYTICS SUMMARY",
        "=" * 60,
        "",
        f"Total Inspections: {total_inspections}",
        f"Total Workers Detected: {total_workers}",
        f"Unique Sites: {len(sites)}",
        "",
        "Worker Statistics:",
        f"  Safe Workers: {safe_workers}",
        f"  Unsafe Workers: {unsafe_workers}",
        f"  Overall Compliance Rate: {overall_compliance}%",
        f"  Average Compliance (per inspection): {avg_compliance}%",
        "",
        "PPE Violations (Total across all inspections):",
    ]
    
    ppe_items = summary["ppe_violations"]
    has_violations = any(count > 0 for count in ppe_items.values())
    
    if has_violations:
        for item, count in ppe_items.items():
            if count > 0:
                text_parts.append(f"  {item}: {count}")
    else:
        text_parts.append("  No PPE violations detected")
    
    text_parts.append("=" * 60)
    
    return summary, "\n".join(text_parts)


# --------------------------------------------------
# Filter Reports by Conditions (Enhanced)
# --------------------------------------------------
def filter_reports(inspection_reports, filters):
    """
    Filter inspection reports based on the provided filters.
    Supports operators for numeric fields.
    """
    if not filters:
        return inspection_reports
    
    filtered = inspection_reports
    
    # Filter by status
    if "status" in filters:
        status = filters["status"].upper()
        filtered = [doc for doc in filtered if doc.get("status") == status]
    
    # Filter by missing PPE
    if "missing_ppe" in filters:
        missing_ppe = filters["missing_ppe"]
        if isinstance(missing_ppe, str):
            missing_ppe = [missing_ppe]
        filtered = [
            doc for doc in filtered 
            if any(item in doc.get("missing", []) for item in missing_ppe)
        ]
    
    # Filter by site
    if "site" in filters:
        site = filters["site"].lower()
        filtered = [
            doc for doc in filtered 
            if site in doc.get("site_name", "").lower()
        ]
    
    # Filter by compliance rate with operator
    if "compliance_rate" in filters:
        comp_filter = filters["compliance_rate"]
        operator = comp_filter.get("operator", ">")
        value = comp_filter.get("value", 0)
        
        if operator == ">":
            filtered = [doc for doc in filtered if doc.get("compliance_rate", 0) > value]
        elif operator == ">=":
            filtered = [doc for doc in filtered if doc.get("compliance_rate", 0) >= value]
        elif operator == "<":
            filtered = [doc for doc in filtered if doc.get("compliance_rate", 0) < value]
        elif operator == "<=":
            filtered = [doc for doc in filtered if doc.get("compliance_rate", 0) <= value]
        elif operator == "==":
            filtered = [doc for doc in filtered if doc.get("compliance_rate", 0) == value]
    
    # Filter by worker count with operator
    if "total_workers" in filters:
        worker_filter = filters["total_workers"]
        operator = worker_filter.get("operator", ">")
        value = worker_filter.get("value", 0)
        
        if operator == ">":
            filtered = [doc for doc in filtered if doc.get("total_workers", 0) > value]
        elif operator == ">=":
            filtered = [doc for doc in filtered if doc.get("total_workers", 0) >= value]
        elif operator == "<":
            filtered = [doc for doc in filtered if doc.get("total_workers", 0) < value]
        elif operator == "<=":
            filtered = [doc for doc in filtered if doc.get("total_workers", 0) <= value]
        elif operator == "==":
            filtered = [doc for doc in filtered if doc.get("total_workers", 0) == value]
    
    return filtered


# --------------------------------------------------
# FIXED: Deduplicate results by image and filename
# --------------------------------------------------
def deduplicate_results(results):
    """
    Deduplicate inspection reports by image and manuals by filename.
    """
    unique_results = []
    seen = set()

    for r in results:
        doc = r["document"]

        if doc.get("type") == "manual":
            key = ("manual", doc.get("filename"))
        else:
            key = ("inspection", doc.get("image"))

        if key in seen:
            continue

        seen.add(key)
        unique_results.append(r)

    return unique_results


# --------------------------------------------------
# RAG Pipeline - Using Multi-Agent Architecture
# --------------------------------------------------
def ask_question(
    question,
    search_scope="all",
    selected_manuals=None
):
    """
    Process a question using the RAG pipeline with multi-agent architecture.
    
    Args:
        question: The user's question
        search_scope: "all" or "selected" - determines which documents to search
        selected_manuals: List of manual filenames to search within (if search_scope is "selected")
    
    Returns:
        dict: Contains mode, answer, sources, source_names, and optionally summary
    """
    
    if selected_manuals is None:
        selected_manuals = []

    # ============================================================
    # DEBUG: Print the exact question received
    # ============================================================
    print("=" * 60)
    print(f"🔍 RAG PIPELINE RECEIVED QUESTION:")
    print(f"   Raw: {repr(question)}")
    print(f"   Search Scope: {search_scope}")
    if search_scope == "selected" and selected_manuals:
        print(f"   Selected Manuals: {', '.join(selected_manuals)}")
    print("=" * 60)

    # --------------------------------------------------
    # Detect Intent (Python-based, no LLM)
    # --------------------------------------------------
    mode, filters = detect_intent(question)
    
    print("=" * 60)
    print("Detected Filters")
    print(filters)
    print("=" * 60)
    
    print(f"📍 Mode: {mode}")
    if filters:
        print(f"   Filters: {filters}")

    # --------------------------------------------------
    # Mode 1: Analytics
    # --------------------------------------------------
    if mode == "analytics":
        print("📊 Analytics Mode")

        if not INSPECTION_REPORTS:
            return {
                "mode": mode,
                "answer": "No inspection reports found.",
                "sources": []
            }

        # Apply filters if any
        if filters:
            filtered_reports = filter_reports(INSPECTION_REPORTS, filters)
            print(f"   Filtered to {len(filtered_reports)} reports")
        else:
            filtered_reports = INSPECTION_REPORTS

        summary, analytics_text = compute_analytics(filtered_reports)
        
        print(f"   ✅ Analytics computed: {summary['total_workers']} workers, {summary['overall_compliance']}% compliance")
        
        return {
            "mode": mode,
            "answer": analytics_text,
            "summary": summary,  # Include summary for frontend to display
            "sources": []  # No sources for analytics
        }

    # --------------------------------------------------
    # Mode 2: Filtered Reports (Updated with Pagination)
    # --------------------------------------------------
    elif mode == "filtered_reports":
        print("📋 Filtered Report Mode")

        if not INSPECTION_REPORTS:
            return {
                "mode": mode,
                "answer": "No inspection reports found.",
                "sources": []
            }

        # Apply filters
        all_filtered_reports = filter_reports(INSPECTION_REPORTS, filters)
        total_matches = len(all_filtered_reports)
        print(f"   Found {total_matches} matching reports")

        if total_matches == 0:
            return {
                "mode": mode,
                "answer": "No reports found matching your criteria.",
                "sources": []
            }

        # Create display reports (limit for LLM context)
        DISPLAY_LIMIT = 10
        display_reports = all_filtered_reports[:DISPLAY_LIMIT]
        
        print(f"   Displaying {len(display_reports)} reports for LLM context")

        # Convert to results format
        results = [
            {
                "score": 1.0,
                "document": doc
            }
            for doc in display_reports
        ]

        # DEDUPLICATE RESULTS BEFORE USING THEM
        results = deduplicate_results(results)

        # Improved filtered reports header
        header = f"""
The search found exactly {total_matches} matching inspection reports.

Only the first {len(display_reports)} reports are provided.

Never say "all reports".

Only describe the reports provided below.

State that additional reports exist if total_matches is greater than {len(display_reports)}.
"""

    # --------------------------------------------------
    # Mode 3: Semantic Search
    # --------------------------------------------------
    else:
        print("🔍 Semantic Search Mode")
        
        # Determine which document type to search based on keywords
        question_lower = question.lower()
        
        # Prevention Keywords
        prevention_keywords = [
            "prevent",
            "prevention",
            "recommend",
            "recommendation",
            "avoid",
            "mitigate",
            "corrective",
            "corrective action",
            "how can",
            "how should",
            "reduce"
        ]
        
        # Worker Lookup Keywords
        worker_lookup_keywords = [
            "who",
            "who was",
            "who were",
            "which worker",
            "which workers",
            "list workers",
            "list worker",
            "whose"
        ]
        
        # Keywords that suggest the user wants inspection reports
        inspection_keywords = [
            "report",
            "reports",
            "inspection",
            "violations",
            "missing",
            "worker",
            "workers",
            "helmet",
            "vest",
            "boots",
            "gloves",
            "goggles",
            "image",
            "unsafe",
            "safe",
            "site",
            "compliance",
            "ppe violations",
            "violation report"
        ]
        
        # Improved manual keywords
        manual_keywords = [
            "manual",
            "pdf",
            "document",
            "documents",
            "summary of the manual",
            "summarize the manual",
            "summary",
            "guide",
            "guideline",
            "guidelines",
            "policy",
            "policies",
            "procedure",
            "procedures",
            "osha",
            "regulation",
            "regulations",
            "standard",
            "standards",
            "requirement",
            "requirements",
            "ppe manual",
            "safety manual"
        ]
        
        # Worker Lookup Branch - Direct retrieval without LLM
        if any(k in question_lower for k in worker_lookup_keywords):
            print("👷 Worker Lookup Question")
            
            # Use RetrievalAgent for document retrieval
            results = retrieval_agent.retrieve(
                query=question,
                top_k=10,
                document_type="inspection",
                search_scope=search_scope,
                selected_manuals=selected_manuals
            )
            
            # DEDUPLICATE RESULTS BEFORE USING THEM
            results = deduplicate_results(results)
            
            # ============================================================
            # DEBUG: Print worker lookup results
            # ============================================================
            print("=" * 60)
            print("DEBUG WORKER LOOKUP")
            print("=" * 60)
            
            for r in results:
                doc = r["document"]
                print("Image:", doc.get("image"))
                print("Missing:", doc.get("missing"))
                print("Workers:", doc.get("workers"))
                print("-" * 40)
            
            # ============================================================
            # END DEBUG
            # ============================================================
            
            # Detect which PPE the user is asking about
            ppe_target = None
            ppe_list = ["helmet", "gloves", "vest", "boots", "goggles"]
            
            for item in ppe_list:
                if item in question_lower:
                    ppe_target = item
                    break
            
            # Default to helmet if no specific PPE mentioned
            if ppe_target is None:
                ppe_target = "helmet"
            
            # Display name for the PPE
            ppe_display = ppe_target.capitalize()
            
            seen_entries = set()
            report_lines = []
            
            for r in results:
                doc = r["document"]
                image = doc.get("image", "Unknown")
                site = doc.get("site_name", "Unknown")
                date = doc.get("date", "")
                missing = doc.get("missing", [])
                
                # Use RiskAssessmentAgent
                assessment = risk_agent.assess(doc)
                risk_level = assessment["risk_level"]
                recommendations = assessment["recommendations"]
                
                # Use ReportAgent
                inspection_report = report_agent.generate(doc, risk_level, recommendations)
                
                # Check individual workers first
                workers = doc.get("workers", [])
                
                if workers:
                    # If we have individual worker data, use it
                    for worker in workers:
                        worker_missing = [
                            m.strip().lower()
                            for m in worker.get("missing", [])
                        ]
                        
                        worker_id = worker.get("worker_id", "Unknown")
                        entry_key = f"{image}_{worker_id}"
                        
                        if entry_key in seen_entries:
                            continue
                        seen_entries.add(entry_key)
                        
                        if ppe_target in worker_missing:
                            print(">>> INDIVIDUAL WORKER PPE MATCH FOUND <<<")
                            report_lines.append(f"""
📋 Inspection Report: {image}

{inspection_report}

----------------------------------
""")
                else:
                    # Fallback: use image-level missing PPE
                    if image in seen_entries:
                        continue
                    seen_entries.add(image)
                    
                    if ppe_target in [m.lower() for m in missing]:
                        print(">>> IMAGE-LEVEL PPE MATCH FOUND <<<")
                        report_lines.append(f"""
📋 Inspection Report: {image}

{inspection_report}

----------------------------------
""")
            
            if report_lines:
                answer = f"Inspection reports containing workers missing {ppe_display}:\n\n"
                answer += "\n".join(report_lines)
            else:
                answer = f"No workers missing {ppe_display} were found in the retrieved inspection reports."
            
            # DEDUPLICATE SOURCES WITH REPORT DATA
            unique_sources = []
            seen_images = set()
            
            for r in results:
                doc = r["document"]
                image = doc.get("image")
                
                if image in seen_images:
                    continue
                
                seen_images.add(image)
                
                missing = doc.get("missing", [])
                
                # Use RiskAssessmentAgent
                assessment = risk_agent.assess(doc)
                risk_level = assessment["risk_level"]
                recommendations = assessment["recommendations"]
                
                # Use ReportAgent
                inspection_report = report_agent.generate(doc, risk_level, recommendations)
                
                unique_sources.append({
                    "type": doc.get("type"),
                    "image": image,
                    "status": doc.get("status"),
                    "missing": missing,
                    "risk_level": risk_level,
                    "recommendations": recommendations,
                    "inspection_report": inspection_report,
                    "date": doc.get("date"),
                    "site_name": doc.get("site_name"),
                    "compliance_rate": doc.get("compliance_rate"),
                    "total_workers": doc.get("total_workers"),
                    "workers": doc.get("workers", []),
                    "score": r.get("score", 0.0),
                    "confidence": r.get("confidence", "Unknown")
                })
            
            sources = unique_sources
            
            return {
                "mode": "worker_lookup",
                "answer": answer,
                "sources": sources,
                "source_names": [s.get("image") for s in sources if s.get("image")]
            }
        
        # Prevention branch - retrieves both inspections and manuals
        elif any(word in question_lower for word in prevention_keywords):
            print("📚 Prevention Question - retrieving inspections + manuals")
            
            # Use RetrievalAgent for inspections
            inspection_results = retrieval_agent.retrieve(
                query=question,
                top_k=3,
                document_type="inspection",
                search_scope="all"
            )
            
            # DEDUPLICATE RESULTS
            inspection_results = deduplicate_results(inspection_results)
            
            # Use RetrievalAgent for manuals
            manual_results = retrieval_agent.retrieve(
                query=question,
                top_k=2,
                document_type="manual",
                search_scope=search_scope,
                selected_manuals=selected_manuals
            )
            
            # DEDUPLICATE RESULTS
            manual_results = deduplicate_results(manual_results)
            
            # Print retrieval counts
            print("=" * 60)
            print("Inspection Results :", len(inspection_results))
            print("Manual Results     :", len(manual_results))
            print("=" * 60)
            
            # Don't sort again - preserve 3+2 mix
            results = inspection_results + manual_results
            
            # DEDUPLICATE RESULTS
            results = deduplicate_results(results)
            
        # Check if query contains inspection keywords
        elif any(word in question_lower for word in inspection_keywords):
            print(f"   🖼️  Detected inspection-related keywords - searching inspection reports")
            results = retrieval_agent.retrieve(
                query=question,
                top_k=20,
                document_type="inspection",
                search_scope=search_scope,
                selected_manuals=selected_manuals
            )
            # DEDUPLICATE RESULTS
            results = deduplicate_results(results)
            
        elif any(word in question_lower for word in manual_keywords):
            print(f"   📄 Detected manual-related keywords - searching manuals only")
            results = retrieval_agent.retrieve(
                query=question,
                top_k=8,
                document_type="manual",
                search_scope=search_scope,
                selected_manuals=selected_manuals
            )
            # DEDUPLICATE RESULTS (manuals don't have images, but keep for consistency)
            results = deduplicate_results(results)
            
        else:
            print(f"   📚 No specific keywords detected - searching both")
            results = retrieval_agent.retrieve(
                query=question,
                top_k=20,
                document_type=None,  # Search both
                search_scope=search_scope,
                selected_manuals=selected_manuals
            )
            # DEDUPLICATE RESULTS
            results = deduplicate_results(results)
        
        header = ""

    # ============================================================
    # Skip LLM prompt for worker lookup (already returned above)
    # ============================================================
    # If we're here, we need to use the LLM for reasoning
    
    print(f"   ✅ Retrieved: {len(results)} documents")
    print("-" * 60)

    if not results:
        return {
            "mode": mode,
            "answer": "No relevant reports were found.",
            "sources": []
        }

    # Sort by confidence
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Dynamic context size
    if mode == "filtered_reports":
        results = results[:10]
    elif any(k in question.lower() for k in ["manual", "osha", "summary"]):
        results = results[:2]
    else:
        results = results[:5]
    
    print(f"📄 Using {len(results)} results for LLM context")

    # --------------------------------------------------
    # Build Context
    # --------------------------------------------------
    context_parts = []

    for i, r in enumerate(results, 1):
        doc = r["document"]
        source = doc.get('filename') or doc.get('image') or 'Unknown'
        
        context_parts.append(f"""
====================================================

DOCUMENT {i}

Document Type:
{doc.get("type")}

Source:
{source}

Site:
{doc.get("site_name","N/A")}

Status:
{doc.get("status","N/A")}

Compliance:
{doc.get("compliance_rate","N/A")}

Missing PPE:
{", ".join(doc.get("missing", []))}

Content
{doc["text"]}

====================================================
""")

    context = "\n\n".join(context_parts)

    # Prepend header if in filtered reports mode
    if mode == "filtered_reports":
        context = header + "\n\n" + context

    # --------------------------------------------------
    # Extract source names - DEDUPLICATED WITH REPORT DATA
    # --------------------------------------------------
    unique_sources = []
    seen_images = set()
    
    for r in results:
        doc = r["document"]
        image = doc.get("image")
        
        # For manuals, use filename as the key
        if doc.get("type") == "manual":
            key = doc.get("filename")
        else:
            key = image
        
        if key in seen_images:
            continue
        
        seen_images.add(key)
        
        missing = doc.get("missing", [])
        
        # Use RiskAssessmentAgent
        assessment = risk_agent.assess(doc)
        risk_level = assessment["risk_level"]
        recommendations = assessment["recommendations"]
        
        # Use ReportAgent
        inspection_report = report_agent.generate(doc, risk_level, recommendations)
        
        unique_sources.append({
            "type": doc.get("type"),
            "filename": doc.get("filename"),
            "document_type": doc.get("document_type"),
            "chunk": doc.get("chunk"),
            "total_chunks": doc.get("total_chunks"),
            "image": image,
            "status": doc.get("status"),
            "missing": missing,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "inspection_report": inspection_report,
            "date": doc.get("date"),
            "site_name": doc.get("site_name"),
            "compliance_rate": doc.get("compliance_rate"),
            "total_workers": doc.get("total_workers"),
            "safe_workers": doc.get("safe_workers"),
            "unsafe_workers": doc.get("unsafe_workers"),
            "workers": doc.get("workers", []),
            "score": r.get("score", 0.0),
            "confidence": r.get("confidence", "Unknown")
        })

    sources = unique_sources

    # Extract source names for the prompt
    source_names = []
    for s in sources:
        if s.get("filename"):
            source_names.append(s["filename"])
        elif s.get("image"):
            source_names.append(s["image"])
    
    # Remove duplicates cleanly
    clean_sources = []
    for s in source_names:
        if s not in clean_sources:
            clean_sources.append(s)
    source_names = clean_sources
    
    # Create source text for the prompt
    source_text = "\n".join(source_names) if source_names else "No sources available"

    # --------------------------------------------------
    # LLM Prompt - Using ResponseAgent
    # --------------------------------------------------
    prompt = f"""
You are SAFESIGHT AI, an expert construction safety assistant.

You MUST answer ONLY from the retrieved context.

=========================
RULES
=========================

1. Never invent workers, worker IDs, counts, PPE items, locations or incidents.

2. Only say "all inspection reports" if EVERY retrieved report explicitly supports that statement.

3. If the evidence is limited, use wording like:
   - "The retrieved reports indicate..."
   - "The available evidence suggests..."
   - "Based on the retrieved inspection reports..."

4. Separate FACTS from RECOMMENDATIONS.

5. Recommendations from manuals ONLY:
   If one or more manuals are retrieved:
   ONLY use those manuals to generate recommendations.
   Always mention the manual filename.
   
   If no manuals are retrieved:
   write exactly:
   "No relevant safety manual was retrieved."
   
   Do NOT generate OSHA recommendations unless a manual is present.

6. Never guess information that is not present.

7. Always cite the source filenames from the source list below.

8. If the retrieved context does not explicitly contain a fact,
reply that the information is not available instead of assuming it.

=========================
RETRIEVED SOURCES
=========================

{source_text}

=========================
RETRIEVED CONTEXT
=========================

{context}

=========================
QUESTION
=========================

{question}

=========================
OUTPUT FORMAT - Return EXACTLY this markdown
=========================

## Observation

Do not mention the number of reports unless it is explicitly provided.
Use phrases such as "The retrieved inspection reports indicate..." instead of "All three inspections..."

## Recommendation

If one or more manuals are retrieved:
Provide recommendations using ONLY the retrieved manuals.
Always mention the manual filename.

If no manuals are retrieved:
Write exactly:
"No relevant safety manual was retrieved."

## Evidence

Mention the important findings from the retrieved reports.

## Sources

List every source filename used from the source list above.
"""

    # Use ResponseAgent instead of direct LLM call
    response = response_agent.answer(prompt)

    # Cleaner logging
    print("=" * 60)
    print(f"Mode: {mode} | Retrieved: {len(results)} | Question: {question[:50]}...")
    print(f"Sources: {', '.join(source_names)}")
    print("=" * 60)

    # Return response directly (not response.content)
    return {
        "mode": mode,
        "answer": response,
        "sources": sources,
        "source_names": source_names
    }


# --------------------------------------------------
# Standalone Testing - UPDATED
# --------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("        SAFESIGHT RAG Assistant")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        question = input("\nAsk: ")

        if question.lower() == "exit":
            break

        # Test with different search scopes
        print("\nSearch Scope Options:")
        print("  1. all (search everything)")
        print("  2. selected (search only selected manuals)")
        choice = input("Choose scope (1 or 2, default=1): ").strip()
        
        if choice == "2":
            # Show available manuals (simplified for testing)
            print("\nAvailable Manuals:")
            for doc in ALL_DOCS:
                if doc.get("type") == "manual":
                    print(f"  - {doc.get('filename', 'Unknown')}")
            
            manual_input = input("\nEnter manual filenames (comma separated): ").strip()
            selected_manuals = [m.strip() for m in manual_input.split(",") if m.strip()]
            search_scope = "selected"
        else:
            selected_manuals = []
            search_scope = "all"

        result = ask_question(
            question=question,
            search_scope=search_scope,
            selected_manuals=selected_manuals
        )

        print("\n" + "=" * 60)
        print(f"Mode: {result.get('mode', 'unknown')}")
        print("Answer:\n")
        print(result["answer"])

        if result.get("sources"):
            print("\n" + "=" * 60)
            print("Retrieved Sources:\n")
            for source in result["sources"]:
                if source.get("type") == "manual":
                    print(f"📄 Manual: {source.get('filename', 'Unknown')}")
                    print(f"   Type: {source.get('document_type', 'Unknown')}")
                    print(f"   Chunk: {source.get('chunk', '?')}/{source.get('total_chunks', '?')}")
                else:
                    print(f"🖼️  Image: {source.get('image', 'Unknown')}")
                    print(f"   Status: {source.get('status', 'Unknown')}")
                    print(f"   Missing: {', '.join(source.get('missing', [])) if source.get('missing') else 'None'}")
                    print(f"   Risk Level: {source.get('risk_level', 'Unknown')}")
                    print(f"   Recommendations:")
                    for rec in source.get('recommendations', []):
                        print(f"     - {rec}")
                    print(f"   Site: {source.get('site_name', 'Unknown')}")
                    print(f"   Date: {source.get('date', 'Unknown')}")
                    print(f"   Compliance: {source.get('compliance_rate', 'N/A')}%")
                    print(f"   Workers: {source.get('total_workers', 'N/A')}")
                    print(f"   Full Report:\n{source.get('inspection_report', 'No report generated')}")
                    if source.get("workers"):
                        print(f"   Individual Workers:")
                        for w in source.get("workers", []):
                            print(f"     - Worker {w.get('worker_id', 'Unknown')}: Missing {', '.join(w.get('missing', [])) if w.get('missing') else 'None'}")
                print(f"   Score: {source.get('score', 0.0):.4f}")
                print(f"   Confidence: {source.get('confidence', 'Unknown')}")
                print("-" * 40)

        print()