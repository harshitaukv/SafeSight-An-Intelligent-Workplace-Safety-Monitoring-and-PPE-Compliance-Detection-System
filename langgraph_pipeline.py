from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import os
import time

from agents import (
    RetrievalAgent,
    RiskAssessmentAgent,
    ManualAgent,
    ReportAgent,
    ResponseAgent,
    EvidenceAgent,
)

from similarity_search import search
from risk_engine import assess_risk, get_recommendations
from violation_checker import compute_compliance_rate
from llm_provider import build_llm, describe as describe_llm
from bson import ObjectId
from bson.errors import InvalidId

# ✅ Import the correct MongoDB collection
from database import detections


# --------------------------------------------------
# Initialize the LLM
# --------------------------------------------------
# Which model runs — local Ollama or a hosted API — is decided entirely by
# environment variables in llm_provider.py, so the same checkout can run
# fully offline on one machine and against a fast hosted endpoint on
# another with no code change. Defaults to local Ollama exactly as before.
# See README.md for the variables.
llm = build_llm()

_llm_info = describe_llm()
print(f"LLM: {_llm_info['model']} via {_llm_info['provider']}")


# --------------------------------------------------
# Initialize your existing agents
# --------------------------------------------------
retrieval_agent = RetrievalAgent(search)

risk_agent = RiskAssessmentAgent(
    assess_risk,
    get_recommendations,
)

manual_agent = ManualAgent()

report_agent = ReportAgent(llm)

response_agent = ResponseAgent(llm)

# ✅ Evidence Agent for extracting key facts
evidence_agent = EvidenceAgent(llm)


# --------------------------------------------------
# Define the LangGraph state
# --------------------------------------------------
class SafeSightState(TypedDict):
    question: str
    search_scope: str
    selected_manuals: List[str]
    mode: str
    results: list
    reports: list
    answer: str
    sources: list
    source_names: list
    summary: dict
    has_inspections: bool
    has_manuals: bool
    # ✅ Updated: Structured evidence and metrics
    evidence: Dict[str, Any]
    metrics: Dict[str, Any]
    is_summary_request: bool  # ✅ Track if this is a summary request
    manual_metadata: List[Dict[str, Any]]  # ✅ Manual metadata


# --------------------------------------------------
# Helper: Build Manual Context
# --------------------------------------------------
def build_manual_context(results):
    """
    Build context from retrieved manual chunks only.
    """
    context = []

    for r in results:
        doc = r.get("document", {})

        if doc.get("type") != "manual":
            continue

        context.append(
            f"""
Filename:
{doc.get("filename")}

Chunk:
{doc.get("chunk")}

Content:
{doc.get("text", "")}
"""
        )

    return "\n\n".join(context)


# --------------------------------------------------
# Node 1: Intent Detection - Fast Rule-Based Routing
# --------------------------------------------------
def intent_node(state: SafeSightState) -> dict:
    """
    Detect the user's intent using fast rule-based routing first,
    then LLM only for ambiguous queries.
    """
    print("🧠 [LangGraph] Detecting intent...")

    question = state["question"].lower().strip()

    # ---------- Fast Rule-Based Routing ----------
    
    # ✅ Manual summary should go to manual_search
    if (
        "manual" in question
        or "manuals" in question
        or "osha" in question
        or "document" in question
    ) and (
        "summary" in question
        or "summarize" in question
        or "overview" in question
        or "brief" in question
    ):
        print("📄 Rule matched → manual_search (manual summary)")
        return {
            "mode": "manual_search",
            "is_summary_request": True,
        }

    # ✅ Inspection summary
    if any(x in question for x in [
        "inspection summary",
        "site summary",
        "today summary",
        "overall summary",
        "give me summary",
        "summarize inspections",
        "overview",
        "brief",
    ]):
        print("📋 Rule matched → inspection_search (summary request)")
        return {
            "mode": "inspection_search",
            "is_summary_request": True,
        }

    # Analytics requests → analytics
    if any(x in question for x in [
        "dashboard",
        "analytics",
        "statistics",
        "graph",
        "chart",
        "overall compliance"
    ]):
        print("📊 Rule matched → analytics")
        return {"mode": "analytics", "is_summary_request": False}

    # Worker lookup requests → worker_lookup
    if any(x in question for x in [
        "worker",
        "employee",
        "person",
        "who"
    ]):
        print("👷 Rule matched → worker_lookup")
        return {"mode": "worker_lookup", "is_summary_request": False}

    # Manual requests → manual_search
    if any(x in question for x in [
        "manual",
        "osha",
        "guide",
        "policy",
        "document"
    ]):
        print("📄 Rule matched → manual_search")
        return {"mode": "manual_search", "is_summary_request": False}

    # Prevention requests → prevention
    if any(x in question for x in [
        "prevent",
        "avoid",
        "reduce risk",
        "how can"
    ]):
        print("📚 Rule matched → prevention")
        return {"mode": "prevention", "is_summary_request": False}

    # ---------- Only ambiguous questions reach the LLM ----------
    print("🧠 [LangGraph] Ambiguous query → using LLM for intent classification")
    
    prompt = f"""
You are an intent classifier for a construction safety assistant.

Return ONLY one of these exact values:

analytics
inspection_search
manual_search
worker_lookup
prevention

Classification Rules:

analytics:
- dashboard
- analytics
- statistics
- compliance statistics
- charts
- graphs
- trends
- KPIs
- overall metrics

inspection_search:
- summary
- summarize
- give me summary
- inspection summary
- today's summary
- latest inspection
- site status
- what happened
- show inspections

manual_search:
- manual
- SOP
- guideline
- OSHA
- policy
- documentation

worker_lookup:
- worker
- employee
- person
- missing PPE
- identify worker

prevention:
- prevent
- prevention
- avoid
- reduce risk
- improve safety

If the user asks for a SUMMARY of inspections,
ALWAYS return:

inspection_search

NOT analytics.

User Question:
{state["question"]}

Think carefully.

If the user is asking for a summary of inspection reports,
classify it as:

inspection_search

Return only the intent.
"""
    
    response = llm.invoke(prompt)
    mode = response.content.strip().lower()
    
    # Validate the response
    valid_modes = ["analytics", "inspection_search", "manual_search", "worker_lookup", "prevention"]
    
    if mode not in valid_modes:
        print(f"⚠️ [LangGraph] Unknown intent: {mode}, defaulting to inspection_search")
        mode = "inspection_search"
    
    # Check if it's a summary request (for LLM-classified queries too)
    question = state["question"].lower()
    is_summary = any(k in question for k in [
        "summary", "summarize", "give me summary", "inspection summary", "site summary", "overall summary",
        "overview", "brief"
    ])
    
    print(f"🧠 [LangGraph] Intent detected: {mode}")
    return {"mode": mode, "is_summary_request": is_summary}


# --------------------------------------------------
# Node 2: Analytics (handled by API)
# --------------------------------------------------
def analytics_node(state: SafeSightState) -> dict:
    """
    Computes real compliance analytics from the detection_history collection
    and returns a natural-language summary.
    """
    print("📊 [LangGraph] Computing analytics from MongoDB")

    docs = list(detections.find())

    if not docs:
        return {
            "answer": (
                "There's no inspection data yet. Run a PPE detection from the "
                "Upload Center to start building compliance analytics."
            ),
            "mode": "analytics",
        }

    totals = {
        "total_inspections": len(docs),
        "total_images": 0,
        "total_workers": 0,
        "safe_workers": 0,
        "unsafe_workers": 0,
    }
    violations = {"helmet": 0, "vest": 0, "gloves": 0, "boots": 0, "goggles": 0}
    compliance_sum = 0.0

    for doc in docs:
        summary = doc.get("summary", {}) or {}
        totals["total_images"] += summary.get("total_images", 0)
        totals["total_workers"] += summary.get("total_workers", 0)
        totals["safe_workers"] += summary.get("safe_workers", 0)
        totals["unsafe_workers"] += summary.get("unsafe_workers", 0)
        compliance_sum += summary.get("compliance", 0) or 0
        for key in violations:
            violations[key] += summary.get(key, 0)

    avg_compliance = round(compliance_sum / totals["total_inspections"], 1)
    top_violation = max(violations, key=violations.get)
    top_violation_count = violations[top_violation]

    answer = (
        f"**Overall Safety Summary**\n\n"
        f"- Inspections logged: {totals['total_inspections']}\n"
        f"- Images processed: {totals['total_images']}\n"
        f"- Workers assessed: {totals['total_workers']}\n"
        f"- Safe: {totals['safe_workers']}  ·  Unsafe: {totals['unsafe_workers']}\n"
        f"- Average compliance rate: {avg_compliance}%\n"
        f"- Most common violation: {top_violation.capitalize()} "
        f"({top_violation_count} occurrences)\n\n"
        f"Violation breakdown — Helmet: {violations['helmet']}, "
        f"Vest: {violations['vest']}, Gloves: {violations['gloves']}, "
        f"Boots: {violations['boots']}, Goggles: {violations['goggles']}"
    )

    return {
        "answer": answer,
        "mode": "analytics",
        "summary": {
            "totals": totals,
            "avg_compliance": avg_compliance,
            "violation_breakdown": violations,
        },
    }


# --------------------------------------------------
# Node 3: Worker Lookup
# --------------------------------------------------
def worker_lookup_node(state: SafeSightState) -> dict:
    """
    Retrieve inspection reports and identify workers missing PPE.
    """
    start_time = time.perf_counter()
    print("👷 [LangGraph] Running worker lookup...")
    
    results = retrieval_agent.retrieve(
        query=state["question"],
        top_k=5,
        document_type="inspection",
        search_scope=state["search_scope"],
        selected_manuals=state["selected_manuals"]
    )
    
    elapsed = time.perf_counter() - start_time
    print(f"👷 [LangGraph] Worker lookup returned {len(results)} results ({elapsed:.2f}s)")
    
    return {
        "results": results,
        "mode": "worker_lookup",
        "has_inspections": True,
        "has_manuals": False
    }


# --------------------------------------------------
# Node 4: Prevention Search
# --------------------------------------------------
def prevention_search_node(state: SafeSightState) -> dict:
    """
    Retrieve both inspection reports and manuals for prevention questions.
    """
    start_time = time.perf_counter()
    print("📚 [LangGraph] Running prevention search...")
    
    inspection_results = retrieval_agent.retrieve(
        query=state["question"],
        top_k=3,
        document_type="inspection",
        search_scope="all"
    )
    
    manual_results = retrieval_agent.retrieve(
        query=state["question"],
        top_k=2,
        document_type="manual",
        search_scope=state["search_scope"],
        selected_manuals=state["selected_manuals"]
    )
    
    results = inspection_results + manual_results
    
    elapsed = time.perf_counter() - start_time
    print(f"📚 [LangGraph] Prevention search returned {len(results)} results ({elapsed:.2f}s)")
    
    return {
        "results": results,
        "mode": "prevention",
        "has_inspections": bool(inspection_results),
        "has_manuals": bool(manual_results)
    }


# --------------------------------------------------
# Node 5: Manual Search (with Summary Support and Sources)
# --------------------------------------------------
def manual_search_node(state: SafeSightState) -> dict:
    """
    Retrieve manual documents and preserve all metadata.
    For summary requests, retrieve ALL manuals (top_k=100).
    """
    start_time = time.perf_counter()
    print("📄 [LangGraph] Running manual search...")
    
    # ✅ For summary requests, retrieve all manuals. Otherwise limit to 5.
    top_k = 100 if state.get("is_summary_request") else 5
    
    results = retrieval_agent.retrieve(
        query=state["question"],
        top_k=top_k,
        document_type="manual",
        search_scope=state["search_scope"],
        selected_manuals=state["selected_manuals"]
    )
    
    # Extract just the documents for ManualAgent
    docs = [r.get("document", {}) for r in results]
    
    # Use ManualAgent for any additional processing
    processed_docs = manual_agent.extract(docs)
    
    # Rebuild results with original scores and confidence
    processed_results = []
    for processed_doc in processed_docs:
        for r in results:
            if r.get("document", {}) == processed_doc:
                processed_results.append({
                    "document": processed_doc,
                    "score": r.get("score", 1.0),
                    "confidence": r.get("confidence", "High")
                })
                break
    
    elapsed = time.perf_counter() - start_time
    print(f"📄 [LangGraph] Manual search returned {len(processed_results)} results ({elapsed:.2f}s)")
    
    # ✅ Build sources from manual results
    source_names = []
    sources = []
    manual_metadata = []

    for r in processed_results:
        doc = r["document"]

        source_names.append(doc.get("filename"))

        sources.append({
            "type": "manual",
            "filename": doc.get("filename"),
            "document_type": doc.get("document_type"),
            "text": doc.get("text"),
            "score": r.get("score"),
            "confidence": r.get("confidence")
        })

        # ✅ Manual metadata for better tracking
        manual_metadata.append({
            "filename": doc.get("filename"),
            "chunk": doc.get("chunk"),
            "score": r["score"]
        })

    return {
        "results": processed_results,
        "mode": "manual_search",
        "has_inspections": False,
        "has_manuals": True,
        "sources": sources,
        "source_names": list(dict.fromkeys(source_names)),
        "manual_metadata": manual_metadata
    }


# --------------------------------------------------
# Node 6: Inspection Search
# --------------------------------------------------
def inspection_search_node(state: SafeSightState) -> dict:
    """
    Retrieve only inspection reports using FAISS.
    Summary mode is now handled by similarity_search.py.
    """
    start_time = time.perf_counter()
    print("🖼️ [LangGraph] Running inspection search...")
    
    question = state["question"].lower()
    is_summary = state.get("is_summary_request", False)
    
    # ✅ Always use FAISS search - summary mode is now handled by similarity_search.py
    results = retrieval_agent.retrieve(
        query=state["question"],
        top_k=5,
        document_type="inspection",
        search_scope=state["search_scope"],
        selected_manuals=state["selected_manuals"]
    )
    
    elapsed = time.perf_counter() - start_time
    print(f"🖼️ [LangGraph] Inspection search returned {len(results)} results ({elapsed:.2f}s)")
    
    return {
        "results": results,
        "mode": "inspection_search",
        "has_inspections": True,
        "has_manuals": False,
        "is_summary_request": is_summary or any(k in question for k in ["summary", "summarize", "give me summary", "inspection summary", "site summary", "overall summary"])
    }


# --------------------------------------------------
# Node 7: Deduplicate Results
# --------------------------------------------------
def deduplicate_node(state: SafeSightState) -> dict:
    """
    Deduplicate results by image and filename.
    """
    start_time = time.perf_counter()
    print("🔍 [LangGraph] Deduplicating results...")
    
    unique_results = []
    seen = set()
    
    for r in state["results"]:
        doc = r.get("document", {})
        
        if doc.get("type") == "manual":
            key = ("manual", doc.get("filename"))
        else:
            # Handle MongoDB documents
            if "images" in doc:
                key = ("mongo", str(doc.get("_id")))
            else:
                key = ("inspection", doc.get("image"))
        
        if key in seen:
            continue
        
        seen.add(key)
        unique_results.append(r)
    
    elapsed = time.perf_counter() - start_time
    print(f"📊 [LangGraph] Deduplicated to {len(unique_results)} unique results ({elapsed:.2f}s)")
    return {"results": unique_results}


# --------------------------------------------------
# Compliance healing for indexed documents
# --------------------------------------------------
# Cleared at the start of every report-generation run: an inspection's
# worker data can change if the video is re-processed, and this only needs
# to deduplicate lookups within a single answer.
_compliance_cache = {}


def _healed_compliance(doc):
    """
    The true compliance % for an indexed inspection document.

    Documents indexed before the compliance formula was unified store a
    safe_workers/total_workers figure. On a single-worker image that is
    binary — a worker missing only their goggles reads as 0% compliant
    despite wearing four of five items — and it's what the AI Assistant's
    report cards were showing.

    Those older documents don't carry a `workers` list either, so there's
    nothing local to recompute from. The document does keep the MongoDB
    inspection id and the image name, so the worker data is fetched from
    the detections collection and compliance recomputed with the shared
    formula. That heals existing indexes in place: no rebuild required, and
    newly indexed documents (which now embed `workers`) skip the lookup
    entirely.

    Any failure here falls back to the stored value — a wrong-looking
    percentage is a far better outcome than a chat request that 500s.
    """
    workers = doc.get("workers")
    if workers:
        return compute_compliance_rate(workers)

    stored = doc.get("compliance_rate")
    inspection_id = doc.get("_id")
    image_name = doc.get("image")
    if not inspection_id or not image_name:
        return stored

    cache_key = (str(inspection_id), image_name)
    if cache_key in _compliance_cache:
        return _compliance_cache[cache_key]

    healed = stored
    try:
        record = detections.find_one({"_id": ObjectId(str(inspection_id))})
        if record:
            for image in record.get("images", []):
                if image.get("name") == image_name and image.get("workers"):
                    healed = compute_compliance_rate(image["workers"])
                    break
    except (InvalidId, TypeError, KeyError) as err:
        print(f"⚠️ Could not heal compliance for {image_name}: {err}")
    except Exception as err:  # pragma: no cover - defensive
        print(f"⚠️ Unexpected error healing compliance for {image_name}: {err}")

    _compliance_cache[cache_key] = healed
    return healed


# --------------------------------------------------
# Node 8: Generate Reports (with Summary Support and inspection_id)
# --------------------------------------------------
def generate_reports_node(state: SafeSightState) -> dict:
    """
    Generate inspection reports for each inspection result.
    For summaries, process ALL results. Otherwise limit to 5.
    """
    _compliance_cache.clear()
    start_time = time.perf_counter()
    print("📋 [LangGraph] Generating inspection reports...")
    
    reports = []
    sources = []
    source_names = []
    
    # ✅ For summary requests, process ALL results. Otherwise limit to 5.
    if state.get("is_summary_request"):
        reports_to_process = state["results"]
    else:
        reports_to_process = state["results"][:5]
    
    for r in reports_to_process:
        doc = r.get("document", {})
        
        # ========== REPORT DEBUG ==========
        print("\n========== REPORT DEBUG ==========")
        print("Image:", doc.get("image"))
        print("_id:", doc.get("_id"))
        print("Keys:", list(doc.keys()))
        print("==================================")
        # ==================================
        
        # Handle MongoDB summary documents with "images" array
        if "images" in doc:
            for image in doc["images"]:
                # Self-heal older records: recompute from this image's own
                # worker data instead of trusting a possibly-stale stored
                # compliance_rate (older records used a safe/total-workers
                # formula that read as a binary 0%/100% for single-worker
                # images — see compute_compliance_rate).
                workers = image.get("workers", [])
                healed_compliance = (
                    compute_compliance_rate(workers) if workers else image.get("compliance_rate")
                )

                # Build inspection document from MongoDB data
                inspection_doc = {
                    "type": "inspection",
                    "image": image.get("name"),
                    "site_name": image.get("site_name"),
                    "workers": workers,
                    "missing": image.get("missing", []),
                    "status": image.get("status"),
                    "compliance_rate": healed_compliance,
                    "safe_workers": image.get("safe_workers", 0),
                    "unsafe_workers": image.get("unsafe_workers", 0),
                    "total_workers": image.get("total_workers", 0),
                    "date": doc.get("timestamp")
                }

                # Assess risk
                assessment = risk_agent.assess(inspection_doc)
                risk_level = assessment["risk_level"]
                recommendations = assessment["recommendations"]

                # Generate report
                inspection_report = report_agent.generate(
                    inspection_doc,
                    risk_level,
                    recommendations
                )

                reports.append(inspection_report)

                # ✅ Change 1: Build source with inspection_id from MongoDB
                source = {
                    "inspection_id": str(doc.get("_id")),  # ✅ NEW

                    **inspection_doc,

                    "risk_level": risk_level,
                    "recommendations": recommendations,
                    "inspection_report": inspection_report,

                    "score": r.get("score", 0.0),
                    "confidence": r.get("confidence", "Unknown")
                }

                sources.append(source)
                source_names.append(image.get("name"))

            continue

        # Existing FAISS logic for inspection documents
        if doc.get("type") != "inspection":
            continue
        
        # Assess risk
        assessment = risk_agent.assess(doc)
        risk_level = assessment["risk_level"]
        recommendations = assessment["recommendations"]
        
        # Generate report
        inspection_report = report_agent.generate(doc, risk_level, recommendations)
        
        # ✅ Change 2: Build source with inspection_id (with fallback)
        source = {
            "inspection_id": str(doc.get("_id", "")),  # ✅ NEW (with fallback)

            "type": doc.get("type"),
            "image": doc.get("image"),
            "status": doc.get("status"),
            "missing": doc.get("missing", []),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "inspection_report": inspection_report,
            "date": doc.get("date"),
            "site_name": doc.get("site_name"),
            "compliance_rate": _healed_compliance(doc),
            "total_workers": doc.get("total_workers"),
            "workers": doc.get("workers", []),
            "score": r.get("score", 0.0),
            "confidence": r.get("confidence", "Unknown")
        }
        
        sources.append(source)
        
        if doc.get("image"):
            source_names.append(doc.get("image"))
        
        reports.append(inspection_report)
    
    # Also include manuals as sources
    for r in state["results"][:5]:
        doc = r.get("document", {})
        if doc.get("type") == "manual":
            sources.append({
                "type": "manual",
                "filename": doc.get("filename"),
                "document_type": doc.get("document_type"),
                "chunk": doc.get("chunk"),
                "total_chunks": doc.get("total_chunks"),
                "score": r.get("score", 0.0),
                "confidence": r.get("confidence", "Unknown")
            })
            if doc.get("filename"):
                source_names.append(doc.get("filename"))
    
    # Remove duplicate source names
    source_names = list(dict.fromkeys(source_names))
    
    elapsed = time.perf_counter() - start_time
    print(f"📊 [LangGraph] Generated {len(reports)} reports with {len(sources)} sources ({elapsed:.2f}s)")
    return {
        "reports": reports,
        "sources": sources,
        "source_names": source_names
    }


# --------------------------------------------------
# Helper: Build structured evidence WITHOUT an LLM call
# --------------------------------------------------
_RISK_RANK = {"SAFE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _build_evidence_from_results(evidence_results):
    """
    Deterministically builds the same evidence shape the EvidenceAgent used
    to produce via a second LLM call (observations / violations / risk /
    key findings), plus the aggregate stats (safe/unsafe worker totals,
    average & overall compliance) that used to be computed by a separate
    loop right below in evidence_node.

    Everything here is already present in structured form in the retrieved
    MongoDB/FAISS documents — missing PPE, compliance, worker counts — so
    asking an LLM to "extract" it back out of freeform text was a full
    extra model round-trip for no real gain in accuracy. Removing it
    roughly halves response time for every inspection-related question,
    since generate_answer_node's own call is now the only LLM call left in
    the non-manual, non-analytics path.

    The old separate stats loop only counted documents with
    doc.get("type") == "inspection" — which no real MongoDB record ever
    has (real records are the "images"-array summary-document shape), so
    it silently summed to zero for every actual question and fed the LLM
    "Overall Compliance: 0%" regardless of the real data. Folding the stats
    into this same per-image loop (which already handles both document
    shapes correctly) fixes that.

    Per-image compliance is recomputed via compute_compliance_rate from
    each image's own worker data rather than trusted from a possibly-stale
    stored compliance_rate field — see violation_checker.compute_compliance_rate.
    """
    observations = []
    violations = []

    worst_risk = "SAFE"
    total_safe = 0
    total_unsafe = 0
    compliance_sum = 0.0
    inspection_count = 0

    for r in evidence_results:
        doc = r.get("document", {})

        if "images" in doc:
            image_list = doc["images"]
        elif doc.get("type") == "inspection":
            image_list = [doc]
        else:
            continue

        for image in image_list:
            site = image.get("site_name") or "Unknown site"
            missing = image.get("missing", [])
            workers = image.get("workers", [])
            compliance = compute_compliance_rate(workers) if workers else image.get("compliance_rate", 0) or 0
            safe = image.get("safe_workers", 0)
            unsafe = image.get("unsafe_workers", 0)

            risk = assess_risk(missing)
            if _RISK_RANK.get(risk, 0) > _RISK_RANK.get(worst_risk, 0):
                worst_risk = risk

            observation = f"{site}: {safe} safe / {unsafe} unsafe workers, {compliance}% compliance"
            if missing:
                observation += f", missing {', '.join(missing)}"
            observations.append(observation)

            if missing:
                violations.append({
                    "type": ", ".join(missing),
                    "details": f"{unsafe} worker(s) missing PPE at {site}",
                    "source": image.get("name", site),
                })

            total_safe += safe
            total_unsafe += unsafe
            compliance_sum += compliance
            inspection_count += 1

    total_workers = total_safe + total_unsafe
    average_compliance = round(compliance_sum / inspection_count, 2) if inspection_count else 0
    overall_compliance = (
        round((total_safe / total_workers) * 100, 2) if total_workers else average_compliance
    )

    evidence = {
        "observations": observations or ["No observations available."],
        "violations": violations,
        "manuals": [],
        "risk_level": worst_risk if evidence_results else "UNKNOWN",
        "key_findings": observations[:5] or ["No key findings."],
        "confidence": "HIGH" if evidence_results else "LOW",
    }

    stats = {
        "total_safe": total_safe,
        "total_unsafe": total_unsafe,
        "total_workers": total_workers,
        "average_compliance": average_compliance,
        "overall_compliance": overall_compliance,
        "inspection_count": inspection_count,
    }

    return evidence, stats


# --------------------------------------------------
# Node 9: Extract Structured Evidence (with Summary Statistics & Debug)
# --------------------------------------------------
def evidence_node(state: SafeSightState) -> dict:
    """
    Build structured evidence + aggregate metrics from the retrieved
    results (see _build_evidence_from_results — no LLM call needed, since
    everything here is already structured data, not freeform text).
    For summaries, process ALL results; otherwise the top 5.
    """
    start_time = time.perf_counter()
    print("📋 [LangGraph] Building structured evidence...")

    # ✅ For summary requests, process ALL results. Otherwise limit to 5.
    if state.get("is_summary_request"):
        evidence_results = state["results"]
    else:
        evidence_results = state["results"][:5]

    evidence, stats = _build_evidence_from_results(evidence_results)

    metrics = {
        "total_observations": len(evidence.get("observations", [])),
        "total_violations": len(evidence.get("violations", [])),
        "total_manuals": len(evidence.get("manuals", [])),
        "risk_level": evidence.get("risk_level", "UNKNOWN"),
        "confidence": evidence.get("confidence", "LOW"),
        "key_findings_count": len(evidence.get("key_findings", [])),
        "overall_compliance": stats["overall_compliance"],
        "average_compliance": stats["average_compliance"],
        "total_safe_workers": stats["total_safe"],
        "total_unsafe_workers": stats["total_unsafe"],
        "total_workers": stats["total_workers"],
        "total_inspections": stats["inspection_count"],
    }

    elapsed = time.perf_counter() - start_time
    print(f"📋 [LangGraph] Structured evidence built ({elapsed:.2f}s)")
    print(f"   Observations: {metrics['total_observations']}")
    print(f"   Violations: {metrics['total_violations']}")
    print(f"   Risk Level: {metrics['risk_level']}")
    print(f"   Overall Compliance: {metrics['overall_compliance']}%")
    print(f"   Total Inspections: {metrics['total_inspections']}")

    return {
        "evidence": evidence,
        "metrics": metrics
    }


# --------------------------------------------------
# Node 10: Generate Final Answer - Complete Manual Support
# --------------------------------------------------
def generate_answer_node(state: SafeSightState) -> dict:
    """
    Generate the final answer using the ResponseAgent with optimized prompt.
    Uses only Evidence + Sources, no full context.
    """
    start_time = time.perf_counter()
    print("🤖 [LangGraph] Generating final answer...")
    
    # If no results, return a helpful message
    if not state.get("results"):
        return {"answer": "No relevant reports were found."}
    
    # Check if this is a summary request
    is_summary = state.get("is_summary_request", False)
    question = state["question"].lower()
    summary_keywords = ["summary", "summarize", "give me summary", "inspection summary", "site summary", "overall summary", "overview", "brief"]
    
    if any(k in question for k in summary_keywords):
        is_summary = True
    
    # Build source text
    source_text = "\n".join(state.get("source_names", [])) if state.get("source_names") else "No sources available"
    
    # ✅ Build manual context for manual searches using helper
    manual_context = ""
    if state["mode"] == "manual_search":
        manual_context = build_manual_context(state["results"])
    
    # ✅ Check if manual context is empty
    if state["mode"] == "manual_search" and not manual_context.strip():
        return {
            "answer": "The uploaded manuals do not contain information related to your question."
        }
    
    # Use structured evidence from EvidenceAgent
    evidence = state.get("evidence", {})
    metrics = state.get("metrics", {})  # ✅ Get metrics
    
    # Format structured evidence for the prompt
    observations = "\n".join([f"  • {obs}" for obs in evidence.get("observations", [])]) if evidence.get("observations") else "  • No observations available"
    
    violations = ""
    for v in evidence.get("violations", []):
        violations += f"  • {v.get('type', 'Unknown')}: {v.get('details', 'No details')} (Source: {v.get('source', 'Unknown')})\n"
    if not violations:
        violations = "  • No violations found"
    
    manuals = "\n".join([f"  • {m.get('name', 'Unknown')}: {m.get('relevant_sections', 'No sections extracted')}" for m in evidence.get("manuals", [])]) if evidence.get("manuals") else "  • No manuals retrieved"
    
    key_findings = "\n".join([f"  • {finding}" for finding in evidence.get("key_findings", [])]) if evidence.get("key_findings") else "  • No key findings"
    
    risk_level = evidence.get("risk_level", "UNKNOWN")
    confidence = evidence.get("confidence", "LOW")
    
    evidence_text = f"""
EVIDENCE SUMMARY
───────────────────────────────────────────────────
Risk Level: {risk_level}
Confidence: {confidence}

OBSERVATIONS:
{observations}

VIOLATIONS:
{violations}

KEY FINDINGS:
{key_findings}

MANUALS RETRIEVED:
{manuals}
"""
    
    # ✅ Summary-specific prompt or regular prompt
    if state["mode"] == "manual_search" and is_summary:
        print("📄 Generating manual summary response...")
        output_format = """
## Manual Summary

For EACH manual provide:

### Manual Name

### Purpose

### Main PPE Requirements

### Important Safety Rules

### Emergency Procedures

### Key Responsibilities

Finally provide

## Overall Summary

Compare all manuals.

Mention similarities.

Mention differences.

## Sources

List filenames.
"""
    elif state["mode"] == "manual_search":
        # ✅ Manual comparison detection
        is_compare = (
            "compare" in question
            or "difference" in question
        )
        
        if is_compare:
            print("📄 Generating manual comparison response...")
            output_format = """
# Manual Comparison

For every manual explain:

- Purpose
- PPE Requirements
- Safety Rules
- Emergency Procedures
- Differences
- Similarities

## Sources
"""
        else:
            print("📄 Generating manual response...")
            output_format = """
## Answer

Provide a clear answer based on the manuals.

## Sources

List the manual filenames.
"""
    elif is_summary:
        print("📊 Generating inspection summary response...")
        output_format = """
## Summary Overview

Provide an overall summary of ALL retrieved inspections.

## Key Statistics

- Total inspections analyzed
- Overall compliance trend
- Most common missing PPE
- Highest observed risk level
- Sites inspected
- Major recurring safety issues

## Key Findings

List the most important findings from the inspections.

## Sources

List all source filenames used.
"""
    else:
        output_format = """
## Observation

Summarize the key findings.

## Recommendation

If manuals are listed above, use them for recommendations.
If no manuals exist, say: "No relevant safety manual was retrieved."

## Evidence

List specific findings.

## Sources

List the source filenames.
"""
    
    # ✅ Build prompt with appropriate context
    if state["mode"] == "manual_search":
        # ✅ Manual-specific prompt
        prompt = f"""
You are an OSHA and Workplace Safety expert.

You are answering ONLY from the manual content below.

If the answer is not found,
say:

"The uploaded manuals do not contain this information."

Never invent OSHA regulations.

Never use outside knowledge.

Summarize exactly what the manuals contain.

========================
MANUAL DOCUMENTS
========================

{manual_context}

========================
SOURCES
========================

{source_text}

========================
QUESTION
========================

{state["question"]}

========================
OUTPUT FORMAT - Return this markdown
========================

{output_format}
"""
    else:
        # ✅ Inspection/Analytics prompt
        prompt = f"""
You are SAFESIGHT AI, an expert construction safety assistant.

Answer the question using ONLY the evidence provided below.

=========================
RULES
=========================

1. Never invent information not present in the evidence.
2. If evidence is limited, say: "The available evidence suggests..."
3. Cite sources when possible.

=========================
EVIDENCE
=========================

{evidence_text}

=========================
SUMMARY METRICS
=========================

Total Inspections: {metrics.get("total_inspections", 0)}
Overall Compliance: {metrics.get("overall_compliance", 0)}%
Safe Workers: {metrics.get("total_safe_workers", 0)}
Unsafe Workers: {metrics.get("total_unsafe_workers", 0)}
Total Workers: {metrics.get("total_workers", 0)}
Average Compliance: {metrics.get("average_compliance", 0)}%

=========================
SOURCES
=========================

{source_text}

=========================
QUESTION
=========================

{state["question"]}

=========================
OUTPUT FORMAT - Return this markdown
=========================

{output_format}
"""
    
    # Generate response using ResponseAgent
    answer = response_agent.answer(prompt)
    
    elapsed = time.perf_counter() - start_time
    print(f"✅ [LangGraph] Answer generated successfully ({elapsed:.2f}s)")
    return {"answer": answer}


# --------------------------------------------------
# Conditional routing helpers
# --------------------------------------------------
def should_generate_reports(state: SafeSightState) -> str:
    """
    Determine if reports should be generated based on the state.
    """
    # ✅ Manual search goes directly to answer
    if state["mode"] == "manual_search":
        print("📄 Manual search → generate answer directly")
        return "generate_answer"

    if state.get("has_inspections", False):
        print("📋 [LangGraph] Has inspections → generating reports")
        return "generate_reports"

    print("📋 [LangGraph] No inspections → evidence")
    return "evidence"


# --------------------------------------------------
# Build the LangGraph workflow with clean routing
# --------------------------------------------------
def build_workflow():
    """
    Build and compile the LangGraph workflow with clean routing.
    """
    workflow = StateGraph(SafeSightState)
    
    # Add all nodes
    workflow.add_node("intent", intent_node)
    workflow.add_node("analytics", analytics_node)
    workflow.add_node("worker_lookup", worker_lookup_node)
    workflow.add_node("prevention_search", prevention_search_node)
    workflow.add_node("manual_search", manual_search_node)
    workflow.add_node("inspection_search", inspection_search_node)
    workflow.add_node("deduplicate", deduplicate_node)
    workflow.add_node("generate_reports", generate_reports_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Set entry point
    workflow.set_entry_point("intent")
    
    # Conditional routing from intent
    workflow.add_conditional_edges(
        "intent",
        lambda state: state["mode"],
        {
            "analytics": "analytics",
            "worker_lookup": "worker_lookup",
            "prevention": "prevention_search",
            "manual_search": "manual_search",
            "inspection_search": "inspection_search",
        },
    )
    
    # Analytics node goes directly to end (handled by API)
    workflow.add_edge("analytics", END)
    
    # All search nodes go through deduplicate
    workflow.add_edge("worker_lookup", "deduplicate")
    workflow.add_edge("prevention_search", "deduplicate")
    workflow.add_edge("manual_search", "deduplicate")
    workflow.add_edge("inspection_search", "deduplicate")
    
    # ✅ Updated conditional routing with generate_answer option
    workflow.add_conditional_edges(
        "deduplicate",
        should_generate_reports,
        {
            "generate_reports": "generate_reports",
            "evidence": "evidence",
            "generate_answer": "generate_answer",
        },
    )
    
    # Clean edges - each node has exactly one outgoing edge
    workflow.add_edge("generate_reports", "evidence")
    workflow.add_edge("evidence", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Compile
    return workflow.compile()


# --------------------------------------------------
# Create the workflow instance
# --------------------------------------------------
workflow = build_workflow()


# --------------------------------------------------
# Visualization helper (for documentation)
# --------------------------------------------------
def visualize_workflow():
    """
    Generate a visual representation of the workflow.
    Requires IPython and matplotlib.
    """
    try:
        from IPython.display import Image, display
        image_data = workflow.get_graph().draw_mermaid_png()
        display(Image(image_data))
        return True
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")
        return False


# --------------------------------------------------
# Standalone Testing with Timing
# --------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("        SAFESIGHT LangGraph Pipeline Test")
    print("=" * 60)
    print("Type 'exit' to quit.\n")
    
    # Try to visualize the workflow
    visualize_workflow()
    
    while True:
        question = input("\nAsk: ")
        
        if question.lower() == "exit":
            break
        
        total_start = time.perf_counter()
        
        result = workflow.invoke(
            {
                "question": question,
                "search_scope": "all",
                "selected_manuals": [],
                "mode": "",
                "results": [],
                "reports": [],
                "answer": "",
                "sources": [],
                "source_names": [],
                "summary": {},
                "has_inspections": False,
                "has_manuals": False,
                "evidence": {},
                "metrics": {},
                "is_summary_request": False,
                "manual_metadata": []
            }
        )
        
        total_elapsed = time.perf_counter() - total_start
        
        print("\n" + "=" * 60)
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"⏱️ Total Time: {total_elapsed:.2f}s")
        print("Answer:\n")
        print(result.get("answer", "No answer generated."))
        
        # Display structured evidence if available
        if result.get("evidence"):
            evidence = result["evidence"]
            print("\n" + "=" * 60)
            print("📋 Structured Evidence:")
            print("=" * 60)
            print(f"Risk Level: {evidence.get('risk_level', 'UNKNOWN')}")
            print(f"Confidence: {evidence.get('confidence', 'LOW')}")
            print(f"Observations: {len(evidence.get('observations', []))}")
            print(f"Violations: {len(evidence.get('violations', []))}")
            print(f"Key Findings: {len(evidence.get('key_findings', []))}")
        
        # ✅ Display metrics
        if result.get("metrics"):
            metrics = result["metrics"]
            print("\n" + "=" * 60)
            print("📊 Summary Metrics:")
            print("=" * 60)
            print(f"Total Inspections: {metrics.get('total_inspections', 0)}")
            print(f"Overall Compliance: {metrics.get('overall_compliance', 0)}%")
            print(f"Safe Workers: {metrics.get('total_safe_workers', 0)}")
            print(f"Unsafe Workers: {metrics.get('total_unsafe_workers', 0)}")
            print(f"Total Workers: {metrics.get('total_workers', 0)}")
        
        if result.get("sources"):
            print("\n" + "=" * 60)
            print("Retrieved Sources:\n")
            for source in result["sources"]:
                if source.get("type") == "manual":
                    print(f"📄 Manual: {source.get('filename', 'Unknown')}")
                else:
                    print(f"🖼️  Image: {source.get('image', 'Unknown')}")
                    print(f"   Status: {source.get('status', 'Unknown')}")
                    print(f"   Risk Level: {source.get('risk_level', 'Unknown')}")
                    # ✅ Display inspection_id if available
                    if source.get("inspection_id"):
                        print(f"   Inspection ID: {source.get('inspection_id')}")
                print("-" * 40)
        
        print()