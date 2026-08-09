from langchain_core.prompts import ChatPromptTemplate
from report_generator import generate_report


class RetrievalAgent:
    """
    Agent responsible for retrieving relevant documents based on a query.
    """
    def __init__(self, retriever):
        """
        Initialize the RetrievalAgent with a retriever function.
        
        Args:
            retriever: A function that takes a query and returns documents.
        """
        self.retriever = retriever

    def retrieve(
        self,
        query,
        top_k=10,
        document_type=None,
        search_scope="all",
        selected_manuals=None,
    ):
        """
        Retrieve documents based on the query.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.
            document_type: Optional filter - "manual" or "inspection".
            search_scope: "all" or "selected" - determines which documents to search.
            selected_manuals: List of manual filenames to search within.
            
        Returns:
            List of retrieved documents.
        """
        return self.retriever(
            query=query,
            top_k=top_k,
            document_type=document_type,
            search_scope=search_scope,
            selected_manuals=selected_manuals,
        )

    def get_recent_inspections(self, collection, limit=20):
        """
        Get latest inspection records directly from MongoDB.
        Works with the detection_history collection structure.
        
        Args:
            collection: MongoDB collection (e.g., detections from database.py)
            limit: Maximum number of recent inspections to return.
            
        Returns:
            List of recent inspection documents with scores.
        """
        # ✅ No type filter - just get the most recent records
        cursor = (
            collection.find({})
            .sort("timestamp", -1)
            .limit(limit)
        )

        results = []

        for doc in cursor:
            results.append({
                "document": doc,
                "score": 1.0,
                "confidence": "High"
            })

        print(f"📊 Retrieved {len(results)} recent inspections for summary")
        
        return results


class RiskAssessmentAgent:
    """
    Agent responsible for assessing risk levels and generating recommendations.
    """
    def __init__(self, assess_risk, get_recommendations):
        """
        Initialize the RiskAssessmentAgent with assessment functions.
        
        Args:
            assess_risk: Function that takes missing PPE list and returns risk level.
            get_recommendations: Function that takes missing PPE list and returns recommendations.
        """
        self.assess_risk = assess_risk
        self.get_recommendations = get_recommendations

    def assess(self, document):
        """
        Assess risk for a single document.
        
        Args:
            document: Document dictionary containing 'missing' field.
            
        Returns:
            Dictionary with risk_level, recommendations, and missing PPE list.
        """
        missing = document.get("missing", [])

        return {
            "risk_level": self.assess_risk(missing),
            "recommendations": self.get_recommendations(missing),
            "missing": missing,
        }


class ManualAgent:
    """
    Agent responsible for extracting manual documents from a collection.
    """
    def extract(self, documents):
        """
        Filter documents to only include manuals.
        
        Args:
            documents: List of document dictionaries.
            
        Returns:
            List of manual documents.
        """
        return [
            doc
            for doc in documents
            if doc.get("type") == "manual"
        ]


class ReportAgent:
    """
    Agent responsible for generating professional inspection reports.
    """
    def __init__(self, llm):
        """
        Initialize the ReportAgent with a language model.
        
        Args:
            llm: Language model instance for report generation.
        """
        self.llm = llm

    def generate(
        self,
        document,
        risk_level=None,
        recommendations=None,
    ):
        """
        Generate a professional inspection report.
        
        Args:
            document: The document containing inspection data.
            risk_level: The assessed risk level (optional, will be added to report).
            recommendations: List of recommendations (optional, will be added to report).
            
        Returns:
            Generated report as a string.
        """
        # Create a copy of the document and add risk data
        report = document.copy()
        report["risk_level"] = risk_level
        report["recommendations"] = recommendations

        return generate_report(report)


class ResponseAgent:
    """
    Agent responsible for generating final responses to user questions.
    """
    def __init__(self, llm):
        """
        Initialize the ResponseAgent with a language model.
        
        Args:
            llm: Language model instance for response generation.
        """
        self.llm = llm

    def answer(self, prompt):
        """
        Generate a response based on the prompt.
        
        Args:
            prompt: The prompt to send to the language model.
            
        Returns:
            Generated response as a string.
        """
        return self.llm.invoke(prompt).content


class EvidenceAgent:
    """
    Agent responsible for extracting structured evidence from retrieved documents.
    """
    def __init__(self, llm):
        """
        Initialize the EvidenceAgent with a language model.
        
        Args:
            llm: Language model instance for evidence extraction.
        """
        self.llm = llm

    def extract(self, question, context):
        """
        Extract structured evidence from the context.
        
        Args:
            question: The user's question.
            context: The retrieved context to analyze.
            
        Returns:
            Structured evidence dictionary.
        """
        prompt = ChatPromptTemplate.from_template(
            """
You are an evidence extraction specialist for construction safety.

Analyze the provided context and extract structured evidence to answer the question.

Return ONLY valid JSON with this exact structure:

{{
    "observations": [
        "Observation 1",
        "Observation 2"
    ],
    "violations": [
        {{
            "type": "PPE type",
            "details": "Details about the violation",
            "source": "image filename or document name"
        }}
    ],
    "manuals": [
        {{
            "name": "manual filename",
            "relevant_sections": "Key sections from the manual"
        }}
    ],
    "risk_level": "HIGH",
    "key_findings": [
        "Key finding 1",
        "Key finding 2"
    ],
    "confidence": "HIGH"
}}

Risk levels: LOW, MEDIUM, HIGH, CRITICAL
Confidence levels: LOW, MEDIUM, HIGH

Context:
{context}

Question:
{question}

JSON:
"""
        )

        chain = prompt | self.llm
        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        ).content

        # Try to parse the JSON response
        try:
            import json
            # Find JSON in the response (in case the model adds extra text)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except:
            # If JSON parsing fails, return a structured fallback
            return {
                "observations": ["Unable to extract structured evidence."],
                "violations": [],
                "manuals": [],
                "risk_level": "UNKNOWN",
                "key_findings": ["No structured evidence available."],
                "confidence": "LOW"
            }