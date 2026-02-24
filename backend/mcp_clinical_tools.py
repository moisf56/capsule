"""
MCP Clinical Tools Server — Proper MCP Implementation

Defines clinical tools using the MCP Python SDK (FastMCP).
Each @mcp.tool() wraps an existing service (Neo4j, FHIR, RxNorm, SNOMED).

Can run standalone:  python mcp_clinical_tools.py  (SSE on port 8083)
Or imported:         from backend.mcp_clinical_tools import <tool_function>

MedGemma is the clinical AI brain that calls these tools autonomously.
"""

import json
import os
import sys

from mcp.server.fastmcp import FastMCP

# Ensure project root is on path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.app.services.neo4j_service import get_neo4j_service
from backend.app.services.terminology_service import (
    get_rxnorm, get_snomed, suggest_icd10_from_soap,
)
from backend.fhir_resources import FHIRClient

# ─── MCP Server ──────────────────────────────────────────

mcp = FastMCP("medgemma-clinical-tools")

_HIGH_RISK_KEYWORDS = [
    "bleeding", "anticoagulant", "qtc", "serotonergic",
    "cardiotoxic", "nephrotoxic", "hepatotoxic", "respiratory",
]


# ─── Tool 1: Drug-Drug Interactions ─────────────────────

@mcp.tool()
def check_drug_interactions(medications: list[str]) -> str:
    """Check for drug-drug interactions between medications using the Neo4j medical knowledge graph.

    Queries 222K+ interaction records from DrugBank. Returns severity-classified
    alerts for physician review.

    Args:
        medications: List of medication names (e.g. ["aspirin", "ibuprofen", "metformin"])

    Returns:
        JSON string with found (bool), summary, and interactions list.
        Each interaction includes drug1, drug2, interaction_type, and severity.
    """
    neo4j = get_neo4j_service()
    result = neo4j.check_interactions(medications)

    interactions = []
    for i in result.interactions:
        severity = "critical" if any(
            kw in i.interaction_type.lower() for kw in _HIGH_RISK_KEYWORDS
        ) else "moderate"
        interactions.append({
            "drug1": i.drug1,
            "drug2": i.drug2,
            "interaction_type": i.interaction_type,
            "severity": severity,
        })

    return json.dumps({
        "found": result.found,
        "summary": result.summary,
        "interactions": interactions,
        "count": len(interactions),
    })


# ─── Tool 2: ICD-10 Code Suggestion from Clinical Text ──

@mcp.tool()
def suggest_icd10_codes(clinical_text: str) -> str:
    """Extract diagnoses from clinical text and suggest ICD-10 billing codes.

    Parses the Assessment section of SOAP notes, extracts diagnosis terms,
    and searches 98K+ ICD-10 codes in Neo4j for matches.

    Args:
        clinical_text: SOAP note text or clinical narrative

    Returns:
        JSON string with suggestions list (code, description, matched_term)
        and terms_searched count.
    """
    result = suggest_icd10_from_soap(clinical_text)
    return json.dumps({
        "suggestions": [
            {
                "code": s.code,
                "description": s.description,
                "matched_term": s.matched_term,
            }
            for s in result.icd10
        ],
        "terms_searched": result.terms_searched,
    })


# ─── Tool 3: ICD-10 Direct Search ───────────────────────

@mcp.tool()
def search_icd10(query: str, limit: int = 5) -> str:
    """Search ICD-10 codes by description in the Neo4j knowledge graph.

    Fulltext search across 98K+ ICD-10-CM codes. Returns billable codes only.

    Args:
        query: Search term (e.g. "chest pain", "type 2 diabetes")
        limit: Maximum results to return (default 5)

    Returns:
        JSON string with results list containing code, description, and score.
    """
    neo4j = get_neo4j_service()
    results = neo4j.search_icd10(query, limit)
    return json.dumps({
        "results": [
            {"code": r["code"], "description": r["description"], "score": r.get("score", 0)}
            for r in results
        ]
    })


# ─── Tool 4: Drug Search ────────────────────────────────

@mcp.tool()
def search_drug(query: str, limit: int = 5) -> str:
    """Search for drugs by name in the Neo4j DrugBank knowledge graph.

    Fulltext search across 1,868 drug names. Returns DrugBank IDs.

    Args:
        query: Drug name or partial name to search
        limit: Maximum results to return (default 5)

    Returns:
        JSON string with results list containing name, id, and relevance score.
    """
    neo4j = get_neo4j_service()
    results = neo4j.search_drug(query, limit)
    return json.dumps({
        "results": [
            {"name": r["name"], "id": r["id"], "score": r.get("score", 0)}
            for r in results
        ]
    })


# ─── Tool 5: RxNorm Normalization ───────────────────────

@mcp.tool()
def normalize_medication(drug_name: str) -> str:
    """Normalize a drug name to its standard RxNorm identifier (RxCUI).

    Uses the NLM RxNorm REST API (free, no key required).
    Falls back to approximate/fuzzy search if exact match fails.

    Args:
        drug_name: Drug name to normalize (e.g. "tylenol", "aspirin")

    Returns:
        JSON string with found (bool), rxcui, name, tty. If not found, includes suggestions.
    """
    rxnorm = get_rxnorm()
    result = rxnorm.normalize(drug_name)
    if result:
        return json.dumps({
            "found": True,
            "rxcui": result.rxcui,
            "name": result.name,
            "tty": result.tty,
        })

    # Fuzzy fallback
    approx = rxnorm.approximate_search(drug_name, max_entries=3)
    return json.dumps({
        "found": False,
        "suggestions": [
            {"rxcui": d.rxcui, "name": d.name, "tty": d.tty}
            for d in approx
        ],
    })


# ─── Tool 6: Patient Lab Results ────────────────────────

@mcp.tool()
def lookup_patient_labs(patient_id: str) -> str:
    """Retrieve lab results for a patient from the FHIR server.

    Returns LOINC-coded lab observations with reference ranges and interpretation
    flags (N=Normal, H=High, L=Low, HH=Critical High, LL=Critical Low).

    Args:
        patient_id: FHIR Patient resource ID

    Returns:
        JSON string with labs list and abnormal_count.
    """
    fhir = FHIRClient()
    try:
        observations = fhir.search_observations(patient_id)
    finally:
        fhir.close()

    abnormal = sum(1 for o in observations if o.get("interpretation", "N") != "N")

    return json.dumps({
        "labs": [
            {
                "loinc_code": o["loinc_code"],
                "loinc_display": o["loinc_display"],
                "value": o["value"],
                "unit": o["unit"],
                "interpretation": o.get("interpretation", "N"),
                "reference_low": o.get("reference_low"),
                "reference_high": o.get("reference_high"),
            }
            for o in observations
        ],
        "abnormal_count": abnormal,
        "total_count": len(observations),
    })


# ─── Tool 7: FHIR Clinical Export ───────────────────────

@mcp.tool()
def export_clinical_encounter(
    patient_id: str,
    reason: str,
    soap_text: str,
    medications: list[str] | None = None,
    icd10_codes: list[dict] | None = None,
    ddi_alerts: list[dict] | None = None,
) -> str:
    """Export a complete clinical encounter to the FHIR R4 server.

    Creates: Encounter, DocumentReference (SOAP note), MedicationRequests
    (RxNorm-coded), Conditions (ICD-10 + SNOMED dual-coded),
    DetectedIssues (DDI safety alerts).

    Args:
        patient_id: FHIR Patient ID
        reason: Visit reason (e.g. "chest pain, diabetes follow-up")
        soap_text: Full SOAP note text
        medications: Optional list of medication names
        icd10_codes: Optional list of {code, description} dicts
        ddi_alerts: Optional list of {drug1, drug2, interaction_type, acknowledged} dicts

    Returns:
        JSON string with created resource IDs and summary count.
    """
    fhir = FHIRClient()
    rxnorm = get_rxnorm()
    snomed = get_snomed()
    medications = medications or []
    icd10_codes = icd10_codes or []
    ddi_alerts = ddi_alerts or []

    try:
        results = {
            "encounter": None,
            "document": None,
            "medications": [],
            "conditions": [],
            "detected_issues": [],
            "errors": [],
        }

        # 1. Create Encounter
        enc = fhir.create_encounter(patient_id, reason)
        results["encounter"] = enc
        encounter_id = enc["id"]

        # 2. Create DocumentReference (SOAP note)
        doc = fhir.create_document_reference(patient_id, encounter_id, soap_text)
        results["document"] = doc

        # 3. MedicationRequests (RxNorm-coded)
        med_id_map = {}
        for drug_name in medications:
            try:
                rx = rxnorm.normalize(drug_name)
                if rx:
                    med = fhir.create_medication_request(
                        patient_id, encounter_id, drug_name, rx.rxcui, rx.name
                    )
                else:
                    med = fhir.create_medication_request(
                        patient_id, encounter_id, drug_name
                    )
                results["medications"].append(med)
                med_id_map[drug_name.lower()] = med["id"]
            except Exception as e:
                results["errors"].append(f"MedicationRequest({drug_name}): {e}")

        # 4. Conditions (ICD-10 + SNOMED dual-coded)
        for icd in icd10_codes:
            try:
                snomed_result = snomed.crosswalk_icd10(icd["code"])
                cond = fhir.create_condition(
                    patient_id, encounter_id,
                    icd["code"], icd["description"],
                    snomed_result.code if snomed_result else None,
                    snomed_result.name if snomed_result else None,
                )
                results["conditions"].append(cond)
            except Exception as e:
                results["errors"].append(f"Condition({icd['code']}): {e}")

        # 5. DetectedIssues (DDI alerts)
        for alert in ddi_alerts:
            try:
                implicated = []
                d1_id = med_id_map.get(alert["drug1"].lower())
                d2_id = med_id_map.get(alert["drug2"].lower())
                if d1_id:
                    implicated.append(d1_id)
                if d2_id:
                    implicated.append(d2_id)

                severity = "high" if any(
                    kw in alert["interaction_type"].lower() for kw in _HIGH_RISK_KEYWORDS
                ) else "moderate"

                di = fhir.create_detected_issue(
                    patient_id,
                    f"{alert['drug1']} + {alert['drug2']}: {alert['interaction_type']}",
                    severity, implicated, alert.get("acknowledged", True),
                )
                results["detected_issues"].append(di)
            except Exception as e:
                results["errors"].append(f"DetectedIssue: {e}")

        total = (
            (1 if results["encounter"] else 0)
            + (1 if results["document"] else 0)
            + len(results["medications"])
            + len(results["conditions"])
            + len(results["detected_issues"])
        )
        results["summary"] = f"{total} FHIR resources created"

        return json.dumps(results)

    finally:
        fhir.close()


# ─── Standalone MCP Server ───────────────────────────────

if __name__ == "__main__":
    print("Starting MedGemma Clinical Tools MCP Server (SSE on port 8083)...")
    print("Connect via MCP Inspector: npx @modelcontextprotocol/inspector")
    # Override default port for standalone mode
    mcp._port = 8083
    mcp.run(transport="sse")
