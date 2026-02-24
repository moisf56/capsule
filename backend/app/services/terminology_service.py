"""
Terminology Service — Iteration 6

Unified medical terminology lookups:
  - RxNorm: Drug name normalization + RxCUI (free REST API, no key needed)
  - SNOMED CT: Clinical concept search (UMLS API, requires API key)
  - ICD-10: Billing code suggestion (Neo4j fulltext + SOAP extraction)
"""

import os
import re
from typing import Optional
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

from backend.app.services.neo4j_service import get_neo4j_service

load_dotenv()

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
UMLS_API_KEY = os.getenv("UMLS_API_KEY", "")


# ─── Data classes ─────────────────────────────────────────

@dataclass
class RxNormDrug:
    rxcui: str
    name: str
    tty: str  # term type (e.g., SBD, SCD, IN, BN)


@dataclass
class SNOMEDConcept:
    cui: str       # UMLS CUI
    code: str      # SNOMED CT code
    name: str
    semantic_type: str


@dataclass
class ICD10Suggestion:
    code: str
    description: str
    matched_term: str


@dataclass
class CodeSuggestResult:
    icd10: list[ICD10Suggestion] = field(default_factory=list)
    snomed: list[SNOMEDConcept] = field(default_factory=list)
    terms_searched: int = 0


# ─── RxNorm (free, no key) ───────────────────────────────

class RxNormClient:
    """Query NLM RxNorm REST API for drug normalization."""

    def __init__(self):
        self._client = httpx.Client(base_url=RXNORM_BASE, timeout=10)

    def normalize(self, drug_name: str) -> Optional[RxNormDrug]:
        """Normalize a drug name to its RxCUI."""
        resp = self._client.get("/rxcui.json", params={"name": drug_name})
        resp.raise_for_status()
        data = resp.json()

        # idGroup.rxnormId contains list of RxCUIs
        ids = data.get("idGroup", {}).get("rxnormId", [])
        if not ids:
            return None

        rxcui = ids[0]
        # Get preferred name
        props = self._get_properties(rxcui)
        return RxNormDrug(
            rxcui=rxcui,
            name=props.get("name", drug_name),
            tty=props.get("tty", ""),
        )

    def approximate_search(self, term: str, max_entries: int = 5) -> list[RxNormDrug]:
        """Fuzzy search for drugs by approximate term."""
        resp = self._client.get(
            "/approximateTerm.json",
            params={"term": term, "maxEntries": max_entries},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for candidate in data.get("approximateGroup", {}).get("candidate", []):
            rxcui = candidate.get("rxcui", "")
            if rxcui:
                props = self._get_properties(rxcui)
                results.append(RxNormDrug(
                    rxcui=rxcui,
                    name=props.get("name", ""),
                    tty=props.get("tty", ""),
                ))
        return results

    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Get RxCUI for a drug name. Returns None if not found."""
        result = self.normalize(drug_name)
        return result.rxcui if result else None

    def _get_properties(self, rxcui: str) -> dict:
        """Get basic properties for an RxCUI."""
        resp = self._client.get(f"/rxcui/{rxcui}/properties.json")
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        return props

    def close(self):
        self._client.close()


# ─── SNOMED CT via UMLS API ──────────────────────────────

class SNOMEDClient:
    """Query UMLS API for SNOMED CT concepts."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or UMLS_API_KEY
        self._client = httpx.Client(base_url=UMLS_BASE, timeout=10)

    def search(self, term: str, limit: int = 5) -> list[SNOMEDConcept]:
        """Search SNOMED CT for clinical concepts matching a term."""
        if not self.api_key:
            return []

        resp = self._client.get(
            "/search/current",
            params={
                "string": term,
                "sabs": "SNOMEDCT_US",
                "returnIdType": "sourceUi",
                "pageSize": limit,
                "apiKey": self.api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("result", {}).get("results", []):
            results.append(SNOMEDConcept(
                cui=item.get("ui", ""),
                code=item.get("ui", ""),
                name=item.get("name", ""),
                semantic_type=item.get("rootSource", "SNOMEDCT_US"),
            ))
        return results

    def crosswalk_icd10(self, icd10_code: str) -> Optional[SNOMEDConcept]:
        """Map ICD-10-CM code to SNOMED CT via UMLS crosswalk API."""
        if not self.api_key:
            return None
        try:
            resp = self._client.get(
                f"/crosswalk/current/source/ICD10CM/{icd10_code}",
                params={
                    "targetSource": "SNOMEDCT_US",
                    "apiKey": self.api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            if results:
                first = results[0]
                return SNOMEDConcept(
                    cui=first.get("ui", ""),
                    code=first.get("ui", ""),
                    name=first.get("name", ""),
                    semantic_type="SNOMEDCT_US",
                )
        except Exception:
            pass  # Graceful degradation
        return None

    def close(self):
        self._client.close()


# ─── ICD-10 Suggestion (Neo4j) ──────────────────────────

def suggest_icd10_from_soap(soap_text: str) -> CodeSuggestResult:
    """
    Extract diagnosis-like phrases from SOAP text, query Neo4j
    for matching ICD-10 codes, and optionally enrich with SNOMED.
    """
    neo4j = get_neo4j_service()

    # Extract Assessment section
    assessment = ""
    lines = soap_text.split("\n")
    in_assessment = False
    for line in lines:
        if re.match(r"(?:^#+\s*|^)(assessment|a[:\.])", line, re.IGNORECASE):
            in_assessment = True
            continue
        elif re.match(r"(?:^#+\s*|^)(plan|p[:\.]|subjective|objective)", line, re.IGNORECASE):
            in_assessment = False
        elif in_assessment:
            assessment += line + " "

    if not assessment.strip():
        assessment = soap_text

    # Diagnosis keyword patterns
    diagnosis_patterns = [
        r"(?:diagnosed? with|assessment[:\s]+|impression[:\s]+|a\.\s*)([\w\s,]+?)(?:\.|$|\n)",
        r"(?:hypertension|diabetes|copd|asthma|chest pain|angina|heart failure|"
        r"pneumonia|uti|infection|anemia|depression|anxiety|"
        r"hyperlipidemia|obesity|gerd|hypothyroidism|"
        r"atrial fibrillation|dvt|pe|stroke|mi|cad|"
        r"chronic kidney disease|acute kidney injury|sepsis|"
        r"type [12] diabetes|htn|dm|chf|ckd|aki|nstemi|stemi|"
        r"coronary artery disease|acute coronary syndrome)",
    ]

    search_terms = set()
    for pattern in diagnosis_patterns:
        for match in re.finditer(pattern, assessment, re.IGNORECASE):
            term = match.group(0).strip().rstrip(".,;:")
            if len(term) >= 3:
                search_terms.add(term.lower())

    # Also try key phrases from assessment
    for phrase in re.split(r"[,;.\n]+", assessment):
        phrase = phrase.strip()
        if 3 <= len(phrase) <= 60:
            search_terms.add(phrase.lower())

    # Query Neo4j for each term
    seen_codes = set()
    icd10_suggestions = []
    for term in list(search_terms)[:10]:
        results = neo4j.search_icd10(term, limit=2)
        for r in results:
            code = r["code"]
            if code not in seen_codes and r.get("score", 0) > 0.5:
                seen_codes.add(code)
                icd10_suggestions.append(ICD10Suggestion(
                    code=code,
                    description=r["description"],
                    matched_term=term,
                ))

    icd10_suggestions.sort(key=lambda x: x.code)

    return CodeSuggestResult(
        icd10=icd10_suggestions[:15],
        terms_searched=len(search_terms),
    )


# ─── Singleton instances ─────────────────────────────────

_rxnorm: Optional[RxNormClient] = None
_snomed: Optional[SNOMEDClient] = None


def get_rxnorm() -> RxNormClient:
    global _rxnorm
    if _rxnorm is None:
        _rxnorm = RxNormClient()
    return _rxnorm


def get_snomed() -> SNOMEDClient:
    global _snomed
    if _snomed is None:
        _snomed = SNOMEDClient()
    return _snomed
