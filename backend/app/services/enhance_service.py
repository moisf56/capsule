"""
Agentic SOAP Enhancement Service

MedGemma autonomously calls MCP tools to enhance a SOAP note with:
  1. Drug-drug interaction detection
  2. ICD-10 billing code suggestions
  3. Lab result correlations
  4. Clinical synthesis

MedGemma is the brain (Steps 1 & 5). MCP tools are the knowledge (Steps 2-4).
"""

import json
import logging
import re
import time
import os
import httpx
from dataclasses import dataclass, field, asdict
from typing import Optional

from backend.mcp_clinical_tools import (
    check_drug_interactions,
    suggest_icd10_codes,
    lookup_patient_labs,
)
from backend.app.services.neo4j_service import get_neo4j_service

LLAMA_URL = os.getenv("LLAMA_VISION_URL", "http://localhost:8081")


def _call_llama(messages: list[dict], max_tokens: int = 150) -> Optional[str]:
    """Call the running llama-server (MedGemma Q4_K_M) via OpenAI-compatible API."""
    try:
        resp = httpx.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "model": "medgemma",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "stop": ["<end_of_turn>", "<eos>"],
            },
            timeout=60.0,
        )
        if resp.status_code != 200:
            return None
        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip thinking traces
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        return raw
    except Exception as e:
        logger.warning(f"llama-server call failed: {e}")
        return None

logger = logging.getLogger("enhance")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[ENHANCE] %(message)s"))
    logger.addHandler(handler)


# ─── Result Types ────────────────────────────────────────

@dataclass
class DDIAlert:
    drug1: str
    drug2: str
    interaction_type: str
    severity: str  # "critical" | "moderate"


@dataclass
class ICD10Suggestion:
    code: str
    description: str
    matched_term: str


@dataclass
class LabFinding:
    test: str
    value: float
    unit: str
    interpretation: str  # N, H, L, HH, LL
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None


@dataclass
class EnhancementResult:
    medications: list[str] = field(default_factory=list)
    ddi_alerts: list[DDIAlert] = field(default_factory=list)
    icd10_suggestions: list[ICD10Suggestion] = field(default_factory=list)
    lab_findings: Optional[list[LabFinding]] = None
    clinical_summary: str = ""
    tools_called: list[str] = field(default_factory=list)
    processing_time_ms: int = 0
    medgemma_available: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ─── Medication Extraction ───────────────────────────────

# Common medication names for regex fallback
_COMMON_MEDS = [
    "aspirin", "ibuprofen", "acetaminophen", "tylenol", "metformin",
    "lisinopril", "atorvastatin", "amlodipine", "metoprolol", "omeprazole",
    "losartan", "gabapentin", "hydrochlorothiazide", "sertraline",
    "simvastatin", "montelukast", "escitalopram", "rosuvastatin",
    "bupropion", "furosemide", "pantoprazole", "duloxetine",
    "prednisone", "tamsulosin", "carvedilol", "trazodone",
    "clopidogrel", "pravastatin", "warfarin", "heparin",
    "insulin", "glipizide", "sitagliptin", "empagliflozin",
    "albuterol", "fluticasone", "tiotropium", "ipratropium",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "levothyroxine", "spironolactone", "digoxin", "diltiazem",
    "verapamil", "apixaban", "rivaroxaban", "enoxaparin",
    "morphine", "oxycodone", "hydrocodone", "tramadol",
    "diazepam", "lorazepam", "alprazolam", "clonazepam",
    "prednisone", "dexamethasone", "methylprednisolone",
]

_MED_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in _COMMON_MEDS) + r')\b',
    re.IGNORECASE,
)

# Also match "Drug Name XXmg" patterns
_MED_DOSE_PATTERN = re.compile(
    r'\b([A-Z][a-z]{3,}(?:\s[A-Z][a-z]+)?)\s+\d+\s*(?:mg|mcg|g|ml|units?)\b'
)


def extract_medications_regex(text: str) -> list[str]:
    """Extract medication names from text using regex (no LLM needed)."""
    found = set()

    # Pattern 1: Known medication names
    for match in _MED_PATTERN.finditer(text):
        found.add(match.group(1).lower())

    # Pattern 2: "DrugName 100mg" pattern
    for match in _MED_DOSE_PATTERN.finditer(text):
        name = match.group(1).strip().lower()
        if len(name) >= 4 and name not in {"patient", "history", "review", "blood", "level"}:
            found.add(name)

    return sorted(found)


def _try_medgemma_extract(text: str) -> Optional[list[str]]:
    """Use MedGemma (via llama-server) for medication extraction. Returns None if unavailable."""
    prompt = (
        "List all medications and drugs mentioned in this clinical note as a JSON array of strings. "
        "Include generic and brand names. "
        "Output ONLY a valid JSON array, nothing else.\n\n"
        f"Clinical note:\n{text[:800]}\n\n"
        'Example output: ["aspirin", "metformin", "ibuprofen"]\n'
        "JSON array:"
    )
    raw = _call_llama([
        {"role": "system", "content": "You are a clinical pharmacist. Extract medication names only. Respond with a JSON array."},
        {"role": "user", "content": prompt},
    ], max_tokens=120)
    if not raw:
        return None
    try:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not match:
            return None
        meds = json.loads(match.group())
        cleaned = []
        skip_phrases = {"medication", "prescribed", "drug", "antiplatelet",
                        "antidiabetic", "antihypertensive", "daily", "twice"}
        for m in meds:
            if not isinstance(m, str):
                continue
            m = m.strip().strip('"').strip("'").strip(".")
            if len(m) < 3 or len(m) > 40:
                continue
            words = m.split()
            if len(words) > 3:
                continue
            if any(sp in m.lower() for sp in skip_phrases):
                continue
            cleaned.append(m.lower())
        return cleaned if cleaned else None
    except Exception:
        return None


# ─── ICD-10: MedGemma-driven diagnosis identification ────

_DIAGNOSIS_KEYWORDS = re.compile(
    r'\b(acute\s+\w+(?:\s+\w+)?|'
    r'\w+\s+syndrome|'
    r'\w+\s+disease|'
    r'type\s+[12]\s+diabetes|'
    r'chest\s+pain|'
    r'hypertension|'
    r'heart\s+failure|'
    r'coronary\s+(?:artery\s+)?disease|'
    r'myocardial\s+infarction|'
    r'angina|'
    r'pneumonia|'
    r'sepsis|'
    r'atrial\s+fibrillation|'
    r'COPD|'
    r'asthma|'
    r'stroke|'
    r'DVT|'
    r'pulmonary\s+embolism)\b',
    re.IGNORECASE,
)


def _regex_identify_diagnoses(soap_text: str) -> list[str]:
    """Fallback: extract diagnoses from SOAP text using regex."""
    # Prefer Assessment/Plan section
    assessment_match = re.search(
        r'(?:\*\*Assessment|\*\*Impression|Assessment:|Diagnosis:)[:\s]*\n?(.*?)(?:\n\n|\*\*Plan|\Z)',
        soap_text, re.IGNORECASE | re.DOTALL,
    )
    text = assessment_match.group(1) if assessment_match else soap_text
    found = []
    seen = set()
    for m in _DIAGNOSIS_KEYWORDS.finditer(text):
        term = m.group(0).strip().lower()
        if term not in seen:
            seen.add(term)
            found.append(term)
    return found[:5]


def _medgemma_identify_diagnoses(soap_text: str) -> list[str]:
    """
    Ask MedGemma (via llama-server) to identify clinical diagnoses from a SOAP note.
    Returns a list of diagnosis strings suitable for Neo4j ICD-10 search.
    Falls back to regex if MedGemma unavailable.
    """
    prompt = (
        "List the clinical diagnoses and conditions from this SOAP note as a JSON array. "
        "Use standard medical terminology (not abbreviations). "
        "Include only confirmed or suspected diagnoses, not symptoms alone. "
        "Output ONLY a valid JSON array of strings.\n\n"
        f"SOAP Note:\n{soap_text[:700]}\n\n"
        'Example: ["acute coronary syndrome", "hypertension", "type 2 diabetes"]\n'
        "JSON array:"
    )
    raw = _call_llama([
        {"role": "system", "content": "You are a clinical coding specialist. Respond with only a valid JSON array."},
        {"role": "user", "content": prompt},
    ], max_tokens=120)

    if raw:
        try:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                diagnoses = json.loads(match.group())
                cleaned = [
                    d.strip() for d in diagnoses
                    if isinstance(d, str) and 3 <= len(d.strip()) <= 60
                ]
                if cleaned:
                    logger.info(f"  MedGemma identified diagnoses: {cleaned}")
                    return cleaned
        except Exception as e:
            logger.warning(f"  MedGemma diagnosis parsing failed: {e}")

    return _regex_identify_diagnoses(soap_text)


# ─── Clinical Summary ────────────────────────────────────

def _try_medgemma_summarize(
    soap_text: str,
    ddi_alerts: list[DDIAlert],
    icd10: list[ICD10Suggestion],
    labs: Optional[list[LabFinding]],
) -> Optional[str]:
    """Use MedGemma (via llama-server) to synthesize a clinical summary."""
    findings_text = ""

    if ddi_alerts:
        critical = [a for a in ddi_alerts if a.severity == "critical"]
        findings_text += f"Drug interactions: {len(ddi_alerts)} found"
        if critical:
            findings_text += f" ({len(critical)} critical)"
        findings_text += ". "
        for a in ddi_alerts[:3]:
            findings_text += f"{a.drug1}+{a.drug2}: {a.interaction_type}. "

    if icd10:
        findings_text += f"ICD-10 codes: {len(icd10)} suggested. "
        for s in icd10[:3]:
            findings_text += f"{s.code} ({s.description}). "

    if labs:
        abnormal = [l for l in labs if l.interpretation != "N"]
        if abnormal:
            findings_text += f"Abnormal labs: {len(abnormal)}. "
            for l in abnormal[:3]:
                findings_text += f"{l.test}: {l.value} {l.unit} [{l.interpretation}]. "

    if not findings_text:
        return None

    prompt = (
        f"Based on this SOAP note and clinical findings, write a brief "
        f"(2-3 sentence) clinical summary highlighting key safety concerns "
        f"and actionable items for the physician.\n\n"
        f"SOAP Note (excerpt):\n{soap_text[:500]}\n\n"
        f"Findings:\n{findings_text}"
    )

    return _call_llama([
        {"role": "system", "content": "You are a clinical decision support assistant. Be concise and actionable."},
        {"role": "user", "content": prompt},
    ], max_tokens=200)


# ─── Main Enhancement Pipeline ───────────────────────────

def enhance_soap(soap_text: str, patient_id: Optional[str] = None) -> EnhancementResult:
    """
    Agentic SOAP enhancement pipeline.

    MedGemma is the brain. MCP tools are the knowledge.

    Steps:
      1. MedGemma extracts medications from SOAP text
      2. check_drug_interactions() on extracted medications
      3. suggest_icd10_codes() on SOAP text
      4. lookup_patient_labs() if patient_id provided
      5. MedGemma synthesizes clinical summary

    Falls back to regex extraction if MedGemma unavailable.
    """
    start = time.time()
    result = EnhancementResult()

    # Step 1: Extract medications (MedGemma + regex combined)
    logger.info("Step 1: Extracting medications from SOAP text...")
    regex_meds = extract_medications_regex(soap_text)
    medgemma_meds = _try_medgemma_extract(soap_text)
    if medgemma_meds:
        result.medgemma_available = True
        # Merge: MedGemma + regex, deduplicated
        all_meds = set(m.lower() for m in medgemma_meds)
        all_meds.update(m.lower() for m in regex_meds)
        result.medications = sorted(all_meds)
        logger.info(f"  MedGemma (llama-server) + regex: {', '.join(result.medications)}")
    else:
        # Still mark as available if llama-server responds later
        result.medications = regex_meds
        logger.info(f"  Regex extracted (llama-server fallback): {', '.join(result.medications)}")

    # Step 2: Check drug interactions (MCP tool)
    if len(result.medications) >= 2:
        logger.info("Step 2: Calling MCP tool: check_drug_interactions")
        try:
            ddi_raw = check_drug_interactions(result.medications)
            ddi_data = json.loads(ddi_raw)
            result.tools_called.append("check_drug_interactions")

            for i in ddi_data.get("interactions", []):
                result.ddi_alerts.append(DDIAlert(
                    drug1=i["drug1"],
                    drug2=i["drug2"],
                    interaction_type=i["interaction_type"],
                    severity=i["severity"],
                ))

            critical = sum(1 for a in result.ddi_alerts if a.severity == "critical")
            logger.info(
                f"  Result: {len(result.ddi_alerts)} interaction(s)"
                f"{f' ({critical} critical)' if critical else ''}"
            )
        except Exception as e:
            logger.warning(f"  DDI check failed: {e}")
    else:
        logger.info("Step 2: Skipped (need 2+ medications for DDI check)")

    # Step 3: ICD-10 suggestions — MedGemma identifies diagnoses, Neo4j finds codes
    logger.info("Step 3: MedGemma identifying diagnoses → Neo4j ICD-10 lookup...")
    try:
        diagnoses = _medgemma_identify_diagnoses(soap_text)
        result.tools_called.append("suggest_icd10_codes")

        if diagnoses:
            neo4j = get_neo4j_service()
            seen_codes: set[str] = set()
            for diagnosis in diagnoses[:6]:  # cap at 6 diagnoses
                hits = neo4j.search_icd10(diagnosis, limit=3)
                if hits:
                    best = hits[0]  # highest fulltext score
                    code = best.get("code", "")
                    if code and code not in seen_codes:
                        seen_codes.add(code)
                        result.icd10_suggestions.append(ICD10Suggestion(
                            code=code,
                            description=best.get("description", ""),
                            matched_term=diagnosis,
                        ))
                        logger.info(f"  {diagnosis} → {code}: {best.get('description', '')}")

        logger.info(f"  Result: {len(result.icd10_suggestions)} ICD-10 codes suggested")
    except Exception as e:
        logger.warning(f"  ICD-10 suggestion failed: {e}")

    # Step 4: Lab correlations (MCP tool, if patient_id provided)
    if patient_id:
        logger.info(f"Step 4: Calling MCP tool: lookup_patient_labs (Patient/{patient_id})")
        try:
            labs_raw = lookup_patient_labs(patient_id)
            labs_data = json.loads(labs_raw)
            result.tools_called.append("lookup_patient_labs")

            result.lab_findings = []
            seen_labs = set()
            for lab in labs_data.get("labs", []):
                if lab.get("interpretation", "N") != "N":
                    # Deduplicate by LOINC code
                    loinc = lab.get("loinc_code", "")
                    if loinc in seen_labs:
                        continue
                    seen_labs.add(loinc)
                    result.lab_findings.append(LabFinding(
                        test=lab["loinc_display"],
                        value=lab["value"],
                        unit=lab["unit"],
                        interpretation=lab["interpretation"],
                        reference_low=lab.get("reference_low"),
                        reference_high=lab.get("reference_high"),
                    ))

            logger.info(
                f"  Result: {labs_data.get('abnormal_count', 0)} abnormal labs "
                f"out of {labs_data.get('total_count', 0)}"
            )
        except Exception as e:
            logger.warning(f"  Lab lookup failed: {e}")
    else:
        logger.info("Step 4: Skipped (no patient_id for lab correlation)")

    # Step 5: MedGemma clinical summary
    logger.info("Step 5: MedGemma synthesizing clinical summary...")
    summary = _try_medgemma_summarize(
        soap_text, result.ddi_alerts, result.icd10_suggestions, result.lab_findings
    )
    if summary:
        result.clinical_summary = summary
        result.medgemma_available = True
        logger.info(f"  Summary generated ({len(summary)} chars)")
    else:
        # Build a simple summary without MedGemma
        parts = []
        if result.medications:
            parts.append(f"{len(result.medications)} medication(s) identified")
        if result.ddi_alerts:
            critical = sum(1 for a in result.ddi_alerts if a.severity == "critical")
            parts.append(f"{len(result.ddi_alerts)} drug interaction(s) found"
                         + (f" ({critical} critical)" if critical else ""))
        if result.icd10_suggestions:
            parts.append(f"{len(result.icd10_suggestions)} ICD-10 code(s) suggested")
        if result.lab_findings:
            parts.append(f"{len(result.lab_findings)} abnormal lab(s) detected")
        result.clinical_summary = ". ".join(parts) + "." if parts else "No findings."
        logger.info("  Used fallback summary (MedGemma unavailable)")

    # Done
    result.processing_time_ms = int((time.time() - start) * 1000)
    logger.info(
        f"Complete: {result.processing_time_ms}ms | "
        f"{len(result.tools_called)} tools called | "
        f"{sum(1 for a in result.ddi_alerts if a.severity == 'critical')} critical alert(s)"
    )

    return result


# ─── Singleton ───────────────────────────────────────────

_enhancement_service = None


def get_enhancement_service():
    """Return the enhance_soap function (module-level, no class needed)."""
    return enhance_soap
