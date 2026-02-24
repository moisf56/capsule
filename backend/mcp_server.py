"""
MCP Server — Iteration 13

FastAPI HTTP bridge for the phone + agentic MedGemma enhancement.
Port 8082, CORS open for phone access over LAN.

MCP tools are defined in backend/mcp_clinical_tools.py using FastMCP (@mcp.tool).
This server bridges HTTP (phone) → MCP tool functions (in-process).

Key endpoints:
  POST /tools/enhance_soap                 — Agentic: MedGemma + MCP tools → findings
  POST /tools/ehr_navigate                 — EHR Navigator Agent (LangGraph, 5-step)
  POST /tools/fhir_create_encounter        — Create FHIR Encounter
  POST /tools/fhir_export_soap             — Export SOAP note as DocumentReference
  POST /tools/fhir_export_full             — Full clinical export (7 resource types)
  POST /tools/check_drug_interactions      — Neo4j drug-drug interaction check
  POST /tools/search_icd10                 — Neo4j ICD-10 fulltext search
  POST /tools/search_drug                  — Neo4j drug name fulltext search
  POST /tools/normalize_drug               — RxNorm drug name → RxCUI normalization
  POST /tools/search_snomed                — SNOMED CT concept search (UMLS API)
  POST /tools/suggest_codes                — Extract diagnoses from SOAP → ICD-10 suggestions
  POST /tools/analyze_medical_image        — MedGemma vision (proxy to llama-server)
  POST /tools/fhir_export_diagnostic_report — Export DiagnosticReport to FHIR
  GET  /tools/list_diagnostic_reports      — List existing DiagnosticReports
  POST /tools/fhir_create_observation      — Create lab result Observation
  GET  /tools/list_observations            — List lab results for patient
  POST /tools/seed_demo_labs               — Seed demo patient with realistic labs
  GET  /tools/list_patients                — List patients from FHIR
  POST /tools/seed_demo_patients           — Seed demo patients with labs

Dashboard:
  GET /dashboard — EMR-style clinical resource browser
"""

import base64
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx as httpx_async

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.fhir_resources import FHIRClient
from backend.app.services.neo4j_service import get_neo4j_service
from backend.app.services.terminology_service import (
    get_rxnorm, get_snomed, suggest_icd10_from_soap,
)
from backend.app.services.enhance_service import enhance_soap

app = FastAPI(title="MedGemma MCP Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

fhir = FHIRClient()


# --- Request schemas ---

class CreateEncounterRequest(BaseModel):
    patient_id: str
    reason: str


class ExportSOAPRequest(BaseModel):
    patient_id: str
    encounter_id: str
    soap_text: str


class CheckDDIRequest(BaseModel):
    medications: list[str]


class SearchICD10Request(BaseModel):
    query: str
    limit: int = 5


class SearchDrugRequest(BaseModel):
    query: str
    limit: int = 5


class NormalizeDrugRequest(BaseModel):
    drug_name: str


class SearchSNOMEDRequest(BaseModel):
    term: str
    limit: int = 5


class SuggestCodesRequest(BaseModel):
    soap_text: str


class AnalyzeImageRequest(BaseModel):
    image_base64: str
    image_type: str = "chest_xray"
    clinical_context: str = ""
    patient_id: str = ""


class ExportDiagnosticReportRequest(BaseModel):
    patient_id: str
    encounter_id: str | None = None
    conclusion: str
    findings_text: str
    image_type: str = "chest_xray"


class ICD10CodeData(BaseModel):
    code: str
    description: str


class DDIAlertData(BaseModel):
    drug1: str
    drug2: str
    interaction_type: str
    acknowledged: bool = True


class CreateObservationRequest(BaseModel):
    patient_id: str
    encounter_id: str | None = None
    loinc_code: str
    loinc_display: str
    value: float
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None


class SeedDemoLabsRequest(BaseModel):
    patient_id: str = "1000"


class FullExportRequest(BaseModel):
    patient_id: str
    reason: str
    soap_text: str
    medications: list[str] = []
    icd10_codes: list[ICD10CodeData] = []
    ddi_alerts: list[DDIAlertData] = []


# --- Endpoints ---

@app.post("/tools/fhir_create_encounter")
def create_encounter(req: CreateEncounterRequest):
    try:
        result = fhir.create_encounter(req.patient_id, req.reason)
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/fhir_export_soap")
def export_soap(req: ExportSOAPRequest):
    try:
        result = fhir.create_document_reference(
            req.patient_id, req.encounter_id, req.soap_text
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_HIGH_RISK_KEYWORDS = [
    "bleeding", "anticoagulant", "qtc", "serotonergic",
    "cardiotoxic", "nephrotoxic", "hepatotoxic", "respiratory",
]


@app.post("/tools/fhir_export_full")
def export_full(req: FullExportRequest):
    """
    Comprehensive FHIR export for a clinical encounter.

    Creates: Encounter → DocumentReference → MedicationRequests (RxNorm) →
             Conditions (ICD-10 + SNOMED) → DetectedIssues (DDI).
    """
    try:
        results = {
            "encounter": None,
            "document": None,
            "medications": [],
            "conditions": [],
            "detected_issues": [],
            "errors": [],
        }

        # 0. Auto-create patient if none selected (new walk-in patient)
        patient_id = req.patient_id
        if not patient_id:
            anon = fhir.create_anonymous_patient()
            patient_id = anon["id"]
            results["new_patient"] = anon

        # 1. Create Encounter
        enc = fhir.create_encounter(patient_id, req.reason)
        results["encounter"] = enc
        encounter_id = enc["id"]

        # 2. Create DocumentReference (SOAP note)
        doc = fhir.create_document_reference(
            patient_id, encounter_id, req.soap_text
        )
        results["document"] = doc

        # 3. Create MedicationRequests (with RxNorm normalization)
        rxnorm = get_rxnorm()
        med_id_map: dict[str, str] = {}  # drug_name_lower -> fhir_id

        for drug_name in req.medications:
            try:
                rx_result = rxnorm.normalize(drug_name)
                if rx_result:
                    med = fhir.create_medication_request(
                        patient_id, encounter_id,
                        drug_name, rx_result.rxcui, rx_result.name,
                    )
                else:
                    med = fhir.create_medication_request(
                        patient_id, encounter_id, drug_name,
                    )
                results["medications"].append(med)
                med_id_map[drug_name.lower()] = med["id"]
            except Exception as e:
                results["errors"].append(f"MedicationRequest({drug_name}): {e}")

        # 4. Create Conditions (with SNOMED crosswalk)
        snomed = get_snomed()
        for icd in req.icd10_codes:
            try:
                snomed_result = snomed.crosswalk_icd10(icd.code)
                cond = fhir.create_condition(
                    patient_id, encounter_id,
                    icd.code, icd.description,
                    snomed_result.code if snomed_result else None,
                    snomed_result.name if snomed_result else None,
                )
                results["conditions"].append(cond)
            except Exception as e:
                results["errors"].append(f"Condition({icd.code}): {e}")

        # 5. Create DetectedIssues (DDI alerts doctor acknowledged)
        for alert in req.ddi_alerts:
            try:
                implicated = []
                d1_id = med_id_map.get(alert.drug1.lower())
                d2_id = med_id_map.get(alert.drug2.lower())
                if d1_id:
                    implicated.append(d1_id)
                if d2_id:
                    implicated.append(d2_id)

                severity = "high" if any(
                    kw in alert.interaction_type.lower()
                    for kw in _HIGH_RISK_KEYWORDS
                ) else "moderate"

                di = fhir.create_detected_issue(
                    patient_id,
                    f"{alert.drug1} + {alert.drug2}: {alert.interaction_type}",
                    severity,
                    implicated,
                    alert.acknowledged,
                )
                results["detected_issues"].append(di)
            except Exception as e:
                results["errors"].append(
                    f"DetectedIssue({alert.drug1}+{alert.drug2}): {e}"
                )

        # Summary
        total = (
            (1 if results["encounter"] else 0)
            + (1 if results["document"] else 0)
            + len(results["medications"])
            + len(results["conditions"])
            + len(results["detected_issues"])
        )
        results["summary"] = f"{total} FHIR resources created"
        if results["errors"]:
            results["summary"] += f" ({len(results['errors'])} warnings)"

        return {"status": "ok", "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/check_drug_interactions")
def check_drug_interactions(req: CheckDDIRequest):
    try:
        neo4j = get_neo4j_service()
        result = neo4j.check_interactions(req.medications)
        return {
            "status": "ok",
            "data": {
                "found": result.found,
                "summary": result.summary,
                "interactions": [
                    {
                        "drug1": i.drug1,
                        "drug2": i.drug2,
                        "interaction_type": i.interaction_type,
                    }
                    for i in result.interactions
                ],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/search_icd10")
def search_icd10(req: SearchICD10Request):
    try:
        neo4j = get_neo4j_service()
        results = neo4j.search_icd10(req.query, req.limit)
        return {"status": "ok", "data": {"results": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/search_drug")
def search_drug(req: SearchDrugRequest):
    """Search drugs via Neo4j fulltext (DrugBank names)."""
    try:
        neo4j = get_neo4j_service()
        results = neo4j.search_drug(req.query, req.limit)
        return {"status": "ok", "data": {"results": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/normalize_drug")
def normalize_drug(req: NormalizeDrugRequest):
    """Normalize a drug name to RxCUI via NLM RxNorm API."""
    try:
        rxnorm = get_rxnorm()
        result = rxnorm.normalize(req.drug_name)
        if result:
            return {
                "status": "ok",
                "data": {
                    "found": True,
                    "rxcui": result.rxcui,
                    "name": result.name,
                    "tty": result.tty,
                },
            }
        # Try fuzzy search as fallback
        approx = rxnorm.approximate_search(req.drug_name, max_entries=3)
        return {
            "status": "ok",
            "data": {
                "found": False,
                "suggestions": [
                    {"rxcui": d.rxcui, "name": d.name, "tty": d.tty}
                    for d in approx
                ],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/search_snomed")
def search_snomed(req: SearchSNOMEDRequest):
    """Search SNOMED CT concepts via UMLS API."""
    try:
        snomed = get_snomed()
        results = snomed.search(req.term, req.limit)
        return {
            "status": "ok",
            "data": {
                "results": [
                    {
                        "cui": r.cui,
                        "code": r.code,
                        "name": r.name,
                        "semantic_type": r.semantic_type,
                    }
                    for r in results
                ],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/suggest_codes")
def suggest_codes(req: SuggestCodesRequest):
    """Extract diagnoses from SOAP text and suggest ICD-10 codes via Neo4j."""
    try:
        result = suggest_icd10_from_soap(req.soap_text)
        return {
            "status": "ok",
            "data": {
                "suggestions": [
                    {
                        "code": s.code,
                        "description": s.description,
                        "matched_term": s.matched_term,
                    }
                    for s in result.icd10
                ],
                "terms_searched": result.terms_searched,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Iteration 8: Vision + DiagnosticReport ---

LLAMA_VISION_URL = os.getenv("LLAMA_VISION_URL", "http://localhost:8081")

_VISION_PROMPTS = {
    "chest_xray": (
        "You are a board-certified radiologist writing a formal radiology report. "
        "Analyze this chest X-ray image thoroughly and provide a DETAILED structured report.\n\n"
        "FINDINGS:\n"
        "Systematically evaluate EACH of the following structures and describe findings in detail:\n"
        "1. LUNGS: Aeration, opacities, consolidation, masses, nodules, interstitial markings, "
        "atelectasis, pleural effusion, pneumothorax. Describe location (RUL/RML/RLL/LUL/LLL) if applicable.\n"
        "2. HEART: Size (CTR), contour, calcifications, pericardial effusion.\n"
        "3. MEDIASTINUM: Width, lymphadenopathy, tracheal deviation, aortic knob.\n"
        "4. HILA: Symmetry, enlargement, masses.\n"
        "5. BONES: Ribs, clavicles, spine — fractures, lytic/blastic lesions, degenerative changes.\n"
        "6. SOFT TISSUES: Subcutaneous emphysema, masses, foreign bodies.\n"
        "7. DEVICES: Lines, tubes, pacemakers if present.\n\n"
        "IMPRESSION:\n"
        "- Primary diagnosis with confidence level (high/moderate/low).\n"
        "- Differential diagnoses ranked by likelihood.\n"
        "- Clinical correlation recommendations.\n"
        "- Suggested follow-up imaging if needed.\n\n"
        "Be thorough. Do NOT be vague. Use proper radiological terminology."
    ),
    "dermatology": (
        "You are an expert dermatologist. Analyze this skin lesion image. "
        "Provide a structured report with:\n\n"
        "FINDINGS:\n"
        "- Describe the lesion: location, size, color, shape, borders, surface.\n"
        "- Apply ABCDE criteria if applicable.\n\n"
        "IMPRESSION:\n"
        "- Most likely diagnosis and differential diagnoses.\n"
        "- Recommended next steps."
    ),
    "pathology": (
        "You are an expert pathologist. Analyze this histopathology image. "
        "Provide a structured report with:\n\n"
        "FINDINGS:\n"
        "- Describe tissue architecture, cell morphology, staining patterns.\n"
        "- Note any abnormal features.\n\n"
        "IMPRESSION:\n"
        "- Diagnosis and grade if applicable.\n"
        "- Recommended additional studies."
    ),
    "mri": (
        "You are a board-certified neuroradiologist writing a formal MRI report. "
        "Analyze this MRI image thoroughly and provide a DETAILED structured report.\n\n"
        "FINDINGS:\n"
        "Systematically evaluate EACH of the following:\n"
        "1. BRAIN PARENCHYMA: Gray-white matter differentiation, signal abnormalities, "
        "masses, edema, midline shift, herniation.\n"
        "2. VENTRICLES: Size, symmetry, hydrocephalus, periventricular changes.\n"
        "3. EXTRA-AXIAL SPACES: Subdural/epidural collections, meningeal enhancement.\n"
        "4. WHITE MATTER: Hyperintensities, demyelination, ischemic changes.\n"
        "5. POSTERIOR FOSSA: Cerebellum, brainstem, fourth ventricle.\n"
        "6. SELLA/PITUITARY: Size, signal characteristics.\n"
        "7. ORBITS & SINUSES: If visible — mucosal thickening, masses.\n"
        "8. CALVARIUM: Bone marrow signal, fractures, lesions.\n\n"
        "Identify the MRI sequence (T1, T2, FLAIR, DWI, etc.) based on signal characteristics.\n\n"
        "IMPRESSION:\n"
        "- Primary diagnosis with confidence level.\n"
        "- Differential diagnoses ranked by likelihood.\n"
        "- Clinical correlation and follow-up recommendations.\n\n"
        "Use proper neuroradiology terminology. Be thorough and precise."
    ),
    "ct": (
        "You are a board-certified radiologist writing a formal CT report. "
        "Analyze this CT image thoroughly and provide a DETAILED structured report.\n\n"
        "FINDINGS:\n"
        "Systematically evaluate:\n"
        "1. TARGET ORGAN: Primary findings in the organ system being imaged.\n"
        "2. ADJACENT STRUCTURES: Any secondary findings.\n"
        "3. CONTRAST: Enhancement patterns if contrast was used.\n"
        "4. MEASUREMENTS: Estimate sizes of any lesions or abnormalities.\n"
        "5. COMPARISON: Note if this appears acute vs chronic.\n\n"
        "IMPRESSION:\n"
        "- Primary diagnosis with confidence level.\n"
        "- Differential diagnoses.\n"
        "- Recommended follow-up.\n\n"
        "Use proper radiological terminology."
    ),
    "fundoscopy": (
        "You are a board-certified ophthalmologist analyzing a fundus photograph. "
        "Provide a DETAILED structured report.\n\n"
        "FINDINGS:\n"
        "Systematically evaluate:\n"
        "1. OPTIC DISC: Color, margins, cup-to-disc ratio, swelling, pallor.\n"
        "2. MACULA: Foveal reflex, edema, drusen, hemorrhage, exudates.\n"
        "3. VESSELS: AV ratio, AV nicking, copper/silver wiring, neovascularization, "
        "microaneurysms, venous beading.\n"
        "4. RETINAL BACKGROUND: Hemorrhages (dot-blot, flame-shaped), cotton wool spots, "
        "hard exudates, soft exudates, pigmentary changes.\n"
        "5. PERIPHERY: If visible — lattice degeneration, tears, detachment.\n\n"
        "IMPRESSION:\n"
        "- Grading (e.g., NPDR mild/moderate/severe, PDR if applicable).\n"
        "- Clinical significance and urgency.\n"
        "- Recommended follow-up (e.g., fluorescein angiography, OCT, referral).\n\n"
        "Use standard ophthalmologic terminology."
    ),
    "ecg": (
        "You are a board-certified cardiologist analyzing an electrocardiogram (ECG/EKG). "
        "Provide a DETAILED structured report.\n\n"
        "FINDINGS:\n"
        "Systematically evaluate:\n"
        "1. RATE: Ventricular rate (bpm), regular vs irregular.\n"
        "2. RHYTHM: Sinus, atrial fibrillation, flutter, etc.\n"
        "3. AXIS: Normal, LAD, RAD.\n"
        "4. INTERVALS: PR, QRS, QTc — normal vs prolonged.\n"
        "5. P WAVES: Morphology, P mitrale, P pulmonale.\n"
        "6. QRS: Bundle branch block, ventricular hypertrophy, pathological Q waves.\n"
        "7. ST SEGMENT: Elevation, depression — leads affected, distribution.\n"
        "8. T WAVES: Inversion, hyperacute, peaked.\n\n"
        "IMPRESSION:\n"
        "- Primary interpretation.\n"
        "- Clinical significance (e.g., STEMI territory, ischemia pattern).\n"
        "- Urgency and recommended action.\n\n"
        "Use standard cardiology terminology."
    ),
    "general": (
        "You are a medical imaging specialist. Analyze this medical image thoroughly. "
        "First identify the imaging modality (X-ray, CT, MRI, ultrasound, etc.) "
        "and the body region. Then provide a DETAILED structured report.\n\n"
        "FINDINGS:\n"
        "- Identify the imaging modality and body region.\n"
        "- Systematically describe all observable findings.\n"
        "- Note normal and abnormal structures.\n"
        "- Describe any pathology with location, size, and characteristics.\n\n"
        "IMPRESSION:\n"
        "- Primary diagnosis with confidence level.\n"
        "- Differential diagnoses if applicable.\n"
        "- Clinical correlation recommendations.\n\n"
        "Use proper medical terminology. Be thorough."
    ),
}


@app.post("/tools/analyze_medical_image")
async def analyze_medical_image(req: AnalyzeImageRequest):
    """Analyze a medical image using MedGemma vision on workstation llama-server."""
    try:
        system_prompt = _VISION_PROMPTS.get(req.image_type, _VISION_PROMPTS["general"])

        text_prompt = "Analyze this medical image."
        if req.clinical_context:
            text_prompt += f"\n\nClinical context: {req.clinical_context}"

        payload = {
            "model": "medgemma",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{req.image_base64}",
                    }},
                    {"type": "text", "text": text_prompt},
                ]},
            ],
            "temperature": 0.4,
            "max_tokens": 4096,
            "stop": ["<end_of_turn>", "<eos>"],
        }

        async with httpx_async.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{LLAMA_VISION_URL}/v1/chat/completions",
                json=payload,
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"llama-server returned {resp.status_code}: {resp.text}",
            )

        result = resp.json()
        raw_report = result["choices"][0]["message"]["content"]

        # Strip thinking traces
        raw_report = re.sub(r"<think>[\s\S]*?</think>", "", raw_report).strip()

        # Parse FINDINGS and IMPRESSION sections
        findings = ""
        impression = ""
        findings_match = re.search(
            r"FINDINGS:\s*\n(.*?)(?=IMPRESSION:|$)", raw_report, re.DOTALL,
        )
        impression_match = re.search(
            r"IMPRESSION:\s*\n(.*?)$", raw_report, re.DOTALL,
        )
        if findings_match:
            findings = findings_match.group(1).strip()
        if impression_match:
            impression = impression_match.group(1).strip()
        if not findings and not impression:
            findings = raw_report

        return {
            "status": "ok",
            "data": {
                "findings": findings,
                "impression": impression,
                "raw_report": raw_report,
                "image_type": req.image_type,
                "model": "medgemma-1.5-4b-it",
            },
        }

    except httpx_async.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to llama-server (vision). Run: ./workstation/start_vision.sh",
        )
    except httpx_async.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail="Image analysis timed out (>120s). The image may be too large.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/fhir_export_diagnostic_report")
def export_diagnostic_report(req: ExportDiagnosticReportRequest):
    """Export a radiology report as a FHIR DiagnosticReport."""
    try:
        result = fhir.create_diagnostic_report(
            req.patient_id, req.encounter_id,
            req.conclusion, req.findings_text, req.image_type,
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/list_diagnostic_reports")
def list_diagnostic_reports(patient_id: str):
    """List existing DiagnosticReports for a patient."""
    try:
        resp = fhir.client.get(
            "/DiagnosticReport",
            params={"subject": f"Patient/{patient_id}", "_count": "50", "_sort": "-_lastUpdated"},
            headers={"Accept": "application/fhir+json"},
        )
        resp.raise_for_status()
        reports = []
        for e in resp.json().get("entry", []):
            r = e["resource"]
            loinc_code = ""
            for c in r.get("code", {}).get("coding", []):
                if "loinc" in c.get("system", ""):
                    loinc_code = c.get("code", "")
            # Map LOINC back to image_type
            loinc_to_type = {
                "36643-5": "chest_xray", "24590-2": "mri", "24727-0": "ct",
                "32451-7": "fundoscopy", "11524-6": "ecg",
                "72170-4": "dermatology", "60567-5": "pathology", "18748-4": "general",
            }
            reports.append({
                "id": r["id"],
                "status": r.get("status", ""),
                "conclusion": r.get("conclusion", ""),
                "date": r.get("issued", "")[:19],
                "image_type": loinc_to_type.get(loinc_code, "general"),
            })
        return {"status": "ok", "data": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/get_diagnostic_report_image")
def get_diagnostic_report_image(report_id: str):
    """Fetch a DiagnosticReport's embedded image (base64) from FHIR."""
    try:
        resp = fhir.client.get(
            f"/DiagnosticReport/{report_id}",
            headers={"Accept": "application/fhir+json"},
        )
        resp.raise_for_status()
        r = resp.json()
        # Find image in presentedForm (first non-text entry)
        for form in r.get("presentedForm", []):
            ct = form.get("contentType", "")
            if ct.startswith("image/"):
                return {
                    "status": "ok",
                    "data": {
                        "image_base64": form["data"],
                        "content_type": ct,
                        "title": form.get("title", ""),
                        "conclusion": r.get("conclusion", ""),
                    },
                }
        raise HTTPException(status_code=404, detail="No image found in DiagnosticReport")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Iteration 9: Lab Results (Observations) ---

DEMO_LABS = [
    # CBC Panel
    ("6690-2", "WBC", 7.2, "10*3/uL", 4.5, 11.0),
    ("789-8", "RBC", 4.8, "10*6/uL", 4.5, 5.5),
    ("718-7", "Hemoglobin", 14.1, "g/dL", 13.5, 17.5),
    ("4544-3", "Hematocrit", 42.0, "%", 38.0, 50.0),
    ("777-3", "Platelets", 245.0, "10*3/uL", 150.0, 400.0),
    # BMP Panel
    ("1558-6", "Glucose (fasting)", 142.0, "mg/dL", 70.0, 100.0),
    ("3094-0", "BUN", 18.0, "mg/dL", 7.0, 20.0),
    ("2160-0", "Creatinine", 1.1, "mg/dL", 0.7, 1.3),
    ("2951-2", "Sodium", 140.0, "mmol/L", 136.0, 145.0),
    ("2823-3", "Potassium", 4.2, "mmol/L", 3.5, 5.1),
    ("2075-0", "Chloride", 102.0, "mmol/L", 98.0, 106.0),
    ("2028-9", "CO2", 24.0, "mmol/L", 23.0, 29.0),
    ("17861-6", "Calcium", 9.4, "mg/dL", 8.5, 10.5),
    # Lipid Panel
    ("2093-3", "Total Cholesterol", 242.0, "mg/dL", 0.0, 200.0),
    ("2089-1", "LDL Cholesterol", 158.0, "mg/dL", 0.0, 100.0),
    ("2085-9", "HDL Cholesterol", 38.0, "mg/dL", 40.0, 60.0),
    ("2571-8", "Triglycerides", 230.0, "mg/dL", 0.0, 150.0),
    # Special
    ("4548-4", "HbA1c", 7.8, "%", 4.0, 5.6),
    ("10839-9", "Troponin I", 0.08, "ng/mL", 0.0, 0.04),
]


@app.post("/tools/fhir_create_observation")
def create_observation(req: CreateObservationRequest):
    """Create a single lab result Observation in FHIR."""
    try:
        result = fhir.create_observation(
            patient_id=req.patient_id,
            encounter_id=req.encounter_id,
            loinc_code=req.loinc_code,
            loinc_display=req.loinc_display,
            value=req.value,
            unit=req.unit,
            reference_low=req.reference_low,
            reference_high=req.reference_high,
        )
        return {"status": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/list_observations")
def list_observations(patient_id: str):
    """List lab result Observations for a patient."""
    try:
        results = fhir.search_observations(patient_id)
        return {"status": "ok", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/seed_demo_labs")
def seed_demo_labs(req: SeedDemoLabsRequest):
    """Seed demo patient with realistic lab values (CBC, BMP, Lipid, HbA1c, Troponin)."""
    try:
        created = []
        for loinc, display, val, unit, ref_lo, ref_hi in DEMO_LABS:
            result = fhir.create_observation(
                patient_id=req.patient_id,
                encounter_id=None,
                loinc_code=loinc,
                loinc_display=display,
                value=val,
                unit=unit,
                reference_low=ref_lo,
                reference_high=ref_hi,
            )
            created.append(result)
        return {"status": "ok", "data": {"created": len(created), "observations": created}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Iteration 11: Patient List + Seed Demo Patients ---

@app.get("/tools/list_patients")
def list_patients():
    """List all patients from FHIR. Returns masked-ready patient summaries."""
    try:
        patients = fhir.list_patients()
        return {"status": "ok", "data": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


DEMO_PATIENTS = [
    # (given, family, gender, birthDate, labs[])
    ("Wei", "Chen", "male", "1968-03-15", DEMO_LABS),
    ("Maria", "Santos", "female", "1979-08-22", [
        ("6690-2", "WBC", 6.8, "10*3/uL", 4.5, 11.0),
        ("718-7", "Hemoglobin", 13.2, "g/dL", 12.0, 16.0),
        ("2093-3", "Total Cholesterol", 228.0, "mg/dL", 0.0, 200.0),
        ("2089-1", "LDL Cholesterol", 145.0, "mg/dL", 0.0, 100.0),
        ("2085-9", "HDL Cholesterol", 52.0, "mg/dL", 40.0, 60.0),
        ("2571-8", "Triglycerides", 155.0, "mg/dL", 0.0, 150.0),
        ("2951-2", "Sodium", 141.0, "mmol/L", 136.0, 145.0),
        ("2823-3", "Potassium", 3.9, "mmol/L", 3.5, 5.1),
        ("2160-0", "Creatinine", 0.9, "mg/dL", 0.6, 1.1),
        ("4548-4", "HbA1c", 5.4, "%", 4.0, 5.6),
    ]),
    ("James", "Wilson", "male", "1962-11-04", [
        ("6690-2", "WBC", 9.8, "10*3/uL", 4.5, 11.0),
        ("718-7", "Hemoglobin", 11.8, "g/dL", 13.5, 17.5),
        ("789-8", "RBC", 4.1, "10*6/uL", 4.5, 5.5),
        ("777-3", "Platelets", 310.0, "10*3/uL", 150.0, 400.0),
        ("6301-6", "INR", 2.8, "ratio", 0.8, 1.1),
        ("3173-2", "aPTT", 42.0, "sec", 25.0, 35.0),
        ("2160-0", "Creatinine", 1.4, "mg/dL", 0.7, 1.3),
        ("3094-0", "BUN", 24.0, "mg/dL", 7.0, 20.0),
        ("2951-2", "Sodium", 138.0, "mmol/L", 136.0, 145.0),
        ("1558-6", "Glucose (fasting)", 98.0, "mg/dL", 70.0, 100.0),
    ]),
    ("Sarah", "Kim", "female", "1990-05-18", [
        ("6690-2", "WBC", 10.2, "10*3/uL", 4.5, 11.0),
        ("718-7", "Hemoglobin", 11.0, "g/dL", 12.0, 16.0),
        ("4544-3", "Hematocrit", 33.0, "%", 36.0, 46.0),
        ("1558-6", "Glucose (fasting)", 132.0, "mg/dL", 70.0, 100.0),
        ("4548-4", "HbA1c", 6.8, "%", 4.0, 5.6),
        ("2093-3", "Total Cholesterol", 195.0, "mg/dL", 0.0, 200.0),
        ("2160-0", "Creatinine", 0.7, "mg/dL", 0.5, 1.0),
        ("17861-6", "Calcium", 9.8, "mg/dL", 8.5, 10.5),
        ("2823-3", "Potassium", 4.0, "mmol/L", 3.5, 5.1),
    ]),
]


@app.post("/tools/seed_demo_patients")
def seed_demo_patients():
    """Seed 4 demo patients with realistic lab data."""
    try:
        results = []
        for given, family, gender, dob, labs in DEMO_PATIENTS:
            # Check if patient already exists by name
            existing = fhir.search_patient(family)
            if any(given in p["name"] for p in existing):
                pid = next(p["id"] for p in existing if given in p["name"])
            else:
                patient = fhir.create_patient(family, given, gender, dob)
                pid = patient["id"]

            # Seed labs for this patient
            lab_count = 0
            for loinc, display, val, unit, ref_lo, ref_hi in labs:
                fhir.create_observation(
                    patient_id=pid, encounter_id=None,
                    loinc_code=loinc, loinc_display=display,
                    value=val, unit=unit,
                    reference_low=ref_lo, reference_high=ref_hi,
                )
                lab_count += 1

            results.append({
                "patient_id": pid,
                "name": f"{given} {family}",
                "labs_created": lab_count,
            })
        return {"status": "ok", "data": {"patients": results, "total": len(results)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Seed Demo Scans (DiagnosticReports with images) ---

DEMO_SCANS_DIR = Path(__file__).resolve().parent.parent / "demo_medical_images"

# Map patient name prefix → list of (filename, image_type, conclusion)
DEMO_SCANS = {
    "Wei": [
        ("wei_chen_chest_xray_chf.jpg", "chest_xray", "Chest X-ray: Cardiomegaly with signs of congestive heart failure. Kerley B lines and cephalization of pulmonary vessels noted."),
        ("wei_chen_ecg_stemi.jpg", "ecg", "ECG: ST-elevation myocardial infarction (STEMI) pattern consistent with acute coronary syndrome."),
        ("wei_chen_coronary_angiogram.jpg", "general", "Coronary angiogram: Severe stenosis in left coronary artery and circumflex artery."),
    ],
    "Maria": [
        ("maria_garcia_fundus_dr.png", "fundoscopy", "Fundus: Moderate non-proliferative diabetic retinopathy with microaneurysms, dot-blot hemorrhages, and hard exudates."),
        ("maria_garcia_fundus_npdr.jpg", "fundoscopy", "Fundus: Early non-proliferative diabetic retinopathy with scattered dot-blot hemorrhages."),
        ("maria_garcia_fundus_pdr.jpg", "fundoscopy", "Fundus: Proliferative diabetic retinopathy with neovascularization. Urgent ophthalmology referral recommended."),
    ],
    "James": [
        ("james_thompson_ct_emphysema.jpg", "ct", "CT Chest: End-stage centrilobular emphysema with extensive parenchymal destruction. Consistent with chronic COPD."),
        ("james_thompson_chest_xray_copd.jpg", "chest_xray", "Chest X-ray: Hyperinflation with flattened diaphragm. Consistent with COPD/emphysema."),
    ],
    "Sarah": [
        ("aisha_patel_brain_mri_t2.jpg", "mri", "MRI Brain T2 Axial: No acute intracranial pathology. No mass, hemorrhage, or midline shift. Normal study."),
        ("aisha_patel_brain_mri_flair.jpg", "mri", "MRI Brain FLAIR Axial: No white matter hyperintensities. No structural abnormality. Normal study for age."),
        ("aisha_patel_brain_mri_t1.jpg", "mri", "MRI Brain T1 Axial: Normal brain parenchyma. No mass effect or enhancement. Unremarkable structural MRI."),
    ],
}


@app.post("/tools/seed_demo_scans")
def seed_demo_scans():
    """Seed demo patients with medical imaging DiagnosticReports (images embedded as base64)."""
    try:
        # First get patient list to map names to IDs
        patients_resp = fhir.client.get("/Patient", params={"_count": "100"})
        patients_resp.raise_for_status()
        patient_map = {}  # name_prefix -> patient_id
        for e in patients_resp.json().get("entry", []):
            r = e["resource"]
            given = r.get("name", [{}])[0].get("given", [""])[0]
            patient_map[given] = r["id"]

        results = []
        for name_prefix, scans in DEMO_SCANS.items():
            pid = patient_map.get(name_prefix)
            if not pid:
                results.append({"name": name_prefix, "error": "Patient not found"})
                continue

            for filename, image_type, conclusion in scans:
                img_path = DEMO_SCANS_DIR / filename
                if not img_path.exists():
                    results.append({"name": name_prefix, "file": filename, "error": "File not found"})
                    continue

                # Read image and encode as base64
                img_data = img_path.read_bytes()
                img_b64 = base64.b64encode(img_data).decode()
                mime = "image/png" if filename.endswith(".png") else "image/jpeg"

                # Determine LOINC code
                loinc_codes = {
                    "chest_xray": ("36643-5", "XR Chest 2 Views"),
                    "dermatology": ("72170-4", "Photographic image"),
                    "pathology": ("60567-5", "Pathology Diagnostic study note"),
                    "general": ("18748-4", "Diagnostic imaging study"),
                }
                code, display = loinc_codes.get(image_type, loinc_codes["general"])

                resource = {
                    "resourceType": "DiagnosticReport",
                    "status": "final",
                    "category": [{"coding": [
                        {"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "RAD", "display": "Radiology"},
                    ]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": display},
                    "subject": {"reference": f"Patient/{pid}"},
                    "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
                    "issued": datetime.now(timezone.utc).isoformat(),
                    "conclusion": conclusion,
                    "presentedForm": [
                        {"contentType": mime, "data": img_b64, "title": filename},
                        {"contentType": "text/plain", "data": base64.b64encode(conclusion.encode()).decode(), "title": "Report"},
                    ],
                }
                resp = fhir.client.post("/DiagnosticReport", json=resource)
                resp.raise_for_status()
                created = resp.json()
                results.append({
                    "patient": name_prefix, "file": filename, "report_id": created["id"],
                    "image_type": image_type, "size_kb": len(img_data) // 1024,
                })

        return {"status": "ok", "data": {"scans_created": len([r for r in results if "report_id" in r]), "details": results}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Iteration 12: Agentic SOAP Enhancement (MedGemma + MCP Tools) ---

class EnhanceSOAPRequest(BaseModel):
    soap_text: str
    patient_id: str | None = None


class EHRNavigateRequest(BaseModel):
    question: str
    patient_id: str


# ─── EHR Navigator Agent (LangGraph) ────────────────────────

from backend.app.services.ehr_navigator import navigate_ehr, navigate_ehr_stream
from fastapi.responses import StreamingResponse


@app.post("/tools/ehr_navigate")
async def ehr_navigate(req: EHRNavigateRequest):
    """
    EHR Navigator Agent — LangGraph multi-step agentic retrieval over FHIR data.

    Inspired by Google's MedGemma EHR Navigator notebook.
    5-step progressive narrowing pipeline:
      1. Discover:  Get FHIR manifest (what resources exist for patient)
      2. Identify:  MedGemma selects relevant resource types
      3. Plan+Fetch: Retrieve relevant resources from HAPI FHIR
      4. Extract:   MedGemma extracts concise facts per resource type
      5. Synthesize: MedGemma produces comprehensive final answer

    Runs on workstation MedGemma (Q4_K_M, GPU) via llama-server on port 8081.
    """
    try:
        result = await navigate_ehr(req.question, req.patient_id)
        return {"status": "ok", "data": result}
    except httpx_async.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to llama-server. Run: ./workstation/start_vision.sh",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/ehr_navigate_stream")
async def ehr_navigate_stream(req: EHRNavigateRequest):
    """
    Streaming EHR Navigator — returns Server-Sent Events (SSE) with step-by-step progress.
    Each line is a JSON object: {"step": "...", "label": "...", "reasoning": "..."}
    Final line: {"step": "done", "data": {full result}}
    """
    return StreamingResponse(
        navigate_ehr_stream(req.question, req.patient_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/tools/enhance_soap")
def enhance_soap_endpoint(req: EnhanceSOAPRequest):
    """
    Agentic SOAP enhancement: MedGemma autonomously calls MCP tools
    to detect drug interactions, suggest ICD-10 codes, correlate labs,
    and synthesize a clinical summary for physician review.
    """
    try:
        result = enhance_soap(req.soap_text, req.patient_id)
        return {"status": "ok", "data": result.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "mcp-server"}


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """EMR-style dashboard showing all clinical FHIR resources."""
    # Fetch all data from HAPI FHIR
    patients_resp = fhir.client.get("/Patient", params={"_count": "100", "_sort": "-_lastUpdated"})
    encounters_resp = fhir.client.get("/Encounter", params={"_count": "100", "_sort": "-_lastUpdated"})
    docs_resp = fhir.client.get("/DocumentReference", params={"_count": "100", "_sort": "-_lastUpdated"})
    conditions_resp = fhir.client.get("/Condition", params={"_count": "100", "_sort": "-_lastUpdated"})
    medrequests_resp = fhir.client.get("/MedicationRequest", params={"_count": "100", "_sort": "-_lastUpdated"})
    issues_resp = fhir.client.get("/DetectedIssue", params={"_count": "100", "_sort": "-_lastUpdated"})
    diagreports_resp = fhir.client.get("/DiagnosticReport", params={"_count": "100", "_sort": "-_lastUpdated"})
    observations_resp = fhir.client.get("/Observation", params={"category": "laboratory", "_count": "100", "_sort": "-_lastUpdated"})

    patients = []
    for e in patients_resp.json().get("entry", []):
        r = e["resource"]
        name = r.get("name", [{}])[0]
        patients.append({
            "id": r["id"],
            "name": " ".join(name.get("given", []) + [name.get("family", "")]),
            "gender": r.get("gender", ""),
            "birthDate": r.get("birthDate", ""),
            "updated": r.get("meta", {}).get("lastUpdated", "")[:19],
        })

    encounters = []
    for e in encounters_resp.json().get("entry", []):
        r = e["resource"]
        reason = ""
        if r.get("reasonCode"):
            reason = r["reasonCode"][0].get("text", "")
        subject_ref = r.get("subject", {}).get("reference", "")
        encounters.append({
            "id": r["id"],
            "patient": subject_ref,
            "status": r.get("status", ""),
            "class": r.get("class", {}).get("display", r.get("class", {}).get("code", "")),
            "reason": reason,
            "period": r.get("period", {}).get("start", "")[:19],
        })

    documents = []
    for e in docs_resp.json().get("entry", []):
        r = e["resource"]
        subject_ref = r.get("subject", {}).get("reference", "")
        enc_refs = r.get("context", {}).get("encounter", [])
        enc_ref = enc_refs[0].get("reference", "") if enc_refs else ""
        # Decode SOAP text
        soap_preview = ""
        soap_full = ""
        try:
            encoded = r["content"][0]["attachment"]["data"]
            full_text = base64.b64decode(encoded).decode()
            soap_preview = full_text[:150].replace("\n", " ") + ("..." if len(full_text) > 150 else "")
            soap_full = full_text
        except (KeyError, IndexError):
            soap_preview = "(no content)"
        documents.append({
            "id": r["id"],
            "patient": subject_ref,
            "encounter": enc_ref,
            "status": r.get("status", ""),
            "date": r.get("date", "")[:19],
            "preview": soap_preview,
            "full_text": soap_full,
        })

    conditions = []
    for e in conditions_resp.json().get("entry", []):
        r = e["resource"]
        subject_ref = r.get("subject", {}).get("reference", "")
        enc_ref = r.get("encounter", {}).get("reference", "")
        codings = r.get("code", {}).get("coding", [])
        icd_code = ""
        snomed_code = ""
        description = r.get("code", {}).get("text", "")
        for c in codings:
            if "icd-10" in c.get("system", ""):
                icd_code = c.get("code", "")
                if not description:
                    description = c.get("display", "")
            elif "snomed" in c.get("system", ""):
                snomed_code = c.get("code", "")
        conditions.append({
            "id": r["id"],
            "patient": subject_ref,
            "encounter": enc_ref,
            "icd10": icd_code,
            "snomed": snomed_code,
            "description": description,
            "date": r.get("recordedDate", "")[:19],
        })

    medrequests = []
    for e in medrequests_resp.json().get("entry", []):
        r = e["resource"]
        subject_ref = r.get("subject", {}).get("reference", "")
        enc_ref = r.get("encounter", {}).get("reference", "")
        med = r.get("medicationCodeableConcept", {})
        drug_name = med.get("text", "")
        rxnorm = ""
        for c in med.get("coding", []):
            if "rxnorm" in c.get("system", ""):
                rxnorm = c.get("code", "")
                if not drug_name:
                    drug_name = c.get("display", "")
        medrequests.append({
            "id": r["id"],
            "patient": subject_ref,
            "encounter": enc_ref,
            "drug": drug_name,
            "rxnorm": rxnorm,
            "status": r.get("status", ""),
            "date": r.get("authoredOn", "")[:19],
        })

    detected_issues = []
    for e in issues_resp.json().get("entry", []):
        r = e["resource"]
        patient_ref = r.get("patient", {}).get("reference", "")
        implicated = ", ".join(
            i.get("reference", "") for i in r.get("implicated", [])
        )
        mitigation = ""
        if r.get("mitigation"):
            mitigation = r["mitigation"][0].get("action", {}).get("text", "")
        detected_issues.append({
            "id": r["id"],
            "patient": patient_ref,
            "severity": r.get("severity", ""),
            "detail": r.get("detail", ""),
            "implicated": implicated,
            "mitigation": mitigation,
            "date": r.get("identifiedDateTime", "")[:19],
        })

    diagreports = []
    for e in diagreports_resp.json().get("entry", []):
        r = e["resource"]
        subject_ref = r.get("subject", {}).get("reference", "")
        loinc_code = ""
        loinc_display = ""
        for c in r.get("code", {}).get("coding", []):
            if "loinc" in c.get("system", ""):
                loinc_code = c.get("code", "")
                loinc_display = c.get("display", "")
        conclusion = r.get("conclusion", "")
        conclusion_short = conclusion[:120] + ("..." if len(conclusion) > 120 else "")
        diagreports.append({
            "id": r["id"],
            "patient": subject_ref,
            "loinc": loinc_code,
            "type": loinc_display,
            "conclusion": conclusion,
            "conclusion_short": conclusion_short,
            "status": r.get("status", ""),
            "date": r.get("issued", "")[:19],
        })

    lab_observations = []
    for e in observations_resp.json().get("entry", []):
        r = e["resource"]
        subject_ref = r.get("subject", {}).get("reference", "")
        coding = r.get("code", {}).get("coding", [{}])[0]
        vq = r.get("valueQuantity", {})
        ref_range = r.get("referenceRange", [{}])[0] if r.get("referenceRange") else {}
        interp = (r.get("interpretation", [{}])[0].get("coding", [{}])[0].get("code", "N"))
        ref_lo = ref_range.get("low", {}).get("value", "")
        ref_hi = ref_range.get("high", {}).get("value", "")
        ref_str = f"{ref_lo}-{ref_hi}" if ref_lo != "" and ref_hi != "" else "--"
        lab_observations.append({
            "id": r["id"],
            "patient": subject_ref,
            "loinc": coding.get("code", ""),
            "test": coding.get("display", r.get("code", {}).get("text", "")),
            "value": vq.get("value", ""),
            "unit": vq.get("unit", ""),
            "ref_range": ref_str,
            "interpretation": interp,
            "date": r.get("effectiveDateTime", "")[:19],
        })

    # Build HTML
    patient_rows = ""
    for p in patients:
        patient_rows += f"""<tr>
            <td>{p['id']}</td><td><strong>{p['name']}</strong></td>
            <td>{p['gender']}</td><td>{p['birthDate']}</td><td>{p['updated']}</td>
        </tr>"""

    encounter_rows = ""
    for e in encounters:
        encounter_rows += f"""<tr>
            <td>{e['id']}</td><td>{e['patient']}</td><td>{e['status']}</td>
            <td>{e['class']}</td><td>{e['reason']}</td><td>{e['period']}</td>
        </tr>"""

    doc_rows = ""
    import html as html_mod
    for i, d in enumerate(documents):
        escaped_full = html_mod.escape(d['full_text']).replace("\n", "<br>") if d['full_text'] else d['preview']
        doc_rows += f"""<tr>
            <td>{d['id']}</td><td>{d['patient']}</td><td>{d['encounter']}</td>
            <td>{d['status']}</td><td>{d['date']}</td>
            <td class="soap-cell">
                <div class="soap-preview" id="preview-{i}" onclick="toggleSoap({i})">{d['preview']} <span class="expand-hint">[expand]</span></div>
                <div class="soap-full" id="full-{i}" onclick="toggleSoap({i})" style="display:none">{escaped_full} <span class="expand-hint">[collapse]</span></div>
            </td>
        </tr>"""

    condition_rows = ""
    for c in conditions:
        condition_rows += f"""<tr>
            <td>{c['id']}</td><td>{c['patient']}</td><td>{c['encounter']}</td>
            <td><strong>{c['icd10']}</strong></td><td>{c['snomed']}</td>
            <td>{c['description']}</td><td>{c['date']}</td>
        </tr>"""

    medrequest_rows = ""
    for m in medrequests:
        medrequest_rows += f"""<tr>
            <td>{m['id']}</td><td>{m['patient']}</td><td>{m['encounter']}</td>
            <td><strong>{m['drug']}</strong></td><td>{m['rxnorm']}</td>
            <td>{m['status']}</td><td>{m['date']}</td>
        </tr>"""

    issue_rows = ""
    for i in detected_issues:
        sev_class = "color:#c62828;font-weight:600" if i['severity'] == 'high' else "color:#e65100;font-weight:600"
        issue_rows += f"""<tr>
            <td>{i['id']}</td><td>{i['patient']}</td>
            <td style="{sev_class}">{i['severity']}</td>
            <td>{i['detail']}</td><td>{i['implicated']}</td>
            <td>{i['mitigation']}</td><td>{i['date']}</td>
        </tr>"""

    diagreport_rows = ""
    for idx, dr in enumerate(diagreports):
        escaped_conclusion = html_mod.escape(dr['conclusion']).replace("\n", "<br>")
        diagreport_rows += f"""<tr>
            <td>{dr['id']}</td><td>{dr['patient']}</td><td>{dr['loinc']}</td>
            <td>{dr['type']}</td><td>{dr['status']}</td><td>{dr['date']}</td>
            <td class="soap-cell">
                <div class="soap-preview" id="dr-preview-{idx}" onclick="toggleDR({idx})">{dr['conclusion_short']} <span class="expand-hint">[expand]</span></div>
                <div class="soap-full" id="dr-full-{idx}" onclick="toggleDR({idx})" style="display:none">{escaped_conclusion} <span class="expand-hint">[collapse]</span></div>
            </td>
        </tr>"""

    lab_rows = ""
    for lab in lab_observations:
        flag = lab['interpretation']
        flag_color = "#c62828" if flag in ("H", "HH") else "#e65100" if flag in ("L", "LL") else "#2e7d32"
        flag_bg = "#ffebee" if flag in ("H", "HH") else "#fff3e0" if flag in ("L", "LL") else "#e8f5e9"
        lab_rows += f"""<tr>
            <td>{lab['id']}</td><td>{lab['patient']}</td><td>{lab['loinc']}</td>
            <td>{lab['test']}</td><td><strong>{lab['value']}</strong> {lab['unit']}</td>
            <td>{lab['ref_range']}</td>
            <td><span style="background:{flag_bg};color:{flag_color};padding:2px 8px;border-radius:4px;font-weight:700;font-size:12px">{flag}</span></td>
            <td>{lab['date']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MedGemma FHIR Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #1a1a2e; }}
        .header {{ background: linear-gradient(135deg, #0f4c75, #1b262c); color: white; padding: 20px 32px; }}
        .header h1 {{ font-size: 22px; font-weight: 600; }}
        .header p {{ font-size: 13px; opacity: 0.8; margin-top: 4px; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        .card {{ background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; overflow: hidden; }}
        .card-header {{ padding: 16px 20px; border-bottom: 1px solid #e8e8e8; display: flex; align-items: center; gap: 10px; }}
        .card-header h2 {{ font-size: 16px; font-weight: 600; }}
        .badge {{ background: #e3f2fd; color: #1565c0; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 10px 16px; background: #fafafa; color: #666; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        td {{ padding: 10px 16px; border-top: 1px solid #f0f0f0; }}
        tr:hover td {{ background: #f8f9ff; }}
        .status-finished {{ color: #2e7d32; font-weight: 600; }}
        .status-current {{ color: #1565c0; font-weight: 600; }}
        .empty {{ padding: 32px; text-align: center; color: #999; }}
        .refresh {{ float: right; background: #0f4c75; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
        .refresh:hover {{ background: #1b6ca8; }}
        .soap-cell {{ max-width: 450px; }}
        .soap-preview {{ cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .soap-full {{ cursor: pointer; white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow-y: auto; padding: 8px; background: #f8f9fa; border-radius: 6px; font-size: 12px; line-height: 1.5; }}
        .expand-hint {{ color: #1565c0; font-size: 11px; font-weight: 600; }}
    </style>
    <script>
        function toggleSoap(i) {{
            var preview = document.getElementById('preview-' + i);
            var full = document.getElementById('full-' + i);
            if (full.style.display === 'none') {{
                preview.style.display = 'none';
                full.style.display = 'block';
            }} else {{
                full.style.display = 'none';
                preview.style.display = 'block';
            }}
        }}
        function toggleDR(i) {{
            var preview = document.getElementById('dr-preview-' + i);
            var full = document.getElementById('dr-full-' + i);
            if (full.style.display === 'none') {{
                preview.style.display = 'none';
                full.style.display = 'block';
            }} else {{
                full.style.display = 'none';
                preview.style.display = 'block';
            }}
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>MedGemma Clinical Dashboard</h1>
        <p>FHIR R4 Store &mdash; HAPI FHIR 8.6.0</p>
    </div>
    <div class="container">
        <button class="refresh" onclick="location.reload()">Refresh</button>

        <div class="card">
            <div class="card-header">
                <h2>Patients</h2>
                <span class="badge">{len(patients)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Name</th><th>Gender</th><th>Birth Date</th><th>Last Updated</th></tr>" + patient_rows + "</table>" if patients else '<div class="empty">No patients yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Encounters</h2>
                <span class="badge">{len(encounters)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>Status</th><th>Class</th><th>Reason</th><th>Date</th></tr>" + encounter_rows + "</table>" if encounters else '<div class="empty">No encounters yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>SOAP Notes (DocumentReference)</h2>
                <span class="badge">{len(documents)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>Encounter</th><th>Status</th><th>Date</th><th>Preview</th></tr>" + doc_rows + "</table>" if documents else '<div class="empty">No documents yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Conditions (Diagnoses)</h2>
                <span class="badge">{len(conditions)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>Encounter</th><th>ICD-10</th><th>SNOMED</th><th>Description</th><th>Date</th></tr>" + condition_rows + "</table>" if conditions else '<div class="empty">No conditions yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Medication Requests</h2>
                <span class="badge">{len(medrequests)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>Encounter</th><th>Drug</th><th>RxNorm</th><th>Status</th><th>Date</th></tr>" + medrequest_rows + "</table>" if medrequests else '<div class="empty">No medication requests yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Detected Issues (Safety Alerts)</h2>
                <span class="badge">{len(detected_issues)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>Severity</th><th>Detail</th><th>Implicated</th><th>Mitigation</th><th>Date</th></tr>" + issue_rows + "</table>" if detected_issues else '<div class="empty">No detected issues yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Diagnostic Reports (Radiology)</h2>
                <span class="badge">{len(diagreports)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>LOINC</th><th>Type</th><th>Status</th><th>Date</th><th>Conclusion</th></tr>" + diagreport_rows + "</table>" if diagreports else '<div class="empty">No diagnostic reports yet</div>'}
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Lab Results (Observations)</h2>
                <span class="badge">{len(lab_observations)}</span>
            </div>
            {"<table><tr><th>ID</th><th>Patient</th><th>LOINC</th><th>Test</th><th>Value</th><th>Ref Range</th><th>Flag</th><th>Date</th></tr>" + lab_rows + "</table>" if lab_observations else '<div class="empty">No lab observations yet</div>'}
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)
