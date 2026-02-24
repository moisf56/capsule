"""
FHIR Resource Client — Iteration 8

Thin wrapper around HAPI FHIR REST API.
Resources: Patient, Encounter, DocumentReference, Condition, MedicationRequest,
           DetectedIssue, DiagnosticReport.
"""

import base64
import os
from datetime import datetime, timezone

import httpx


FHIR_BASE = os.getenv("FHIR_BASE", "http://localhost:8080/fhir")

HEADERS = {
    "Content-Type": "application/fhir+json",
    "Accept": "application/fhir+json",
}


class FHIRClient:
    def __init__(self, base_url: str = FHIR_BASE):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, headers=HEADERS, timeout=10)

    def close(self):
        self.client.close()

    # --- List / Search ---

    def list_patients(self) -> list[dict]:
        """List all patients. Returns [{id, name, gender, birthDate}]."""
        resp = self.client.get("/Patient", params={"_count": "100"})
        resp.raise_for_status()
        patients = []
        for entry in resp.json().get("entry", []):
            r = entry["resource"]
            human_name = r.get("name", [{}])[0]
            display = " ".join(human_name.get("given", []) + [human_name.get("family", "")])
            patients.append({
                "id": r["id"],
                "name": display.strip(),
                "gender": r.get("gender"),
                "birthDate": r.get("birthDate"),
            })
        return patients

    def search_patient(self, name: str) -> list[dict]:
        """Search patients by name. Returns list of {id, name, birthDate, gender}."""
        resp = self.client.get("/Patient", params={"name": name})
        resp.raise_for_status()
        bundle = resp.json()

        patients = []
        for entry in bundle.get("entry", []):
            r = entry["resource"]
            human_name = r.get("name", [{}])[0]
            display = " ".join(human_name.get("given", []) + [human_name.get("family", "")])
            patients.append({
                "id": r["id"],
                "name": display.strip(),
                "birthDate": r.get("birthDate"),
                "gender": r.get("gender"),
            })
        return patients

    # --- Create ---

    def create_patient(self, family: str, given: str, gender: str, birth_date: str) -> dict:
        """Create a Patient resource. Returns {id, name}."""
        resource = {
            "resourceType": "Patient",
            "name": [{"family": family, "given": [given]}],
            "gender": gender,
            "birthDate": birth_date,
        }
        resp = self.client.post("/Patient", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "name": f"{given} {family}"}

    def create_anonymous_patient(self) -> dict:
        """Create an anonymous walk-in patient with a timestamp-based identifier.
        Returns {id, name}. The patient appears in the FHIR patient list immediately.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        resource = {
            "resourceType": "Patient",
            "identifier": [{"system": "urn:capsule:anonymous", "value": ts}],
            "name": [{"family": "Anonymous", "given": ["Patient"]}],
            "gender": "unknown",
        }
        resp = self.client.post("/Patient", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "name": f"Anonymous Patient ({ts})"}

    def create_encounter(self, patient_id: str, reason: str) -> dict:
        """Create an ambulatory Encounter. Returns {id, status, reason}."""
        resource = {
            "resourceType": "Encounter",
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "reasonCode": [{"text": reason}],
            "period": {
                "start": datetime.now(timezone.utc).isoformat(),
            },
        }
        resp = self.client.post("/Encounter", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "status": created["status"], "reason": reason}

    def create_document_reference(
        self, patient_id: str, encounter_id: str, soap_text: str
    ) -> dict:
        """Store a SOAP note as a DocumentReference (base64 encoded). Returns {id, status}."""
        encoded = base64.b64encode(soap_text.encode()).decode()
        resource = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11506-3",
                        "display": "Progress note",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "context": {
                "encounter": [{"reference": f"Encounter/{encounter_id}"}],
            },
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": encoded,
                    }
                }
            ],
            "date": datetime.now(timezone.utc).isoformat(),
        }
        resp = self.client.post("/DocumentReference", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "status": created["status"]}

    # --- Iteration 7: Condition, MedicationRequest, DetectedIssue ---

    def create_condition(
        self,
        patient_id: str,
        encounter_id: str,
        icd_code: str,
        icd_display: str,
        snomed_code: str | None = None,
        snomed_display: str | None = None,
    ) -> dict:
        """Create a Condition with ICD-10 coding, optionally dual-coded with SNOMED CT."""
        codings = [
            {
                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                "code": icd_code,
                "display": icd_display,
            }
        ]
        if snomed_code and snomed_display:
            codings.append({
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "display": snomed_display,
            })

        resource = {
            "resourceType": "Condition",
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                    "display": "Active",
                }]
            },
            "verificationStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                    "display": "Confirmed",
                }]
            },
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                    "code": "encounter-diagnosis",
                    "display": "Encounter Diagnosis",
                }]
            }],
            "code": {
                "coding": codings,
                "text": icd_display,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "recordedDate": datetime.now(timezone.utc).isoformat(),
        }
        resp = self.client.post("/Condition", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "icd_code": icd_code, "snomed_code": snomed_code}

    def create_medication_request(
        self,
        patient_id: str,
        encounter_id: str,
        drug_name: str,
        rxnorm_code: str | None = None,
        rxnorm_display: str | None = None,
    ) -> dict:
        """Create a MedicationRequest, optionally RxNorm-coded."""
        med_concept: dict = {"text": drug_name}
        if rxnorm_code and rxnorm_display:
            med_concept["coding"] = [{
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": rxnorm_code,
                "display": rxnorm_display,
            }]

        resource = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": med_concept,
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "authoredOn": datetime.now(timezone.utc).isoformat(),
        }
        resp = self.client.post("/MedicationRequest", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "drug": drug_name, "rxnorm_code": rxnorm_code}

    def create_detected_issue(
        self,
        patient_id: str,
        detail: str,
        severity: str,
        implicated_med_ids: list[str],
        acknowledged: bool = True,
    ) -> dict:
        """Create a DetectedIssue for a drug-drug interaction alert."""
        resource = {
            "resourceType": "DetectedIssue",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "DRG",
                    "display": "Drug Interaction Alert",
                }]
            },
            "severity": severity,
            "detail": detail,
            "patient": {"reference": f"Patient/{patient_id}"},
            "implicated": [
                {"reference": f"MedicationRequest/{mid}"}
                for mid in implicated_med_ids
            ],
            "identifiedDateTime": datetime.now(timezone.utc).isoformat(),
        }
        if acknowledged:
            resource["mitigation"] = [{
                "action": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                        "code": "EMAUTH",
                        "display": "emergency authorization override",
                    }],
                    "text": "Physician acknowledged interaction and approved export",
                },
                "date": datetime.now(timezone.utc).isoformat(),
            }]

        resp = self.client.post("/DetectedIssue", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "detail": detail, "severity": severity}

    # --- Iteration 8: DiagnosticReport ---

    def create_diagnostic_report(
        self,
        patient_id: str,
        encounter_id: str | None,
        conclusion: str,
        findings_text: str,
        image_type: str = "chest_xray",
    ) -> dict:
        """Create a DiagnosticReport for medical image analysis (FHIR R4)."""
        loinc_codes = {
            "chest_xray": ("36643-5", "XR Chest 2 Views"),
            "mri": ("24590-2", "MR Brain"),
            "ct": ("24727-0", "CT study"),
            "fundoscopy": ("32451-7", "Physical findings of Eye"),
            "ecg": ("11524-6", "EKG study"),
            "dermatology": ("72170-4", "Photographic image"),
            "pathology": ("60567-5", "Pathology Diagnostic study note"),
            "general": ("18748-4", "Diagnostic imaging study"),
        }
        code, display = loinc_codes.get(image_type, loinc_codes["general"])

        full_report = f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{conclusion}"
        encoded_report = base64.b64encode(full_report.encode()).decode()

        resource: dict = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "category": [{
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    },
                    {
                        "system": "http://snomed.info/sct",
                        "code": "394914008",
                        "display": "Radiology",
                    },
                ]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": code,
                    "display": display,
                }],
                "text": display,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
            "issued": datetime.now(timezone.utc).isoformat(),
            "conclusion": conclusion,
            "presentedForm": [{
                "contentType": "text/plain",
                "data": encoded_report,
                "title": f"MedGemma Image Analysis ({image_type})",
            }],
        }
        if encounter_id:
            resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}

        resp = self.client.post("/DiagnosticReport", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {"id": created["id"], "status": created["status"], "conclusion": conclusion}

    # --- Iteration 9: Observation (Lab Results) ---

    def create_observation(
        self,
        patient_id: str,
        encounter_id: str | None,
        loinc_code: str,
        loinc_display: str,
        value: float,
        unit: str,
        reference_low: float | None = None,
        reference_high: float | None = None,
        status: str = "final",
    ) -> dict:
        """Create an Observation for a lab result (FHIR R4, LOINC coded)."""
        # Compute interpretation
        interpretation = "N"
        if reference_low is not None and reference_high is not None:
            if value > reference_high:
                interpretation = "HH" if value > reference_high * 1.5 else "H"
            elif value < reference_low:
                interpretation = "LL" if value < reference_low * 0.5 else "L"

        interp_display = {
            "N": "Normal", "H": "High", "L": "Low",
            "HH": "Critically high", "LL": "Critically low",
        }

        resource: dict = {
            "resourceType": "Observation",
            "status": status,
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "laboratory",
                    "display": "Laboratory",
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": loinc_display,
                }],
                "text": loinc_display,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
            "valueQuantity": {
                "value": value,
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit,
            },
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": interpretation,
                    "display": interp_display[interpretation],
                }]
            }],
        }

        if encounter_id:
            resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}

        if reference_low is not None and reference_high is not None:
            resource["referenceRange"] = [{
                "low": {"value": reference_low, "unit": unit, "system": "http://unitsofmeasure.org"},
                "high": {"value": reference_high, "unit": unit, "system": "http://unitsofmeasure.org"},
            }]

        resp = self.client.post("/Observation", json=resource)
        resp.raise_for_status()
        created = resp.json()
        return {
            "id": created["id"],
            "loinc_code": loinc_code,
            "value": value,
            "unit": unit,
            "interpretation": interpretation,
        }

    def search_observations(self, patient_id: str) -> list[dict]:
        """Search Observations (lab results) for a patient. Returns parsed list."""
        resp = self.client.get(
            "/Observation",
            params={
                "subject": f"Patient/{patient_id}",
                "category": "laboratory",
                "_count": "100",
            },
        )
        resp.raise_for_status()
        bundle = resp.json()

        results = []
        for entry in bundle.get("entry", []):
            r = entry["resource"]
            code_obj = r.get("code", {})
            coding = code_obj.get("coding", [{}])[0]
            vq = r.get("valueQuantity", {})
            ref_range = r.get("referenceRange", [{}])[0] if r.get("referenceRange") else {}
            interp_coding = (r.get("interpretation", [{}])[0]
                             .get("coding", [{}])[0])

            results.append({
                "id": r["id"],
                "loinc_code": coding.get("code", ""),
                "loinc_display": coding.get("display", code_obj.get("text", "")),
                "value": vq.get("value"),
                "unit": vq.get("unit", ""),
                "reference_low": ref_range.get("low", {}).get("value"),
                "reference_high": ref_range.get("high", {}).get("value"),
                "interpretation": interp_coding.get("code", "N"),
                "effective_date": r.get("effectiveDateTime", ""),
            })
        return results

    # --- Read ---

    def get_document_text(self, doc_id: str) -> str:
        """Retrieve a DocumentReference and decode its base64 content."""
        resp = self.client.get(f"/DocumentReference/{doc_id}")
        resp.raise_for_status()
        doc = resp.json()
        encoded = doc["content"][0]["attachment"]["data"]
        return base64.b64decode(encoded).decode()
