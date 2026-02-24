# MedGemma Clinical Assistant - Architecture Diagrams

---

## 1. Full System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         DOCTOR'S MOBILE PHONE                            ║
║                    (Tecno Spark 40, 8GB RAM, Android)                    ║
║                                                                          ║
║  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────┐ ║
║  │  Microphone   │→│  MedASR      │→│  Transcript Preview (Editable) │ ║
║  │  16kHz PCM    │  │  101MB INT8  │  │  Human Checkpoint #1           │ ║
║  │  30s audio    │  │  ONNX Runtime│  └────────────────────────────────┘ ║
║  └──────────────┘  └──────────────┘            ↓                         ║
║                                     ┌────────────────────────────────┐  ║
║                                     │  MedGemma 1.5 4B (Q3_K_M)     │  ║
║                                     │  2.0 GB GGUF, llama.rn         │  ║
║                                     │  SOAP Note Generation          │  ║
║                                     │  Lab Interpretation (local)    │  ║
║                                     │  General Clinical Chat         │  ║
║                                     └────────────┬───────────────────┘  ║
║                                                  ↓                       ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  Human Checkpoint #2: Review SOAP Note [Approve/Edit/Regenerate] │  ║
║  └───────────────────────┬───────────────────────────────────────────┘  ║
║                          ↓ HTTP (entities only, no PHI)                  ║
╚══════════════════════════╪═══════════════════════════════════════════════╝
                           │
          WiFi LAN (192.168.x.x:8082)
                           │
╔══════════════════════════╪═══════════════════════════════════════════════╗
║              HOSPITAL WORKSTATION (Laptop)                               ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────────┐║
║  │  MCP Server (FastAPI, port 8082)                                    │║
║  │  backend/mcp_server.py — 24 HTTP endpoints                          │║
║  │                                                                     │║
║  │  POST /tools/enhance_soap ← Agentic MedGemma pipeline              │║
║  │    Step 1: MedGemma extracts medications                            │║
║  │    Step 2: MCP Tool → check_drug_interactions (Neo4j)               │║
║  │    Step 3: MCP Tool → suggest_icd10_codes (Neo4j)                   │║
║  │    Step 4: MCP Tool → lookup_patient_labs (FHIR)                    │║
║  │    Step 5: MedGemma synthesizes clinical summary                    │║
║  │                                                                     │║
║  │  POST /tools/analyze_medical_image ← Radiology/Derm/Pathology      │║
║  │  POST /tools/fhir_export_full ← 7-resource FHIR export             │║
║  │  POST /tools/ehr_navigate ← EHR Navigator Agent                    │║
║  │  GET  /tools/list_patients, list_observations, etc.                 │║
║  └─────────┬──────────────────┬──────────────────┬─────────────────────┘║
║            │                  │                  │                       ║
║            ↓                  ↓                  ↓                       ║
║  ┌─────────────────┐ ┌──────────────┐  ┌─────────────────────────────┐ ║
║  │ llama-server     │ │ HAPI FHIR    │  │ MCP Clinical Tools          │ ║
║  │ (port 8081)      │ │ (port 8080)  │  │ (Python, in-process)        │ ║
║  │                  │ │              │  │                             │ ║
║  │ MedGemma Q4_K_M  │ │ Patient      │  │ check_drug_interactions()  │ ║
║  │ + mmproj (vision)│ │ Encounter    │  │ suggest_icd10_codes()      │ ║
║  │ + FunctionGemma  │ │ Observation  │  │ normalize_medication()     │ ║
║  │                  │ │ Condition    │  │ search_drug()              │ ║
║  │ GPU: -ngl 99     │ │ MedicationRx │  │ search_icd10()             │ ║
║  │ Context: 4096    │ │ DocReference │  │ lookup_patient_labs()      │ ║
║  │                  │ │ DiagReport   │  │ export_clinical_encounter()│ ║
║  │ Radiology AI     │ │ DetectedIssue│  │                             │ ║
║  │ Derm AI          │ │              │  │ Backed by:                  │ ║
║  │ Pathology AI     │ │ Docker       │  │ ├─ Neo4j Aura (cloud)      │ ║
║  │ Text inference   │ │ hapiproject/ │  │ ├─ RxNorm API (free)       │ ║
║  └─────────────────┘ │ hapi:latest   │  │ └─ UMLS/SNOMED API         │ ║
║                       └──────────────┘  └─────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════╝
                                                    │
                              Internet (entities only, no PHI)
                                                    │
╔═══════════════════════════════════════════════════╪══════════════════════╗
║                    CLOUD SERVICES (No PHI)                               ║
║                                                                          ║
║  ┌───────────────────────┐  ┌──────────────────┐  ┌──────────────────┐ ║
║  │ Neo4j Aura             │  │ NLM RxNorm API   │  │ UMLS API         │ ║
║  │                        │  │ (Free, no key)   │  │ (API key)        │ ║
║  │ Drug nodes: 1,751      │  │                  │  │                  │ ║
║  │ DDI edges: 222,271     │  │ Drug → RxCUI     │  │ SNOMED CT search │ ║
║  │ ICD-10 codes: 98,186   │  │ Fuzzy matching   │  │ ICD-10 crosswalk │ ║
║  │ Fulltext indexes       │  │                  │  │                  │ ║
║  └───────────────────────┘  └──────────────────┘  └──────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Lab Results: 3-Button Design

```
┌─────────────────────────────────────────────────────────────┐
│  Lab Results — Wei Chen                                      │
│  19 results loaded                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ View Results │  │ AI Summary   │  │ EHR Navigator     │  │
│  │ (Table)      │  │ (On-Device)  │  │ (Workstation)     │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                │                    │              │
│         ↓                ↓                    ↓              │
│  Show lab table   MedGemma on phone    Send to workstation  │
│  with flags       pastes all labs      EHR Navigator Agent  │
│  (H/L/HH/LL)     into chat prompt     fetches relevant     │
│                   Single LLM call      FHIR resources only  │
│                   ~60s response        Multi-step agentic   │
│                                        Deep clinical Q&A    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Privacy Architecture

```
┌─────────────────────────────┐    ┌───────────────────────────────┐
│  ON-DEVICE (Phone)           │    │  WORKSTATION (Hospital LAN)    │
│  100% Private                │    │  Hospital-controlled           │
│                              │    │                               │
│  - Audio recording           │    │  - HAPI FHIR (patient data)   │
│  - MedASR transcription      │    │  - MedGemma vision (images)   │
│  - MedGemma SOAP generation  │    │  - Agentic enhancement        │
│  - MedGemma lab summary      │    │  - EHR Navigator Agent        │
│  - General clinical chat     │    │  - FHIR export                │
│                              │    │                               │
│  PHI stays here              │    │  PHI stays here               │
└──────────────────────────────┘    └───────────────┬───────────────┘
                                                     │
                                    Only de-identified entities
                                    (drug names, symptom terms)
                                                     ↓
                                    ┌───────────────────────────────┐
                                    │  CLOUD (No PHI)               │
                                    │  - Neo4j Aura (DDI, ICD-10)  │
                                    │  - RxNorm API (drug normalize)│
                                    │  - UMLS API (SNOMED CT)       │
                                    └───────────────────────────────┘
```

---

## 4. Data Flow: Dictate SOAP (Fully On-Device)

```
Doctor speaks → Microphone (16kHz PCM)
  → MedASR (ONNX, 101MB, pre-loaded) → Transcript
  → Human reviews/edits transcript [Checkpoint #1]
  → MedGemma (GGUF Q3_K_M, 2.0GB) → SOAP Note
  → Human reviews/approves SOAP [Checkpoint #2]
  → [Optional] Send to workstation for enhancement
```

---

## 5. Data Flow: SOAP Enhancement (Workstation Agentic)

```
Phone sends SOAP text → Workstation MCP Server (8082)
  → Step 1: MedGemma extracts medications
  → Step 2: MCP Tool → check_drug_interactions → Neo4j (222K DDI)
  → Step 3: MCP Tool → suggest_icd10_codes → Neo4j (98K codes)
  → Step 4: MCP Tool → lookup_patient_labs → HAPI FHIR
  → Step 5: MedGemma synthesizes summary
  → Phone shows findings for human review [Checkpoint #3]
  → Doctor accepts/dismisses each finding
  → FHIR export with all accepted findings
```

---

## 6. Data Flow: Lab Results (3 Paths)

```
Phone loads labs from FHIR (via workstation) → 19 results

Path A: View Results → Table with H/L/HH/LL flags (instant)

Path B: AI Summary → MedGemma on phone
  → Paste all labs into system prompt
  → Single LLM call → clinical interpretation (~60s)

Path C: EHR Navigator → Workstation agent
  → Discover available FHIR resources for patient
  → MedGemma plans which to retrieve
  → Fetch relevant resources (labs, conditions, meds)
  → Extract facts from each
  → Synthesize comprehensive answer (~10-20s)
```

---

## 7. Data Flow: Radiology (Workstation Vision)

```
Doctor selects patient → Views scans
  → Picks image (X-ray, derm, pathology)
  → Phone sends image to workstation (8082)
  → MCP Server forwards to llama-server (8081)
  → MedGemma Q4_K_M + mmproj analyzes image
  → Returns FINDINGS + IMPRESSION
  → Doctor reviews report
  → [Optional] Export as DiagnosticReport to FHIR
```

---

## 8. EHR Navigator Agent Flow (Google Pattern)

```
┌────────────────────────────────────────────────────────────┐
│  Doctor asks: "What's causing this patient's anemia?"      │
└─────────────────────────┬──────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: DISCOVER                                           │
│  Get FHIR resource manifest for patient                     │
│  → Patient has: 19 Observations, 3 Conditions,              │
│    2 MedicationRequests, 1 Encounter                        │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: PLAN                                               │
│  MedGemma decides relevant resource types:                  │
│  → "For anemia, I need: Observations (CBC, Iron, B12,       │
│     Ferritin), Conditions (diagnoses), MedicationRequests"  │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: FETCH + EXTRACT (per resource type)                │
│  Fetch Observations → Extract: "Hgb 11 (L), MCV 72 (L)"   │
│  Fetch Conditions → Extract: "Iron deficiency, T2DM"       │
│  Fetch Medications → Extract: "Metformin (can affect B12)" │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: SYNTHESIZE                                         │
│  MedGemma combines all facts:                               │
│  "Patient has microcytic anemia (Hgb 11, MCV 72) likely    │
│   due to iron deficiency. Metformin may also contribute     │
│   to B12 deficiency. Recommend: iron studies, B12 level,   │
│   reticulocyte count."                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Port Map

```
┌───────────────┬────────┬──────────────────────────────┐
│ Service       │ Port   │ Purpose                       │
├───────────────┼────────┼──────────────────────────────┤
│ HAPI FHIR     │ 8080   │ Clinical data store (R4)     │
│ llama-server  │ 8081   │ MedGemma vision + text       │
│ MCP Server    │ 8082   │ HTTP bridge to MCP tools     │
│ MCP SSE       │ 8083   │ Standalone MCP (for judges)  │
│ Metro (dev)   │ 8081*  │ React Native dev server      │
└───────────────┴────────┴──────────────────────────────┘
* Metro shares port with llama-server (not run simultaneously)
```
