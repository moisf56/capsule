# MedGemma Clinical Assistant — System Summary

**Project**: Edge AI Medical Documentation System
**Competition**: MedGemma Impact Challenge — Edge AI Prize
**Device Tested**: Tecno Spark 40 (8GB RAM, Android 15, arm64-v8a)
**Current Milestone**: M4 — Performance optimization + EHR Navigator Agent

---

## What It Does

Record doctor dictation → Auto-transcribe with MedASR → Generate structured SOAP note with MedGemma → Autonomous clinical enhancement (DDI, ICD-10, lab correlations) → EHR Navigator for deep clinical Q&A → Doctor reviews and approves → FHIR R4 export.

**All patient data stays on-device or on hospital workstation. No PHI to cloud.**

---

## Full System Architecture

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
║  │  MCP Server (FastAPI, port 8082) — 25 HTTP endpoints                │║
║  │                                                                     │║
║  │  POST /tools/enhance_soap ← Agentic MedGemma pipeline              │║
║  │    Step 1: MedGemma extracts medications                            │║
║  │    Step 2: MCP Tool → check_drug_interactions (Neo4j)               │║
║  │    Step 3: MCP Tool → suggest_icd10_codes (Neo4j)                   │║
║  │    Step 4: MCP Tool → lookup_patient_labs (FHIR)                    │║
║  │    Step 5: MedGemma synthesizes clinical summary                    │║
║  │                                                                     │║
║  │  POST /tools/ehr_navigate ← EHR Navigator Agent (LangGraph)        │║
║  │    Step 1: Discover FHIR manifest                                   │║
║  │    Step 2: MedGemma identifies relevant resource types              │║
║  │    Step 3: Fetch relevant FHIR resources                            │║
║  │    Step 4: MedGemma extracts concise facts per resource type        │║
║  │    Step 5: MedGemma synthesizes comprehensive answer                │║
║  │                                                                     │║
║  │  POST /tools/analyze_medical_image ← Radiology/Derm/Pathology      │║
║  │  POST /tools/fhir_export_full ← 7-resource FHIR export             │║
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
║  │ EHR Navigator    │ │ hapi:latest  │  │ └─ UMLS/SNOMED API         │ ║
║  └─────────────────┘ └──────────────┘  └─────────────────────────────┘ ║
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

## Models

### MedASR (Speech-to-Text) — On-Device

| Property | Value |
|----------|-------|
| Source | `google/medasr` |
| Architecture | 105M param Conformer CTC |
| Original size | 402MB (PyTorch) |
| On-device size | **101MB** (INT8 quantized ONNX) |
| Runtime | onnxruntime-react-native 1.24.1 |
| Vocabulary | 512 SentencePiece tokens |
| Input | Mel spectrogram (128 mel bins, 400-sample window, 160-sample hop) |
| Output | CTC logits → greedy decode |
| Performance | 20-30s for 30s audio on Tecno Spark 40 |

### MedGemma (SOAP Generation) — On-Device

| Property | Value |
|----------|-------|
| Source | `google/medgemma-1.5-4b-it` |
| Architecture | Gemma 3, 3.9B params, 34 layers |
| On-device size | **2.0GB** (Q3_K_M GGUF) |
| Runtime | llama.rn v0.11.0-rc.3 (llama.cpp bindings) |
| Context window | 1024 tokens (optimized from 2048) |
| Generation params | n_predict=700, temp=0.3, top_k=40, top_p=0.9, repeat_penalty=1.2 |
| System prompt | Focused SOAP-only prompt to reduce thinking overhead |
| Thinking trace | Stripped: `<think>`, `<unused94>`, `model_output\n` prefixes |
| Performance | ~60-120s SOAP generation on Tecno Spark 40 |

### MedGemma (Workstation) — GPU Accelerated

| Property | Value |
|----------|-------|
| Format | Q4_K_M GGUF + mmproj (vision) |
| Runtime | llama-server (llama.cpp) on port 8081 |
| GPU offload | `-ngl 99` (full offload) |
| Context | 4096 tokens |
| Capabilities | Text inference, vision (radiology/derm/pathology), EHR Navigator |

### Memory Management (Phone)

Both models cannot coexist in 8GB RAM. Sequential loading:
1. MedASR loads (101MB) → transcribes → unloads
2. MedGemma loads (2.0GB) → generates SOAP / chat → stays loaded

---

## Key Features

### 1. Dictate SOAP (Fully On-Device)
```
Doctor speaks → Microphone (16kHz PCM)
  → MedASR (ONNX, 101MB) → Transcript
  → Human reviews/edits transcript
  → MedGemma (GGUF Q3_K_M, 2.0GB) → SOAP Note
  → Human reviews/approves SOAP
```

### 2. SOAP Enhancement (Workstation Agentic)
```
Phone sends SOAP text → Workstation MCP Server (8082)
  → MedGemma extracts medications
  → MCP Tool: check_drug_interactions → Neo4j (222K DDI)
  → MCP Tool: suggest_icd10_codes → Neo4j (98K codes)
  → MCP Tool: lookup_patient_labs → HAPI FHIR
  → MedGemma synthesizes summary
  → Phone shows findings for human review
  → Doctor accepts/dismisses each finding
```

### 3. Lab Results (3 Paths)
```
Phone loads labs from FHIR (via workstation) → N results

Path A: View Results → Table with H/L/HH/LL flags (instant)

Path B: AI Summary → MedGemma on phone
  → Paste all labs into system prompt
  → Single LLM call → clinical interpretation (~60s)

Path C: EHR Navigator → Workstation LangGraph agent
  → Discover available FHIR resources for patient
  → MedGemma plans which to retrieve
  → Fetch relevant resources (labs, conditions, meds)
  → Extract facts from each resource type
  → Synthesize comprehensive answer (~10-20s)
```

### 4. EHR Navigator Agent (LangGraph)

Inspired by [Google's MedGemma EHR Navigator notebook](https://github.com/google-health/medgemma/blob/main/notebooks/ehr_navigator_agent.ipynb).

**Architecture**: 5-node LangGraph StateGraph with progressive narrowing:

| Node | Function | LLM Temp |
|------|----------|----------|
| `discover_manifest` | Scan all FHIR resource types | N/A |
| `identify_relevant_types` | LLM selects relevant types | 0.0 |
| `execute_and_extract` | Fetch + LLM fact extraction | 0.6 |
| `synthesize_answer` | LLM comprehensive answer | 0.1 |

**Key design patterns** (from Google notebook):
- Progressive narrowing: scan everything → identify relevant → fetch only needed
- Per-resource-type fact extraction before synthesis (prevents context overflow)
- "Fact extractor" vs "question answerer" separation
- "Think silently if needed" system instruction for MedGemma

**Tech stack**: LangGraph + LangChain Core + langchain-openai (→ llama-server)

### 5. Radiology (Workstation Vision)
```
Doctor selects patient → Picks image (X-ray, derm, pathology)
  → Phone sends image to workstation (8082)
  → MCP Server forwards to llama-server (8081)
  → MedGemma Q4_K_M + mmproj analyzes image
  → Returns FINDINGS + IMPRESSION
  → Doctor reviews report
  → Optional: Export as DiagnosticReport to FHIR
```

### 6. FHIR R4 Export (7 Resource Types)
```
Phone triggers full export → Workstation MCP Server
  → Creates: Encounter
  → Creates: DocumentReference (SOAP note, base64)
  → Creates: MedicationRequests (RxNorm coded via NLM API)
  → Creates: Conditions (ICD-10 + SNOMED CT dual-coded via UMLS API)
  → Creates: DetectedIssues (DDI alerts, physician-acknowledged)
  → Creates: Observations (lab results)
  → Creates: DiagnosticReports (radiology/pathology)
```

---

## Human-in-the-Loop Checkpoints

| Checkpoint | Screen | Requirement |
|------------|--------|-------------|
| #1 | Transcript | Review/edit MedASR output (optional) |
| #2 | SOAP Note | Approve/edit/regenerate MedGemma output (required) |
| #3 | Agent Findings | Review DDI alerts, ICD-10 suggestions, lab correlations (required for critical alerts) |

---

## Privacy Architecture (3 Tiers)

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

## Mobile App

| Property | Value |
|----------|-------|
| Framework | React Native 0.83.1 + TypeScript |
| APK size | 176.4MB (169.5MB = llama.cpp native libs) |
| Audio | react-native-live-audio-stream (16kHz mono 16-bit PCM) |
| ML inference | onnxruntime-react-native (MedASR), llama.rn (MedGemma) |
| MCP Client | HTTP fetch to workstation (FHIR, Neo4j, vision, EHR Navigator) |

### Screens

| Screen | Purpose |
|--------|---------|
| Home | Feature grid: Dictate SOAP, Lab Results, Radiology Center |
| Recording | Live audio capture with waveform timer |
| Transcript | Editable MedASR output (Checkpoint #1) |
| SOAP | MedGemma-generated SOAP note (Checkpoint #2) |
| Alerts | Agent findings: DDI, ICD-10, labs (Checkpoint #3) |
| Export | FHIR export summary |
| Labs | Patient list → lab table → AI Summary / EHR Navigator |
| Radiology | Patient list → scan history → new analysis → export |
| Chat | General clinical chat with MedGemma (voice + text) |

---

## Backend Services

| Service | Port | Purpose |
|---------|------|---------|
| HAPI FHIR | 8080 | FHIR R4 store (Docker: `hapiproject/hapi:latest`) |
| llama-server | 8081 | MedGemma workstation (Q4_K_M + mmproj, GPU) |
| MCP Server | 8082 | FastAPI bridge: 25 endpoints |

### MCP Server Endpoints (25 total)

| Endpoint | Category |
|----------|----------|
| `POST /tools/enhance_soap` | Agentic SOAP enhancement |
| `POST /tools/ehr_navigate` | EHR Navigator Agent (LangGraph) |
| `POST /tools/analyze_medical_image` | Vision AI (radiology/derm/pathology) |
| `POST /tools/fhir_export_full` | Full clinical FHIR export |
| `POST /tools/check_drug_interactions` | Neo4j DDI check |
| `POST /tools/search_icd10` | Neo4j ICD-10 search |
| `POST /tools/search_drug` | Neo4j drug search |
| `POST /tools/normalize_drug` | RxNorm normalization |
| `POST /tools/search_snomed` | SNOMED CT search (UMLS) |
| `POST /tools/suggest_codes` | ICD-10 code suggestions from SOAP |
| + 15 FHIR CRUD endpoints | Patient, Encounter, Observation, etc. |

### Neo4j Knowledge Graph

| Data | Count |
|------|-------|
| Drug nodes | 1,751 |
| Drug-drug interaction edges | 222,271 |
| ICD-10 codes | 98,186 |
| Fulltext indexes | Drug names, ICD-10 descriptions |

### Cloud APIs (No PHI)

| API | Purpose | Auth |
|-----|---------|------|
| NLM RxNorm | Drug → RxCUI normalization | Free, no key |
| UMLS | SNOMED CT search, ICD-10 crosswalk | API key |
| Neo4j Aura | DDI, ICD-10, drug search | Credentials |

---

## Quantization Reference

| Format | Bits | Size | Quality | Use Case |
|--------|------|------|---------|----------|
| Q4_K_M | 4.83 | 2.4GB | Best | Workstation (backed up on phone) |
| **Q3_K_M** | **3.07** | **2.0GB** | **Good** | **Phone (current)** |
| Q2_K | 2.96 | 1.5GB | Poor (+3.5 PPL) | Too low quality |
| IQ2_M | 2.7 | 1.3GB | Medium | Too slow on mobile CPU |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `mobile/MedGemmaApp/App.tsx` | Main app: all screens, inference orchestration |
| `mobile/MedGemmaApp/src/MelSpectrogram.ts` | Mel spectrogram (matches LasrFeatureExtractor) |
| `mobile/MedGemmaApp/src/CTCDecoder.ts` | CTC greedy decode + punctuation formatting |
| `mobile/MedGemmaApp/src/MCPClient.ts` | MCP client: FHIR, Neo4j, vision, EHR Navigator |
| `backend/mcp_server.py` | MCP Server: 25 FastAPI endpoints |
| `backend/app/services/ehr_navigator.py` | EHR Navigator Agent (LangGraph, 5-step) |
| `backend/app/services/enhance_service.py` | Agentic SOAP enhancement pipeline |
| `backend/app/services/neo4j_service.py` | Neo4j Aura connection + DDI/ICD-10 queries |
| `backend/app/services/terminology_service.py` | RxNorm + SNOMED + ICD-10 terminology |
| `backend/fhir_resources.py` | FHIR R4 client (HAPI FHIR REST wrapper) |
| `backend/mcp_clinical_tools.py` | 7 MCP tools (FastMCP SDK) |

---

## Quick Start

```bash
# 1. Start HAPI FHIR (Docker)
cd workstation && docker compose up -d

# 2. Start llama-server (workstation MedGemma, GPU)
./workstation/start_vision.sh

# 3. Start MCP Server
cd project && source medgemma-env/bin/activate
python -m uvicorn backend.mcp_server:app --host 0.0.0.0 --port 8082

# 4. Seed demo patients
curl -X POST http://localhost:8082/tools/seed_demo_patients

# 5. Start mobile app
cd mobile/MedGemmaApp && npx react-native start --port 8081
# (In another terminal) npx react-native run-android

# 6. Push models to phone
adb push ml-models/gguf/medgemma-1.5-4b-it-Q3_K_M.gguf /data/local/tmp/medgemma.gguf
adb push ml-models/onnx/medasr_int8.onnx /data/local/tmp/medasr_int8.onnx
```

---

## Milestones

| Milestone | Date | Achievement |
|-----------|------|-------------|
| M1 | 2026-02-06 | MedGemma running on-device on Android phone |
| M2 | 2026-02-08 | Full pipeline: Record → MedASR → MedGemma SOAP |
| M3 | 2026-02-21 | MCP Server + Agentic MedGemma Enhancement |
| M4 | 2026-02-21 | Q3_K_M optimization + EHR Navigator Agent (LangGraph) |

---

## What's Completed

- [x] MedASR on-device (INT8 ONNX, 101MB)
- [x] MedGemma on-device (Q3_K_M GGUF, 2.0GB)
- [x] Full dictation pipeline (Record → Transcribe → SOAP)
- [x] Human-in-the-loop checkpoints (#1, #2, #3)
- [x] MCP Server with 25 endpoints
- [x] Neo4j Aura (222K DDI, 98K ICD-10)
- [x] RxNorm + SNOMED CT + UMLS integration
- [x] Agentic SOAP enhancement (MedGemma + MCP tools)
- [x] EHR Navigator Agent (LangGraph, 5-step)
- [x] Radiology/Derm/Pathology vision AI
- [x] FHIR R4 export (7 resource types, RxNorm + SNOMED dual-coded)
- [x] Lab results with 3-path analysis (View/AI Summary/EHR Navigator)
- [x] Clinical chat with voice input (MedASR)
- [x] FHIR Dashboard (web UI)

## What's Remaining

- [ ] Demo video (3 minutes)
- [ ] Technical writeup (3 pages)
- [ ] Architecture diagrams (draw.io/Figma)
- [ ] End-to-end testing on phone (all 5 flows)
- [ ] Code cleanup + final polish
