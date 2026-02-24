# Capsule — AI Clinical Documentation Assistant

> **MedGemma Impact Challenge Submission**
> Edge AI · Privacy-First · Human-in-the-Loop

Capsule turns a 30-second voice dictation into a complete, safety-reviewed clinical note — running MedGemma and MedASR **entirely on the doctor's devices**, with no patient data ever leaving the clinic.

---

## Demo

[![Demo Video](docs/assets/demo-thumbnail.png)](https://youtube.com/TODO)

---

## What It Does

| Step | Where it runs | What happens |
|------|--------------|--------------|
| 1. Doctor dictates | **Phone (on-device)** | MedASR transcribes speech in real time |
| 2. SOAP note generated | **Phone (on-device)** | MedGemma 4B GGUF produces structured clinical note |
| 3. Human checkpoint | **Phone** | Doctor reviews and approves base note |
| 4. Agentic enhancement | **Workstation GPU** | MedGemma identifies meds & diagnoses, queries knowledge graph |
| 5. Safety review | **Phone** | Doctor reviews DDI alerts, ICD-10 codes, abnormal labs |
| 6. FHIR export | **Local FHIR server** | Full R4 bundle (Encounter, Conditions, MedicationRequests) |

**Key privacy guarantee**: Steps 1–3 and 5–6 run entirely on the doctor's phone. Step 4 sends only de-identified medical terms (drug names, diagnoses) to the local workstation — never patient names, dates, or identifiers.

---

## Architecture

```
┌─────────────────────────────────────────┐
│           MOBILE (Android/iOS)          │
│                                         │
│  Microphone → MedASR (ONNX, INT8)      │
│            → MedGemma (GGUF Q3_K_M)   │
│            → Human Checkpoint UI       │
│                                         │
│  PHI never leaves the phone             │
└──────────────┬──────────────────────────┘
               │ de-identified terms only
               │ (drug names, ICD queries)
┌──────────────▼──────────────────────────┐
│         WORKSTATION (Local GPU)         │
│                                         │
│  MCP Server (FastAPI, port 8082)       │
│    ├─ Neo4j: 1,868 drugs               │
│    │         222,271 DDI edges         │
│    │         98,186 ICD-10 codes       │
│    ├─ HAPI FHIR R4 (port 8080)        │
│    └─ MedGemma llama-server (8081)     │
│                                        │
│  EHR Navigator (LangGraph agent)       │
└────────────────────────────────────────┘
```

---

## Clinical Intelligence Features

### Drug-Drug Interaction (DDI) Detection
MedGemma extracts medication names from the SOAP note and queries a Neo4j graph of 222,271 interaction edges loaded from real DDI data. Critical interactions surface as a mandatory physician review step before export.

### Agentic ICD-10 Coding
MedGemma identifies clinical diagnoses from the Assessment/Plan section, then queries Neo4j's 98,186 ICD-10 code index per diagnosis. The doctor can accept, edit, or reject each suggestion before it is written to FHIR.

### Lab Correlation
Abnormal lab results from the patient's FHIR record are surfaced alongside the note — flagging values outside reference ranges with H/L/HH/LL markers.

### EHR Navigator Agent
A LangGraph 5-step agent answers natural language questions about the patient's record: "What were the last 3 HbA1c values?" Powered by MedGemma on the workstation GPU.

### Radiology AI
Chest X-rays and other images are analyzed by MedGemma's vision encoder (mmproj) running on the workstation. Reports are saved as FHIR DiagnosticReports.

---

## Models Used

| Model | Size on device | Task |
|-------|---------------|------|
| MedGemma 4B (GGUF Q3_K_M) | 2.0 GB | SOAP generation, enhancement reasoning |
| MedGemma 4B + mmproj | 3.2 GB | Radiology image analysis (workstation) |
| MedASR (ONNX INT8) | 101 MB | Medical speech recognition |

Both models run without internet access after initial download.

---

## Repository Structure

```
capsule/
├── mobile/MedGemmaApp/
│   ├── App.tsx              # Complete React Native app (~1800 lines)
│   ├── src/
│   │   ├── MCPClient.ts     # Backend API client (all 25+ endpoints)
│   │   ├── MelSpectrogram.ts # On-device audio → mel spectrogram
│   │   ├── CTCDecoder.ts    # CTC greedy decode for MedASR output
│   │   └── theme.ts         # Design system (WCAG AA compliant)
│   └── package.json
│
├── backend/
│   ├── mcp_server.py        # FastAPI server (25+ MCP tool endpoints)
│   ├── mcp_clinical_tools.py # Neo4j query wrappers
│   ├── fhir_resources.py    # FHIR R4 resource creation
│   └── app/services/
│       ├── enhance_service.py    # Agentic enhancement pipeline
│       ├── neo4j_service.py      # Graph DB queries + drug aliases
│       ├── ehr_navigator.py      # LangGraph EHR agent
│       └── terminology_service.py # SNOMED/RxNorm crosswalk
│
├── neo4j/
│   ├── scripts/
│   │   ├── load_ddi.py      # Load 222K drug interaction edges
│   │   └── load_icd10.py    # Load 98K ICD-10 codes
│   ├── queries/             # Cypher query library
│   └── SCHEMA.md            # Graph schema documentation
│
├── workstation/
│   ├── docker-compose.yml   # Neo4j + HAPI FHIR containers
│   └── start_vision.sh      # Launch MedGemma llama-server with GPU
│
└── docs/
    ├── system-summary.md    # Full technical architecture
    └── architecture-diagrams.md
```

---

## Setup

### Prerequisites
- Android phone (6 GB RAM+) or iOS device
- Workstation with NVIDIA GPU (8 GB VRAM+)
- Docker Desktop
- Node.js 18+, Python 3.11+

### 1. Start Workstation Services

```bash
# Clone and enter repo
git clone https://github.com/mo-saif/capsule
cd capsule

# Start Neo4j + HAPI FHIR
cd workstation
docker-compose up -d

# Create Python environment
cd ..
python -m venv medgemma-env && source medgemma-env/bin/activate
pip install -r backend/requirements.txt

# Download MedGemma GGUF (requires HuggingFace access)
# Place at: ml-models/gguf/medgemma-1.5-4b-it-Q3_K_M.gguf
#           ml-models/gguf/medgemma-1.5-4b-it-mmproj.gguf

# Start MedGemma inference server
bash workstation/start_vision.sh

# Start MCP server
python -m uvicorn backend.mcp_server:app --host 0.0.0.0 --port 8082
```

### 2. Load Knowledge Graph

```bash
source medgemma-env/bin/activate

# Load drug-drug interactions (~222K edges)
python neo4j/scripts/load_ddi.py

# Load ICD-10 codes (~98K codes)
python neo4j/scripts/load_icd10.py
```

### 3. Build Mobile App

```bash
cd mobile/MedGemmaApp
npm install

# Set your workstation IP in src/MCPClient.ts:
# const DEFAULT_LOCAL_URL = 'http://YOUR_IP:8082'

# Download MedASR ONNX model
# Place at android/app/src/main/assets/medasr_int8.onnx

# Download MedGemma GGUF for on-device inference
# Place at /data/local/tmp/medgemma.gguf on device

# Build and run
npx react-native run-android
```

---

## Key Design Decisions

**Why GGUF over ONNX for MedGemma?**
MediaPipe's LLM inference API only supports models up to 1B parameters. MedGemma 4B requires llama.cpp (GGUF format) via the `llama.rn` React Native binding.

**Why local Neo4j over cloud?**
Neo4j Aura requires a paid subscription for production use. A local Docker instance gives the same query performance with zero operational cost and keeps all knowledge queries on-premises.

**Why MCP protocol for backend?**
The Model Context Protocol gives a clean tool-calling interface that MedGemma can reason about. Each Neo4j query, FHIR operation, and lab lookup is a named MCP tool — making the agent's decision-making transparent and auditable.

---

## Competition: MedGemma Impact Challenge
- **Track**: Edge AI Prize
- **Models**: MedGemma 4B (multimodal) + MedASR
- **Privacy**: 100% on-device PHI — no cloud dependency
- **FHIR**: Full R4 compliance (Epic/Cerner compatible)
