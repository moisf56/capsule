/**
 * MCP Client — Iteration 13
 *
 * Connects the phone to the workstation MCP server.
 * Operations: FHIR (full clinical export), Neo4j (DDI, ICD-10, drug search),
 *             Terminology (RxNorm, SNOMED, code suggestions),
 *             EHR Navigator (LangGraph agentic retrieval).
 *
 * Set MCP_BASE to your workstation's LAN IP before building.
 */

// Server URLs — updated at runtime via setMcpBase()
export const DEMO_SERVER_URL = 'https://capsule-med-demo.hf.space'; // HuggingFace Space (placeholder)
export const DEFAULT_LOCAL_URL = 'http://192.168.1.149:8082';

let MCP_BASE = DEFAULT_LOCAL_URL;

/** Switch the active server at runtime (called from settings screen) */
export function setMcpBase(url: string): void {
  MCP_BASE = url.replace(/\/$/, ''); // strip trailing slash
}

/** Test if the server is reachable. Returns true if online. */
export async function testConnection(): Promise<boolean> {
  try {
    const resp = await fetch(`${MCP_BASE}/health`, {method: 'GET'});
    return resp.ok;
  } catch {
    // Try /tools/list_patients as fallback health check
    try {
      const resp = await fetch(`${MCP_BASE}/tools/list_patients`, {method: 'GET'});
      return resp.ok;
    } catch {
      return false;
    }
  }
}

interface MCPResponse<T> {
  status: string;
  data: T;
}

interface EncounterResult {
  id: string;
  status: string;
  reason: string;
}

interface DocumentResult {
  id: string;
  status: string;
}

interface DrugInteraction {
  drug1: string;
  drug2: string;
  interaction_type: string;
  severity?: string;
}

interface DDIResult {
  found: boolean;
  summary: string;
  interactions: DrugInteraction[];
}

async function mcpPost<T>(path: string, body: object): Promise<T> {
  const resp = await fetch(`${MCP_BASE}${path}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`MCP ${path} failed (${resp.status}): ${text}`);
  }
  const json: MCPResponse<T> = await resp.json();
  return json.data;
}

export async function createEncounter(
  patientId: string,
  reason: string,
): Promise<EncounterResult> {
  return mcpPost<EncounterResult>('/tools/fhir_create_encounter', {
    patient_id: patientId,
    reason,
  });
}

export async function exportSOAP(
  patientId: string,
  encounterId: string,
  soapText: string,
): Promise<DocumentResult> {
  return mcpPost<DocumentResult>('/tools/fhir_export_soap', {
    patient_id: patientId,
    encounter_id: encounterId,
    soap_text: soapText,
  });
}

export async function checkDrugInteractions(
  medications: string[],
): Promise<DDIResult> {
  return mcpPost<DDIResult>('/tools/check_drug_interactions', {
    medications,
  });
}

// --- Iteration 6: Terminology ---

interface ICD10Result {
  code: string;
  description: string;
  score: number;
}

interface DrugSearchResult {
  name: string;
  id: string;
  score: number;
}

interface CodeSuggestion {
  code: string;
  description: string;
  matched_term: string;
}

interface SuggestCodesResult {
  suggestions: CodeSuggestion[];
  terms_searched: number;
}

export async function searchICD10(
  query: string,
  limit: number = 5,
): Promise<{results: ICD10Result[]}> {
  return mcpPost<{results: ICD10Result[]}>('/tools/search_icd10', {query, limit});
}

export async function searchDrug(
  query: string,
  limit: number = 5,
): Promise<{results: DrugSearchResult[]}> {
  return mcpPost<{results: DrugSearchResult[]}>('/tools/search_drug', {query, limit});
}

export async function suggestCodes(
  soapText: string,
): Promise<SuggestCodesResult> {
  return mcpPost<SuggestCodesResult>('/tools/suggest_codes', {soap_text: soapText});
}

// --- Iteration 6: RxNorm + SNOMED ---

interface RxNormResult {
  found: boolean;
  rxcui?: string;
  name?: string;
  tty?: string;
  suggestions?: Array<{rxcui: string; name: string; tty: string}>;
}

interface SNOMEDResult {
  cui: string;
  code: string;
  name: string;
  semantic_type: string;
}

export async function normalizeDrug(
  drugName: string,
): Promise<RxNormResult> {
  return mcpPost<RxNormResult>('/tools/normalize_drug', {drug_name: drugName});
}

export async function searchSNOMED(
  term: string,
  limit: number = 5,
): Promise<{results: SNOMEDResult[]}> {
  return mcpPost<{results: SNOMEDResult[]}>('/tools/search_snomed', {term, limit});
}

// --- Iteration 7: Full FHIR Export ---

interface ICD10CodeData {
  code: string;
  description: string;
}

interface DDIAlertData {
  drug1: string;
  drug2: string;
  interaction_type: string;
  acknowledged: boolean;
}

interface MedicationExportResult {
  id: string;
  drug: string;
  rxnorm_code: string | null;
}

interface ConditionExportResult {
  id: string;
  icd_code: string;
  snomed_code: string | null;
}

interface DetectedIssueResult {
  id: string;
  detail: string;
  severity: string;
}

interface FullExportResult {
  encounter: EncounterResult;
  document: DocumentResult;
  medications: MedicationExportResult[];
  conditions: ConditionExportResult[];
  detected_issues: DetectedIssueResult[];
  errors: string[];
  summary: string;
}

export async function exportFull(
  patientId: string,
  reason: string,
  soapText: string,
  medications: string[],
  icd10Codes: ICD10CodeData[],
  ddiAlerts: DDIAlertData[],
): Promise<FullExportResult> {
  return mcpPost<FullExportResult>('/tools/fhir_export_full', {
    patient_id: patientId,
    reason,
    soap_text: soapText,
    medications,
    icd10_codes: icd10Codes,
    ddi_alerts: ddiAlerts,
  });
}

// --- Iteration 8: Medical Image Analysis (Vision) ---

interface ImageAnalysisResult {
  findings: string;
  impression: string;
  raw_report: string;
  image_type: string;
  model: string;
}

interface DiagnosticReportResult {
  id: string;
  status: string;
  conclusion: string;
}

interface DiagnosticReportSummary {
  id: string;
  status: string;
  conclusion: string;
  date: string;
  image_type: string;
}

export async function analyzeImage(
  imageBase64: string,
  imageType: string = 'chest_xray',
  clinicalContext: string = '',
): Promise<ImageAnalysisResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);
  try {
    const resp = await fetch(`${MCP_BASE}/tools/analyze_medical_image`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image_base64: imageBase64,
        image_type: imageType,
        clinical_context: clinicalContext,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Image analysis failed (${resp.status}): ${text}`);
    }
    const json: MCPResponse<ImageAnalysisResult> = await resp.json();
    return json.data;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Image analysis timed out (>120s). Try a smaller image.');
    }
    throw err;
  }
}

export async function exportDiagnosticReport(
  patientId: string,
  conclusion: string,
  findingsText: string,
  imageType: string = 'chest_xray',
  encounterId?: string,
): Promise<DiagnosticReportResult> {
  return mcpPost<DiagnosticReportResult>('/tools/fhir_export_diagnostic_report', {
    patient_id: patientId,
    encounter_id: encounterId || null,
    conclusion,
    findings_text: findingsText,
    image_type: imageType,
  });
}

export async function getDiagnosticReportImage(
  reportId: string,
): Promise<{image_base64: string; content_type: string; title: string; conclusion: string}> {
  const resp = await fetch(
    `${MCP_BASE}/tools/get_diagnostic_report_image?report_id=${encodeURIComponent(reportId)}`,
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Get report image failed (${resp.status}): ${text}`);
  }
  const json: MCPResponse<{image_base64: string; content_type: string; title: string; conclusion: string}> = await resp.json();
  return json.data;
}

export async function listDiagnosticReports(
  patientId: string,
): Promise<DiagnosticReportSummary[]> {
  const resp = await fetch(
    `${MCP_BASE}/tools/list_diagnostic_reports?patient_id=${encodeURIComponent(patientId)}`,
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`List reports failed (${resp.status}): ${text}`);
  }
  const json: MCPResponse<DiagnosticReportSummary[]> = await resp.json();
  return json.data;
}

// --- Iteration 9: Lab Results (Observations) ---

interface LabObservation {
  id: string;
  loinc_code: string;
  loinc_display: string;
  value: number;
  unit: string;
  reference_low: number | null;
  reference_high: number | null;
  interpretation: string; // 'N' | 'H' | 'L' | 'HH' | 'LL'
  effective_date: string;
}

interface SeedDemoLabsResult {
  created: number;
  observations: Array<{id: string; loinc_code: string; value: number; unit: string; interpretation: string}>;
}

export async function listObservations(
  patientId: string,
): Promise<LabObservation[]> {
  const resp = await fetch(
    `${MCP_BASE}/tools/list_observations?patient_id=${encodeURIComponent(patientId)}`,
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`List observations failed (${resp.status}): ${text}`);
  }
  const json: MCPResponse<LabObservation[]> = await resp.json();
  return json.data;
}

export async function seedDemoLabs(
  patientId: string = '1000',
): Promise<SeedDemoLabsResult> {
  return mcpPost<SeedDemoLabsResult>('/tools/seed_demo_labs', {patient_id: patientId});
}

// --- Iteration 11: Patient List + Name Masking ---

interface PatientSummary {
  id: string;
  name: string;
  gender: string;
  birthDate: string;
}

interface SeedDemoPatientsResult {
  patients: Array<{patient_id: string; name: string; labs_created: number}>;
  total: number;
}

export async function listPatients(): Promise<PatientSummary[]> {
  const resp = await fetch(`${MCP_BASE}/tools/list_patients`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`List patients failed (${resp.status}): ${text}`);
  }
  const json: MCPResponse<PatientSummary[]> = await resp.json();
  return json.data;
}

export async function seedDemoPatients(): Promise<SeedDemoPatientsResult> {
  return mcpPost<SeedDemoPatientsResult>('/tools/seed_demo_patients', {});
}

export function maskName(fullName: string): string {
  const parts = fullName.trim().split(/\s+/);
  if (parts.length === 0) return '****';
  if (parts.length === 1) return parts[0][0] + '****';
  return parts[0][0] + '. ' + parts[parts.length - 1][0] + '****';
}

// --- Iteration 12: Agentic SOAP Enhancement (MedGemma + MCP Tools) ---

interface EnhanceDDIAlert {
  drug1: string;
  drug2: string;
  interaction_type: string;
  severity: string; // "critical" | "moderate"
}

interface EnhanceICD10Suggestion {
  code: string;
  description: string;
  matched_term: string;
}

interface EnhanceLabFinding {
  test: string;
  value: number;
  unit: string;
  interpretation: string; // N, H, L, HH, LL
  reference_low: number | null;
  reference_high: number | null;
}

interface EnhanceSOAPResult {
  medications: string[];
  ddi_alerts: EnhanceDDIAlert[];
  icd10_suggestions: EnhanceICD10Suggestion[];
  lab_findings: EnhanceLabFinding[] | null;
  clinical_summary: string;
  tools_called: string[];
  processing_time_ms: number;
  medgemma_available: boolean;
}

export async function enhanceSoap(
  soapText: string,
  patientId?: string,
): Promise<EnhanceSOAPResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60_000);
  try {
    const resp = await fetch(`${MCP_BASE}/tools/enhance_soap`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        soap_text: soapText,
        patient_id: patientId || null,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Enhance SOAP failed (${resp.status}): ${text}`);
    }
    const json: MCPResponse<EnhanceSOAPResult> = await resp.json();
    return json.data;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('AI enhancement timed out (>60s). Try again.');
    }
    throw err;
  }
}

// --- Iteration 13: EHR Navigator Agent (LangGraph) ---

interface EHRNavigateResult {
  answer: string;
  reasoning: string;
  resources_consulted: string[];
  facts_extracted: number;
  processing_time_ms: number;
}

export async function ehrNavigate(
  question: string,
  patientId: string,
): Promise<EHRNavigateResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);
  try {
    const resp = await fetch(`${MCP_BASE}/tools/ehr_navigate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        patient_id: patientId,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`EHR Navigator failed (${resp.status}): ${text}`);
    }
    const json: MCPResponse<EHRNavigateResult> = await resp.json();
    return json.data;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('EHR Navigator timed out (>120s). Try again.');
    }
    throw err;
  }
}

interface EHRStreamEvent {
  step: string;
  label?: string;
  reasoning?: string;
  data?: EHRNavigateResult;
}

export async function ehrNavigateStream(
  question: string,
  patientId: string,
  onStep: (event: EHRStreamEvent) => void,
): Promise<EHRNavigateResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180_000);
  try {
    const resp = await fetch(`${MCP_BASE}/tools/ehr_navigate_stream`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, patient_id: patientId}),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`EHR Navigator failed (${resp.status}): ${text}`);
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';
    let finalResult: EHRNavigateResult | null = null;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';  // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const event: EHRStreamEvent = JSON.parse(trimmed);
          if (event.step === 'done' && event.data) {
            finalResult = event.data;
          }
          onStep(event);
        } catch {
          // Skip malformed lines
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      try {
        const event: EHRStreamEvent = JSON.parse(buffer.trim());
        if (event.step === 'done' && event.data) {
          finalResult = event.data;
        }
        onStep(event);
      } catch {}
    }

    if (!finalResult) throw new Error('No final result from EHR Navigator');
    return finalResult;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('EHR Navigator timed out (>180s). Try again.');
    }
    throw err;
  }
}

export {MCP_BASE};
export type {
  EncounterResult,
  DocumentResult,
  DDIResult,
  DrugInteraction,
  ICD10Result,
  DrugSearchResult,
  CodeSuggestion,
  SuggestCodesResult,
  RxNormResult,
  SNOMEDResult,
  ICD10CodeData,
  DDIAlertData,
  MedicationExportResult,
  ConditionExportResult,
  DetectedIssueResult,
  FullExportResult,
  ImageAnalysisResult,
  DiagnosticReportResult,
  DiagnosticReportSummary,
  LabObservation,
  SeedDemoLabsResult,
  PatientSummary,
  SeedDemoPatientsResult,
  EnhanceDDIAlert,
  EnhanceICD10Suggestion,
  EnhanceLabFinding,
  EnhanceSOAPResult,
  EHRNavigateResult,
};
