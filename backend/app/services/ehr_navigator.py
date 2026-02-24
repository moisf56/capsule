"""
EHR Navigator Agent — LangGraph Implementation

Inspired by Google's MedGemma EHR Navigator notebook:
  https://github.com/google-health/medgemma/blob/main/notebooks/ehr_navigator_agent.ipynb

5-step progressive narrowing pipeline:
  1. Discover:  Get FHIR manifest (what resources exist for patient)
  2. Identify:  LLM selects relevant resource types for the question
  3. Plan:      LLM generates specific FHIR queries for each type
  4. Execute:   Fetch resources + LLM extracts concise facts per result
  5. Synthesize: LLM produces comprehensive final answer

Uses workstation MedGemma (Q4_K_M, GPU) via llama-server OpenAI-compatible API.
"""

import base64
import json
import logging
import operator
import os
import re
import time
from typing import Annotated, Optional, TypedDict

import httpx
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("ehr_navigator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[EHR-NAV] %(message)s"))
    logger.addHandler(handler)


# ─── Configuration ───────────────────────────────────────

LLAMA_SERVER_URL = os.getenv("LLAMA_VISION_URL", "http://localhost:8081")
FHIR_BASE = os.getenv("FHIR_BASE", "http://localhost:8080/fhir")

FHIR_RESOURCE_TYPES = [
    "Observation",
    "Condition",
    "MedicationRequest",
    "Encounter",
    "DiagnosticReport",
    "DocumentReference",
    "DetectedIssue",
]

FHIR_HEADERS = {
    "Accept": "application/fhir+json",
    "Content-Type": "application/fhir+json",
}


# ─── LLM Setup ───────────────────────────────────────────

def _get_llm(temperature: float = 0.1, max_tokens: int = 1024) -> ChatOpenAI:
    """Get MedGemma LLM via llama-server OpenAI-compatible API."""
    return ChatOpenAI(
        model="medgemma",
        base_url=f"{LLAMA_SERVER_URL}/v1",
        api_key="not-needed",  # llama-server doesn't need a key
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<end_of_turn>", "<eos>"],
    )


def _strip_thinking(text: str) -> str:
    """Strip MedGemma thinking traces (multiple formats)."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Strip <unused94>...<unused95> thinking blocks
    text = re.sub(r"<unused\d*>.*?<unused\d*>", "", text, flags=re.DOTALL).strip()
    # Strip model_output prefix (MedGemma 1.5 thinking)
    idx = text.find("model_output\n")
    if idx >= 0:
        text = text[idx + len("model_output\n"):]
    # Strip plain "thought\n..." prefix (llama-server output)
    if text.startswith("thought\n") or text.startswith("thought\r\n"):
        # Find the actual answer after the thinking block
        # Look for common section starters
        for marker in ["\n## ", "\n# ", "\n**", "\nBased on", "\nThe patient",
                       "\nHere is", "\nClinical", "\n---"]:
            pos = text.find(marker)
            if pos > 0:
                text = text[pos:].strip()
                break
        else:
            # Fallback: skip the first paragraph (the "thought" section)
            parts = text.split("\n\n", 1)
            if len(parts) > 1:
                text = parts[1]
    return text.strip()


def _strip_json_decoration(text: str) -> str:
    """Strip markdown code fences from JSON output."""
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[7:-3].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[3:-3].strip()
    return cleaned


# ─── FHIR Helpers ────────────────────────────────────────

_fhir_client: Optional[httpx.Client] = None


def _get_fhir_client() -> httpx.Client:
    global _fhir_client
    if _fhir_client is None:
        _fhir_client = httpx.Client(base_url=FHIR_BASE, headers=FHIR_HEADERS, timeout=10)
    return _fhir_client


def get_patient_data_manifest(patient_id: str) -> dict[str, list[str]]:
    """
    Discovery tool: scan all FHIR resource types for a patient.
    Returns {resource_type: [list of "display=code" strings]}.
    """
    client = _get_fhir_client()
    manifest: dict[str, list[str]] = {}

    for rtype in FHIR_RESOURCE_TYPES:
        try:
            params: dict[str, str] = {"_count": "100"}
            if rtype == "DetectedIssue":
                params["patient"] = f"Patient/{patient_id}"
            else:
                params["subject"] = f"Patient/{patient_id}"

            resp = client.get(f"/{rtype}", params=params)
            resp.raise_for_status()
            entries = resp.json().get("entry", [])

            if not entries:
                continue

            codes: list[str] = []
            for entry in entries:
                resource = entry["resource"]
                summary = _summarize_resource_brief(resource)
                if summary:
                    codes.append(summary)

            if codes:
                manifest[rtype] = codes

        except Exception as e:
            logger.warning(f"  Manifest scan {rtype}: {e}")

    return manifest


def get_patient_fhir_resource(
    patient_id: str,
    fhir_resource: str,
    filter_code: Optional[str] = None,
) -> list[dict]:
    """
    Data retrieval tool: fetch FHIR resources for a patient.
    Optionally filter by code (comma-separated).
    """
    client = _get_fhir_client()
    params: dict[str, str] = {"_count": "50"}

    if fhir_resource == "DetectedIssue":
        params["patient"] = f"Patient/{patient_id}"
    else:
        params["subject"] = f"Patient/{patient_id}"

    if filter_code:
        params["code"] = filter_code

    resp = client.get(f"/{fhir_resource}", params=params)
    resp.raise_for_status()
    bundle = resp.json()
    entries = bundle.get("entry", [])

    # Fallback: if code filter returned nothing, try category
    if not entries and filter_code:
        params.pop("code", None)
        params["category"] = filter_code
        resp = client.get(f"/{fhir_resource}", params=params)
        resp.raise_for_status()
        entries = resp.json().get("entry", [])

    return [e["resource"] for e in entries]


def _summarize_resource_brief(resource: dict) -> str:
    """Brief summary for manifest (display=code style)."""
    rtype = resource.get("resourceType", "")

    if rtype == "Observation":
        code = resource.get("code", {}).get("coding", [{}])[0]
        display = code.get("display", resource.get("code", {}).get("text", ""))
        loinc = code.get("code", "")
        return f"{display}={loinc}" if display else ""

    if rtype == "Condition":
        codings = resource.get("code", {}).get("coding", [])
        text = resource.get("code", {}).get("text", "")
        for c in codings:
            if "icd-10" in c.get("system", ""):
                return f"{c.get('display', text)}={c.get('code', '')}"
        return text

    if rtype == "MedicationRequest":
        med = resource.get("medicationCodeableConcept", {})
        drug = med.get("text", "")
        for c in med.get("coding", []):
            if "rxnorm" in c.get("system", ""):
                return f"{drug}={c.get('code', '')}"
        return drug

    if rtype == "Encounter":
        reason = ""
        if resource.get("reasonCode"):
            reason = resource["reasonCode"][0].get("text", "")
        return f"{reason}={resource.get('status', '')}" if reason else ""

    if rtype == "DiagnosticReport":
        conclusion = resource.get("conclusion", "")[:100]
        return conclusion

    if rtype == "DocumentReference":
        return "SOAP Note"

    if rtype == "DetectedIssue":
        return f"{resource.get('detail', '')}={resource.get('severity', '')}"

    return ""


def _summarize_resource_full(resource: dict) -> str:
    """Full text summary of a FHIR resource for LLM context."""
    rtype = resource.get("resourceType", "Unknown")

    if rtype == "Observation":
        code = resource.get("code", {}).get("coding", [{}])[0]
        vq = resource.get("valueQuantity", {})
        interp = (resource.get("interpretation", [{}])[0]
                  .get("coding", [{}])[0].get("code", ""))
        ref_range = resource.get("referenceRange", [{}])[0] if resource.get("referenceRange") else {}
        ref_lo = ref_range.get("low", {}).get("value", "")
        ref_hi = ref_range.get("high", {}).get("value", "")
        date = resource.get("effectiveDateTime", "")[:10]
        return (
            f"Lab [{resource.get('id', '')}]: {code.get('display', '')} = "
            f"{vq.get('value', '')} {vq.get('unit', '')} [{interp}] "
            f"(ref: {ref_lo}-{ref_hi}) date: {date}"
        )

    if rtype == "Condition":
        codings = resource.get("code", {}).get("coding", [])
        parts = []
        for c in codings:
            sys = c.get("system", "")
            if "icd-10" in sys:
                parts.append(f"ICD-10: {c.get('code', '')} ({c.get('display', '')})")
            elif "snomed" in sys:
                parts.append(f"SNOMED: {c.get('code', '')} ({c.get('display', '')})")
        text = resource.get("code", {}).get("text", "")
        date = resource.get("recordedDate", "")[:10]
        return f"Condition [{resource.get('id', '')}]: {text or '; '.join(parts)} date: {date}"

    if rtype == "MedicationRequest":
        med = resource.get("medicationCodeableConcept", {})
        drug = med.get("text", "")
        rxnorm = ""
        for c in med.get("coding", []):
            if "rxnorm" in c.get("system", ""):
                rxnorm = c.get("code", "")
        date = resource.get("authoredOn", "")[:10]
        return f"Medication [{resource.get('id', '')}]: {drug}" + (f" (RxNorm: {rxnorm})" if rxnorm else "") + f" date: {date}"

    if rtype == "Encounter":
        reason = ""
        if resource.get("reasonCode"):
            reason = resource["reasonCode"][0].get("text", "")
        date = resource.get("period", {}).get("start", "")[:10]
        return f"Encounter [{resource.get('id', '')}]: {resource.get('status', '')} — {reason} date: {date}"

    if rtype == "DiagnosticReport":
        conclusion = resource.get("conclusion", "")[:300]
        date = resource.get("issued", "")[:10]
        return f"DiagnosticReport [{resource.get('id', '')}]: {conclusion} date: {date}"

    if rtype == "DocumentReference":
        try:
            encoded = resource["content"][0]["attachment"]["data"]
            text = base64.b64decode(encoded).decode()[:400]
            return f"Document [{resource.get('id', '')}]: {text}"
        except (KeyError, IndexError):
            return f"Document [{resource.get('id', '')}]: (no content)"

    if rtype == "DetectedIssue":
        return f"DetectedIssue [{resource.get('id', '')}] [{resource.get('severity', '')}]: {resource.get('detail', '')}"

    return f"{rtype} [{resource.get('id', '')}]: {str(resource)[:200]}"


# ─── LangGraph Agent State ──────────────────────────────

class AgentState(TypedDict, total=False):
    """EHR Navigator agent state (TypedDict for proper LangGraph state management)."""
    question: str
    patient_id: str
    manifest: dict
    relevant_types: list[str]
    facts: Annotated[list, operator.add]  # accumulated across nodes
    resources_consulted: list[str]
    reasoning: str
    answer: str


# ─── LangGraph Nodes ────────────────────────────────────

def discover_manifest(state: dict) -> dict:
    """Step 1: Discover what FHIR resources exist for the patient."""
    patient_id = state["patient_id"]
    logger.info(f"Step 1: Discovering FHIR manifest for Patient/{patient_id}")

    manifest = get_patient_data_manifest(patient_id)

    if not manifest:
        return {
            "manifest": {},
            "answer": "No clinical data found for this patient in the EHR.",
            "facts": [],
            "resources_consulted": [],
        }

    manifest_summary = {k: len(v) for k, v in manifest.items()}
    logger.info(f"  Found: {manifest_summary}")
    return {"manifest": manifest}


def identify_relevant_types(state: dict) -> dict:
    """Step 2: LLM identifies which resource types are relevant to the question."""
    manifest = state.get("manifest", {})
    if not manifest:
        return {"relevant_types": []}

    question = state["question"]
    logger.info("Step 2: Identifying relevant resource types")

    manifest_text = "\n".join(
        f"- {rtype} ({len(codes)} records): {', '.join(codes[:5])}"
        + (f" ... and {len(codes)-5} more" if len(codes) > 5 else "")
        for rtype, codes in manifest.items()
    )

    llm = _get_llm(temperature=0.0, max_tokens=500)
    try:
        response = llm.invoke([
            SystemMessage(content="SYSTEM INSTRUCTION: think silently if needed."),
            HumanMessage(content=(
                f"USER QUESTION: {question}\n\n"
                f"PATIENT DATA MANIFEST:\n{manifest_text}\n\n"
                "You are a medical assistant analyzing a patient's FHIR data manifest "
                "to answer a user question.\n"
                "Based on the user question, identify the specific FHIR resource types "
                "from the manifest that are most likely to contain the information needed.\n"
                'Output a JSON list of the relevant resource types. No other text.\n'
                'Example: ["Observation", "Condition", "MedicationRequest"]'
            )),
        ])

        raw = _strip_thinking(response.content)
        raw = _strip_json_decoration(raw)

        # Parse the JSON list
        try:
            types = json.loads(raw)
            if isinstance(types, list):
                # Filter to only types that exist in manifest
                relevant = [t for t in types if t in manifest]
                logger.info(f"  LLM selected: {relevant}")
                return {"relevant_types": relevant}
        except json.JSONDecodeError:
            pass

    except Exception as e:
        logger.warning(f"  LLM type identification failed: {e}")

    # Fallback: use all manifest types
    fallback = list(manifest.keys())
    logger.info(f"  Fallback: using all types: {fallback}")
    return {"relevant_types": fallback}


def execute_and_extract(state: dict) -> dict:
    """Step 3+4: Fetch relevant resources and extract facts per resource type."""
    relevant_types = state.get("relevant_types", [])
    manifest = state.get("manifest", {})
    patient_id = state["patient_id"]
    question = state["question"]

    if not relevant_types:
        return {"facts": [], "resources_consulted": []}

    logger.info(f"Step 3-4: Fetching and extracting facts from {relevant_types}")

    all_facts: list[str] = []
    resources_consulted: list[str] = []
    llm = _get_llm(temperature=0.6, max_tokens=2048)

    for rtype in relevant_types:
        try:
            # Fetch resources
            resources = get_patient_fhir_resource(patient_id, rtype)
            if not resources:
                continue

            resources_consulted.append(f"{rtype} ({len(resources)})")
            logger.info(f"  Fetched {len(resources)} {rtype} resources")

            # Build text summaries of all resources of this type
            resource_text = "\n".join(
                _summarize_resource_full(r) for r in resources
            )

            # LLM extracts concise facts relevant to the question
            try:
                response = llm.invoke([
                    SystemMessage(content=(
                        "You are a concise fact extractor. Output ONLY a short bullet list. "
                        "No explanations. No reasoning. No repetition. Max 10 bullets."
                    )),
                    HumanMessage(content=(
                        f"USER QUESTION: {question}\n\n"
                        f"FHIR {rtype} DATA:\n{resource_text}\n\n"
                        "Extract ONLY facts relevant to the question as a bullet list. "
                        "Each fact on one line starting with '- '. Include values and units. "
                        "Do NOT repeat any fact. Do NOT answer the question."
                    )),
                ])

                facts = _strip_thinking(response.content)
                if facts:
                    all_facts.append(f"--- {rtype} ---\n{facts}")
                    logger.info(f"  Extracted facts from {rtype} ({len(facts)} chars)")

            except Exception as e:
                # Fallback: use raw summaries as facts
                logger.warning(f"  LLM fact extraction failed for {rtype}: {e}")
                all_facts.append(f"--- {rtype} ---\n{resource_text}")

        except Exception as e:
            logger.warning(f"  Fetch {rtype} failed: {e}")

    return {"facts": all_facts, "resources_consulted": resources_consulted}


def synthesize_answer(state: dict) -> dict:
    """Step 5: LLM synthesizes all facts into a comprehensive final answer."""
    facts = state.get("facts", [])
    question = state["question"]

    if not facts:
        return {"answer": "No relevant clinical data found to answer this question.", "reasoning": ""}

    logger.info(f"Step 5: Synthesizing answer from {len(facts)} fact groups")

    joined_facts = "\n\n".join(facts)
    llm = _get_llm(temperature=0.1, max_tokens=2048)

    # The extracted facts ARE the reasoning — show the agent's intermediate work
    reasoning = joined_facts

    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You are MedGemma, a clinical assistant. Answer the question directly. "
                "Do NOT show your reasoning steps. Do NOT number your thought process. "
                "Just give the final clinical answer using markdown formatting. "
                "Reference specific values from the data. Be concise and organized."
            )),
            HumanMessage(content=(
                f"QUESTION: {question}\n\n"
                f"PATIENT DATA:\n{joined_facts}\n\n"
                "Answer directly:"
            )),
        ])

        answer = _strip_thinking(response.content)
        logger.info(f"  Answer generated ({len(answer)} chars, reasoning: {len(reasoning)} chars)")
        return {"answer": answer, "reasoning": reasoning}

    except Exception as e:
        logger.warning(f"  Synthesis failed: {e}, using raw facts")
        return {"answer": f"**Extracted Facts:**\n\n{joined_facts}", "reasoning": ""}


# ─── Build LangGraph ────────────────────────────────────

def _should_continue_after_discover(state: dict) -> str:
    """Route: if manifest is empty, skip to END."""
    if not state.get("manifest"):
        return "end"
    return "identify"


def build_ehr_navigator_graph() -> StateGraph:
    """Build the EHR Navigator LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("discover_manifest", discover_manifest)
    workflow.add_node("identify_relevant_types", identify_relevant_types)
    workflow.add_node("execute_and_extract", execute_and_extract)
    workflow.add_node("synthesize_answer", synthesize_answer)

    # Set entry point
    workflow.set_entry_point("discover_manifest")

    # Add edges
    workflow.add_conditional_edges(
        "discover_manifest",
        _should_continue_after_discover,
        {"identify": "identify_relevant_types", "end": END},
    )
    workflow.add_edge("identify_relevant_types", "execute_and_extract")
    workflow.add_edge("execute_and_extract", "synthesize_answer")
    workflow.add_edge("synthesize_answer", END)

    return workflow.compile()


# ─── Public API ──────────────────────────────────────────

# Singleton compiled graph
_agent = None


def get_ehr_navigator():
    """Get (or create) the compiled EHR Navigator agent."""
    global _agent
    if _agent is None:
        _agent = build_ehr_navigator_graph()
    return _agent


async def navigate_ehr_stream(question: str, patient_id: str):
    """
    Stream the EHR Navigator agent step-by-step via SSE.
    Yields JSON lines: {"step": str, "data": str}
    Final yield: {"step": "done", "data": {full result}}
    """
    import asyncio
    start = time.time()

    agent = get_ehr_navigator()
    initial_state = {
        "question": question,
        "patient_id": patient_id,
        "manifest": {},
        "relevant_types": [],
        "facts": [],
        "resources_consulted": [],
        "reasoning": "",
        "answer": "",
    }

    step_names = {
        "discover_manifest": "Discovering patient records...",
        "identify_relevant_types": "Identifying relevant data...",
        "execute_and_extract": "Extracting clinical facts...",
        "synthesize_answer": "Synthesizing answer...",
    }

    final_state = initial_state.copy()

    # LangGraph stream yields (node_name, state_update) pairs
    for step_output in agent.stream(initial_state):
        for node_name, state_update in step_output.items():
            # Merge update into final_state
            for k, v in state_update.items():
                if k == "facts" and isinstance(v, list):
                    final_state.setdefault("facts", [])
                    final_state["facts"].extend(v)
                else:
                    final_state[k] = v

            label = step_names.get(node_name, node_name)

            # Build reasoning text from facts so far
            reasoning_text = ""
            if node_name == "discover_manifest":
                manifest = state_update.get("manifest", {})
                if manifest:
                    summary = ", ".join(f"{k} ({len(v)})" for k, v in manifest.items())
                    reasoning_text = f"Found: {summary}"
                else:
                    reasoning_text = "No data found"
            elif node_name == "identify_relevant_types":
                types = state_update.get("relevant_types", [])
                reasoning_text = f"Selected: {', '.join(types)}" if types else "No types selected"
            elif node_name == "execute_and_extract":
                facts = final_state.get("facts", [])
                reasoning_text = "\n".join(facts) if facts else "No facts extracted"
            elif node_name == "synthesize_answer":
                reasoning_text = "Complete"

            yield json.dumps({
                "step": node_name,
                "label": label,
                "reasoning": reasoning_text,
            }) + "\n"

        # Small yield delay so frontend can process
        await asyncio.sleep(0.05)

    elapsed = int((time.time() - start) * 1000)

    yield json.dumps({
        "step": "done",
        "data": {
            "answer": final_state.get("answer", "No answer generated."),
            "reasoning": final_state.get("reasoning", ""),
            "resources_consulted": final_state.get("resources_consulted", []),
            "facts_extracted": len(final_state.get("facts", [])),
            "processing_time_ms": elapsed,
        },
    }) + "\n"


async def navigate_ehr(question: str, patient_id: str) -> dict:
    """
    Run the EHR Navigator agent.

    Returns:
      {
        "answer": str,
        "resources_consulted": list[str],
        "facts_extracted": int,
        "processing_time_ms": int,
      }
    """
    start = time.time()

    agent = get_ehr_navigator()
    initial_state = {
        "question": question,
        "patient_id": patient_id,
        "manifest": {},
        "relevant_types": [],
        "facts": [],
        "resources_consulted": [],
        "reasoning": "",
        "answer": "",
    }

    # Run the graph (synchronous invoke, LangGraph handles it)
    final_state = agent.invoke(initial_state)

    elapsed = int((time.time() - start) * 1000)

    return {
        "answer": final_state.get("answer", "No answer generated."),
        "reasoning": final_state.get("reasoning", ""),
        "resources_consulted": final_state.get("resources_consulted", []),
        "facts_extracted": len(final_state.get("facts", [])),
        "processing_time_ms": elapsed,
    }
