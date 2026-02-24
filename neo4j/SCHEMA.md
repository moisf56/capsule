# Neo4j Knowledge Graph Schema

## Overview

This knowledge graph powers three clinical AI agents:
1. **Safety Agent** - Drug interaction checking
2. **Coding Agent** - ICD-10 code suggestion
3. **Differential Agent** - Symptom → Disease mapping

---

## Node Types

### 1. Drug (from DDI dataset)
```cypher
(:Drug {
    drugbank_id: "DB00945",     // Primary key (unique)
    name: "Aspirin"              // Drug name
})
```
**Source:** `DDI_data.csv` (Mendeley)
**Count:** ~1,751 drugs

---

### 2. ICD10 (from CMS dataset)
```cypher
(:ICD10 {
    code: "I10",                 // Primary key (unique)
    short_desc: "Essential hypertension",
    long_desc: "Essential (primary) hypertension",
    billable: true,              // true = can bill, false = category header
    chapter: "I"                 // First letter = chapter
})
```
**Source:** `icd10cm_order_2026.txt` (CMS)
**Count:** ~74,719 codes (~72K billable)

---

### 3. Disease (from Kaggle dataset)
```cypher
(:Disease {
    name: "Diabetes",            // Primary key
    description: "A metabolic disease..."
})
```
**Source:** Kaggle Disease-Symptom dataset
**Count:** ~400 diseases (pending download)

---

### 4. Symptom (from Kaggle dataset)
```cypher
(:Symptom {
    name: "chest_pain",          // Primary key (normalized)
    display_name: "Chest Pain"   // Human-readable
})
```
**Source:** Kaggle Disease-Symptom dataset
**Count:** ~130 symptoms (pending download)

---

## Relationship Types

### 1. INTERACTS_WITH (Drug → Drug)
```cypher
(:Drug)-[:INTERACTS_WITH {
    type: "risk or severity of bleeding",
    severity: "high"             // high, moderate, low
}]->(:Drug)
```
**Source:** DDI_data.csv
**Count:** ~222,696 interactions

**Severity Classification:**
| Severity | Interaction Types | Clinical Action |
|----------|------------------|-----------------|
| `high` | bleeding, QTc prolongation, serotonin syndrome, cardiotoxic | **ALERT** - Requires review |
| `moderate` | adverse effects, hypotension, CNS depression | **WARNING** - Monitor |
| `low` | metabolism, serum concentration, absorption | **INFO** - May need adjustment |

---

### 2. IS_CHILD_OF (ICD10 → ICD10)
```cypher
(:ICD10 {code: "I10.1"})-[:IS_CHILD_OF]->(:ICD10 {code: "I10"})
```
Represents ICD-10 hierarchy. Derived from code structure:
- `I10` is parent of `I10.1`, `I10.2`, etc.
- `I` (chapter) is parent of `I10`, `I11`, etc.

---

### 3. INDICATES (Symptom → Disease)
```cypher
(:Symptom)-[:INDICATES {
    weight: 0.8                  // Strength of association (0-1)
}]->(:Disease)
```
**Source:** Kaggle Disease-Symptom dataset
**Status:** Pending data download

---

### 4. CODED_AS (Disease → ICD10)
```cypher
(:Disease)-[:CODED_AS]->(:ICD10)
```
Maps diseases to their ICD-10 billing codes.
**Status:** Will create after loading both datasets

---

## Visual Schema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE GRAPH SCHEMA                          │
│                                                                         │
│   ┌──────────┐                                    ┌──────────┐         │
│   │  :Drug   │──────INTERACTS_WITH───────────────→│  :Drug   │         │
│   │          │  {type, severity}                  │          │         │
│   └──────────┘                                    └──────────┘         │
│                                                                         │
│   ┌──────────┐         INDICATES          ┌───────────┐                │
│   │ :Symptom │───────────────────────────→│ :Disease  │                │
│   │          │  {weight}                  │           │                │
│   └──────────┘                            └─────┬─────┘                │
│                                                 │                       │
│                                                 │ CODED_AS              │
│                                                 ↓                       │
│   ┌──────────┐         IS_CHILD_OF        ┌───────────┐                │
│   │  :ICD10  │←───────────────────────────│  :ICD10   │                │
│   │ (parent) │                            │  (child)  │                │
│   └──────────┘                            └───────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Indexes & Constraints

```cypher
// Unique constraints (auto-creates index)
CREATE CONSTRAINT drug_id IF NOT EXISTS
FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE;

CREATE CONSTRAINT icd10_code IF NOT EXISTS
FOR (i:ICD10) REQUIRE i.code IS UNIQUE;

CREATE CONSTRAINT disease_name IF NOT EXISTS
FOR (d:Disease) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT symptom_name IF NOT EXISTS
FOR (s:Symptom) REQUIRE s.name IS UNIQUE;

// Additional indexes for fast lookup
CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name);
CREATE INDEX icd10_desc IF NOT EXISTS FOR (i:ICD10) ON (i.short_desc);
CREATE INDEX icd10_billable IF NOT EXISTS FOR (i:ICD10) ON (i.billable);

// Full-text indexes for fuzzy search
CREATE FULLTEXT INDEX drug_name_fulltext IF NOT EXISTS
FOR (d:Drug) ON EACH [d.name];

CREATE FULLTEXT INDEX icd10_fulltext IF NOT EXISTS
FOR (i:ICD10) ON EACH [i.short_desc, i.long_desc];

CREATE FULLTEXT INDEX symptom_fulltext IF NOT EXISTS
FOR (s:Symptom) ON EACH [s.name, s.display_name];
```

---

## Sample Queries

### Safety Agent: Check Drug Interactions
```cypher
// Find all interactions between patient's medications
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $medications
  AND toLower(d2.name) IN $medications
RETURN d1.name, d2.name, r.type, r.severity
ORDER BY CASE r.severity WHEN 'high' THEN 1 WHEN 'moderate' THEN 2 ELSE 3 END;
```

### Coding Agent: Suggest ICD-10 Codes
```cypher
// Search for ICD-10 codes matching symptoms/diagnoses
CALL db.index.fulltext.queryNodes("icd10_fulltext", $search_terms)
YIELD node, score
WHERE node.billable = true
RETURN node.code, node.short_desc, score
ORDER BY score DESC
LIMIT 10;
```

### Differential Agent: Symptoms → Diseases
```cypher
// Find diseases matching multiple symptoms
MATCH (s:Symptom)-[r:INDICATES]->(d:Disease)
WHERE s.name IN $symptoms
WITH d, count(s) AS matching_symptoms, sum(r.weight) AS total_weight
RETURN d.name, matching_symptoms, total_weight
ORDER BY matching_symptoms DESC, total_weight DESC
LIMIT 10;
```

---

## Data Loading Order

1. **Load Drugs** - `load_ddi.py` (creates Drug nodes)
2. **Load Interactions** - `load_ddi.py` (creates INTERACTS_WITH)
3. **Load ICD-10** - `load_icd10.py` (creates ICD10 nodes + IS_CHILD_OF)
4. **Load Symptoms/Diseases** - `load_symptoms.py` (pending data)
5. **Create Disease→ICD10 mappings** - `link_disease_icd10.py` (after step 4)

---

## Statistics (Expected)

| Node Type | Count | Source |
|-----------|-------|--------|
| Drug | ~1,751 | DDI_data.csv |
| ICD10 | ~74,719 | icd10cm_order_2026.txt |
| Disease | ~400 | Kaggle (pending) |
| Symptom | ~130 | Kaggle (pending) |

| Relationship | Count | Source |
|--------------|-------|--------|
| INTERACTS_WITH | ~222,696 | DDI_data.csv |
| IS_CHILD_OF | ~74,000 | Derived from ICD-10 codes |
| INDICATES | ~4,000 | Kaggle (pending) |
| CODED_AS | ~400 | Manual/SNOMED (pending) |

---

## Privacy Note

**This graph contains NO patient data.**

Only generic medical knowledge:
- Drug names and interactions
- ICD-10 codes and descriptions
- Disease-symptom relationships

Patient data stays **on-device** in the mobile app.
