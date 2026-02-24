// ===========================================
// Drug Interaction Queries for Clinical Agent
// ===========================================

// -------------------------------------------
// 1. CHECK DRUG INTERACTIONS (Core Query)
// -------------------------------------------
// Input: List of medication names
// Output: All interactions between the drugs with severity
// Used by: ClinicalEnhancementAgent

// Find all interactions between a list of drugs
// Replace $drug_names with actual list like ["Aspirin", "Ibuprofen", "Metformin"]
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS interaction_type,
       r.severity AS severity
ORDER BY CASE r.severity
           WHEN 'high' THEN 1
           WHEN 'moderate' THEN 2
           WHEN 'low' THEN 3
         END;

// -------------------------------------------
// 2. HIGH SEVERITY INTERACTIONS ONLY
// -------------------------------------------
// For critical alerts that REQUIRE user action
MATCH (d1:Drug)-[r:INTERACTS_WITH {severity: 'high'}]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS interaction_type,
       'CRITICAL: Requires immediate attention' AS alert_level;

// -------------------------------------------
// 3. FIND ALL INTERACTIONS FOR A SINGLE DRUG
// -------------------------------------------
// Useful for: "What interacts with Aspirin?"
MATCH (d1:Drug {name: $drug_name})-[r:INTERACTS_WITH]-(d2:Drug)
RETURN d2.name AS interacting_drug,
       r.type AS interaction_type,
       r.severity AS severity
ORDER BY r.severity;

// -------------------------------------------
// 4. BLEEDING RISK CHECK (Common Alert)
// -------------------------------------------
// Specifically check for bleeding-related interactions
// Important for: NSAIDs, anticoagulants, antiplatelet drugs
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
  AND (r.type CONTAINS 'bleeding' OR r.type CONTAINS 'anticoagulant' OR r.type CONTAINS 'antiplatelet')
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS risk_type,
       'HIGH' AS severity,
       'GI bleeding, hemorrhage risk' AS clinical_note;

// -------------------------------------------
// 5. QTc PROLONGATION CHECK (Cardiac Safety)
// -------------------------------------------
// Important for: Many psychiatric drugs, antibiotics
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
  AND r.type CONTAINS 'QTc'
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS risk_type,
       'HIGH' AS severity,
       'Risk of cardiac arrhythmia - consider ECG monitoring' AS clinical_note;

// -------------------------------------------
// 6. SEROTONIN SYNDROME CHECK
// -------------------------------------------
// Important for: SSRIs, SNRIs, MAOIs, triptans
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
  AND r.type CONTAINS 'serotonergic'
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS risk_type,
       'HIGH' AS severity,
       'Risk of serotonin syndrome - monitor for hyperthermia, agitation, tremor' AS clinical_note;

// -------------------------------------------
// 7. HYPOGLYCEMIA CHECK (Diabetes Patients)
// -------------------------------------------
// Important for: Insulin, sulfonylureas, metformin
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
  AND r.type CONTAINS 'hypoglycemic'
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS risk_type,
       'MODERATE' AS severity,
       'Monitor blood glucose closely' AS clinical_note;

// -------------------------------------------
// 8. CNS DEPRESSION CHECK
// -------------------------------------------
// Important for: Opioids, benzodiazepines, sedatives
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
  AND (r.type CONTAINS 'CNS depressant' OR r.type CONTAINS 'sedative' OR r.type CONTAINS 'respiratory')
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS risk_type,
       'HIGH' AS severity,
       'Risk of excessive sedation and respiratory depression' AS clinical_note;

// -------------------------------------------
// 9. FUZZY DRUG NAME SEARCH
// -------------------------------------------
// When exact name doesn't match, find similar drugs
CALL db.index.fulltext.queryNodes("drug_name_fulltext", $search_term)
YIELD node, score
RETURN node.name AS drug_name, node.drugbank_id AS drugbank_id, score
ORDER BY score DESC
LIMIT 5;

// -------------------------------------------
// 10. GET DRUG BY DRUGBANK ID
// -------------------------------------------
MATCH (d:Drug {drugbank_id: $drugbank_id})
RETURN d.name AS name, d.drugbank_id AS id;

// -------------------------------------------
// 11. COMPREHENSIVE SAFETY CHECK
// -------------------------------------------
// Run all critical checks at once
// Returns structured results for the agent
MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
WHERE toLower(d1.name) IN $drug_names
  AND toLower(d2.name) IN $drug_names
  AND r.severity IN ['high', 'moderate']
WITH d1, d2, r,
     CASE
       WHEN r.type CONTAINS 'bleeding' OR r.type CONTAINS 'anticoagulant' THEN 'BLEEDING_RISK'
       WHEN r.type CONTAINS 'QTc' THEN 'CARDIAC_RISK'
       WHEN r.type CONTAINS 'serotonergic' THEN 'SEROTONIN_SYNDROME'
       WHEN r.type CONTAINS 'hypoglycemic' THEN 'HYPOGLYCEMIA'
       WHEN r.type CONTAINS 'CNS depressant' OR r.type CONTAINS 'sedative' THEN 'CNS_DEPRESSION'
       WHEN r.type CONTAINS 'hypotensive' OR r.type CONTAINS 'hypotension' THEN 'HYPOTENSION'
       WHEN r.type CONTAINS 'nephrotoxic' THEN 'KIDNEY_RISK'
       ELSE 'OTHER'
     END AS risk_category
RETURN d1.name AS drug1,
       d2.name AS drug2,
       r.type AS interaction_type,
       r.severity AS severity,
       risk_category
ORDER BY CASE r.severity WHEN 'high' THEN 1 ELSE 2 END,
         risk_category;

// -------------------------------------------
// STATISTICS QUERIES
// -------------------------------------------

// Count total drugs and interactions
MATCH (d:Drug) WITH count(d) AS drug_count
MATCH ()-[r:INTERACTS_WITH]->()
RETURN drug_count, count(r) AS interaction_count;

// Count by severity
MATCH ()-[r:INTERACTS_WITH]->()
RETURN r.severity AS severity, count(r) AS count
ORDER BY count DESC;

// Count by interaction type (top 20)
MATCH ()-[r:INTERACTS_WITH]->()
RETURN r.type AS interaction_type, count(r) AS count
ORDER BY count DESC
LIMIT 20;
