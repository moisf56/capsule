// =====================================================
// NEO4J DATABASE EXPLORATION COMMANDS
// =====================================================
// Run these in Neo4j Browser or Aura Console
// Connection: neo4j+s://bf5f8749.databases.neo4j.io
// =====================================================

// -------------------------------------------
// 1. DATABASE OVERVIEW
// -------------------------------------------

// Count all nodes by label
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS count
ORDER BY count DESC;

// Count all relationships by type
MATCH ()-[r]->()
RETURN type(r) AS relationship, count(r) AS count
ORDER BY count DESC;

// Database schema visualization
CALL db.schema.visualization();

// -------------------------------------------
// 2. EXPLORE DRUGS
// -------------------------------------------

// List first 20 drugs
MATCH (d:Drug)
RETURN d.drugbank_id AS id, d.name AS name
LIMIT 20;

// Search drug by name (case insensitive)
MATCH (d:Drug)
WHERE toLower(d.name) CONTAINS 'aspirin'
RETURN d.drugbank_id, d.name;

// Find drug by DrugBank ID
MATCH (d:Drug {drugbank_id: 'DB00945'})
RETURN d;

// Drugs with most interactions
MATCH (d:Drug)-[r:INTERACTS_WITH]-()
RETURN d.name AS drug, count(r) AS interactions
ORDER BY interactions DESC
LIMIT 20;

// -------------------------------------------
// 3. EXPLORE DRUG INTERACTIONS
// -------------------------------------------

// All interactions for a specific drug
MATCH (d1:Drug {name: 'Acetylsalicylic acid'})-[r:INTERACTS_WITH]-(d2:Drug)
RETURN d2.name AS interacts_with, r.interaction_type AS type
ORDER BY r.interaction_type
LIMIT 50;

// Check interaction between two drugs
MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) = 'warfarin' AND toLower(d2.name) = 'acetylsalicylic acid'
RETURN d1.name, d2.name, r.interaction_type;

// Find interactions for multiple drugs (like patient med list)
MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) IN ['warfarin', 'acetylsalicylic acid', 'metformin']
  AND toLower(d2.name) IN ['warfarin', 'acetylsalicylic acid', 'metformin']
  AND d1.drugbank_id < d2.drugbank_id  // Avoid duplicates
RETURN d1.name AS drug1, d2.name AS drug2, r.interaction_type;

// Count interactions by type
MATCH ()-[r:INTERACTS_WITH]->()
RETURN r.interaction_type AS type, count(r) AS count
ORDER BY count DESC
LIMIT 20;

// High-risk interactions (bleeding, cardiac, etc.)
MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
WHERE r.interaction_type CONTAINS 'bleeding'
   OR r.interaction_type CONTAINS 'QTc'
   OR r.interaction_type CONTAINS 'anticoagulant'
RETURN d1.name, d2.name, r.interaction_type
LIMIT 50;

// -------------------------------------------
// 4. EXPLORE ICD-10 CODES
// -------------------------------------------

// List first 20 ICD-10 codes
MATCH (i:ICD10)
RETURN i.code, i.short_desc, i.billable
ORDER BY i.code
LIMIT 20;

// Search ICD-10 by description (fulltext)
CALL db.index.fulltext.queryNodes("icd10_fulltext", "hypertension")
YIELD node, score
RETURN node.code AS code, node.short_desc AS description, node.billable, score
ORDER BY score DESC
LIMIT 10;

// Search ICD-10 by code prefix
MATCH (i:ICD10)
WHERE i.code STARTS WITH 'I10'
RETURN i.code, i.short_desc, i.billable
ORDER BY i.code;

// Count billable vs non-billable codes
MATCH (i:ICD10)
RETURN i.billable AS billable, count(i) AS count;

// Codes by chapter
MATCH (i:ICD10)
RETURN i.chapter AS chapter, count(i) AS codes
ORDER BY chapter;

// -------------------------------------------
// 5. EXPLORE ICD-10 HIERARCHY
// -------------------------------------------

// Get parent of a code
MATCH (child:ICD10 {code: 'I10'})-[:IS_CHILD_OF]->(parent:ICD10)
RETURN child.code, child.short_desc, parent.code AS parent, parent.short_desc;

// Get children of a code
MATCH (parent:ICD10 {code: 'I10'})<-[:IS_CHILD_OF]-(child:ICD10)
RETURN child.code, child.short_desc, child.billable
ORDER BY child.code;

// Full hierarchy path (up to 5 levels)
MATCH path = (child:ICD10 {code: 'E11.65'})-[:IS_CHILD_OF*0..5]->(ancestor:ICD10)
RETURN [n IN nodes(path) | n.code + ': ' + n.short_desc] AS hierarchy;

// Find root codes (no parent)
MATCH (i:ICD10)
WHERE NOT (i)-[:IS_CHILD_OF]->()
RETURN i.code, i.short_desc
ORDER BY i.code
LIMIT 30;

// -------------------------------------------
// 6. USEFUL CLINICAL QUERIES
// -------------------------------------------

// Find common diabetes codes
CALL db.index.fulltext.queryNodes("icd10_fulltext", "diabetes mellitus type 2")
YIELD node, score
WHERE node.billable = true
RETURN node.code, node.short_desc, score
ORDER BY score DESC
LIMIT 10;

// Find chest pain codes
CALL db.index.fulltext.queryNodes("icd10_fulltext", "chest pain")
YIELD node, score
WHERE node.billable = true
RETURN node.code, node.short_desc, score
LIMIT 10;

// Find hypertension codes
CALL db.index.fulltext.queryNodes("icd10_fulltext", "hypertension essential primary")
YIELD node, score
WHERE node.billable = true
RETURN node.code, node.short_desc, score
LIMIT 10;

// -------------------------------------------
// 7. STATISTICS & HEALTH CHECKS
// -------------------------------------------

// Total database statistics
MATCH (d:Drug) WITH count(d) AS drugs
MATCH (i:ICD10) WITH drugs, count(i) AS icd10_codes
MATCH ()-[r:INTERACTS_WITH]->() WITH drugs, icd10_codes, count(r) AS interactions
MATCH ()-[h:IS_CHILD_OF]->() WITH drugs, icd10_codes, interactions, count(h) AS hierarchy
RETURN drugs, icd10_codes, interactions, hierarchy;

// Check for orphan nodes (no relationships)
MATCH (d:Drug)
WHERE NOT (d)-[:INTERACTS_WITH]-()
RETURN count(d) AS orphan_drugs;

// Check constraints
SHOW CONSTRAINTS;

// Check indexes
SHOW INDEXES;

// -------------------------------------------
// 8. SAMPLE DEMO QUERIES
// -------------------------------------------

// Demo: Patient on Warfarin + Aspirin + Metoprolol
// Check all interactions
WITH ['warfarin', 'acetylsalicylic acid', 'metoprolol'] AS meds
MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) IN meds AND toLower(d2.name) IN meds
  AND d1.drugbank_id < d2.drugbank_id
RETURN d1.name AS drug1, d2.name AS drug2, r.interaction_type AS interaction;

// Demo: Code lookup for "chest pain, shortness of breath, hypertension"
UNWIND ['chest pain', 'shortness of breath', 'hypertension'] AS term
CALL db.index.fulltext.queryNodes("icd10_fulltext", term)
YIELD node, score
WHERE node.billable = true
WITH term, node, score
ORDER BY score DESC
WITH term, collect({code: node.code, desc: node.short_desc, score: score})[0..3] AS top_codes
RETURN term, top_codes;

// -------------------------------------------
// 9. DATA CLEANUP (USE WITH CAUTION)
// -------------------------------------------

// Delete all data (DANGEROUS - uncomment only if sure)
// MATCH (n) DETACH DELETE n;

// Delete only Drug nodes and relationships
// MATCH (d:Drug) DETACH DELETE d;

// Delete only ICD10 nodes and relationships
// MATCH (i:ICD10) DETACH DELETE i;
