"""
Neo4j Service - Drug Interaction Agent

Queries Neo4j for drug interactions.
Concise responses: result first, details on demand.
"""

import os
from typing import Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DrugInteraction:
    drug1: str
    drug2: str
    interaction_type: str


@dataclass
class InteractionResult:
    found: bool
    interactions: list[DrugInteraction]
    summary: str  # One-liner for quick view


# Common brand/trade names → DrugBank canonical names
# (fulltext search can't match "aspirin" to "Acetylsalicylic acid")
_DRUG_ALIASES: dict[str, str] = {
    "aspirin": "Acetylsalicylic acid",
    "tylenol": "Acetaminophen",
    "paracetamol": "Acetaminophen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    "aleve": "Naproxen",
    "coumadin": "Warfarin",
    "lipitor": "Atorvastatin",
    "zocor": "Simvastatin",
    "crestor": "Rosuvastatin",
    "nexium": "Esomeprazole",
    "prilosec": "Omeprazole",
    "plavix": "Clopidogrel",
    "eliquis": "Apixaban",
    "xarelto": "Rivaroxaban",
    "lasix": "Furosemide",
    "norvasc": "Amlodipine",
    "lisinopril": "Lisinopril",
    "losartan": "Losartan",
    "synthroid": "Levothyroxine",
    "glucophage": "Metformin",
    "januvia": "Sitagliptin",
    "lantus": "Insulin glargine",
    "xanax": "Alprazolam",
    "valium": "Diazepam",
    "zoloft": "Sertraline",
    "prozac": "Fluoxetine",
    "lexapro": "Escitalopram",
    "ambien": "Zolpidem",
    "prednisone": "Prednisone",
    "prednisolone": "Prednisolone",
    "amoxicillin": "Amoxicillin",
    "azithromycin": "Azithromycin",
    "ciprofloxacin": "Ciprofloxacin",
    "metoprolol": "Metoprolol",
    "atenolol": "Atenolol",
    "propranolol": "Propranolol",
    "digoxin": "Digoxin",
    "gabapentin": "Gabapentin",
    "tramadol": "Tramadol",
    "morphine": "Morphine",
    "oxycodone": "Oxycodone",
    "hydrocodone": "Hydrocodone",
    "heparin": "Heparin",
    "enoxaparin": "Enoxaparin",
    "lovenox": "Enoxaparin",
}


class Neo4jService:
    """Neo4j connection and query service."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def _resolve_drug(self, session, name: str) -> str | None:
        """Resolve a drug name to its DrugBank ID.

        Tries: alias table → exact match → fulltext fuzzy search.
        """
        # Check alias table first (aspirin → Acetylsalicylic acid, etc.)
        canonical = _DRUG_ALIASES.get(name.lower(), name)

        # Exact match (case-insensitive)
        result = session.run(
            "MATCH (d:Drug) WHERE toLower(d.name) = $name RETURN d.drugbank_id AS id LIMIT 1",
            name=canonical.lower(),
        )
        rec = result.single()
        if rec:
            return rec["id"]

        # Also try the original name if different from canonical
        if canonical.lower() != name.lower():
            result = session.run(
                "MATCH (d:Drug) WHERE toLower(d.name) = $name RETURN d.drugbank_id AS id LIMIT 1",
                name=name.lower(),
            )
            rec = result.single()
            if rec:
                return rec["id"]

        # Fulltext fuzzy match
        result = session.run(
            """CALL db.index.fulltext.queryNodes("drug_name_fulltext", $term)
               YIELD node, score
               WHERE score > 1.0
               RETURN node.drugbank_id AS id LIMIT 1""",
            term=canonical,
        )
        rec = result.single()
        return rec["id"] if rec else None

    def check_interactions(self, medications: list[str]) -> InteractionResult:
        """
        Check drug interactions for a list of medications.

        Uses fulltext search to resolve common drug names (e.g. "aspirin"
        → "Acetylsalicylic acid") before querying interaction edges.
        """
        if len(medications) < 2:
            return InteractionResult(
                found=False,
                interactions=[],
                summary="Need 2+ medications to check interactions."
            )

        with self.driver.session() as session:
            # Resolve each medication name to a DrugBank ID
            resolved: dict[str, str] = {}  # drugbank_id → original name
            for med in medications:
                db_id = self._resolve_drug(session, med)
                if db_id:
                    resolved[db_id] = med

            if len(resolved) < 2:
                return InteractionResult(
                    found=False,
                    interactions=[],
                    summary=f"Could only resolve {len(resolved)}/{len(medications)} medications in DrugBank."
                )

            db_ids = list(resolved.keys())

            result = session.run("""
                MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
                WHERE d1.drugbank_id IN $ids AND d2.drugbank_id IN $ids
                  AND d1.drugbank_id < d2.drugbank_id
                RETURN d1.name AS drug1, d2.name AS drug2, r.interaction_type AS type
                ORDER BY r.interaction_type
            """, ids=db_ids)

            interactions = [
                DrugInteraction(
                    drug1=r["drug1"],
                    drug2=r["drug2"],
                    interaction_type=r["type"]
                )
                for r in result
            ]

        if not interactions:
            return InteractionResult(
                found=False,
                interactions=[],
                summary=f"No interactions found between {len(medications)} medications."
            )

        # Generate concise summary
        high_risk = [i for i in interactions if self._is_high_risk(i.interaction_type)]

        if high_risk:
            summary = f"⚠️ {len(high_risk)} high-risk interaction(s) found."
        else:
            summary = f"{len(interactions)} interaction(s) found."

        return InteractionResult(
            found=True,
            interactions=interactions,
            summary=summary
        )

    def _is_high_risk(self, interaction_type: str) -> bool:
        """Check if interaction type is high risk."""
        high_risk_keywords = [
            'bleeding', 'anticoagulant', 'qtc', 'serotonergic',
            'cardiotoxic', 'nephrotoxic', 'hepatotoxic', 'respiratory'
        ]
        return any(kw in interaction_type.lower() for kw in high_risk_keywords)

    def search_drug(self, search_term: str, limit: int = 5) -> list[dict]:
        """Search for drugs by name."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("drug_name_fulltext", $term)
                YIELD node, score
                RETURN node.name AS name, node.drugbank_id AS id, score
                ORDER BY score DESC
                LIMIT $lim
            """, term=search_term, lim=limit)
            return [dict(r) for r in result]

    def search_icd10(self, search_term: str, limit: int = 5) -> list[dict]:
        """Search ICD-10 codes by description."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("icd10_fulltext", $term)
                YIELD node, score
                WHERE node.billable = true
                RETURN node.code AS code, node.short_desc AS description, score
                ORDER BY score DESC
                LIMIT $lim
            """, term=search_term, lim=limit)
            return [dict(r) for r in result]


# Singleton instance
_neo4j_service: Optional[Neo4jService] = None


def get_neo4j_service() -> Neo4jService:
    """Get singleton Neo4j service instance."""
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service
