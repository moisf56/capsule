"""
Drug-Drug Interaction Loader for Neo4j

Loads DDI_data.csv into Neo4j with schema:
(:Drug {drugbank_id, name})-[:INTERACTS_WITH {type, severity}]->(:Drug)

Usage:
    python load_ddi.py --uri bolt://localhost:7687 --user neo4j --password <password>

Or set environment variables:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Generator
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def read_ddi_csv(file_path: Path, batch_size: int = 5000) -> Generator[list, None, None]:
    """Read DDI CSV in batches to avoid memory issues."""
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class DDILoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create unique constraints and indexes for performance."""
        with self.driver.session() as session:
            # Unique constraint on Drug drugbank_id
            session.run("""
                CREATE CONSTRAINT drug_id IF NOT EXISTS
                FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE
            """)

            # Index on drug name for fast lookups
            session.run("""
                CREATE INDEX drug_name IF NOT EXISTS
                FOR (d:Drug) ON (d.name)
            """)

            # Full-text index for fuzzy drug name search
            session.run("""
                CREATE FULLTEXT INDEX drug_name_fulltext IF NOT EXISTS
                FOR (d:Drug) ON EACH [d.name]
            """)

            print("Created constraints and indexes")

    def load_drugs_and_interactions(self, file_path: Path):
        """Load all drugs and their interactions from DDI CSV."""
        total_drugs = set()
        total_interactions = 0

        with self.driver.session() as session:
            for batch_num, batch in enumerate(read_ddi_csv(file_path)):
                # Collect unique drugs from batch
                drugs = {}
                for row in batch:
                    drugs[row['drug1_id']] = row['drug1_name']
                    drugs[row['drug2_id']] = row['drug2_name']

                # Create drug nodes
                session.run("""
                    UNWIND $drugs AS drug
                    MERGE (d:Drug {drugbank_id: drug.id})
                    ON CREATE SET d.name = drug.name
                """, drugs=[{"id": k, "name": v} for k, v in drugs.items()])

                # Create interactions (using exact CSV column name)
                interactions = [
                    {
                        "drug1_id": row['drug1_id'],
                        "drug2_id": row['drug2_id'],
                        "interaction_type": row['interaction_type']
                    }
                    for row in batch
                ]

                session.run("""
                    UNWIND $interactions AS i
                    MATCH (d1:Drug {drugbank_id: i.drug1_id})
                    MATCH (d2:Drug {drugbank_id: i.drug2_id})
                    MERGE (d1)-[r:INTERACTS_WITH]->(d2)
                    ON CREATE SET r.interaction_type = i.interaction_type
                    ON MATCH SET r.interaction_type = i.interaction_type
                """, interactions=interactions)

                total_drugs.update(drugs.keys())
                total_interactions += len(batch)

                print(f"Batch {batch_num + 1}: Loaded {len(drugs)} drugs, {len(batch)} interactions")

        return len(total_drugs), total_interactions

    def verify_load(self):
        """Verify the data was loaded correctly."""
        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (d:Drug) RETURN count(d) as count")
            drug_count = result.single()["count"]

            # Count relationships
            result = session.run("MATCH ()-[r:INTERACTS_WITH]->() RETURN count(r) as count")
            interaction_count = result.single()["count"]

            # Count by interaction_type (top 10)
            result = session.run("""
                MATCH ()-[r:INTERACTS_WITH]->()
                RETURN r.interaction_type as interaction_type, count(r) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            top_types = {row["interaction_type"]: row["count"] for row in result}

            return drug_count, interaction_count, top_types

    def test_query(self, drug_names: list[str]):
        """Test query: Find interactions between given drugs."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
                WHERE toLower(d1.name) IN $names AND toLower(d2.name) IN $names
                RETURN d1.name as drug1, d2.name as drug2,
                       r.interaction_type as interaction_type
            """, names=[n.lower() for n in drug_names])
            return list(result)


def main():
    parser = argparse.ArgumentParser(description="Load DDI data into Neo4j")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--data-file", default=None, help="Path to DDI_data.csv")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    parser.add_argument("--test-drugs", nargs="+", help="Test query with drug names")
    args = parser.parse_args()

    if not args.password:
        print("Error: Neo4j password required. Set NEO4J_PASSWORD env var or use --password")
        return 1

    # Default data file path
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        data_file = Path(__file__).parent.parent / "data" / "DDI_data.csv"

    loader = DDILoader(args.uri, args.user, args.password)

    try:
        if args.test_drugs:
            print(f"\nTesting query for drugs: {args.test_drugs}")
            results = loader.test_query(args.test_drugs)
            if results:
                for r in results:
                    print(f"  {r['drug1']} + {r['drug2']}: {r['interaction_type']}")
            else:
                print("  No interactions found")
            return 0

        if args.verify_only:
            drug_count, interaction_count, top_types = loader.verify_load()
            print(f"\nVerification Results:")
            print(f"  Total Drugs: {drug_count}")
            print(f"  Total Interactions: {interaction_count}")
            print(f"  Top Interaction Types:")
            for itype, count in top_types.items():
                print(f"    {itype}: {count}")
            return 0

        if not data_file.exists():
            print(f"Error: Data file not found: {data_file}")
            return 1

        print(f"Loading DDI data from: {data_file}")
        print(f"Connecting to: {args.uri}")

        # Create constraints first
        loader.create_constraints()

        # Load data
        drug_count, interaction_count = loader.load_drugs_and_interactions(data_file)
        print(f"\nLoaded {drug_count} drugs and {interaction_count} interactions")

        # Verify
        drug_count, interaction_count, top_types = loader.verify_load()
        print(f"\nVerification:")
        print(f"  Drugs in DB: {drug_count}")
        print(f"  Interactions in DB: {interaction_count}")
        print(f"  Top Interaction Types:")
        for itype, count in top_types.items():
            print(f"    {itype}: {count}")

        # Test query
        print("\nTest: Aspirin + Ibuprofen interactions")
        results = loader.test_query(["Aspirin", "Ibuprofen"])
        if results:
            for r in results:
                print(f"  {r['drug1']} + {r['drug2']}: {r['interaction_type']}")
        else:
            print("  No direct interaction found (may need name variants)")

    finally:
        loader.close()

    return 0


if __name__ == "__main__":
    exit(main())
