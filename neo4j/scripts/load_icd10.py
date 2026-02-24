"""
ICD-10-CM Loader for Neo4j

Loads icd10cm_order_2026.txt into Neo4j with schema:
(:ICD10 {code, short_desc, long_desc, billable, chapter})
(:ICD10)-[:IS_CHILD_OF]->(:ICD10)

File format (fixed-width):
- Positions 0-5:   Order number
- Positions 6-13:  ICD-10 code
- Position 14:     Billable flag (0=header, 1=billable)
- Positions 16-76: Short description
- Positions 77+:   Long description

Usage:
    python load_icd10.py --uri bolt://localhost:7687 --user neo4j --password <password>
"""

import os
import argparse
from pathlib import Path
from typing import Generator
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def parse_icd10_line(line: str) -> dict | None:
    """Parse a single line from icd10cm_order file."""
    if len(line) < 20:
        return None

    try:
        # Fixed-width parsing
        code = line[6:14].strip()
        billable = line[14:15].strip() == '1'
        short_desc = line[16:77].strip()
        long_desc = line[77:].strip() if len(line) > 77 else short_desc

        if not code:
            return None

        # Extract chapter (first character for letter codes)
        chapter = code[0] if code[0].isalpha() else None

        return {
            'code': code,
            'billable': billable,
            'short_desc': short_desc,
            'long_desc': long_desc,
            'chapter': chapter
        }
    except Exception:
        return None


def get_parent_code(code: str) -> str | None:
    """
    Get parent code for ICD-10 hierarchy.

    Examples:
    - I10.1 -> I10
    - I10 -> I (chapter)
    - A000 -> A00
    - A00 -> A0
    - A0 -> A
    - A -> None (top level)
    """
    if not code:
        return None

    # If has decimal, remove last part after decimal
    if '.' in code:
        return code.rsplit('.', 1)[0]

    # If length > 1, remove last character
    if len(code) > 1:
        return code[:-1]

    # Single character = top level, no parent
    return None


def read_icd10_file(file_path: Path, batch_size: int = 5000) -> Generator[list, None, None]:
    """Read ICD-10 file in batches."""
    batch = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parsed = parse_icd10_line(line)
            if parsed:
                batch.append(parsed)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


class ICD10Loader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create constraints and indexes."""
        with self.driver.session() as session:
            # Unique constraint on code
            session.run("""
                CREATE CONSTRAINT icd10_code IF NOT EXISTS
                FOR (i:ICD10) REQUIRE i.code IS UNIQUE
            """)

            # Index on description for search
            session.run("""
                CREATE INDEX icd10_short_desc IF NOT EXISTS
                FOR (i:ICD10) ON (i.short_desc)
            """)

            # Index on billable flag
            session.run("""
                CREATE INDEX icd10_billable IF NOT EXISTS
                FOR (i:ICD10) ON (i.billable)
            """)

            # Index on chapter
            session.run("""
                CREATE INDEX icd10_chapter IF NOT EXISTS
                FOR (i:ICD10) ON (i.chapter)
            """)

            # Full-text index for fuzzy search
            session.run("""
                CREATE FULLTEXT INDEX icd10_fulltext IF NOT EXISTS
                FOR (i:ICD10) ON EACH [i.short_desc, i.long_desc]
            """)

            print("Created ICD-10 constraints and indexes")

    def load_codes(self, file_path: Path) -> tuple[int, int]:
        """Load ICD-10 codes from file."""
        total_codes = 0
        billable_codes = 0

        with self.driver.session() as session:
            for batch_num, batch in enumerate(read_icd10_file(file_path)):
                # Create ICD10 nodes
                session.run("""
                    UNWIND $codes AS c
                    MERGE (i:ICD10 {code: c.code})
                    ON CREATE SET
                        i.short_desc = c.short_desc,
                        i.long_desc = c.long_desc,
                        i.billable = c.billable,
                        i.chapter = c.chapter
                    ON MATCH SET
                        i.short_desc = c.short_desc,
                        i.long_desc = c.long_desc,
                        i.billable = c.billable,
                        i.chapter = c.chapter
                """, codes=batch)

                total_codes += len(batch)
                billable_codes += sum(1 for c in batch if c['billable'])

                print(f"Batch {batch_num + 1}: Loaded {len(batch)} codes (running total: {total_codes})")

        return total_codes, billable_codes

    def create_hierarchy(self):
        """Create IS_CHILD_OF relationships based on code structure."""
        with self.driver.session() as session:
            # This query finds parent-child relationships
            # A code's parent is derived by removing the last character (or decimal part)
            result = session.run("""
                MATCH (child:ICD10)
                WHERE size(child.code) > 1
                WITH child,
                     CASE
                         WHEN child.code CONTAINS '.'
                         THEN substring(child.code, 0, size(child.code) -
                              size(split(child.code, '.')[-1]) - 1)
                         ELSE substring(child.code, 0, size(child.code) - 1)
                     END AS parent_code
                MATCH (parent:ICD10 {code: parent_code})
                MERGE (child)-[:IS_CHILD_OF]->(parent)
                RETURN count(*) AS relationships_created
            """)

            count = result.single()["relationships_created"]
            print(f"Created {count} IS_CHILD_OF relationships")
            return count

    def verify_load(self) -> dict:
        """Verify the data was loaded correctly."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:ICD10)
                RETURN
                    count(i) AS total_codes,
                    sum(CASE WHEN i.billable THEN 1 ELSE 0 END) AS billable_codes,
                    count(DISTINCT i.chapter) AS chapters
            """)
            stats = result.single()

            # Count relationships
            rel_result = session.run("""
                MATCH ()-[r:IS_CHILD_OF]->()
                RETURN count(r) AS hierarchy_relationships
            """)
            rel_count = rel_result.single()["hierarchy_relationships"]

            return {
                "total_codes": stats["total_codes"],
                "billable_codes": stats["billable_codes"],
                "chapters": stats["chapters"],
                "hierarchy_relationships": rel_count
            }

    def search_codes(self, search_term: str, limit: int = 10) -> list:
        """Search for ICD-10 codes by description."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("icd10_fulltext", $term)
                YIELD node, score
                WHERE node.billable = true
                RETURN node.code AS code, node.short_desc AS description, score
                ORDER BY score DESC
                LIMIT $limit
            """, term=search_term, limit=limit)
            return list(result)

    def get_code_with_parents(self, code: str) -> list:
        """Get a code and all its parents in the hierarchy."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (child:ICD10 {code: $code})-[:IS_CHILD_OF*0..10]->(ancestor:ICD10)
                RETURN ancestor.code AS code, ancestor.short_desc AS description,
                       ancestor.billable AS billable, length(path) AS depth
                ORDER BY depth
            """, code=code)
            return list(result)


def main():
    parser = argparse.ArgumentParser(description="Load ICD-10-CM data into Neo4j")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--data-file", default=None, help="Path to icd10cm_order_2026.txt")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--search", help="Search for ICD-10 codes")
    parser.add_argument("--hierarchy", help="Show hierarchy for a code")
    args = parser.parse_args()

    if not args.password:
        print("Error: Neo4j password required. Set NEO4J_PASSWORD env var or use --password")
        return 1

    # Default data file path
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        data_file = Path(__file__).parent.parent.parent / "data" / "ICD-10-CM" / "Code Descriptions" / "icd10cm_order_2026.txt"

    loader = ICD10Loader(args.uri, args.user, args.password)

    try:
        if args.search:
            print(f"\nSearching for: {args.search}")
            results = loader.search_codes(args.search)
            for r in results:
                print(f"  {r['code']}: {r['description']} (score: {r['score']:.2f})")
            return 0

        if args.hierarchy:
            print(f"\nHierarchy for: {args.hierarchy}")
            results = loader.get_code_with_parents(args.hierarchy)
            for r in results:
                indent = "  " * r['depth']
                billable = "✓" if r['billable'] else " "
                print(f"{indent}[{billable}] {r['code']}: {r['description']}")
            return 0

        if args.verify_only:
            stats = loader.verify_load()
            print(f"\nVerification Results:")
            print(f"  Total codes: {stats['total_codes']}")
            print(f"  Billable codes: {stats['billable_codes']}")
            print(f"  Chapters: {stats['chapters']}")
            print(f"  Hierarchy relationships: {stats['hierarchy_relationships']}")
            return 0

        if not data_file.exists():
            print(f"Error: Data file not found: {data_file}")
            return 1

        print(f"Loading ICD-10-CM data from: {data_file}")
        print(f"Connecting to: {args.uri}")

        # Create constraints
        loader.create_constraints()

        # Load codes
        total, billable = loader.load_codes(data_file)
        print(f"\nLoaded {total} codes ({billable} billable)")

        # Create hierarchy
        loader.create_hierarchy()

        # Verify
        stats = loader.verify_load()
        print(f"\nVerification:")
        print(f"  Total codes: {stats['total_codes']}")
        print(f"  Billable codes: {stats['billable_codes']}")
        print(f"  Hierarchy relationships: {stats['hierarchy_relationships']}")

        # Test search
        print("\nTest search: 'hypertension'")
        results = loader.search_codes("hypertension", limit=5)
        for r in results:
            print(f"  {r['code']}: {r['description']}")

    finally:
        loader.close()

    return 0


if __name__ == "__main__":
    exit(main())
