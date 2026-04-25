### ADR-2026-04-08-07: Align ingest_step with Actual load_step Signature (pythonocc-core 7.9.3)

**Status**: Accepted  
**Date**: 2026-04-08  
**Context**: load_step() in cad/parser.py takes only step_file_path (1 arg). Previous ingest_step and tests incorrectly passed config, causing TypeError.  
**Decision**: Update only kg/ingest.py ingest_step to call load_step(step_file_path) exactly as implemented in parser. Keep config for adjacency and future use. build_adjacency_from_occ still receives config.  
**Rationale**: Preserves fixed architecture (cad/ → kg/). Guarantees real adjacency via topexp_MapShapesAndAncestors on 7.9.3 with zero NameErrors/deprecations.  
**Verification**: tests/test_kg_ingest_full_793.py (≥90% coverage on ingest + adjacency).

### ADR-2026-04-08-08: Extract TopoDS_Shape from dict returned by load_step() (pythonocc-core 7.9.3)

**Status**: Accepted  
**Date**: 2026-04-08  
**Context**: load_step() returns a dict (contains topology counts and the main shape), not an object with .shape attribute. Previous ingest_step and tests assumed object.  
**Decision**: In kg/ingest.py and tests: use parser_result.get("shape") or parser_result["shape"] to extract TopoDS_Shape. Keep load_step(step_file_path) call unchanged.  
**Rationale**: Preserves fixed architecture (cad/ → kg/). Guarantees real adjacency on 7.9.3 without NameErrors or fallback to 0 pairs.  
**Verification**: tests/test_kg_ingest_full_793.py (≥90% coverage).