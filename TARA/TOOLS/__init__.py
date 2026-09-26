"""
TARA/TOOLS Package

Standard, safe, production-grade utility tools for TARA:
- file_inspector: Safe metadata inspection with secret boundary isolation
- hash_verifier: Cryptographic hash generation and verification
- knowledge_retriever: Structured interface to GlobalKnowledgeBase
- provenance_tracker: Data provenance and audit token generation
"""

from .file_inspector import inspect_file
from .hash_verifier import compute_hash, verify_file_hash
from .knowledge_retriever import search_knowledge
from .provenance_tracker import create_provenance_record

__all__ = [
    "inspect_file",
    "compute_hash",
    "verify_file_hash",
    "search_knowledge",
    "create_provenance_record"
]
