"""
TARA/KNOWLEDGE/knowledge_base.py

Unified Global Knowledge Base for TARA AI.
Acts as the single, common repository for all verified factual knowledge learned
by either Normal Users or Creator ROOT_OPERATOR.
Enforces:
- Single common store (no separate user vs creator stores)
- Semantic deduplication
- Automatic version increment and update history preservation
- Multi-source citation tracking and reliability ratings
- Explicit conflict and contradiction handling
- Provenance recording (role, trigger, session)
- Disk persistence across restarts
"""

import os
import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any

class GlobalKnowledgeBase:
    """Master single-store Global Knowledge Base for TARA."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.base_dir = base_dir
        self.entries_dir = os.path.join(self.base_dir, "entries")
        self.topics_dir = os.path.join(self.base_dir, "topics")
        self.provenance_dir = os.path.join(self.base_dir, "provenance")
        self.candidates_dir = os.path.join(self.base_dir, "candidates")
        self.index_file = os.path.join(self.base_dir, "knowledge_index.json")

        os.makedirs(self.entries_dir, exist_ok=True)
        os.makedirs(self.topics_dir, exist_ok=True)
        os.makedirs(self.provenance_dir, exist_ok=True)
        os.makedirs(self.candidates_dir, exist_ok=True)

        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def _generate_knowledge_id(self, topic: str, key_phrase: str) -> str:
        clean_topic = "".join(c for c in topic if c.isalnum()).upper()[:8] or "GEN"
        hash_suffix = hashlib.sha256(key_phrase.lower().strip().encode("utf-8")).hexdigest()[:8].upper()
        return f"KB-{clean_topic}-{hash_suffix}"

    def store_or_update_knowledge(
        self,
        topic: str,
        subject: str,
        content: str,
        sources: List[Dict[str, Any]],
        learned_by_role: str,
        trigger: str,
        confidence: float = 0.9,
        verification_status: str = "VERIFIED",
        session_id: Optional[str] = None,
        conflicting_views: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Stores or updates a piece of knowledge into the single global repository.
        Deduplicates, preserves history on update, tracks conflicts if present.
        """
        knowledge_id = self._generate_knowledge_id(topic, subject)
        entry_file = os.path.join(self.entries_dir, f"{knowledge_id}.json")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        existing_entry = None
        if os.path.exists(entry_file):
            try:
                with open(entry_file, "r", encoding="utf-8") as f:
                    existing_entry = json.load(f)
            except Exception:
                existing_entry = None

        if existing_entry:
            # Check for exact duplicate content
            existing_content_hash = hashlib.sha256(existing_entry["content"].encode("utf-8")).hexdigest()
            new_content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            if existing_content_hash == new_content_hash and not conflicting_views:
                # Duplicate content: update verification and merge new sources, do not duplicate!
                existing_source_urls = {s.get("url") for s in existing_entry.get("sources", [])}
                for s in sources:
                    if s.get("url") and s.get("url") not in existing_source_urls:
                        existing_entry["sources"].append(s)
                existing_entry["last_verified_at"] = now
                existing_entry["verification_count"] = existing_entry.get("verification_count", 1) + 1
                
                with open(entry_file, "w", encoding="utf-8") as f:
                    json.dump(existing_entry, f, indent=2)

                self.index[knowledge_id]["last_verified_at"] = now
                self.index[knowledge_id]["source_count"] = len(existing_entry["sources"])
                self._save_index()

                return {
                    "action": "DEDUPLICATED_AND_VERIFIED",
                    "knowledge_id": knowledge_id,
                    "version": existing_entry["version"],
                    "entry": existing_entry
                }

            # Update existing knowledge with newer / revised information
            old_version = existing_entry["version"]
            new_version = old_version + 1

            # Archive superseded version into update_history
            archive_item = {
                "version": old_version,
                "content": existing_entry["content"],
                "archived_at": now,
                "confidence": existing_entry["confidence"],
                "sources": existing_entry["sources"]
            }
            history = existing_entry.get("update_history", [])
            history.append(archive_item)

            # Check for conflicting claims
            combined_conflicts = existing_entry.get("conflicting_views", [])
            if conflicting_views:
                combined_conflicts.extend(conflicting_views)

            status = "CONTRADICTORY" if combined_conflicts else verification_status

            updated_entry = {
                "knowledge_id": knowledge_id,
                "topic": topic,
                "subject": subject,
                "content": content,
                "version": new_version,
                "sources": sources,
                "confidence": confidence,
                "verification_status": status,
                "first_learned_at": existing_entry["first_learned_at"],
                "last_updated_at": now,
                "last_verified_at": now,
                "provenance": {
                    "learned_by_role": learned_by_role,
                    "trigger": trigger,
                    "session_id": session_id or "default"
                },
                "update_history": history,
                "conflicting_views": combined_conflicts
            }

            with open(entry_file, "w", encoding="utf-8") as f:
                json.dump(updated_entry, f, indent=2)

            self.index[knowledge_id] = {
                "topic": topic,
                "subject": subject,
                "version": new_version,
                "last_updated_at": now,
                "source_count": len(sources),
                "status": status
            }
            self._save_index()

            return {
                "action": "UPDATED_AND_VERSIONED",
                "knowledge_id": knowledge_id,
                "old_version": old_version,
                "new_version": new_version,
                "entry": updated_entry
            }

        # Brand new knowledge entry
        new_entry = {
            "knowledge_id": knowledge_id,
            "topic": topic,
            "subject": subject,
            "content": content,
            "version": 1,
            "sources": sources,
            "confidence": confidence,
            "verification_status": "CONTRADICTORY" if conflicting_views else verification_status,
            "first_learned_at": now,
            "last_updated_at": now,
            "last_verified_at": now,
            "provenance": {
                "learned_by_role": learned_by_role,
                "trigger": trigger,
                "session_id": session_id or "default"
            },
            "update_history": [],
            "conflicting_views": conflicting_views or []
        }

        with open(entry_file, "w", encoding="utf-8") as f:
            json.dump(new_entry, f, indent=2)

        self.index[knowledge_id] = {
            "topic": topic,
            "subject": subject,
            "version": 1,
            "last_updated_at": now,
            "source_count": len(sources),
            "status": new_entry["verification_status"]
        }
        self._save_index()

        return {
            "action": "CREATED",
            "knowledge_id": knowledge_id,
            "version": 1,
            "entry": new_entry
        }

    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        if not knowledge_id or ".." in knowledge_id or "/" in knowledge_id or "\\" in knowledge_id:
            return None
        entry_file = os.path.join(self.entries_dir, f"{knowledge_id}.json")
        if os.path.exists(entry_file):
            try:
                with open(entry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def query_knowledge(self, query: str, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        q_lower = query.lower()
        STOP_WORDS = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to", "for", "of", "and", "or", "who", "what", "where", "when", "why", "how", "with"}
        terms = [t for t in re.findall(r'\b\w+\b', q_lower) if len(t) >= 3 and t not in STOP_WORDS]
        if not terms:
            terms = q_lower.split()

        for kid, meta in self.index.items():
            if topic and meta.get("topic", "").lower() != topic.lower():
                continue
            subj = meta.get("subject", "").lower()
            topic_str = meta.get("topic", "").lower()
            if any(term in subj or term in topic_str for term in terms):
                entry = self.get_knowledge(kid)
                if entry:
                    results.append(entry)
        return results

    def store_candidate(
        self,
        topic: str,
        subject: str,
        content: str,
        sources: List[Dict[str, Any]],
        learned_by_role: str = "USER",
        trigger: str = "USER_SEARCH",
        confidence: float = 0.85,
        session_id: Optional[str] = None,
        source_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quarantines an unverified search/research fact as a candidate.
        DOES NOT add candidate to knowledge_index.json or verified entries.
        Candidate status is strictly 'PENDING_APPROVAL'.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        candidate_id = "cand_" + hashlib.sha256(f"{topic}_{subject}_{time.time()}".encode("utf-8")).hexdigest()[:12]
        candidate_file = os.path.join(self.candidates_dir, f"{candidate_id}.json")

        candidate_record = {
            "candidate_id": candidate_id,
            "topic": topic,
            "subject": subject,
            "source_query": source_query or subject,
            "content": content,
            "sources": sources,
            "confidence": confidence,
            "status": "PENDING_APPROVAL",
            "provenance": {
                "learned_by_role": learned_by_role,
                "trigger": trigger,
                "session_id": session_id or "default"
            },
            "created_at": now
        }

        with open(candidate_file, "w", encoding="utf-8") as f:
            json.dump(candidate_record, f, indent=2)

        return candidate_record

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a quarantined candidate by ID."""
        if not candidate_id or ".." in candidate_id or "/" in candidate_id or "\\" in candidate_id:
            return None
        candidate_file = os.path.join(self.candidates_dir, f"{candidate_id}.json")
        if os.path.exists(candidate_file):
            try:
                with open(candidate_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def approve_candidate(
        self,
        candidate_id: str,
        creator_id: str = "ROOT_OPERATOR",
        creator_auth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Approves a quarantined candidate and promotes it into verified knowledge.
        Strictly requires Creator authority (ROOT_OPERATOR).
        Unauthorized callers fail closed with PermissionError.
        """
        CANONICAL_CREATOR_ID = "ROOT_OPERATOR"
        if creator_id != CANONICAL_CREATOR_ID:
            raise PermissionError(f"Access Denied: Only Creator ({CANONICAL_CREATOR_ID}) can approve knowledge candidates.")

        candidate = self.get_candidate(candidate_id)
        if not candidate:
            raise ValueError(f"Candidate '{candidate_id}' not found.")

        if candidate.get("status") == "APPROVED":
            # Already promoted
            return candidate

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Promote to verified knowledge
        verified_entry = self.store_or_update_knowledge(
            topic=candidate["topic"],
            subject=candidate["subject"],
            content=candidate["content"],
            sources=candidate["sources"],
            learned_by_role="ROOT_CREATOR",
            trigger="CREATOR_APPROVAL",
            confidence=candidate.get("confidence", 0.95),
            verification_status="VERIFIED",
            session_id="creator_approval"
        )

        # Mark candidate as approved
        candidate["status"] = "APPROVED"
        candidate["approved_by"] = creator_id
        candidate["approved_at"] = now
        candidate["promoted_knowledge_id"] = verified_entry.get("knowledge_id")

        candidate_file = os.path.join(self.candidates_dir, f"{candidate_id}.json")
        with open(candidate_file, "w", encoding="utf-8") as f:
            json.dump(candidate, f, indent=2)

        return verified_entry

    def save_topic_dossier(
        self,
        topic_slug: str,
        overview: Dict[str, Any],
        sources: List[Dict[str, Any]],
        research_report_md: str
    ) -> str:
        """
        Saves a structured deep topic dossier into TARA/KNOWLEDGE/topics/<topic_slug>/
        Enforces strict path boundary check and slug sanitization against directory traversal.
        """
        if not topic_slug or ".." in topic_slug or "/" in topic_slug or "\\" in topic_slug or ":" in topic_slug:
            raise ValueError(f"Path traversal detected: illegal characters in topic slug '{topic_slug}'")

        clean_slug = re.sub(r'[^a-zA-Z0-9_\-]+', '_', topic_slug.strip()).strip('_')
        if not clean_slug:
            raise ValueError(f"Invalid topic slug: '{topic_slug}'")

        real_topics_dir = os.path.realpath(os.path.abspath(self.topics_dir))
        topic_path = os.path.realpath(os.path.abspath(os.path.join(self.topics_dir, clean_slug)))

        if os.path.commonpath([real_topics_dir, topic_path]) != real_topics_dir or topic_path == real_topics_dir:
            raise ValueError(f"Path traversal detected: destination '{topic_path}' escapes '{real_topics_dir}'")

        os.makedirs(topic_path, exist_ok=True)

        # 1. overview.json
        with open(os.path.join(topic_path, "overview.json"), "w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2)

        # 2. sources.json
        with open(os.path.join(topic_path, "sources.json"), "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2)

        # 3. research_report.md
        with open(os.path.join(topic_path, "research_report.md"), "w", encoding="utf-8") as f:
            f.write(research_report_md)

        # 4. versions.json
        versions_file = os.path.join(topic_path, "versions.json")
        versions = []
        if os.path.exists(versions_file):
            try:
                with open(versions_file, "r", encoding="utf-8") as f:
                    versions = json.load(f)
            except Exception:
                versions = []

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        versions.append({
            "version": len(versions) + 1,
            "timestamp": now,
            "summary": overview.get("summary", ""),
            "subtopics": overview.get("subtopics", [])
        })

        with open(versions_file, "w", encoding="utf-8") as f:
            json.dump(versions, f, indent=2)

        return topic_path

    def get_topic_dossier(self, topic_slug: str) -> Optional[Dict[str, Any]]:
        topic_path = os.path.join(self.topics_dir, topic_slug)
        if not os.path.exists(topic_path):
            return None

        result = {"topic_slug": topic_slug}
        try:
            with open(os.path.join(topic_path, "overview.json"), "r", encoding="utf-8") as f:
                result["overview"] = json.load(f)
            with open(os.path.join(topic_path, "sources.json"), "r", encoding="utf-8") as f:
                result["sources"] = json.load(f)
            with open(os.path.join(topic_path, "research_report.md"), "r", encoding="utf-8") as f:
                result["report_md"] = f.read()
            with open(os.path.join(topic_path, "versions.json"), "r", encoding="utf-8") as f:
                result["versions"] = json.load(f)
            return result
        except Exception:
            return None
