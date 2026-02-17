import chromadb
import uuid
import json
from typing import List, Dict, Any
from datetime import datetime

class MemorySystem:
    """
    Manages both short-term (RAM) and long-term (Vector DB) memory.
    """
    def __init__(self, db_path="./data/memory_db"):
        self.short_term_memory: List[Dict[str, str]] = []
        self.max_short_term = 10
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="agent_memory")
        self.rel_collection = self.chroma_client.get_or_create_collection(name="agent_relationships")

    def add_short_term(self, role: str, content: str, source: str = "unknown"):
        """Adds a message to the short-term conversation history."""
        self.short_term_memory.append({"role": role, "content": content, "source": source})
        # Pruning handled explicitly by the Agent loop via prune_short_term()

    def prune_short_term(self) -> str:
        """
        If memory exceeds limit (15), pops the oldest chunk (5) and returns it as a string.
        Returns None if no pruning needed.
        """
        limit = 15
        prune_size = 5
        
        if len(self.short_term_memory) > limit:
            pruned = self.short_term_memory[:prune_size]
            self.short_term_memory = self.short_term_memory[prune_size:]
            
            # Format with clear role labels for better LLM summarization
            text = ""
            for msg in pruned:
                role = msg.get('role', 'unknown')
                label = '[User]' if role == 'user' else '[Assistant]'
                source = msg.get('source', '')
                source_tag = f' ({source})' if source and source != 'unknown' else ''
                text += f"{label}{source_tag} {msg.get('content', '')}\n"
            return text
        return None

    def get_short_term(self) -> List[Dict[str, str]]:
        return self.short_term_memory

    def save_long_term(self, content: str, metadata: Dict[str, Any] = None, deduplicate: bool = True) -> str:
        """Saves a piece of information to the vector database.
        Returns the ID of the saved memory."""
        if metadata is None:
            metadata = {}
        
        # Dedup check: skip if near-duplicate exists
        if deduplicate and content.strip():
            try:
                existing = self.collection.query(
                    query_texts=[content],
                    n_results=1
                )
                if (existing and existing['distances'] and existing['distances'][0]
                        and existing['distances'][0][0] < 0.15):
                    return existing['ids'][0][0] # Return existing ID
            except Exception:
                pass  # On error, save anyway
        
        metadata["timestamp"] = datetime.now().isoformat()
        mem_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[mem_id]
        )
        return mem_id

    def save_relationship(self, source_id: str, target_id: str, rel_type: str):
        """Saves a relationship between two memories."""
        try:
            self.rel_collection.add(
                documents=[f"{source_id} {rel_type} {target_id}"],
                metadatas=[{"source": source_id, "target": target_id, "type": rel_type}],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"Error saving relationship: {e}")

    def retrieve_relevant(self, query: str, n_results: int = 3) -> str:
        """Retrieves the most relevant memories from the vector DB for a given query."""
        if not query or not query.strip():
            return ""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if not results or not results.get('documents') or not results['documents'][0]:
                return ""

            docs = results['documents'][0]
            metas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            parts = []
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = distances[i] if i < len(distances) else 1.0
                # Skip very low-relevance results (distance >= 1.0 is essentially unrelated)
                if dist >= 1.0:
                    continue
                mem_type = meta.get("type", "memory")
                timestamp = meta.get("timestamp", "")
                ts_str = f" [{timestamp[:10]}]" if timestamp else ""
                parts.append(f"[{mem_type}{ts_str}] {doc}")

            return "\n".join(parts) if parts else ""
        except Exception as e:
            print(f"[Memory] retrieve_relevant failed: {e}")
            return ""

    def clear_short_term(self):
        """Clears the short-term (in-memory) conversation history."""
        self.short_term_memory = []

    def get_all_memories(self) -> Dict[str, Any]:
        """Retrieves all memories and relationships for visualization."""
        try:
            results = self.collection.get()
            nodes = []
            if results and results['ids']:
                for i, id in enumerate(results['ids']):
                    nodes.append({
                        "id": id,
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i] if results['metadatas'] else {}
                    })
            
            rel_results = self.rel_collection.get()
            edges = []
            if rel_results and rel_results['ids']:
                for i in range(len(rel_results['ids'])):
                    meta = rel_results['metadatas'][i]
                    edges.append({
                        "from": meta["source"],
                        "to": meta["target"],
                        "label": meta["type"]
                    })
            
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            print(f"Error getting memories: {e}")
            return {"nodes": [], "edges": []}

    def wipe_memory(self):
        """Clears ALL memory (Dangerous)."""
        try:
            self.chroma_client.delete_collection("agent_memory")
            self.collection = self.chroma_client.get_or_create_collection(name="agent_memory")
            self.short_term_memory = []
            return "Memory wiped successfully."
        except Exception as e:
            return f"Error wiping memory: {e}"

    def delete_memories_containing(self, substring: str) -> int:
        """Deletes memories containing the given substring. Returns count of deleted items."""
        try:
            # 1. Find IDs
            results = self.collection.get(where_document={"$contains": substring})
            if not results or not results['ids']:
                return 0
            
            ids_to_delete = results['ids']
            count = len(ids_to_delete)
            
            # 2. Delete
            self.collection.delete(ids=ids_to_delete)
            return count
        except Exception as e:
            print(f"Error deleting memories: {e}")
            return 0
