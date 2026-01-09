"""
Agent Tools for CogniDoc Agentic RAG.

Provides a set of tools that the agent can use to interact with
the knowledge base and perform multi-step reasoning.

Tools:
- RETRIEVE_VECTOR: Semantic search in vector index
- RETRIEVE_GRAPH: Knowledge graph traversal
- LOOKUP_ENTITY: Get information about a specific entity
- COMPARE_ENTITIES: Compare two or more entities
- SYNTHESIZE: Combine multiple contexts into a coherent summary
- VERIFY_CLAIM: Fact-check a claim against sources
- ASK_CLARIFICATION: Request clarification from user (special tool)
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .utils.llm_client import llm_chat
from .utils.logger import logger


# =============================================================================
# Tool Cache (persistent SQLite backend)
# =============================================================================

from .utils.tool_cache import ToolCache, get_tool_cache

if TYPE_CHECKING:
    from .hybrid_retriever import HybridRetriever
    from .knowledge_graph import KnowledgeGraph, GraphNode
    from .graph_retrieval import GraphRetriever


class ToolName(str, Enum):
    """Available agent tools."""
    RETRIEVE_VECTOR = "retrieve_vector"
    RETRIEVE_GRAPH = "retrieve_graph"
    LOOKUP_ENTITY = "lookup_entity"
    COMPARE_ENTITIES = "compare_entities"
    SYNTHESIZE = "synthesize"
    VERIFY_CLAIM = "verify_claim"
    ASK_CLARIFICATION = "ask_clarification"
    FINAL_ANSWER = "final_answer"
    DATABASE_STATS = "database_stats"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool: ToolName
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def observation(self) -> str:
        """Format result as observation string for the agent."""
        if not self.success:
            return f"Error: {self.error}"

        if self.tool == ToolName.RETRIEVE_VECTOR:
            docs = self.data or []
            if not docs:
                return "No relevant documents found."
            return f"Found {len(docs)} relevant documents:\n" + "\n".join(
                f"[{i+1}] {doc.get('text', '')[:200]}..."
                for i, doc in enumerate(docs[:5])
            )

        elif self.tool == ToolName.RETRIEVE_GRAPH:
            if not self.data:
                return "No graph results found."
            return self.data.get("context", "Graph context retrieved.")

        elif self.tool == ToolName.LOOKUP_ENTITY:
            entity = self.data
            if not entity:
                return "Entity not found."
            return (
                f"Entity: {entity.get('name', 'Unknown')}\n"
                f"Type: {entity.get('type', 'Unknown')}\n"
                f"Description: {entity.get('description', 'No description')}\n"
                f"Attributes: {json.dumps(entity.get('attributes', {}), indent=2)}"
            )

        elif self.tool == ToolName.COMPARE_ENTITIES:
            return self.data or "Comparison completed."

        elif self.tool == ToolName.SYNTHESIZE:
            return self.data or "Synthesis completed."

        elif self.tool == ToolName.VERIFY_CLAIM:
            result = self.data or {}
            verified = result.get("verified", False)
            evidence = result.get("evidence", "No evidence")
            return f"Claim {'VERIFIED' if verified else 'NOT VERIFIED'}: {evidence}"

        elif self.tool == ToolName.ASK_CLARIFICATION:
            return f"CLARIFICATION_NEEDED: {self.data}"

        elif self.tool == ToolName.FINAL_ANSWER:
            return self.data or ""

        elif self.tool == ToolName.DATABASE_STATS:
            stats = self.data or {}
            parts = []
            if "total_documents" in stats:
                parts.append(f"Total documents: {stats['total_documents']}")
            if "total_chunks" in stats:
                parts.append(f"Total chunks: {stats['total_chunks']}")
            if "graph_nodes" in stats:
                parts.append(f"Graph nodes: {stats['graph_nodes']}")
            if "graph_edges" in stats:
                parts.append(f"Graph edges: {stats['graph_edges']}")
            if "document_names" in stats and stats["document_names"]:
                names = stats["document_names"]
                parts.append(f"Document names: {', '.join(names)}")
            return "DATABASE STATS: " + ", ".join(parts) if parts else "No stats available"

        return str(self.data)


@dataclass
class ToolCall:
    """Represents a tool call from the agent."""
    tool: ToolName
    arguments: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


# =============================================================================
# Base Tool Class
# =============================================================================

class BaseTool(ABC):
    """Base class for agent tools."""

    name: ToolName
    description: str
    parameters: Dict[str, str]  # param_name -> description

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": self.name.value,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": "string", "description": desc}
                    for name, desc in self.parameters.items()
                },
                "required": list(self.parameters.keys()),
            },
        }


# =============================================================================
# Tool Implementations
# =============================================================================

class RetrieveVectorTool(BaseTool):
    """Semantic search in vector index."""

    name = ToolName.RETRIEVE_VECTOR
    description = "Search for relevant documents using semantic similarity. Use when you need factual information from documents."
    parameters = {
        "query": "The search query to find relevant documents",
        "top_k": "Number of results to return (default: 5)",
    }

    def __init__(self, retriever: "HybridRetriever"):
        self.retriever = retriever

    def execute(self, query: str = "", top_k: str = "5") -> ToolResult:
        try:
            if not query or not query.strip():
                return ToolResult(
                    tool=self.name,
                    success=False,
                    error="Query parameter is required for vector search",
                )

            k = int(top_k)

            # Check cache first
            cached = ToolCache.get("retrieve_vector", query=query, top_k=k)
            if cached is not None:
                return ToolResult(
                    tool=self.name,
                    success=True,
                    data=cached,
                    metadata={"query": query, "count": len(cached), "cached": True},
                )

            if not self.retriever.is_loaded():
                self.retriever.load()

            # Use vector-only retrieval
            results = self.retriever._vector_index.search(query, top_k=k)

            docs = []
            for nws in results:
                docs.append({
                    "text": nws.node.text,
                    "score": nws.score,
                    "metadata": nws.node.metadata,
                })

            # Store in cache
            ToolCache.set("retrieve_vector", docs, query=query, top_k=k)

            return ToolResult(
                tool=self.name,
                success=True,
                data=docs,
                metadata={"query": query, "count": len(docs)},
            )
        except Exception as e:
            logger.error(f"RetrieveVectorTool error: {e}")
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


class RetrieveGraphTool(BaseTool):
    """Knowledge graph traversal."""

    name = ToolName.RETRIEVE_GRAPH
    description = "Search the knowledge graph for entities and relationships. Use for questions about relationships, connections, or entity properties."
    parameters = {
        "query": "The query to search in the knowledge graph",
    }

    def __init__(self, graph_retriever: "GraphRetriever"):
        self.graph_retriever = graph_retriever

    def execute(self, query: str) -> ToolResult:
        try:
            if not self.graph_retriever.is_loaded():
                self.graph_retriever.load()

            result = self.graph_retriever.retrieve(query)

            return ToolResult(
                tool=self.name,
                success=True,
                data={
                    "context": result.context or result.get_context_text(),
                    "entities": [
                        {"name": e.name, "type": e.type, "description": e.description}
                        for e in result.entities
                    ],
                    "relationships": result.relationships,
                    "confidence": result.confidence,
                },
                metadata={"query": query, "retrieval_type": result.retrieval_type},
            )
        except Exception as e:
            logger.error(f"RetrieveGraphTool error: {e}")
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


class LookupEntityTool(BaseTool):
    """Get detailed information about a specific entity."""

    name = ToolName.LOOKUP_ENTITY
    description = "Look up detailed information about a specific named entity. Use when you need attributes, description, or relationships of a specific entity."
    parameters = {
        "entity_name": "The name of the entity to look up",
    }

    def __init__(self, graph_retriever: "GraphRetriever"):
        self.graph_retriever = graph_retriever

    def execute(self, entity_name: str) -> ToolResult:
        try:
            if not self.graph_retriever.is_loaded():
                self.graph_retriever.load()

            kg = self.graph_retriever.kg
            if not kg:
                return ToolResult(
                    tool=self.name,
                    success=False,
                    error="Knowledge graph not available",
                )

            # Search for entity by name (case-insensitive)
            entity_name_lower = entity_name.lower()
            found_node = None

            for node_id, node in kg.nodes.items():
                if node.name.lower() == entity_name_lower:
                    found_node = node
                    break
                # Also check if entity_name is a substring
                if entity_name_lower in node.name.lower():
                    found_node = node

            if not found_node:
                return ToolResult(
                    tool=self.name,
                    success=True,
                    data=None,
                    metadata={"searched_for": entity_name},
                )

            # Get relationships for this entity
            relationships = []
            for edge in kg.edges:
                if edge.source_id == found_node.id:
                    target = kg.nodes.get(edge.target_id)
                    if target:
                        relationships.append({
                            "relation": edge.relationship_type,
                            "target": target.name,
                            "direction": "outgoing",
                        })
                elif edge.target_id == found_node.id:
                    source = kg.nodes.get(edge.source_id)
                    if source:
                        relationships.append({
                            "relation": edge.relationship_type,
                            "source": source.name,
                            "direction": "incoming",
                        })

            return ToolResult(
                tool=self.name,
                success=True,
                data={
                    "name": found_node.name,
                    "type": found_node.type,
                    "description": found_node.description,
                    "attributes": found_node.attributes,
                    "relationships": relationships[:10],
                    "source_chunks": found_node.source_chunks[:5],
                },
                metadata={"entity_id": found_node.id},
            )
        except Exception as e:
            logger.error(f"LookupEntityTool error: {e}")
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


class CompareEntitiesTool(BaseTool):
    """Compare two or more entities."""

    name = ToolName.COMPARE_ENTITIES
    description = "Compare two or more entities to identify similarities and differences. Use for comparative questions."
    parameters = {
        "entities": "Comma-separated list of entity names to compare",
        "aspects": "Optional: specific aspects to compare (e.g., 'performance,cost')",
    }

    def __init__(self, graph_retriever: "GraphRetriever"):
        self.graph_retriever = graph_retriever

    def execute(self, entities: str, aspects: str = "") -> ToolResult:
        try:
            entity_names = [e.strip() for e in entities.split(",")]
            if len(entity_names) < 2:
                return ToolResult(
                    tool=self.name,
                    success=False,
                    error="Need at least 2 entities to compare",
                )

            if not self.graph_retriever.is_loaded():
                self.graph_retriever.load()

            kg = self.graph_retriever.kg
            if not kg:
                return ToolResult(
                    tool=self.name,
                    success=False,
                    error="Knowledge graph not available",
                )

            # Find entities
            found_entities = []
            for name in entity_names:
                name_lower = name.lower()
                for node_id, node in kg.nodes.items():
                    if node.name.lower() == name_lower or name_lower in node.name.lower():
                        found_entities.append({
                            "name": node.name,
                            "type": node.type,
                            "description": node.description,
                            "attributes": node.attributes,
                        })
                        break

            if len(found_entities) < 2:
                return ToolResult(
                    tool=self.name,
                    success=False,
                    error=f"Could not find enough entities. Found: {[e['name'] for e in found_entities]}",
                )

            # Generate comparison using LLM
            comparison_prompt = f"""Compare the following entities:

{json.dumps(found_entities, indent=2)}

{"Focus on these aspects: " + aspects if aspects else "Compare all relevant aspects."}

Provide a structured comparison highlighting:
1. Similarities
2. Differences
3. Key distinguishing features"""

            comparison = llm_chat([
                {"role": "user", "content": comparison_prompt}
            ])

            return ToolResult(
                tool=self.name,
                success=True,
                data=comparison,
                metadata={"entities_compared": [e["name"] for e in found_entities]},
            )
        except Exception as e:
            logger.error(f"CompareEntitiesTool error: {e}")
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


class SynthesizeTool(BaseTool):
    """Synthesize multiple pieces of information."""

    name = ToolName.SYNTHESIZE
    description = "Combine multiple pieces of information into a coherent synthesis. Use when you have gathered enough context and need to formulate a comprehensive answer."
    parameters = {
        "contexts": "The information to synthesize (can be multiple paragraphs)",
        "focus": "The main question or focus for the synthesis",
    }

    def execute(self, contexts: str, focus: str) -> ToolResult:
        try:
            synthesis_prompt = f"""Synthesize the following information to answer the question.

QUESTION/FOCUS: {focus}

INFORMATION:
{contexts}

Provide a clear, comprehensive synthesis that:
1. Addresses the main question directly
2. Integrates all relevant information
3. Notes any gaps or uncertainties
4. Is well-structured and easy to understand"""

            synthesis = llm_chat([
                {"role": "user", "content": synthesis_prompt}
            ])

            return ToolResult(
                tool=self.name,
                success=True,
                data=synthesis,
                metadata={"focus": focus},
            )
        except Exception as e:
            logger.error(f"SynthesizeTool error: {e}")
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


class VerifyClaimTool(BaseTool):
    """Verify a claim against sources."""

    name = ToolName.VERIFY_CLAIM
    description = "Fact-check a specific claim against the available sources. Use to verify accuracy of generated statements."
    parameters = {
        "claim": "The claim to verify",
        "sources": "The source text to check against",
    }

    def execute(self, claim: str, sources: str) -> ToolResult:
        try:
            verify_prompt = f"""Verify whether the following claim is supported by the sources.

CLAIM: {claim}

SOURCES:
{sources}

Respond in JSON format:
{{
    "verified": true/false,
    "confidence": 0.0-1.0,
    "evidence": "Quote or description of supporting/contradicting evidence",
    "reasoning": "Brief explanation of verification logic"
}}"""

            response = llm_chat([
                {"role": "user", "content": verify_prompt}
            ])

            # Parse JSON response
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {"verified": False, "evidence": response}
            except json.JSONDecodeError:
                result = {"verified": False, "evidence": response}

            return ToolResult(
                tool=self.name,
                success=True,
                data=result,
                metadata={"claim": claim},
            )
        except Exception as e:
            logger.error(f"VerifyClaimTool error: {e}")
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


class AskClarificationTool(BaseTool):
    """Request clarification from the user."""

    name = ToolName.ASK_CLARIFICATION
    description = "Ask the user for clarification when the query is ambiguous or more information is needed. IMPORTANT: The question MUST be in the SAME LANGUAGE as the user's original query. Use sparingly."
    parameters = {
        "question": "The clarification question to ask the user (MUST be in the same language as the user's query)",
    }

    def execute(self, question: str) -> ToolResult:
        """
        This tool is special - it signals that the agent needs user input.
        The actual clarification flow is handled by the agent orchestrator.
        """
        return ToolResult(
            tool=self.name,
            success=True,
            data=question,
            metadata={"requires_user_input": True},
        )


class FinalAnswerTool(BaseTool):
    """Provide the final answer to the user."""

    name = ToolName.FINAL_ANSWER
    description = "Provide the final, complete answer to the user's question. IMPORTANT: The answer MUST be in the SAME LANGUAGE as the user's original query. Use when you have gathered enough information and are ready to respond."
    parameters = {
        "answer": "The complete answer to provide to the user (MUST be in the same language as the user's query)",
    }

    def execute(self, answer: str) -> ToolResult:
        return ToolResult(
            tool=self.name,
            success=True,
            data=answer,
            metadata={"is_final": True},
        )


class DatabaseStatsTool(BaseTool):
    """Get statistics about the document database."""

    name = ToolName.DATABASE_STATS
    description = "Get statistics about the document database: total number of documents, chunks, graph nodes, document names/titles. Use this when the user asks about the database size, document count, to list documents, or similar meta-questions about the knowledge base."
    parameters = {
        "list_documents": "Set to true to get the list of document names/titles (default: false)",
    }

    def __init__(self, retriever=None):
        self.retriever = retriever

    def execute(self, list_documents: bool = False) -> ToolResult:
        """Get database statistics.

        Returns unique source document count (not chunk count).
        - total_documents: Number of unique source files (PDFs)
        - total_chunks: Number of chunks in the index
        - document_names: List of unique source document names (if list_documents=True)
        """
        try:
            # Normalize list_documents to bool
            if isinstance(list_documents, str):
                list_documents = list_documents.lower() in ('true', '1', 'yes')

            # Check cache first
            cached = ToolCache.get("database_stats", list_documents=list_documents)
            if cached is not None:
                return ToolResult(
                    tool=self.name,
                    success=True,
                    data=cached,
                    metadata={"cached": True},
                )

            stats = {}

            if self.retriever:
                # Get keyword index stats (parent documents) using get_all_documents()
                if hasattr(self.retriever, '_keyword_index') and self.retriever._keyword_index:
                    ki = self.retriever._keyword_index
                    if hasattr(ki, 'get_all_documents'):
                        docs = ki.get_all_documents()
                        stats["total_chunks"] = len(docs)  # This is chunk count

                        # Extract unique source document names from ALL chunks
                        unique_sources = set()
                        for doc in docs:
                            if hasattr(doc, 'metadata') and doc.metadata:
                                # Try to get source document name from metadata
                                source = doc.metadata.get('source', {})
                                if isinstance(source, dict):
                                    name = source.get('document')
                                else:
                                    name = str(source) if source else None

                                if not name:
                                    # Fallback to name or title
                                    name = doc.metadata.get('name') or doc.metadata.get('title')

                                if name:
                                    unique_sources.add(name)

                        # Total documents = unique source files
                        stats["total_documents"] = len(unique_sources)

                        # List document names if requested
                        if list_documents:
                            stats["document_names"] = sorted(list(unique_sources))

                # Get graph stats
                if hasattr(self.retriever, '_graph_retriever') and self.retriever._graph_retriever:
                    gr = self.retriever._graph_retriever
                    if hasattr(gr, 'kg') and gr.kg:
                        stats["graph_nodes"] = len(gr.kg.nodes) if hasattr(gr.kg, 'nodes') else 0
                        stats["graph_edges"] = len(gr.kg.edges) if hasattr(gr.kg, 'edges') else 0

            # Store in cache
            ToolCache.set("database_stats", stats, list_documents=list_documents)

            return ToolResult(
                tool=self.name,
                success=True,
                data=stats,
            )

        except Exception as e:
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(e),
            )


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self):
        self.tools: Dict[ToolName, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name.value}")

    def get(self, name: ToolName) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        tool = self.tools.get(tool_call.tool)
        if not tool:
            return ToolResult(
                tool=tool_call.tool,
                success=False,
                error=f"Unknown tool: {tool_call.tool}",
            )

        logger.info(f"Executing tool: {tool_call.tool.value} with args: {tool_call.arguments}")
        return tool.execute(**tool_call.arguments)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools."""
        lines = ["Available tools:"]
        for tool in self.tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            lines.append(f"- {tool.name.value}: {tool.description}")
            lines.append(f"  Parameters: {params}")
        return "\n".join(lines)


def create_tool_registry(
    retriever: "HybridRetriever",
    graph_retriever: Optional["GraphRetriever"] = None,
) -> ToolRegistry:
    """
    Create a tool registry with all available tools.

    Args:
        retriever: HybridRetriever instance for vector search
        graph_retriever: GraphRetriever instance for graph operations

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    # Vector search tool
    registry.register(RetrieveVectorTool(retriever))

    # Graph tools (if graph retriever available)
    if graph_retriever is None:
        graph_retriever = retriever._graph_retriever

    if graph_retriever:
        registry.register(RetrieveGraphTool(graph_retriever))
        registry.register(LookupEntityTool(graph_retriever))
        registry.register(CompareEntitiesTool(graph_retriever))

    # LLM-based tools (always available)
    registry.register(SynthesizeTool())
    registry.register(VerifyClaimTool())

    # Database stats tool (for meta-questions)
    registry.register(DatabaseStatsTool(retriever))

    # Special tools
    registry.register(AskClarificationTool())
    registry.register(FinalAnswerTool())

    logger.info(f"Created tool registry with {len(registry.tools)} tools")
    return registry


__all__ = [
    "ToolName",
    "ToolResult",
    "ToolCall",
    "BaseTool",
    "RetrieveVectorTool",
    "RetrieveGraphTool",
    "LookupEntityTool",
    "CompareEntitiesTool",
    "SynthesizeTool",
    "VerifyClaimTool",
    "AskClarificationTool",
    "FinalAnswerTool",
    "DatabaseStatsTool",
    "ToolRegistry",
    "create_tool_registry",
]
