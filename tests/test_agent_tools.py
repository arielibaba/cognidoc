"""
Unit tests for agent tools module.
"""

import pytest
from unittest.mock import MagicMock, patch

from cognidoc.agent_tools import (
    ToolName,
    ToolResult,
    ToolCall,
    BaseTool,
    ToolRegistry,
    SynthesizeTool,
    VerifyClaimTool,
    AskClarificationTool,
    FinalAnswerTool,
    RetrieveVectorTool,
    RetrieveGraphTool,
    LookupEntityTool,
    CompareEntitiesTool,
    create_tool_registry,
)


class TestToolName:
    """Tests for ToolName enum."""

    def test_all_tools_defined(self):
        """All expected tools are defined."""
        expected = [
            "retrieve_vector",
            "retrieve_graph",
            "lookup_entity",
            "compare_entities",
            "synthesize",
            "verify_claim",
            "ask_clarification",
            "final_answer",
            "database_stats",
        ]
        actual = [t.value for t in ToolName]
        assert set(expected) == set(actual)

    def test_tool_name_is_string_enum(self):
        """ToolName values are strings."""
        assert ToolName.RETRIEVE_VECTOR.value == "retrieve_vector"
        assert isinstance(ToolName.SYNTHESIZE.value, str)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_successful_result(self):
        """Test successful tool result."""
        result = ToolResult(
            tool=ToolName.SYNTHESIZE,
            success=True,
            data="Test data",
        )
        assert result.success is True
        assert result.error is None
        assert result.data == "Test data"

    def test_failed_result(self):
        """Test failed tool result."""
        result = ToolResult(
            tool=ToolName.RETRIEVE_VECTOR,
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert "Connection failed" in result.observation

    def test_observation_vector_results(self):
        """Test observation formatting for vector results."""
        result = ToolResult(
            tool=ToolName.RETRIEVE_VECTOR,
            success=True,
            data=[
                {"text": "First document content", "score": 0.9},
                {"text": "Second document content", "score": 0.8},
            ],
        )
        obs = result.observation
        assert "Found 2 relevant documents" in obs
        assert "First document" in obs

    def test_observation_empty_vector(self):
        """Test observation for empty vector results."""
        result = ToolResult(
            tool=ToolName.RETRIEVE_VECTOR,
            success=True,
            data=[],
        )
        assert "No relevant documents found" in result.observation

    def test_observation_entity_lookup(self):
        """Test observation for entity lookup."""
        result = ToolResult(
            tool=ToolName.LOOKUP_ENTITY,
            success=True,
            data={
                "name": "Gemini",
                "type": "MODEL",
                "description": "Google's LLM",
                "attributes": {"version": "2.0"},
            },
        )
        obs = result.observation
        assert "Gemini" in obs
        assert "MODEL" in obs
        assert "Google's LLM" in obs

    def test_observation_entity_not_found(self):
        """Test observation for entity not found."""
        result = ToolResult(
            tool=ToolName.LOOKUP_ENTITY,
            success=True,
            data=None,
        )
        assert "Entity not found" in result.observation

    def test_observation_verify_claim(self):
        """Test observation for claim verification."""
        result = ToolResult(
            tool=ToolName.VERIFY_CLAIM,
            success=True,
            data={"verified": True, "evidence": "Source confirms this."},
        )
        assert "VERIFIED" in result.observation

        result_false = ToolResult(
            tool=ToolName.VERIFY_CLAIM,
            success=True,
            data={"verified": False, "evidence": "No evidence found."},
        )
        assert "NOT VERIFIED" in result_false.observation

    def test_observation_clarification(self):
        """Test observation for clarification request."""
        result = ToolResult(
            tool=ToolName.ASK_CLARIFICATION,
            success=True,
            data="What time period are you interested in?",
        )
        assert "CLARIFICATION_NEEDED" in result.observation
        assert "time period" in result.observation


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            tool=ToolName.RETRIEVE_VECTOR,
            arguments={"query": "test query", "top_k": "5"},
            reasoning="Need to find relevant documents",
        )
        assert call.tool == ToolName.RETRIEVE_VECTOR
        assert call.arguments["query"] == "test query"
        assert call.reasoning == "Need to find relevant documents"

    def test_tool_call_default_arguments(self):
        """Test tool call with default arguments."""
        call = ToolCall(tool=ToolName.SYNTHESIZE)
        assert call.arguments == {}
        assert call.reasoning == ""


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = SynthesizeTool()
        registry.register(tool)
        assert ToolName.SYNTHESIZE in registry.tools
        assert registry.get(ToolName.SYNTHESIZE) == tool

    def test_get_unknown_tool(self):
        """Test getting unknown tool returns None."""
        registry = ToolRegistry()
        assert registry.get(ToolName.RETRIEVE_VECTOR) is None

    def test_execute_registered_tool(self):
        """Test executing a registered tool."""
        registry = ToolRegistry()
        registry.register(FinalAnswerTool())

        call = ToolCall(
            tool=ToolName.FINAL_ANSWER,
            arguments={"answer": "The answer is 42."},
        )
        result = registry.execute(call)
        assert result.success is True
        assert result.data == "The answer is 42."

    def test_execute_unknown_tool(self):
        """Test executing unknown tool fails."""
        registry = ToolRegistry()
        call = ToolCall(
            tool=ToolName.RETRIEVE_VECTOR,
            arguments={"query": "test"},
        )
        result = registry.execute(call)
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_get_all_schemas(self):
        """Test getting all tool schemas."""
        registry = ToolRegistry()
        registry.register(SynthesizeTool())
        registry.register(FinalAnswerTool())

        schemas = registry.get_all_schemas()
        assert len(schemas) == 2
        assert all("name" in s for s in schemas)
        assert all("description" in s for s in schemas)
        assert all("parameters" in s for s in schemas)

    def test_get_tool_descriptions(self):
        """Test getting formatted tool descriptions."""
        registry = ToolRegistry()
        registry.register(SynthesizeTool())

        desc = registry.get_tool_descriptions()
        assert "synthesize" in desc
        assert "Parameters:" in desc


class TestSynthesizeTool:
    """Tests for SynthesizeTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = SynthesizeTool()
        assert tool.name == ToolName.SYNTHESIZE
        assert "contexts" in tool.parameters
        assert "focus" in tool.parameters

    @patch("cognidoc.agent_tools.llm_chat")
    def test_execute_success(self, mock_llm):
        """Test successful synthesis."""
        mock_llm.return_value = "Synthesized answer"

        tool = SynthesizeTool()
        result = tool.execute(
            contexts="Context 1. Context 2.",
            focus="What is the main point?",
        )

        assert result.success is True
        assert result.data == "Synthesized answer"
        mock_llm.assert_called_once()

    @patch("cognidoc.agent_tools.llm_chat")
    def test_execute_error(self, mock_llm):
        """Test synthesis error handling."""
        mock_llm.side_effect = Exception("LLM error")

        tool = SynthesizeTool()
        result = tool.execute(contexts="test", focus="test")

        assert result.success is False
        assert "LLM error" in result.error


class TestVerifyClaimTool:
    """Tests for VerifyClaimTool."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = VerifyClaimTool()
        assert tool.name == ToolName.VERIFY_CLAIM
        assert "claim" in tool.parameters
        assert "sources" in tool.parameters

    @patch("cognidoc.agent_tools.llm_chat")
    def test_execute_verified(self, mock_llm):
        """Test claim verification - verified."""
        mock_llm.return_value = '{"verified": true, "confidence": 0.9, "evidence": "Source confirms"}'

        tool = VerifyClaimTool()
        result = tool.execute(
            claim="The sky is blue",
            sources="Scientific text about sky color...",
        )

        assert result.success is True
        assert result.data["verified"] is True

    @patch("cognidoc.agent_tools.llm_chat")
    def test_execute_not_verified(self, mock_llm):
        """Test claim verification - not verified."""
        mock_llm.return_value = '{"verified": false, "evidence": "No supporting evidence"}'

        tool = VerifyClaimTool()
        result = tool.execute(claim="claim", sources="sources")

        assert result.success is True
        assert result.data["verified"] is False


class TestAskClarificationTool:
    """Tests for AskClarificationTool."""

    def test_execute_returns_question(self):
        """Test clarification tool returns the question."""
        tool = AskClarificationTool()
        result = tool.execute(question="Could you specify the date range?")

        assert result.success is True
        assert result.data == "Could you specify the date range?"
        assert result.metadata["requires_user_input"] is True


class TestFinalAnswerTool:
    """Tests for FinalAnswerTool."""

    def test_execute_returns_answer(self):
        """Test final answer tool returns the answer."""
        tool = FinalAnswerTool()
        result = tool.execute(answer="The final answer is 42.")

        assert result.success is True
        assert result.data == "The final answer is 42."
        assert result.metadata["is_final"] is True


class TestRetrieveVectorTool:
    """Tests for RetrieveVectorTool with mocked retriever."""

    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        mock_retriever = MagicMock()
        tool = RetrieveVectorTool(mock_retriever)
        assert tool.name == ToolName.RETRIEVE_VECTOR
        assert "query" in tool.parameters

    def test_execute_success(self):
        """Test successful vector retrieval."""
        mock_retriever = MagicMock()
        mock_retriever.is_loaded.return_value = True

        # Mock search results
        mock_node = MagicMock()
        mock_node.text = "Document content"
        mock_node.metadata = {"source": "test.pdf"}
        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.9
        mock_retriever._vector_index.search.return_value = [mock_nws]

        tool = RetrieveVectorTool(mock_retriever)
        result = tool.execute(query="test query", top_k="3")

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["text"] == "Document content"


class TestLookupEntityTool:
    """Tests for LookupEntityTool with mocked graph."""

    def test_execute_entity_found(self):
        """Test entity lookup when found."""
        mock_graph_retriever = MagicMock()
        mock_graph_retriever.is_loaded.return_value = True

        # Mock knowledge graph
        mock_node = MagicMock()
        mock_node.id = "gemini-1"
        mock_node.name = "Gemini"
        mock_node.type = "MODEL"
        mock_node.description = "Google's LLM"
        mock_node.attributes = {"version": "2.0"}
        mock_node.source_chunks = ["chunk1"]

        mock_kg = MagicMock()
        mock_kg.nodes = {"gemini-1": mock_node}
        mock_kg.edges = []
        mock_graph_retriever.kg = mock_kg

        tool = LookupEntityTool(mock_graph_retriever)
        result = tool.execute(entity_name="Gemini")

        assert result.success is True
        assert result.data["name"] == "Gemini"
        assert result.data["type"] == "MODEL"

    def test_execute_entity_not_found(self):
        """Test entity lookup when not found."""
        mock_graph_retriever = MagicMock()
        mock_graph_retriever.is_loaded.return_value = True

        mock_kg = MagicMock()
        mock_kg.nodes = {}
        mock_graph_retriever.kg = mock_kg

        tool = LookupEntityTool(mock_graph_retriever)
        result = tool.execute(entity_name="NonExistent")

        assert result.success is True
        assert result.data is None


class TestCreateToolRegistry:
    """Tests for create_tool_registry factory function."""

    def test_creates_registry_with_tools(self):
        """Test factory creates registry with expected tools."""
        mock_retriever = MagicMock()
        mock_graph_retriever = MagicMock()

        registry = create_tool_registry(mock_retriever, mock_graph_retriever)

        # Should have all tools registered
        assert ToolName.RETRIEVE_VECTOR in registry.tools
        assert ToolName.RETRIEVE_GRAPH in registry.tools
        assert ToolName.LOOKUP_ENTITY in registry.tools
        assert ToolName.COMPARE_ENTITIES in registry.tools
        assert ToolName.SYNTHESIZE in registry.tools
        assert ToolName.VERIFY_CLAIM in registry.tools
        assert ToolName.ASK_CLARIFICATION in registry.tools
        assert ToolName.FINAL_ANSWER in registry.tools

    def test_creates_registry_without_graph(self):
        """Test factory works without graph retriever."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None

        registry = create_tool_registry(mock_retriever, graph_retriever=None)

        # Should have vector and LLM tools
        assert ToolName.RETRIEVE_VECTOR in registry.tools
        assert ToolName.SYNTHESIZE in registry.tools
        # Graph tools should not be present
        assert ToolName.RETRIEVE_GRAPH not in registry.tools


class TestBaseTool:
    """Tests for BaseTool schema generation."""

    def test_get_schema(self):
        """Test schema generation."""
        tool = SynthesizeTool()
        schema = tool.get_schema()

        assert schema["name"] == "synthesize"
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "contexts" in schema["parameters"]["properties"]
        assert "focus" in schema["parameters"]["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
