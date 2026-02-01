"""
Unit tests for CogniDocAgent module.
"""

import pytest
from unittest.mock import MagicMock, patch

from cognidoc.agent import (
    AgentState,
    AgentStep,
    AgentContext,
    AgentResult,
    CogniDocAgent,
    create_agent,
)
from cognidoc.agent_tools import ToolName, ToolCall


class TestAgentState:
    """Tests for AgentState enum."""

    def test_all_states_defined(self):
        """All expected states are defined."""
        expected = [
            "thinking",
            "acting",
            "observing",
            "reflecting",
            "finished",
            "needs_clarification",
            "error",
        ]
        actual = [s.value for s in AgentState]
        assert set(expected) == set(actual)


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_step_creation(self):
        """Test creating a step."""
        step = AgentStep(
            step_number=1,
            thought="I need to search for documents",
            observation="Found 5 documents",
        )
        assert step.step_number == 1
        assert "search" in step.thought
        assert step.action is None

    def test_step_with_action(self):
        """Test step with action."""
        action = ToolCall(
            tool=ToolName.RETRIEVE_VECTOR,
            arguments={"query": "test", "top_k": "5"},
        )
        step = AgentStep(
            step_number=2,
            thought="Searching...",
            action=action,
            observation="Results found",
        )
        assert step.action.tool == ToolName.RETRIEVE_VECTOR

    def test_to_text(self):
        """Test text formatting."""
        step = AgentStep(
            step_number=1,
            thought="Analyzing query",
            observation="Found relevant info",
            reflection="Good progress",
        )
        text = step.to_text()
        assert "Step 1:" in text
        assert "Thought:" in text
        assert "Observation:" in text
        assert "Reflection:" in text

    def test_to_text_with_action(self):
        """Test text formatting includes action."""
        action = ToolCall(
            tool=ToolName.LOOKUP_ENTITY,
            arguments={"entity_name": "Gemini"},
        )
        step = AgentStep(
            step_number=1,
            thought="Looking up entity",
            action=action,
        )
        text = step.to_text()
        assert "Action:" in text
        assert "lookup_entity" in text


class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_context_creation(self):
        """Test creating context."""
        context = AgentContext(query="What is X?")
        assert context.query == "What is X?"
        assert len(context.steps) == 0
        assert context.current_state == AgentState.THINKING

    def test_add_step(self):
        """Test adding steps."""
        context = AgentContext(query="test")
        step1 = AgentStep(step_number=1, thought="First")
        step2 = AgentStep(step_number=2, thought="Second")

        context.add_step(step1)
        context.add_step(step2)

        assert len(context.steps) == 2

    def test_add_context(self):
        """Test adding gathered context."""
        context = AgentContext(query="test")
        context.add_context("First piece of info")
        context.add_context("Second piece of info")
        context.add_context("First piece of info")  # Duplicate

        assert len(context.gathered_context) == 2  # No duplicate

    def test_add_context_empty(self):
        """Test empty context not added."""
        context = AgentContext(query="test")
        context.add_context("")
        context.add_context(None)

        assert len(context.gathered_context) == 0

    def test_get_history_text(self):
        """Test history text generation."""
        context = AgentContext(query="test")
        context.add_step(AgentStep(step_number=1, thought="First"))
        context.add_step(AgentStep(step_number=2, thought="Second"))

        history = context.get_history_text()
        assert "Step 1:" in history
        assert "Step 2:" in history

    def test_get_gathered_context(self):
        """Test gathered context retrieval."""
        context = AgentContext(query="test")
        context.add_context("Info A")
        context.add_context("Info B")

        gathered = context.get_gathered_context()
        assert "Info A" in gathered
        assert "Info B" in gathered
        assert "---" in gathered  # Separator


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_successful_result(self):
        """Test successful result."""
        result = AgentResult(
            query="What is X?",
            answer="X is a concept.",
            steps=[],
            success=True,
        )
        assert result.success is True
        assert result.needs_clarification is False
        assert result.error is None

    def test_clarification_result(self):
        """Test result needing clarification."""
        result = AgentResult(
            query="What is X?",
            answer="",
            steps=[],
            success=False,
            needs_clarification=True,
            clarification_question="Which X do you mean?",
        )
        assert result.needs_clarification is True
        assert "Which X" in result.clarification_question

    def test_error_result(self):
        """Test error result."""
        result = AgentResult(
            query="test",
            answer="Error occurred",
            steps=[],
            success=False,
            error="Connection timeout",
        )
        assert result.success is False
        assert "timeout" in result.error

    def test_stream(self):
        """Test streaming answer."""
        result = AgentResult(
            query="test",
            answer="Hello",
            steps=[],
            success=True,
        )
        streamed = list(result.stream())
        assert streamed == ["H", "e", "l", "l", "o"]


class TestCogniDocAgent:
    """Tests for CogniDocAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = MagicMock()

        agent = CogniDocAgent(
            retriever=mock_retriever,
            max_steps=5,
            temperature=0.2,
        )

        assert agent.max_steps == 5
        assert agent.temperature == 0.2
        assert agent.tools is not None

    def test_agent_default_values(self):
        """Test agent default values."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None

        agent = CogniDocAgent(retriever=mock_retriever)

        assert agent.max_steps == 7
        assert agent.temperature == 0.3

    @patch("cognidoc.agent.llm_chat")
    def test_parse_thought_action(self, mock_llm):
        """Test parsing thought and action from LLM response."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        response = """THOUGHT: I need to search for documents about X.
ACTION: retrieve_vector
ARGUMENTS: {"query": "what is X", "top_k": "5"}"""

        thought, action = agent._parse_thought_action(response)

        assert "search for documents" in thought
        assert action is not None
        assert action.tool == ToolName.RETRIEVE_VECTOR
        assert action.arguments["query"] == "what is X"

    @patch("cognidoc.agent.llm_chat")
    def test_parse_final_answer(self, mock_llm):
        """Test parsing final answer action."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        response = """THOUGHT: I have enough information.
ACTION: final_answer
ARGUMENTS: {"answer": "The answer is 42."}"""

        thought, action = agent._parse_thought_action(response)

        assert action.tool == ToolName.FINAL_ANSWER
        assert action.arguments["answer"] == "The answer is 42."

    @patch("cognidoc.agent.llm_chat")
    def test_parse_malformed_response(self, mock_llm):
        """Test parsing malformed response."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        response = "I don't know what to do"

        thought, action = agent._parse_thought_action(response)

        assert thought == ""
        assert action is None

    @patch("cognidoc.agent.llm_chat")
    def test_run_simple_query(self, mock_llm):
        """Test running a simple query."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None
        mock_retriever.is_loaded.return_value = True

        agent = CogniDocAgent(retriever=mock_retriever, max_steps=3)

        # Mock LLM to return final answer immediately
        mock_llm.return_value = """THOUGHT: I can answer this directly.
ACTION: final_answer
ARGUMENTS: {"answer": "The answer is 42."}"""

        result = agent.run("What is the meaning of life?")

        assert result.success is True
        assert "42" in result.answer
        assert len(result.steps) == 1

    @patch("cognidoc.agent.llm_chat")
    def test_run_clarification_needed(self, mock_llm):
        """Test query needing clarification."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        mock_llm.return_value = """THOUGHT: The query is ambiguous.
ACTION: ask_clarification
ARGUMENTS: {"question": "Which topic are you asking about?"}"""

        result = agent.run("Tell me about it")

        assert result.success is False
        assert result.needs_clarification is True
        assert "topic" in result.clarification_question

    @patch("cognidoc.agent.llm_chat")
    def test_run_max_steps_reached(self, mock_llm):
        """Test max steps forces conclusion."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None
        mock_retriever.is_loaded.return_value = True

        # Mock vector search
        mock_node = MagicMock()
        mock_node.text = "Some info"
        mock_node.metadata = {}
        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.9
        mock_retriever._vector_index.search.return_value = [mock_nws]

        agent = CogniDocAgent(retriever=mock_retriever, max_steps=2)

        # Mock LLM to always search (never conclude)
        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 4:  # During thinking/reflecting
                return """THOUGHT: Need more info.
ACTION: retrieve_vector
ARGUMENTS: {"query": "test", "top_k": "3"}"""
            else:  # Force conclusion
                return "Based on the information, the answer is X."

        mock_llm.side_effect = mock_response

        result = agent.run("Complex query")

        assert result.success is True
        assert result.metadata.get("forced_conclusion", False) is True

    @patch("cognidoc.agent.llm_chat")
    def test_run_error_handling(self, mock_llm):
        """Test error handling during run."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        mock_llm.side_effect = Exception("LLM API error")

        result = agent.run("Test query")

        assert result.success is False
        assert result.error is not None
        assert "error" in result.error.lower()


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_create_agent(self):
        """Test factory creates agent."""
        mock_retriever = MagicMock()

        agent = create_agent(
            retriever=mock_retriever,
            max_steps=10,
            temperature=0.5,
        )

        assert isinstance(agent, CogniDocAgent)
        assert agent.max_steps == 10
        assert agent.temperature == 0.5

    def test_create_agent_defaults(self):
        """Test factory with defaults."""
        mock_retriever = MagicMock()

        agent = create_agent(retriever=mock_retriever)

        assert agent.max_steps == 7
        assert agent.temperature == 0.3


class TestContextTrimming:
    """Tests for AgentContext.get_trimmed_history()."""

    def _make_step(
        self,
        num,
        thought="thinking",
        action_tool=ToolName.RETRIEVE_VECTOR,
        action_args=None,
        observation="obs " * 100,
        reflection="refl " * 50,
    ):
        """Helper to create a fully-populated step."""
        action = ToolCall(
            tool=action_tool,
            arguments=action_args or {"query": f"q{num}"},
        )
        return AgentStep(
            step_number=num,
            thought=thought,
            action=action,
            observation=observation,
            reflection=reflection,
        )

    def test_all_steps_kept_when_fewer_than_cutoff(self):
        """When steps <= keep_full, all are returned in full."""
        context = AgentContext(query="test")
        context.add_step(self._make_step(1))
        context.add_step(self._make_step(2))

        result = context.get_trimmed_history(keep_full=3)
        assert "Observation:" in result
        assert "Reflection:" in result

    def test_older_steps_trimmed(self):
        """Steps older than keep_full have observation and reflection dropped."""
        context = AgentContext(query="test")
        for i in range(1, 6):
            context.add_step(self._make_step(i, thought=f"thought_{i}"))

        result = context.get_trimmed_history(keep_full=2)

        # Steps 1-3 should be trimmed (no Observation/Reflection)
        # Steps 4-5 should be full
        lines = result.split("\n")

        # Older steps: thought + action only
        step1_block = result.split("Step 2:")[0]
        assert "thought_1" in step1_block
        assert "Action:" in step1_block
        assert "Observation:" not in step1_block
        assert "Reflection:" not in step1_block

        # Recent steps: full detail
        step5_block = result.split("Step 5:")[1] if "Step 5:" in result else ""
        assert "Observation:" in step5_block
        assert "Reflection:" in step5_block

    def test_empty_steps(self):
        """Empty context returns empty string."""
        context = AgentContext(query="test")
        assert context.get_trimmed_history() == ""

    def test_single_step_kept_full(self):
        """Single step is always kept in full."""
        context = AgentContext(query="test")
        context.add_step(self._make_step(1))

        result = context.get_trimmed_history(keep_full=2)
        assert "Observation:" in result
        assert "Reflection:" in result

    def test_default_keep_full_is_two(self):
        """Default keep_full parameter is 2."""
        context = AgentContext(query="test")
        for i in range(1, 5):
            context.add_step(self._make_step(i, thought=f"thought_{i}"))

        result = context.get_trimmed_history()  # default keep_full=2

        # Steps 1-2 trimmed, steps 3-4 full
        step1_block = result.split("Step 2:")[0]
        assert "Observation:" not in step1_block

        step4_block = result.split("Step 4:")[1] if "Step 4:" in result else ""
        assert "Observation:" in step4_block


class TestBindingReflection:
    """Tests for reflection binding (should_conclude signal)."""

    @patch("cognidoc.agent.llm_chat")
    def test_reflect_returns_tuple(self, mock_llm):
        """_reflect returns (str, bool) tuple."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        mock_llm.return_value = "THOUGHT: We have enough info.\nACTION: final_answer"

        context = AgentContext(query="test query")
        reflection, should_conclude = agent._reflect(context, "some observation")

        assert isinstance(reflection, str)
        assert isinstance(should_conclude, bool)
        assert should_conclude is True

    @patch("cognidoc.agent.llm_chat")
    def test_reflect_continue_signal(self, mock_llm):
        """_reflect returns should_conclude=False when ACTION: continue."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        mock_llm.return_value = "THOUGHT: Need more data.\nACTION: continue"

        context = AgentContext(query="test query")
        reflection, should_conclude = agent._reflect(context, "partial data")

        assert should_conclude is False
        assert "Need more data" in reflection

    @patch("cognidoc.agent.llm_chat")
    def test_reflect_no_action_means_continue(self, mock_llm):
        """_reflect returns should_conclude=False when no ACTION keyword."""
        mock_retriever = MagicMock()
        agent = CogniDocAgent(retriever=mock_retriever)

        mock_llm.return_value = "I think we need more information about the topic."

        context = AgentContext(query="test query")
        reflection, should_conclude = agent._reflect(context, "obs")

        assert should_conclude is False

    @patch("cognidoc.agent.llm_chat")
    def test_binding_reflection_concludes_early(self, mock_llm):
        """Agent concludes early when reflection signals should_conclude."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None
        mock_retriever.is_loaded.return_value = True

        # Mock vector search
        mock_node = MagicMock()
        mock_node.text = "Relevant information about the topic."
        mock_node.metadata = {}
        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.9
        mock_retriever._vector_index.search.return_value = [mock_nws]

        agent = CogniDocAgent(retriever=mock_retriever, max_steps=5)

        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Think step: do a retrieval
                return """THOUGHT: Let me search.
ACTION: retrieve_vector
ARGUMENTS: {"query": "topic", "top_k": "3"}"""
            elif call_count[0] == 2:
                # Reflect step: signal conclusion
                return "THOUGHT: We have all the info needed.\nACTION: final_answer"
            else:
                # Force conclusion LLM call
                return "The topic is about X based on gathered evidence."

        mock_llm.side_effect = mock_response

        result = agent.run("What is the topic?")

        assert result.success is True
        assert result.metadata.get("reflection_concluded") is True
        assert result.metadata["total_steps"] == 1

    @patch("cognidoc.agent.llm_chat")
    def test_binding_reflection_does_not_conclude_on_continue(self, mock_llm):
        """Agent continues when reflection signals ACTION: continue."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None
        mock_retriever.is_loaded.return_value = True

        mock_node = MagicMock()
        mock_node.text = "Some info"
        mock_node.metadata = {}
        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.9
        mock_retriever._vector_index.search.return_value = [mock_nws]

        agent = CogniDocAgent(retriever=mock_retriever, max_steps=2)

        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return """THOUGHT: Let me search.
ACTION: retrieve_vector
ARGUMENTS: {"query": "topic", "top_k": "3"}"""
            elif call_count[0] == 2:
                # Reflect: continue
                return "THOUGHT: Need more info.\nACTION: continue"
            elif call_count[0] == 3:
                # Second think: now conclude
                return """THOUGHT: I have enough now.
ACTION: final_answer
ARGUMENTS: {"answer": "The answer."}"""
            else:
                return "Forced answer."

        mock_llm.side_effect = mock_response

        result = agent.run("What is the topic?")

        assert result.success is True
        assert result.metadata.get("reflection_concluded") is not True


class TestAgentIntegration:
    """Integration-style tests for agent behavior."""

    @patch("cognidoc.agent.llm_chat")
    def test_multi_step_reasoning(self, mock_llm):
        """Test agent performs multi-step reasoning."""
        mock_retriever = MagicMock()
        mock_retriever._graph_retriever = None
        mock_retriever.is_loaded.return_value = True

        # Mock vector search
        mock_node = MagicMock()
        mock_node.text = "Gemini is a multimodal AI model."
        mock_node.metadata = {}
        mock_nws = MagicMock()
        mock_nws.node = mock_node
        mock_nws.score = 0.95
        mock_retriever._vector_index.search.return_value = [mock_nws]

        agent = CogniDocAgent(retriever=mock_retriever, max_steps=5)

        # All LLM calls return final answer to keep test simple
        # (think_and_decide + reflect both call llm_chat)
        mock_llm.return_value = """THOUGHT: I have enough information.
ACTION: final_answer
ARGUMENTS: {"answer": "Gemini is a multimodal AI model developed by Google."}"""

        result = agent.run("What is Gemini?")

        assert result.success is True
        assert "Gemini" in result.answer
        assert len(result.steps) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
