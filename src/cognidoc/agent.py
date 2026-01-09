"""
CogniDoc Agent - ReAct-style reasoning agent for complex queries.

Implements a Reasoning + Acting loop that:
1. THINK: Analyze the current situation
2. DECIDE: Choose the next action (tool to use)
3. ACT: Execute the chosen tool
4. OBSERVE: Process the result
5. REFLECT: Determine if goal is achieved

The agent is triggered only for complex queries (via complexity evaluation)
and uses the standard RAG pipeline for simple queries.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

from .agent_tools import (
    ToolName,
    ToolResult,
    ToolCall,
    ToolRegistry,
    create_tool_registry,
)
from .complexity import ComplexityScore
from .utils.llm_client import llm_chat, llm_stream
from .utils.logger import logger

if TYPE_CHECKING:
    from .hybrid_retriever import HybridRetriever


class AgentState(str, Enum):
    """Agent execution states."""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    FINISHED = "finished"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_number: int
    thought: str = ""
    action: Optional[ToolCall] = None
    observation: str = ""
    reflection: str = ""

    def to_text(self) -> str:
        """Format step as text for context."""
        parts = [f"Step {self.step_number}:"]
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action.tool.value}({self.action.arguments})")
        if self.observation:
            parts.append(f"Observation: {self.observation[:500]}...")
        if self.reflection:
            parts.append(f"Reflection: {self.reflection}")
        return "\n".join(parts)


@dataclass
class AgentContext:
    """Context accumulated during agent execution."""
    query: str
    steps: List[AgentStep] = field(default_factory=list)
    gathered_context: List[str] = field(default_factory=list)
    entities_found: List[Dict[str, Any]] = field(default_factory=list)
    current_state: AgentState = AgentState.THINKING

    def add_step(self, step: AgentStep):
        """Add a step to the context."""
        self.steps.append(step)

    def add_context(self, text: str):
        """Add gathered context."""
        if text and text not in self.gathered_context:
            self.gathered_context.append(text)

    def get_history_text(self) -> str:
        """Get formatted history for LLM context."""
        return "\n\n".join(step.to_text() for step in self.steps)

    def get_gathered_context(self) -> str:
        """Get all gathered context as text."""
        return "\n\n---\n\n".join(self.gathered_context)


@dataclass
class AgentResult:
    """Final result from agent execution."""
    query: str
    answer: str
    steps: List[AgentStep]
    success: bool
    needs_clarification: bool = False
    clarification_question: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def stream(self) -> Generator[str, None, None]:
        """Stream the answer character by character."""
        for char in self.answer:
            yield char


# =============================================================================
# ReAct Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an intelligent research assistant that answers questions by gathering information step by step.

You have access to these tools:
{tool_descriptions}

## Language Rules
- ALWAYS respond in the SAME LANGUAGE as the user's question.
- If the user asks in French, your final_answer MUST be in French.
- If the user asks in English, your final_answer MUST be in English.
- Your THOUGHT can be in any language, but the final_answer must match the user's language.
- If you cannot understand the language, say so in English and ask for clarification.

## Instructions

1. For each step, you must output your reasoning in this EXACT format:
```
THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
ARGUMENTS: <valid JSON object with the tool's parameters>
```

2. After receiving an observation, reflect on whether you have enough information.

3. When you have gathered enough information, use the `final_answer` tool to provide your complete answer.
   IMPORTANT: You MUST provide the answer in the ARGUMENTS, like this:
   ```
   THOUGHT: I have enough information to answer.
   ACTION: final_answer
   ARGUMENTS: {{"answer": "Your complete answer here in the user's language"}}
   ```

4. If the query is ambiguous, use `ask_clarification` to request more information.

## Important Rules
- Always think before acting
- Use retrieve_vector for factual lookups
- Use retrieve_graph for relationship questions
- Use lookup_entity when you need details about a specific entity
- Use compare_entities for comparative questions
- Use synthesize to combine information before final answer
- Use verify_claim to fact-check important statements
- Use database_stats for questions about the database itself (document count, size, etc.)
- Maximum {max_steps} steps allowed
"""

THINK_PROMPT = """Current query: {query}

{history}

Based on the above, what should be the next step? Think carefully about:
1. What information do we still need?
2. Which tool would be most helpful?
3. Are we ready to provide a final answer?

Respond with:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ARGUMENTS: <json arguments>"""

REFLECT_PROMPT = """Query: {query}

Steps taken:
{history}

Latest observation: {observation}

Gathered context so far:
{context}

Reflect on the progress:
1. Did the last action provide useful information?
2. Do we have enough information to answer the query?
3. What gaps remain?

If ready to answer, respond with:
THOUGHT: I have enough information to answer.
ACTION: final_answer
ARGUMENTS: {{"answer": "<your complete answer>"}}

If more information is needed, respond with:
THOUGHT: <what's still missing>
ACTION: <next_tool>
ARGUMENTS: <json arguments>"""


# =============================================================================
# CogniDocAgent
# =============================================================================

class CogniDocAgent:
    """
    ReAct-style agent for complex query handling.

    Uses a Think → Act → Observe → Reflect loop to gather information
    and construct comprehensive answers.
    """

    def __init__(
        self,
        retriever: "HybridRetriever",
        max_steps: int = 7,
        temperature: float = 0.3,
    ):
        """
        Initialize the agent.

        Args:
            retriever: HybridRetriever instance for search operations
            max_steps: Maximum reasoning steps before forcing conclusion
            temperature: LLM temperature for reasoning (lower = more focused)
        """
        self.retriever = retriever
        self.max_steps = max_steps
        self.temperature = temperature

        # Create tool registry
        self.tools = create_tool_registry(
            retriever=retriever,
            graph_retriever=retriever._graph_retriever if hasattr(retriever, '_graph_retriever') else None,
        )

        logger.info(f"CogniDocAgent initialized with {len(self.tools.tools)} tools, max_steps={max_steps}")

    def run(
        self,
        query: str,
        complexity: Optional[ComplexityScore] = None,
    ) -> AgentResult:
        """
        Run the agent on a query.

        Args:
            query: User query to process
            complexity: Pre-computed complexity score (optional)

        Returns:
            AgentResult with answer and execution trace
        """
        logger.info(f"Agent starting for query: {query[:100]}...")

        context = AgentContext(query=query)
        step_count = 0

        try:
            while step_count < self.max_steps:
                step_count += 1
                step = AgentStep(step_number=step_count)

                # 1. THINK + DECIDE
                context.current_state = AgentState.THINKING
                thought, action = self._think_and_decide(context)
                step.thought = thought
                step.action = action

                if action is None:
                    logger.warning(f"Step {step_count}: No action decided")
                    break

                logger.info(f"Step {step_count}: {action.tool.value}")

                # 2. Check for terminal actions
                if action.tool == ToolName.FINAL_ANSWER:
                    # Extract answer, with fallback to any string value in arguments
                    answer = action.arguments.get("answer", "")
                    if not answer and action.arguments:
                        # Fallback: use first string value if "answer" key not found
                        for v in action.arguments.values():
                            if isinstance(v, str) and v:
                                answer = v
                                break
                    step.observation = "Final answer provided"
                    context.add_step(step)
                    context.current_state = AgentState.FINISHED

                    return AgentResult(
                        query=query,
                        answer=answer,
                        steps=context.steps,
                        success=True,
                        metadata={
                            "total_steps": step_count,
                            "tools_used": [s.action.tool.value for s in context.steps if s.action],
                        },
                    )

                if action.tool == ToolName.ASK_CLARIFICATION:
                    question = action.arguments.get("question", "Could you clarify your question?")
                    step.observation = f"Clarification requested: {question}"
                    context.add_step(step)
                    context.current_state = AgentState.NEEDS_CLARIFICATION

                    return AgentResult(
                        query=query,
                        answer="",
                        steps=context.steps,
                        success=False,
                        needs_clarification=True,
                        clarification_question=question,
                        metadata={"total_steps": step_count},
                    )

                # 3. ACT
                context.current_state = AgentState.ACTING
                result = self.tools.execute(action)
                step.observation = result.observation

                # Store useful context
                if result.success and result.data:
                    if action.tool == ToolName.RETRIEVE_VECTOR:
                        for doc in result.data[:3]:
                            context.add_context(doc.get("text", ""))
                    elif action.tool == ToolName.RETRIEVE_GRAPH:
                        context.add_context(result.data.get("context", ""))
                    elif action.tool == ToolName.LOOKUP_ENTITY:
                        if result.data:
                            context.entities_found.append(result.data)
                    elif action.tool in (ToolName.SYNTHESIZE, ToolName.COMPARE_ENTITIES):
                        context.add_context(str(result.data))

                # 4. OBSERVE + REFLECT
                context.current_state = AgentState.REFLECTING
                reflection = self._reflect(context, step.observation)
                step.reflection = reflection

                context.add_step(step)

            # Max steps reached - force conclusion
            logger.warning(f"Agent reached max_steps ({self.max_steps}), forcing conclusion")
            final_answer = self._force_conclusion(context)

            return AgentResult(
                query=query,
                answer=final_answer,
                steps=context.steps,
                success=True,
                metadata={
                    "total_steps": step_count,
                    "forced_conclusion": True,
                    "tools_used": [s.action.tool.value for s in context.steps if s.action],
                },
            )

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            context.current_state = AgentState.ERROR

            return AgentResult(
                query=query,
                answer=f"An error occurred while processing your query: {str(e)}",
                steps=context.steps,
                success=False,
                error=str(e),
            )

    def _think_and_decide(self, context: AgentContext) -> Tuple[str, Optional[ToolCall]]:
        """
        Think about the current state and decide on next action.

        Returns:
            Tuple of (thought, action)
        """
        # Build prompt
        if not context.steps:
            # First step
            prompt = THINK_PROMPT.format(
                query=context.query,
                history="No previous steps.",
            )
        else:
            prompt = THINK_PROMPT.format(
                query=context.query,
                history=context.get_history_text(),
            )

        system = SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_tool_descriptions(),
            max_steps=self.max_steps,
        )

        # Get LLM response
        response = llm_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        # Parse response
        thought, action = self._parse_thought_action(response)
        return thought, action

    def _reflect(self, context: AgentContext, observation: str) -> str:
        """
        Reflect on the latest observation.

        Returns:
            Reflection text
        """
        prompt = REFLECT_PROMPT.format(
            query=context.query,
            history=context.get_history_text(),
            observation=observation[:1000],
            context=context.get_gathered_context()[:2000],
        )

        response = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # Extract just the thought/reflection part
        if "THOUGHT:" in response:
            thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
            if thought_match:
                return thought_match.group(1).strip()

        return response[:500]

    def _parse_thought_action(self, response: str) -> Tuple[str, Optional[ToolCall]]:
        """
        Parse LLM response to extract thought and action.

        Expected format:
        THOUGHT: ...
        ACTION: tool_name
        ARGUMENTS: {"key": "value"}
        """
        thought = ""
        action = None

        # Extract thought
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        if action_match:
            tool_name = action_match.group(1).lower()

            # Map to ToolName enum
            try:
                tool = ToolName(tool_name)
            except ValueError:
                logger.warning(f"Unknown tool: {tool_name}")
                return thought, None

            # Extract arguments
            args = {}
            args_match = re.search(r"ARGUMENTS:\s*(\{.+?\})", response, re.DOTALL | re.IGNORECASE)
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    # Try to extract key-value pairs manually
                    args = self._parse_args_fallback(args_match.group(1))

            action = ToolCall(tool=tool, arguments=args, reasoning=thought)

        return thought, action

    def _parse_args_fallback(self, args_str: str) -> Dict[str, str]:
        """Fallback argument parsing for malformed JSON."""
        args = {}
        # Try to find key: value patterns
        pairs = re.findall(r'"?(\w+)"?\s*:\s*"([^"]*)"', args_str)
        for key, value in pairs:
            args[key] = value
        return args

    def _force_conclusion(self, context: AgentContext) -> str:
        """
        Force a conclusion when max steps reached.
        """
        prompt = f"""You must now provide a final answer based on the information gathered.

Query: {context.query}

Information gathered:
{context.get_gathered_context()[:3000]}

Provide the best possible answer with the available information. If some aspects couldn't be fully answered, acknowledge this."""

        response = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        return response

    def run_streaming(
        self,
        query: str,
        complexity: Optional[ComplexityScore] = None,
    ) -> Generator[Tuple[AgentState, str], None, AgentResult]:
        """
        Run the agent with streaming output.

        Yields:
            Tuples of (state, message) during execution
            Returns AgentResult when complete
        """
        logger.info(f"Agent streaming for query: {query[:100]}...")

        context = AgentContext(query=query)
        step_count = 0

        try:
            while step_count < self.max_steps:
                step_count += 1
                step = AgentStep(step_number=step_count)

                # 1. THINK
                yield (AgentState.THINKING, f"Step {step_count}: Analyzing...")

                thought, action = self._think_and_decide(context)
                step.thought = thought

                if action is None:
                    break

                yield (AgentState.THINKING, f"Thought: {thought[:100]}...")

                # 2. Check terminal actions
                if action.tool == ToolName.FINAL_ANSWER:
                    # Extract answer, with fallback to any string value in arguments
                    answer = action.arguments.get("answer", "")
                    if not answer and action.arguments:
                        # Fallback: use first string value if "answer" key not found
                        for v in action.arguments.values():
                            if isinstance(v, str) and v:
                                answer = v
                                break
                    step.observation = "Final answer provided"
                    context.add_step(step)

                    yield (AgentState.FINISHED, "Generating final answer...")

                    return AgentResult(
                        query=query,
                        answer=answer,
                        steps=context.steps,
                        success=True,
                    )

                if action.tool == ToolName.ASK_CLARIFICATION:
                    question = action.arguments.get("question", "")
                    context.add_step(step)

                    yield (AgentState.NEEDS_CLARIFICATION, question)

                    return AgentResult(
                        query=query,
                        answer="",
                        steps=context.steps,
                        success=False,
                        needs_clarification=True,
                        clarification_question=question,
                    )

                # 3. ACT
                yield (AgentState.ACTING, f"Using tool: {action.tool.value}")

                step.action = action
                result = self.tools.execute(action)
                step.observation = result.observation

                # Store context
                if result.success and result.data:
                    if action.tool == ToolName.RETRIEVE_VECTOR:
                        for doc in result.data[:3]:
                            context.add_context(doc.get("text", ""))
                    elif action.tool == ToolName.RETRIEVE_GRAPH:
                        context.add_context(result.data.get("context", ""))

                yield (AgentState.OBSERVING, f"Observation: {step.observation[:100]}...")

                # 4. REFLECT
                reflection = self._reflect(context, step.observation)
                step.reflection = reflection
                context.add_step(step)

                yield (AgentState.REFLECTING, f"Reflection: {reflection[:100]}...")

            # Max steps - force conclusion
            yield (AgentState.FINISHED, "Reaching conclusion...")
            final_answer = self._force_conclusion(context)

            return AgentResult(
                query=query,
                answer=final_answer,
                steps=context.steps,
                success=True,
                metadata={"forced_conclusion": True},
            )

        except Exception as e:
            logger.error(f"Agent streaming error: {e}")
            yield (AgentState.ERROR, str(e))

            return AgentResult(
                query=query,
                answer=f"Error: {str(e)}",
                steps=context.steps,
                success=False,
                error=str(e),
            )


# =============================================================================
# Factory Function
# =============================================================================

def create_agent(
    retriever: "HybridRetriever",
    max_steps: int = 7,
    temperature: float = 0.3,
) -> CogniDocAgent:
    """
    Create a CogniDocAgent instance.

    Args:
        retriever: HybridRetriever for search operations
        max_steps: Maximum reasoning steps
        temperature: LLM temperature

    Returns:
        Configured CogniDocAgent
    """
    return CogniDocAgent(
        retriever=retriever,
        max_steps=max_steps,
        temperature=temperature,
    )


__all__ = [
    "AgentState",
    "AgentStep",
    "AgentContext",
    "AgentResult",
    "CogniDocAgent",
    "create_agent",
]
