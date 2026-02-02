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

import atexit
import json
import re
from concurrent.futures import ThreadPoolExecutor, Future
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

# Thread pool for parallel reflection (reused across agent calls)
_reflection_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent_reflect")
atexit.register(_reflection_executor.shutdown, wait=False)

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
    actions: List[ToolCall] = field(default_factory=list)
    observation: str = ""
    reflection: str = ""

    @property
    def action(self) -> Optional[ToolCall]:
        """First action (backward compatibility)."""
        return self.actions[0] if self.actions else None

    @action.setter
    def action(self, value: Optional[ToolCall]):
        """Set single action (backward compatibility)."""
        if value is not None:
            self.actions = [value]
        else:
            self.actions = []

    def to_text(self) -> str:
        """Format step as text for context."""
        parts = [f"Step {self.step_number}:"]
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        for act in self.actions:
            parts.append(f"Action: {act.tool.value}({act.arguments})")
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
        """Get formatted history for LLM context (full detail, used for logging)."""
        return "\n\n".join(step.to_text() for step in self.steps)

    def get_trimmed_history(self, keep_full: int = 2) -> str:
        """Get history with older steps summarized to save context window.

        Keeps the last `keep_full` steps in full detail (thought, action,
        observation, reflection). Older steps are reduced to thought + action
        only, dropping the bulky observation and reflection fields.
        """
        parts = []
        cutoff = len(self.steps) - keep_full
        for i, step in enumerate(self.steps):
            if i < cutoff:
                lines = [f"Step {step.step_number}:"]
                if step.thought:
                    lines.append(f"Thought: {step.thought}")
                for act in step.actions:
                    lines.append(f"Action: {act.tool.value}({act.arguments})")
                parts.append("\n".join(lines))
            else:
                parts.append(step.to_text())
        return "\n\n".join(parts)

    def get_gathered_context(self) -> str:
        """Get all gathered context as text."""
        return "\n\n---\n\n".join(self.gathered_context)


@dataclass
class _ContextSnapshot:
    """Lightweight snapshot for rollback — stores list lengths only."""

    steps_len: int
    gathered_context_len: int
    entities_found_len: int


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

    def stream(self, chunk_size: int = 3) -> Generator[str, None, None]:
        """Stream the answer in word chunks for progressive display."""
        words = self.answer.split()
        accumulated = ""
        for i, word in enumerate(words):
            accumulated += (" " if accumulated else "") + word
            if i % chunk_size == 0 or i == len(words) - 1:
                yield accumulated


# =============================================================================
# ReAct Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an efficient research assistant. Your goal is to answer questions QUICKLY with MINIMAL steps.

## Tools Available
{tool_descriptions}

## Language Rules
- ALWAYS respond in the SAME LANGUAGE as the user's question.
- If the user asks in French, your final_answer MUST be in French.
- If the user asks in English, your final_answer MUST be in English.
- If the user asks in Spanish, your final_answer MUST be in Spanish.
- If the user asks in German, your final_answer MUST be in German.

## Response Format
For each step, output EXACTLY:
```
THOUGHT: <brief reasoning>
ACTION: <tool_name>
ARGUMENTS: <JSON object>
```

For parallel actions (e.g., comparing two things), use numbered format:
```
THOUGHT: <brief reasoning>
ACTION_1: <tool_name>
ARGUMENTS_1: <JSON object>
ACTION_2: <tool_name>
ARGUMENTS_2: <JSON object>
```
Use parallel actions ONLY for independent operations (e.g., retrieving info about X and Y separately). Do NOT use parallel actions for terminal actions (final_answer, ask_clarification).

## Efficiency Guidelines - CRITICAL
1. **One retrieval is usually enough.** After ONE successful retrieve_vector or retrieve_graph call, you likely have enough information. Proceed to final_answer.
2. **Skip synthesis for simple questions.** Use final_answer directly after getting relevant documents.
3. **Use database_stats ONLY for meta-questions** about the database itself (document count, listing documents, etc.).
4. **Avoid redundant lookups.** If retrieve_vector gave you info, don't also call retrieve_graph for the same question.
5. **Target: 2-3 steps max for most queries.** Complex comparisons may need 3 steps with parallel retrieval.
6. **Use parallel actions for comparisons.** When comparing X and Y, retrieve both in parallel rather than sequentially.

## When to use each tool
- `database_stats`: ONLY for "how many documents?", "list documents", etc.
- `exhaustive_search`: For corpus-wide keyword questions: "how many documents mention X?", "list all documents about Y", "does any document discuss Z?" Use `source_filter` to restrict to a specific document.
- `retrieve_vector`: Factual questions about document content. Use `source_filter` to target a specific document (e.g., when comparing documents or when the user asks about a particular report).
- `retrieve_graph`: Questions about relationships between entities
- `lookup_entity`: Only if you need detailed info about ONE specific entity
- `compare_entities`: Only for explicit comparison questions
- `synthesize`: Only for multi-part questions requiring integration
- `final_answer`: Use as soon as you have sufficient information!

## Final Answer Format
```
THOUGHT: I have enough information.
ACTION: final_answer
ARGUMENTS: {{"answer": "Your complete answer in the user's language"}}
```

Maximum {max_steps} steps. Aim for 2-3.
"""

THINK_PROMPT = """Query: {query}

{history}

What's the MOST EFFICIENT next action? If you already have relevant information, use final_answer immediately.

THOUGHT:
ACTION:
ARGUMENTS:"""

REFLECT_PROMPT = """Query: {query}

Observation: {observation}

Context gathered: {context}

Can you answer the query NOW with the information gathered?
- If YES: respond with ACTION: final_answer
- If NO (critical information is still missing): respond with ACTION: continue

THOUGHT:
ACTION:"""


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
            graph_retriever=(
                retriever._graph_retriever if hasattr(retriever, "_graph_retriever") else None
            ),
        )

        logger.info(
            f"CogniDocAgent initialized with {len(self.tools.tools)} tools, max_steps={max_steps}"
        )

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
        gen = self._run_loop(query, complexity)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result: AgentResult = e.value
            return result

    def _think_and_decide(
        self, context: AgentContext, backtrack_hint: str = ""
    ) -> Tuple[str, List[ToolCall]]:
        """
        Think about the current state and decide on next action(s).

        Args:
            context: Current agent context
            backtrack_hint: If non-empty, describes a previous failed attempt

        Returns:
            Tuple of (thought, actions) where actions may contain 1-2 tool calls.
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
                history=context.get_trimmed_history(),
            )

        if backtrack_hint:
            prompt += (
                f"\n\nIMPORTANT — PREVIOUS ATTEMPT FAILED:\n"
                f"{backtrack_hint}\nTry a DIFFERENT approach."
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
        thought, actions = self._parse_thought_actions(response)
        return thought, actions

    def _reflect(self, context: AgentContext, observation: str) -> Tuple[str, bool]:
        """
        Reflect on the latest observation.

        Returns:
            Tuple of (reflection_text, should_conclude).
            should_conclude is True when the reflection determines enough
            information has been gathered to produce a final answer.
        """
        prompt = REFLECT_PROMPT.format(
            query=context.query,
            history=context.get_trimmed_history(),
            observation=observation[:1000],
            context=context.get_gathered_context()[:3000],
        )

        response = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # Determine if reflection recommends concluding
        should_conclude = bool(re.search(r"ACTION:\s*final_answer", response, re.IGNORECASE))

        # Extract just the thought/reflection part
        if "THOUGHT:" in response:
            thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
            if thought_match:
                return thought_match.group(1).strip(), should_conclude

        return response[:500], should_conclude

    def _parse_thought_actions(self, response: str) -> Tuple[str, List[ToolCall]]:
        """
        Parse LLM response to extract thought and action(s).

        Supports two formats:
        - Single: THOUGHT / ACTION / ARGUMENTS
        - Parallel: THOUGHT / ACTION_1 / ARGUMENTS_1 / ACTION_2 / ARGUMENTS_2
        """
        thought = ""
        actions: List[ToolCall] = []

        # Extract thought
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=ACTION|$)", response, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Try numbered actions first: ACTION_1, ACTION_2
        for i in range(1, 3):  # max 2 parallel actions
            action_match = re.search(rf"ACTION_{i}:\s*(\w+)", response, re.IGNORECASE)
            if action_match:
                tool_name = action_match.group(1).lower()
                try:
                    tool = ToolName(tool_name)
                except ValueError:
                    logger.warning(f"Unknown tool in parallel action {i}: {tool_name}")
                    continue
                args = {}
                args_match = re.search(
                    rf"ARGUMENTS_{i}:\s*(\{{.+?\}})", response, re.DOTALL | re.IGNORECASE
                )
                if args_match:
                    try:
                        args = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        args = self._parse_args_fallback(args_match.group(1))
                actions.append(ToolCall(tool=tool, arguments=args, reasoning=thought))

        # Filter out terminal actions from parallel responses
        terminal_tools = {ToolName.FINAL_ANSWER, ToolName.ASK_CLARIFICATION}
        if len(actions) > 1:
            terminal = [a for a in actions if a.tool in terminal_tools]
            if terminal:
                # Terminal action found in parallel — keep only the terminal one
                actions = [terminal[0]]

        # Fallback to single ACTION: format
        if not actions:
            action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
            if action_match:
                tool_name = action_match.group(1).lower()
                try:
                    tool = ToolName(tool_name)
                except ValueError:
                    logger.warning(f"Unknown tool: {tool_name}")
                    return thought, []
                args = {}
                args_match = re.search(
                    r"ARGUMENTS:\s*(\{.+?\})", response, re.DOTALL | re.IGNORECASE
                )
                if args_match:
                    try:
                        args = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        args = self._parse_args_fallback(args_match.group(1))
                actions.append(ToolCall(tool=tool, arguments=args, reasoning=thought))

        return thought, actions

    def _parse_args_fallback(self, args_str: str) -> Dict[str, str]:
        """Fallback argument parsing for malformed JSON."""
        args = {}
        # Try to find key: value patterns
        pairs = re.findall(r'"?(\w+)"?\s*:\s*"([^"]*)"', args_str)
        for key, value in pairs:
            args[key] = value
        return args

    def _is_dead_end(self, tool_results: List[Tuple["ToolCall", "ToolResult"]]) -> bool:
        """Check if tool results indicate a dead end worth backtracking from."""
        for action, result in tool_results:
            # Terminal and meta tools never trigger backtracking
            if action.tool in (
                ToolName.FINAL_ANSWER,
                ToolName.ASK_CLARIFICATION,
                ToolName.SYNTHESIZE,
                ToolName.DATABASE_STATS,
            ):
                return False

            if not result.success:
                return True

            if action.tool == ToolName.RETRIEVE_VECTOR:
                if result.data is not None and len(result.data) == 0:
                    return True

            if action.tool == ToolName.LOOKUP_ENTITY:
                if result.data is None:
                    return True

            if action.tool == ToolName.RETRIEVE_GRAPH:
                if not result.data:
                    return True

        return False

    def _build_backtrack_hint(self, tool_results: List[Tuple["ToolCall", "ToolResult"]]) -> str:
        """Build a human-readable hint describing what failed."""
        parts = []
        for action, result in tool_results:
            args_str = ", ".join(f"{k}={v!r}" for k, v in action.arguments.items())
            if not result.success:
                parts.append(f"{action.tool.value}({args_str}) failed: {result.error}")
            elif action.tool == ToolName.RETRIEVE_VECTOR and not result.data:
                parts.append(f"{action.tool.value}({args_str}) returned no documents")
            elif action.tool == ToolName.LOOKUP_ENTITY and result.data is None:
                parts.append(f"{action.tool.value}({args_str}) found no matching entity")
            elif action.tool == ToolName.RETRIEVE_GRAPH and not result.data:
                parts.append(f"{action.tool.value}({args_str}) returned no graph results")
            else:
                parts.append(f"{action.tool.value}({args_str}) produced empty results")
        return "; ".join(parts)

    def _rollback_context(self, context: AgentContext, snapshot: _ContextSnapshot):
        """Roll back context to a previous snapshot."""
        context.steps[:] = context.steps[: snapshot.steps_len]
        context.gathered_context[:] = context.gathered_context[: snapshot.gathered_context_len]
        context.entities_found[:] = context.entities_found[: snapshot.entities_found_len]

    def _force_conclusion(self, context: AgentContext) -> str:
        """
        Force a conclusion when max steps reached.
        """
        prompt = f"""You must now provide a final answer based on the information gathered.

Query: {context.query}

Reasoning steps:
{context.get_trimmed_history()}

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
        return (yield from self._run_loop(query, complexity))

    def _run_loop(
        self,
        query: str,
        complexity: Optional[ComplexityScore] = None,
    ) -> Generator[Tuple[AgentState, str], None, AgentResult]:
        """
        Core agent loop shared by run() and run_streaming().

        Yields (AgentState, message) tuples for streaming progress.
        Returns AgentResult when complete.
        """
        context = AgentContext(query=query)
        step_count = 0
        has_backtracked = False
        backtrack_hint = ""

        try:
            while step_count < self.max_steps:
                step_count += 1
                step = AgentStep(step_number=step_count)

                # Snapshot for potential rollback
                snapshot = _ContextSnapshot(
                    steps_len=len(context.steps),
                    gathered_context_len=len(context.gathered_context),
                    entities_found_len=len(context.entities_found),
                )

                # 1. THINK + DECIDE
                context.current_state = AgentState.THINKING
                yield (
                    AgentState.THINKING,
                    f"[Step {step_count}/{self.max_steps}] Analyzing query...",
                )

                thought, actions = self._think_and_decide(context, backtrack_hint)
                backtrack_hint = ""  # Clear after use
                step.thought = thought
                step.actions = actions

                if not actions:
                    logger.warning(f"Step {step_count}: No action decided")
                    yield (AgentState.THINKING, "No action decided, finishing...")
                    break

                action_names = ", ".join(a.tool.value for a in actions)
                logger.info(f"Step {step_count}: {action_names}")

                thought_preview = thought[:150].replace("\n", " ") if thought else "..."
                yield (AgentState.THINKING, f"Thought: {thought_preview}")

                # 2. Check for terminal actions (only valid as single action)
                if len(actions) == 1:
                    action = actions[0]
                    if action.tool == ToolName.FINAL_ANSWER:
                        answer = action.arguments.get("answer", "")
                        if not answer and action.arguments:
                            for v in action.arguments.values():
                                if isinstance(v, str) and v:
                                    answer = v
                                    break
                        step.observation = "Final answer provided"
                        context.add_step(step)
                        context.current_state = AgentState.FINISHED

                        yield (AgentState.FINISHED, "Preparing final answer...")

                        return AgentResult(
                            query=query,
                            answer=answer,
                            steps=context.steps,
                            success=True,
                            metadata={
                                "total_steps": step_count,
                                "backtracked": has_backtracked,
                                "tools_used": list(
                                    {a.tool.value for s in context.steps for a in s.actions}
                                ),
                            },
                        )

                    if action.tool == ToolName.ASK_CLARIFICATION:
                        question = action.arguments.get(
                            "question", "Could you clarify your question?"
                        )
                        step.observation = f"Clarification requested: {question}"
                        context.add_step(step)
                        context.current_state = AgentState.NEEDS_CLARIFICATION

                        yield (AgentState.NEEDS_CLARIFICATION, question)

                        return AgentResult(
                            query=query,
                            answer="",
                            steps=context.steps,
                            success=False,
                            needs_clarification=True,
                            clarification_question=question,
                            metadata={"total_steps": step_count},
                        )

                # 3. ACT — execute actions (parallel if multiple)
                context.current_state = AgentState.ACTING

                if len(actions) == 1:
                    # Single action
                    action = actions[0]
                    tool_args_preview = ", ".join(
                        f"{k}={v}" for k, v in list(action.arguments.items())[:2]
                    )
                    yield (
                        AgentState.ACTING,
                        f"Calling {action.tool.value}({tool_args_preview[:50]})",
                    )
                    result = self.tools.execute(action)
                    tool_results = [(action, result)]
                else:
                    # Parallel execution
                    yield (
                        AgentState.ACTING,
                        f"Executing {len(actions)} actions in parallel: {action_names}",
                    )
                    futures = [
                        (act, _reflection_executor.submit(self.tools.execute, act))
                        for act in actions
                    ]
                    tool_results = [(act, fut.result(timeout=30.0)) for act, fut in futures]

                # Combine observations and store context
                observations = []
                for act, res in tool_results:
                    obs_text = res.observation
                    if len(tool_results) > 1:
                        obs_text = f"[{act.tool.value}] {obs_text}"
                    observations.append(obs_text)

                    # Store useful context
                    if res.success and res.data:
                        if act.tool == ToolName.RETRIEVE_VECTOR:
                            for doc in res.data[:3]:
                                context.add_context(doc.get("text", ""))
                        elif act.tool == ToolName.RETRIEVE_GRAPH:
                            context.add_context(res.data.get("context", ""))
                        elif act.tool == ToolName.LOOKUP_ENTITY:
                            if res.data:
                                context.entities_found.append(res.data)
                        elif act.tool == ToolName.EXHAUSTIVE_SEARCH:
                            total = res.data.get("total_matches", 0)
                            docs = res.data.get("source_documents", [])
                            context.add_context(
                                f"Exhaustive search: {total} matches across "
                                f"{len(docs)} documents: {', '.join(docs[:10])}"
                            )
                        elif act.tool in (ToolName.SYNTHESIZE, ToolName.COMPARE_ENTITIES):
                            context.add_context(str(res.data))

                step.observation = "\n".join(observations)

                # Dead-end detection + backtrack (max 1 per run)
                if not has_backtracked and self._is_dead_end(tool_results):
                    has_backtracked = True
                    backtrack_hint = self._build_backtrack_hint(tool_results)
                    logger.info(
                        f"Step {step_count}: dead end detected, backtracking. "
                        f"Hint: {backtrack_hint[:120]}"
                    )
                    yield (
                        AgentState.THINKING,
                        "Dead end detected, trying a different approach...",
                    )
                    self._rollback_context(context, snapshot)
                    continue  # re-enter loop — step_count already incremented

                # 4. OBSERVE
                obs_preview = (
                    step.observation[:120].replace("\n", " ") if step.observation else "No result"
                )
                cached_indicator = ""
                if len(tool_results) == 1 and tool_results[0][1].metadata.get("cached"):
                    cached_indicator = " [cached]"
                yield (AgentState.OBSERVING, f"Result{cached_indicator}: {obs_preview}")

                # 5. REFLECT (parallel) - Start reflection while context is fresh
                context.current_state = AgentState.REFLECTING
                reflection_future: Future = _reflection_executor.submit(
                    self._reflect, context, step.observation
                )

                # Wait for reflection to complete (usually already done by now)
                step.reflection, should_conclude = reflection_future.result(timeout=30.0)
                context.add_step(step)

                if step.reflection and len(step.reflection) > 10:
                    refl_preview = step.reflection[:100].replace("\n", " ")
                    yield (AgentState.REFLECTING, f"Analysis: {refl_preview}")

                # Binding reflection: if reflection says we can conclude, do it now
                if should_conclude:
                    logger.info("Reflection binding: concluding based on reflection signal")
                    yield (AgentState.FINISHED, "Sufficient information gathered, concluding...")
                    conclusion = self._force_conclusion(context)
                    return AgentResult(
                        query=query,
                        answer=conclusion,
                        steps=context.steps,
                        success=True,
                        metadata={
                            "total_steps": step_count,
                            "reflection_concluded": True,
                            "backtracked": has_backtracked,
                            "tools_used": list(
                                {a.tool.value for s in context.steps for a in s.actions}
                            ),
                            "parallel_steps": sum(1 for s in context.steps if len(s.actions) > 1),
                        },
                    )

            # Max steps reached - force conclusion
            logger.warning(f"Agent reached max_steps ({self.max_steps}), forcing conclusion")
            yield (AgentState.FINISHED, "Reaching conclusion...")
            final_answer = self._force_conclusion(context)

            return AgentResult(
                query=query,
                answer=final_answer,
                steps=context.steps,
                success=True,
                metadata={
                    "total_steps": step_count,
                    "forced_conclusion": True,
                    "backtracked": has_backtracked,
                    "tools_used": list({a.tool.value for s in context.steps for a in s.actions}),
                    "parallel_steps": sum(1 for s in context.steps if len(s.actions) > 1),
                },
            )

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            context.current_state = AgentState.ERROR

            yield (AgentState.ERROR, str(e))

            return AgentResult(
                query=query,
                answer=f"An error occurred while processing your query: {str(e)}",
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
