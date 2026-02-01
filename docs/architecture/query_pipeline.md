# Query Pipeline - Architecture Details

This document explains the four core mechanisms of the CogniDoc query pipeline: complexity evaluation, retrieval caching, vector/graph fusion, and the ReAct agent loop. These components are interdependent and form a single end-to-end flow.

## Overview

```
User Query
    |
    v
Query Rewriter (adds conversation context)
    |
    +---> Query Classifier (query type + weights)
    +---> Complexity Evaluator (routing decision)
    |
    v
score < 0.35 ───> FAST PATH ──────────┐
0.35 - 0.55 ───> ENHANCED PATH ───────┤
score >= 0.55 ──> AGENT PATH (ReAct) ──┤
    |                                  |
    v                                  v
                              Final Response
```

The classifier determines *how* to retrieve (vector/graph weights). The complexity evaluator determines *which path* to use (fast, enhanced, or agent). These two evaluations run in parallel.

---

## 1. Complexity Evaluation

**File:** `src/cognidoc/complexity.py` (function `evaluate_complexity`, lines 275-387)

### Why This Component Exists

Without complexity evaluation, the system would have to choose a single path for all queries. Sending everything through the fast path means complex queries (comparisons, multi-entity analysis) get shallow, incomplete answers. Sending everything through the agent means simple factual queries take 5-12s instead of 2-3s, with no improvement in answer quality. The complexity evaluator is the router that makes the two-path architecture viable.

### Advantages

- **Cost control**: the agent path consumes 4-10x more LLM tokens per query. Routing only complex queries through it keeps API costs proportional to actual need.
- **Latency optimization**: simple queries get fast answers (~2-3s) without paying the agent overhead.
- **Deterministic**: the formula is a pure function of observable signals (query type, entity count, keywords). No LLM call is needed for the routing decision itself, unlike systems that use an LLM to decide whether to use an LLM.
- **Debuggable**: the `ComplexityScore` dataclass exposes all factors and reasoning, making it easy to understand why a query was routed to a specific path.

### Limitations and Possible Improvements

- **~~The scoring is coarse~~** *(addressed)*: factors now use continuous scoring functions (`min(n/4, 1.0)`, linear ramps) instead of discrete 0.0/0.5/1.0 steps. See "Factor Scoring" below for the formulas.
- **No feedback loop**: the thresholds (0.35, 0.55) are static. If the agent consistently produces better answers than the fast path for queries scoring 0.40, there is no mechanism to learn and lower the threshold. An adaptive threshold based on user satisfaction signals (thumbs up/down, follow-up questions) could improve routing over time.
- **Keyword matching is brittle**: regex patterns are language-specific and must be maintained manually. A query like "quelles sont les différences fondamentales" triggers on "différences" but a rephrasing like "en quoi X et Y divergent-ils" does not. A lightweight embedding-based classifier could replace the keyword factor for more robust detection.
- **No distinction between MODERATE and SIMPLE in practice**: the enhanced path uses the same pipeline as the fast path. The MODERATE level exists as a conceptual category but does not currently trigger different behavior. It could be used to enable specific features (e.g., query expansion, additional retrieval passes) without the full agent loop.

### The Formula

The complexity score is a weighted sum of 5 factors, each scored continuously in [0.0, 1.0]:

```
score = 0.25 * query_type
      + 0.20 * entity_count
      + 0.20 * subquestion_count
      + 0.20 * keyword_matches
      + 0.15 * low_confidence
```

Score range: 0.0 (trivially simple) to 1.0 (maximally complex).

### Factor Scoring

Four of the five factors use continuous scoring functions. `query_type` remains discrete because it is categorical (enum-based).

| Factor | Weight | Formula | Example values |
|--------|--------|---------|----------------|
| **query_type** | 25% | Discrete: 0.0 (FACTUAL, PROCEDURAL), 0.5 (RELATIONAL, EXPLORATORY), 1.0 (ANALYTICAL, COMPARATIVE) | — |
| **entity_count** | 20% | `min(count / 4, 1.0)` | 0→0.0, 1→0.25, 2→0.5, 3→0.75, 4+→1.0 |
| **subquestion_count** | 20% | `min((count - 1) / 3, 1.0)` | 1→0.0, 2→0.33, 3→0.67, 4+→1.0 |
| **keyword_matches** | 20% | `min(count / 4, 1.0)` | 0→0.0, 1→0.25, 2→0.5, 3→0.75, 4+→1.0 |
| **low_confidence** | 15% | `clamp((0.7 - confidence) / 0.5)` | 0.7+→0.0, 0.6→0.2, 0.45→0.5, 0.2→1.0 |

**Why continuous scoring:** The previous discrete scoring (0.0/0.5/1.0) lost information at the boundaries. A query with 3 entities scored the same as one with 10 (both 1.0). A query with 1 entity scored the same as one with 0 (both 0.0). Continuous functions preserve the proportional signal: 3 entities (0.75) is meaningfully different from 2 (0.5) and 4 (1.0). This improves routing precision for borderline queries in the 0.35-0.55 range.

**Why these specific functions:** Each function is designed to match the previous behavior at the key breakpoints (2 entities ≈ 0.5, ≥4 = 1.0) while filling in the gaps. The denominators (4 for entities/keywords, 3 for sub-questions, 0.5 for confidence) were chosen so that the saturation point (score = 1.0) corresponds to a value that genuinely indicates high complexity. Beyond 4 entities or 4 keywords, additional ones don't meaningfully increase complexity — the query is already clearly complex.

**query_type** comes from the classifier (`QueryOrchestrator`). ANALYTICAL and COMPARATIVE queries inherently require multi-step reasoning. This factor remains discrete because query types are categorical, not ordinal.

**entity_count** is the number of entities detected by the classifier. More entities = more likely to need cross-referencing.

**subquestion_count** is derived from the rewritten query. The query rewriter decomposes complex questions into bullet points; more sub-questions means more retrieval steps needed. The formula subtracts 1 because a single question (count=1) should score 0.0 — it's the baseline, not a complexity signal.

**keyword_matches** counts regex matches against ~45 patterns in French, English, Spanish, and German covering causal reasoning ("pourquoi", "why"), analysis ("analyser", "evaluate"), comparison ("comparer", "avantages"), multi-step ("étapes", "processus"), and synthesis ("résumer", "overview").

**low_confidence** uses the classifier's own confidence score. The linear ramp starts at 0.7 (above which the classifier is confident enough that its uncertainty is not a useful signal) and reaches 1.0 at 0.2 (very uncertain). When the classifier is unsure about the query type, the agent is more likely to produce a better result through iterative retrieval.

### Why These Weights

`query_type` has the highest weight (25%) because it is the strongest single signal: an ANALYTICAL query almost always needs multi-step reasoning regardless of other factors. The remaining four factors are equally weighted at 15-20% because none is individually decisive -- they act as converging evidence. `low_confidence` is slightly lower (15%) because low classifier confidence is a weak signal on its own (the classifier may just be uncertain between FACTUAL and PROCEDURAL, both of which are simple).

### Routing Thresholds

- `score < 0.35` --> **SIMPLE** (fast path: standard single-pass RAG)
- `0.35 <= score < 0.55` --> **MODERATE** (enhanced path: same pipeline, better fusion)
- `score >= 0.55` --> **COMPLEX** (agent path: ReAct loop)

The thresholds were chosen so that a single strong signal (e.g., query_type=1.0 alone = 0.25) does not trigger the agent path, but two or more converging signals do. With continuous scoring, each factor contributes proportionally, so borderline queries get more nuanced routing. For example: query_type=COMPARATIVE (1.0×0.25=0.25) + 2 entities (0.5×0.20=0.10) + 2 keywords (0.5×0.20=0.10) = 0.45 (moderate), while adding a third entity pushes it to 0.50 and a third keyword to 0.55 (agent). This gradual accumulation prevents over-triggering on queries with a single complex signal.

### Overrides

Two cases bypass the formula and force the agent path:

1. **Database meta-questions** (detected by multilingual regex patterns in `DATABASE_META_PATTERNS`): queries like "combien de documents ?", "list all docs", "wie viele Dokumente" force the score to `max(computed_score, 0.75)`. These questions can only be answered by the `database_stats` agent tool, not by document retrieval.

2. **Ambiguous queries** (< 15 characters, multiple `?`, repeated "ou...ou"): the score is forced to `max(computed_score, 0.65)` and the level becomes AMBIGUOUS. The agent will use `ask_clarification` to request more information.

### Worked Example

Query: *"Compare les avantages et inconvenients du processus X par rapport a Y"*

- `query_type` = COMPARATIVE --> **1.0**
- `entity_count` = 2 (X, Y) --> `min(2/4, 1.0)` = **0.5**
- `subquestion_count` = likely 3 --> `min((3-1)/3, 1.0)` = **0.67**
- `keyword_matches` = "compare", "avantages", "inconvenients", "processus" (4 matches) --> `min(4/4, 1.0)` = **1.0**
- `low_confidence` = assuming confidence 0.7 --> `clamp((0.7-0.7)/0.5)` = **0.0**

Score: `0.25*1.0 + 0.20*0.5 + 0.20*0.67 + 0.20*1.0 + 0.15*0.0 = 0.684` --> agent path.

---

## 2. Retrieval Cache (LRU)

**File:** `src/cognidoc/hybrid_retriever.py` (class `RetrievalCache`, lines 86-169)

### Why This Component Exists

A single retrieval pass involves: query classification (1 LLM call), vector search (Qdrant query + BM25), optional graph traversal, reranking (1 LLM call), and fusion. This costs ~1-3s and several thousand LLM tokens. In an interactive session, users frequently rephrase or repeat queries ("What is X?" followed by "Tell me about X"). Without caching, each variation that produces the same effective query would re-execute the full pipeline. The cache short-circuits this entirely.

### Advantages

- **Significant latency reduction**: a cache hit returns in < 1ms vs ~1-3s for a full pipeline execution.
- **Zero-dependency**: uses Python's `OrderedDict`, no external cache service (Redis, Memcached) needed. This keeps the deployment simple for a single-process application.
- **Transparent**: the `from_cache=True` metadata flag lets the UI and logs distinguish cached from fresh results.
- **Self-bounding**: the combination of max_size (50) and TTL (5 min) ensures the cache never grows unbounded or serves stale data indefinitely.

### Limitations and Possible Improvements

- **Exact match only**: "What is X?" and "what is X" produce different MD5 hashes. The cache does not normalize queries (lowercasing, stripping punctuation, stemming). Adding a normalization step before hashing would increase hit rates for trivially equivalent queries.
- **No semantic similarity**: "What is X?" and "Tell me about X" are cache misses despite being semantically identical. An embedding-based cache key (cosine similarity above a threshold) could catch these, at the cost of an embedding computation per lookup.
- **In-memory only**: the cache is lost on process restart. For long-running production deployments, persisting the cache (e.g., SQLite, like the embedding cache already does) would preserve it across restarts. However, the 5-minute TTL makes this less critical — most entries would expire anyway.
- **No invalidation on re-ingestion**: if documents are re-ingested while the app is running, cached results may reference stale content. An explicit `cache.clear()` after ingestion would solve this, but there is currently no hook between the ingestion pipeline and the query cache.

### Implementation

The cache is an `OrderedDict` (Python standard library) instantiated once as a module-level global (line 173). Each entry stores a tuple `(HybridRetrievalResult, timestamp)`.

**Parameters:**
- Capacity: 50 entries max
- TTL: 300 seconds (5 minutes)

**Why these values:** 50 entries is sized for a typical interactive session where a user asks 20-50 questions. Beyond that, the oldest entries are likely stale anyway. The 5-minute TTL balances freshness (the underlying index could change after re-ingestion) with usefulness (users often rephrase the same question within a few minutes). A longer TTL would risk serving stale results after an index update; a shorter one would defeat the purpose for slow exploratory sessions.

### Cache Key

The key is an MD5 hash of four concatenated parameters (line 108-111):

```
MD5("{query}|{top_k}|{use_reranking}|{strategy}")
```

MD5 is used here not for security but as a fast, fixed-length key generator. The alternative (using the raw concatenated string as key) would work but wastes memory on long queries. Collision risk is irrelevant at 50 entries.

The same query with different parameters (e.g., reranking on vs off) produces distinct cache entries. This prevents returning a non-reranked result when reranking was requested.

### LRU Behavior

- **On hit**: the entry is moved to the end of the `OrderedDict` via `move_to_end()`, marking it as most recently used.
- **On expiration**: the entry is deleted at read time (lazy eviction), not by a background timer. This keeps the implementation simple with no background threads.
- **On capacity overflow**: `popitem(last=False)` removes the entry at the head of the `OrderedDict` (least recently used).

### When the Cache is Bypassed

The cache is checked at the beginning of `retrieve()` (line 508-513), but is **skipped** in two cases:

1. **`metadata_filters` is set**: the query targets specific documents/sources, so cached results from a different filter context would be wrong.
2. **`pre_computed_routing` is provided**: the routing decision was made externally (typically by the agent), so the cached result may have used different routing.

When the cache returns a valid result, the **entire pipeline is skipped** (classification, retrieval, reranking, fusion) and the `HybridRetrievalResult` is returned directly with `from_cache=True` in metadata.

Results are cached at the very end of the pipeline (line 848), after fusion and compression.

### Statistics

The cache exposes `stats()` returning current size, hit rate, hits/misses. This is visible in the Gradio UI via the "refresh metrics" button and in debug logs.

---

## 3. Weighted Vector/Graph Fusion

**Files:** `src/cognidoc/query_orchestrator.py` (routing + weights), `src/cognidoc/hybrid_retriever.py` (execution + fusion)

### Why This Component Exists

CogniDoc has two fundamentally different retrieval systems: vector search (finds semantically similar text chunks) and graph retrieval (finds entities, relationships, and community summaries). Neither alone is sufficient for all query types. A factual question like "What is the deadline for project X?" is best answered by the exact document chunk containing that information (vector). A relational question like "How are X and Y connected?" is best answered by the knowledge graph that explicitly models entity relationships. The fusion mechanism decides how much to rely on each system per query.

Without fusion, the system would have to pick one retriever per query (losing information) or always use both with equal weight (adding noise from the irrelevant one and wasting latency).

### Advantages

- **Query-adaptive**: the weight configuration means a PROCEDURAL query gets 80% vector (where step-by-step instructions live) while a RELATIONAL query gets 80% graph (where connections are modeled). This is more nuanced than a binary "use vector or use graph" decision.
- **Graceful degradation**: the post-hoc confidence adjustment means a poorly performing retriever gets automatically down-weighted. If the knowledge graph is sparse or the query entities are not in the graph, the system compensates by boosting vector weight rather than returning a half-empty result.
- **Parallel execution**: running both retrievers concurrently means the latency cost of fusion is `max(vector_time, graph_time)` rather than `vector_time + graph_time`. In practice, this saves ~0.5-1s per query.
- **Skip logic reduces waste**: queries that clearly need only one retriever (high-confidence factual, purely exploratory) skip the other entirely, saving both latency and LLM tokens.

### Limitations and Possible Improvements

- **Fusion is binary, not proportional**: despite the weights (e.g., 0.7/0.3), the actual fusion concatenates both full contexts when both are included. A weight of 0.3 vs 0.7 makes no difference in the fused text — the LLM sees both sections in full. The weights only matter at the skip threshold boundary (< 15% = excluded). A more sophisticated approach would truncate the lower-weighted context proportionally (e.g., graph_weight=0.3 means only the top 30% of graph results are included).
- **No cross-retriever reranking**: after fusion, the vector and graph results are presented as separate sections. There is no unified reranking that interleaves the best results from both sources. A cross-retriever reranker could produce a more coherent context by ordering all results by relevance regardless of source.
- **Static weight table**: the weights are hardcoded per query type. Different corpora may benefit from different balances (e.g., a highly structured corpus with rich entity relationships might benefit from higher graph weights across the board). Per-corpus weight tuning or learned weights based on retrieval feedback could improve quality.
- **The confidence adjustment is symmetric**: both retrievers get the same +/-0.3 adjustment regardless of how poor the confidence actually is. A proportional adjustment (larger shift for very low confidence) would be more responsive.

### Phase A: Weight Assignment (`query_orchestrator.py`, lines 145-153)

A static `WEIGHT_CONFIG` table assigns vector/graph weights based on query type:

| Query Type | Vector | Graph | Mode |
|------------|--------|-------|------|
| FACTUAL | 0.7 | 0.3 | HYBRID |
| RELATIONAL | 0.2 | 0.8 | HYBRID |
| COMPARATIVE | 0.3 | 0.7 | HYBRID |
| EXPLORATORY | 0.1 | 0.9 | GRAPH_ONLY |
| PROCEDURAL | 0.8 | 0.2 | HYBRID |
| ANALYTICAL | 0.5 | 0.5 | HYBRID |
| UNKNOWN | 0.6 | 0.4 | HYBRID |

Three conditional overrides apply after the table lookup (`query_orchestrator.py`, lines 307-328):

1. **FACTUAL + confidence > 0.7**: forces vector=1.0, graph=0.0. High-confidence factual queries don't need the graph; skipping it reduces latency.
2. **EXPLORATORY**: forces vector=0.0, graph=1.0. Exploratory queries are best served by entity/community summaries.
3. **RELATIONAL with no detected entities**: rebalances to 0.5/0.5. If the classifier couldn't identify entities, the graph retriever won't know what to look for, so falling back to equal weights is safer.

### Phase B: Skip Logic and Parallel Execution

If a weight is **< 15%** (`skip_threshold=0.15` in `OrchestratorConfig`), the corresponding retriever is not executed at all. The 15% threshold is a pragmatic cutoff: at that weight, the retriever's contribution to the final context would be marginal, but its execution cost (graph traversal or vector search + reranking) is not. Skipping it saves ~0.5-1s of latency per query.

When both retrievers are needed, they run **in parallel** via `ThreadPoolExecutor(max_workers=2)` (`hybrid_retriever.py`, lines 756-773). When only one is needed, it runs alone.

### Phase C: Post-hoc Confidence Adjustment

After retrieval completes, `should_fallback()` (`query_orchestrator.py`, lines 351-398) adjusts weights based on the **confidence of the actual results**:

- Vector confidence low (< threshold) --> graph_weight += 0.3, vector_weight -= 0.3
- Graph confidence low --> vector_weight += 0.3, graph_weight -= 0.3
- Both low --> mode switches to ADAPTIVE

The +/-0.3 adjustment is large enough to meaningfully shift the balance (e.g., from 0.7/0.3 to 0.4/0.6) but not so large that it completely inverts the routing decision. It acts as a correction, not an override.

This compensates for cases where a retriever returns poor results. For example, if the graph has no matching entities for a RELATIONAL query, the vector weight gets boosted so the final context still contains useful information.

### Phase D: The Fusion Itself (`fuse_results`, lines 236-288)

The fusion is **not a numerical score interpolation** -- it is a **textual context concatenation**. Concretely:

1. If `graph_weight > 0` and graph returned results: a section `=== KNOWLEDGE GRAPH CONTEXT ===` is added containing entity descriptions, relationships, and community summaries. Source chunks are capped at `MAX_SOURCE_CHUNKS_FROM_GRAPH`.
2. If `vector_weight > 0` and vector returned results: a section `=== DOCUMENT CONTEXT ===` is added containing the reranked document chunks.
3. Source chunks are deduplicated via `dict.fromkeys()` (preserves order).

The weights serve as **inclusion/exclusion gates**, not as numerical multipliers on the content. If a weight is > 0, the full context from that retriever is included. The LLM then sees both sections and synthesizes the final answer, implicitly giving more attention to whichever section is more relevant.

The real impact of the weights is upstream: in the **skip logic** (whether a retriever runs at all) and in the **confidence adjustment** (whether to compensate for a weak retriever).

---

## 4. ReAct Agent Loop

**File:** `src/cognidoc/agent.py` (class `CogniDocAgent`)

### Why This Component Exists

The fast/enhanced path does a single retrieval pass: it retrieves the top-K chunks, reranks them, and generates an answer. This works for questions whose answer is contained within a few chunks. But some queries require information that cannot be gathered in one pass:

- **Comparisons** ("Compare X and Y"): need separate retrievals for X and for Y, then a synthesis step.
- **Multi-entity analysis** ("How are X, Y, and Z related?"): need entity lookups in the graph, possibly followed by vector retrieval for supporting evidence.
- **Aggregation questions** ("What are all the advantages of X?"): may need multiple retrieval passes with different phrasings to achieve better coverage.
- **Meta-questions** ("How many documents are in the database?"): cannot be answered by document retrieval at all — they need a dedicated tool.

The agent provides a controlled loop where the LLM can reason about what information it still needs and choose which tool to call next, rather than being limited to a single retrieve-then-generate pass.

### Advantages

- **Adaptive information gathering**: the agent decides at each step whether it has enough information or needs another retrieval. This produces more complete answers for complex queries.
- **Tool composability**: the 9 tools can be combined in any order. The agent might retrieve from vector, then look up an entity in the graph, then synthesize — a sequence that the fixed pipeline cannot express.
- **Transparency**: every step (thought, action, observation, reflection) is logged and displayed in the UI, making the reasoning process auditable. Users can see *why* the agent called a specific tool.
- **Safety net**: the forced conclusion at max_steps and the fallback to the fast path on error ensure the user always gets a response, even if the agent encounters issues.
- **Efficiency via prompting**: the system prompt aggressively targets "2-3 steps max" and discourages redundant tool calls. In practice, most queries resolve in 2-3 steps despite the 7-step ceiling.

### Limitations and Possible Improvements

- **No backtracking**: if step 3 leads to a dead end, the agent cannot revert to step 2 and try a different tool. It can only move forward, accumulating potentially irrelevant context. A state-machine architecture with explicit backtracking would allow the agent to prune unproductive paths.
- **~~Reflection is advisory, not binding~~** *(addressed)*: the reflection step now returns a structured `(text, should_conclude)` signal. When reflection determines enough information has been gathered (`ACTION: final_answer`), the loop exits immediately via `_force_conclusion` without an additional think step. See "Binding Reflection" below.
- **~~No parallel tool execution~~** *(addressed)*: the agent now supports up to 2 simultaneous tool calls per step using a numbered `ACTION_1`/`ACTION_2` format. See "Parallel Tool Execution" below.
- **~~Context accumulation is unbounded within a run~~** *(addressed)*: `get_trimmed_history()` now replaces `get_history_text()` in all LLM prompts. Older steps are reduced to thought + action only, while the last 2 steps retain full detail. See "Context Trimming" below.
- **Aggregate/exhaustive queries**: the agent retrieves top-K chunks per tool call. For questions like "how many X are mentioned across all documents?", multiple retrieval passes still only cover a fraction of the corpus. A dedicated exhaustive search tool or pre-computed aggregation indexes would be needed to answer these reliably (see discussion in the "Known Limitations" section).

### Code Structure

The agent loop is implemented as a single private generator `_run_loop()` that yields `(AgentState, message)` tuples for streaming progress and returns an `AgentResult`. The two public methods are thin wrappers:

- **`run()`**: consumes the generator silently (discards yielded events), returns the `AgentResult` directly.
- **`run_streaming()`**: delegates via `yield from`, propagating both the yielded events and the return value to the caller.

This avoids duplicating the loop logic across two methods. The streaming events are always generated (even in `run()`), but the cost is negligible since they are just string formatting — no I/O or LLM calls.

### Loop Structure (`_run_loop`)

```
+--- THINK + DECIDE ----> LLM generates a thought + chooses 1-2 tools
|         |
|         +-- final_answer?       --> END (return the answer)
|         +-- ask_clarification?  --> END (request clarification)
|         +-- actions = []?       --> END (break)
|         |
|    --- ACT ---------------> Execute tool(s) via ToolRegistry
|         |                   (parallel if 2 actions)
|    --- OBSERVE ------------> Store result(s) in context   \
|    --- REFLECT ------------> LLM evaluates sufficiency    / in parallel
|         |
|         +-- should_conclude?    --> _force_conclusion --> END
|         |
+------- loop (max 7 iterations) ---------------------------+
```

### Step 1: THINK + DECIDE (`_think_and_decide`)

The LLM receives:
- A **system prompt** (`SYSTEM_PROMPT`) listing available tools, language rules, and efficiency guidelines ("2-3 steps max", "one retrieval is usually enough").
- A **user prompt** (`THINK_PROMPT`) containing the original query + the trimmed history of previous steps (via `get_trimmed_history()` — see "Context Trimming" below).

The LLM must respond in one of two formats:

**Single action** (standard):
```
THOUGHT: <reasoning>
ACTION: <tool_name>
ARGUMENTS: <JSON>
```

**Parallel actions** (for independent operations like comparing X and Y):
```
THOUGHT: <reasoning>
ACTION_1: <tool_name>
ARGUMENTS_1: <JSON>
ACTION_2: <tool_name>
ARGUMENTS_2: <JSON>
```

Parsing (`_parse_thought_actions`) tries the numbered format first, then falls back to the single format. If the JSON is malformed, a fallback parser (`_parse_args_fallback`) attempts to extract key-value pairs via regex. Terminal actions (`final_answer`, `ask_clarification`) are filtered from parallel responses — if a terminal action appears alongside a non-terminal one, only the terminal action is kept.

The format is intentionally text-based rather than using native function calling. This forces the LLM to reason out loud in the THOUGHT section before choosing an action, producing better tool selection decisions.

### Step 2: Terminal Action Check

Before executing any tool, two cases exit the loop immediately:

- **`final_answer`**: the answer is extracted from the `answer` field in arguments (with a fallback to the first string value if the key is missing). Returns `AgentResult(success=True)`.
- **`ask_clarification`**: returns `AgentResult(needs_clarification=True)` with the clarification question. The UI displays this to the user.

### Step 3: ACT

For a single action, `self.tools.execute(action)` dispatches to the concrete tool implementation in `agent_tools.py`. For parallel actions (2 tools), both are submitted to the `_reflection_executor` thread pool and executed concurrently. Results are collected with a 30s timeout per tool. Observations from parallel actions are prefixed with `[tool_name]` and concatenated.

Each tool returns a `ToolResult` with:
- `observation`: text summary of the result (truncated for prompt inclusion)
- `success`: boolean
- `data`: structured data (documents, entities, etc.)

The 9 available tools:

| Tool | Purpose | Cached |
|------|---------|--------|
| `retrieve_vector` | Semantic search over document chunks | Yes (30 min) |
| `retrieve_graph` | Entity/relationship search in knowledge graph | No |
| `lookup_entity` | Get detailed info about one specific entity | No |
| `compare_entities` | Compare 2+ entities side by side | No |
| `synthesize` | Combine multiple contexts into a coherent answer | No |
| `verify_claim` | Fact-check a statement against the corpus | No |
| `database_stats` | Database metadata (document count, names) | Yes (30 min) |
| `ask_clarification` | Request more information from the user | N/A |
| `final_answer` | Return the final response | N/A |

### Step 4: OBSERVE + REFLECT in Parallel

This is the most architecturally interesting part. Two operations happen **simultaneously**:

**Main thread** -- stores useful context:
- `retrieve_vector`: top 3 document texts added to `gathered_context`
- `retrieve_graph`: graph context added
- `lookup_entity`: entity added to `entities_found`
- `synthesize` / `compare_entities`: result added to context

This is instantaneous (just list appends).

**Background thread** (`_reflect`) -- a separate LLM call with `REFLECT_PROMPT` that asks: *"Can you answer the query NOW? If YES, respond with ACTION: final_answer. If NO, respond with ACTION: continue."* Returns `Tuple[str, bool]` — the reflection text and a `should_conclude` flag parsed from the response.

The `ThreadPoolExecutor` is **module-level** (line 34), reused across agent calls -- no thread creation overhead per invocation. The main thread waits for the reflection future with a 30s timeout. Since context storage is near-instant and the LLM call takes ~0.5-1s, the parallelism saves the full duration of context storage (minimal) but more importantly overlaps the two operations cleanly.

After both operations complete, if `should_conclude` is True, the loop exits immediately via `_force_conclusion` (see "Binding Reflection" above).

### Binding Reflection

The reflection step now returns a structured signal: `_reflect()` returns `Tuple[str, bool]` where the boolean indicates whether the reflection concluded that enough information has been gathered.

The `REFLECT_PROMPT` explicitly asks the LLM to choose between two actions:
- `ACTION: final_answer` — enough information is available to answer the query
- `ACTION: continue` — critical information is still missing

The response is parsed via regex (`ACTION:\s*final_answer`). When `should_conclude` is True, the loop skips the next `_think_and_decide` call entirely and goes straight to `_force_conclusion()`, which synthesizes a final answer from the accumulated context. The result is marked `reflection_concluded=True` in metadata.

**Why this matters:** Without binding reflection, the agent could waste 1-2 extra steps after gathering sufficient information. The `_think_and_decide` LLM call might not "notice" the reflection in the history suggesting it should conclude, leading to redundant tool calls. Binding reflection saves one LLM call (~0.5-1.5s) and avoids accumulating irrelevant context.

**Why `_force_conclusion` instead of extracting the answer from the reflection:** The reflection prompt is designed for evaluation, not answer generation. It lacks the full system prompt, tool descriptions, and language instructions. Routing through `_force_conclusion` ensures the final answer is synthesized with the full context and proper formatting.

### Context Trimming

As the agent progresses through steps, the prompt history grows with each observation (retrieval results can be hundreds of tokens each) and reflection. Without trimming, step 7's prompt would contain all previous observations verbatim, risking context window pressure and diluting the signal with old, potentially superseded information.

`AgentContext.get_trimmed_history(keep_full=2)` addresses this with a heuristic approach:

- **Last 2 steps**: kept in full detail (thought, action, observation, reflection) — the LLM needs recent results to make informed decisions.
- **Older steps**: reduced to thought + action only — the LLM knows *what was done* and *why*, but not the raw results, which are already summarized in `gathered_context`.

This method is used in three places:
1. `_think_and_decide()` — the LLM sees trimmed history when deciding the next action
2. `_reflect()` — the reflection evaluates sufficiency based on trimmed history + gathered context (capped at 3000 chars)
3. `_force_conclusion()` — the forced conclusion includes both trimmed history (reasoning chain) and gathered context (information)

The original `get_history_text()` (full detail, no trimming) is preserved for logging and debugging.

**Why keep_full=2:** The last 2 steps represent the most recent reasoning cycle. The LLM needs to see the latest observation to avoid re-doing the same retrieval, and the previous step provides continuity. Beyond 2 steps back, the thought+action summary is sufficient context — the detailed observations are already captured in `gathered_context`.

**Latency cost:** Zero. This is pure string formatting — no LLM call, no computation beyond string slicing. The prompt is shorter, so if anything, LLM inference is slightly faster.

### Parallel Tool Execution

The agent can execute up to 2 independent tools simultaneously within a single step. This is primarily useful for comparison queries ("Compare X and Y") where two separate retrievals are needed.

**Format:** The system prompt instructs the LLM to use a numbered format (`ACTION_1`/`ARGUMENTS_1`, `ACTION_2`/`ARGUMENTS_2`) when two operations are independent. The prompt explicitly restricts parallel actions to non-terminal operations — `final_answer` and `ask_clarification` must be single actions.

**Execution:** When `_parse_thought_actions` returns 2 tool calls, they are submitted to the existing `_reflection_executor` thread pool (max_workers=2) and run concurrently. This reuses the same thread pool used for parallel reflection, since tool execution and reflection never overlap (tools complete before reflection starts).

**Data model:** `AgentStep` stores actions as a `List[ToolCall]` in the `actions` field. A backward-compatible `action` property (getter/setter) returns `actions[0]` for single-action code paths. `to_text()` and `get_trimmed_history()` iterate over all actions, so both full and trimmed history correctly display parallel steps.

**Observations:** When multiple tools return results, observations are prefixed with `[tool_name]` and joined with newlines. The reflection receives this combined observation as a single string, which is sufficient since `_reflect` already caps observation input at 1000 characters.

**Metadata:** `AgentResult.metadata` includes `parallel_steps` — a count of steps where more than one action was executed. This allows tracking how often the parallel path is actually used.

**Why max 2 actions:** Two is sufficient for the primary use case (comparing two entities). Allowing more would increase prompt complexity, make parsing less reliable, and risk the LLM submitting redundant or conflicting actions. The cap is enforced in the parsing loop (`range(1, 3)`).

**Why not always parallel:** Most queries only need one tool per step. The system prompt's efficiency guidelines ("one retrieval is usually enough") mean the LLM only uses parallel actions when it genuinely needs two independent pieces of information. The single-action format remains the default and requires no behavioral change.

**Latency impact:** For comparison queries, parallel retrieval saves one full step (~1.5-3s) by collapsing two sequential retrievals into one parallel step. The typical profile for comparisons drops from 4 steps to 3 steps.

### Forced Conclusion

If the loop reaches 7 iterations without a `final_answer`, `_force_conclusion` makes one last LLM call with the trimmed reasoning history and accumulated context (capped at 3000 characters) and asks for the best possible answer. The result is marked `forced_conclusion=True` in metadata.

### Fallback to Fast Path

If the agent fails entirely (exception), the calling code in `cognidoc_app.py` falls through to the fast path as a safety net. The user still gets an answer, just without multi-step reasoning.

### Why max_steps=7 and temperature=0.3

**max_steps=7**: Most queries resolve in 2-3 steps. 7 is a safety net for genuinely complex multi-entity comparisons that need several retrievals. Beyond 7, the accumulated context becomes noisy and the LLM tends to loop rather than converge. The prompt reinforces this by explicitly targeting "2-3 steps max" -- the limit exists as a hard ceiling, not as a target.

**temperature=0.3**: Lower than the default (typically 0.7-1.0) to make tool selection more deterministic. At higher temperatures, the agent occasionally makes erratic tool choices (e.g., calling `verify_claim` when `final_answer` is appropriate). 0.3 is low enough for consistent behavior while still allowing some flexibility in reasoning.

### Typical Execution Profile

- Most queries: 2-3 steps (one retrieval + final_answer)
- Complex comparisons: 2-3 steps (parallel retrieval of both entities + compare/synthesize + final_answer)
- Worst case: 7 steps + forced conclusion
- Per-step latency: ~1.5-4s (dominated by LLM calls)
- Total agent latency: ~5-12s for typical queries, ~3-8s for comparisons with parallel retrieval

---

## How the Components Connect

```
User Query
    |
    v
[Complexity Evaluator] ----score >= 0.55----> [ReAct Agent Loop]
    |                                              |
    | score < 0.55                                 | uses tools that call
    v                                              v
[LRU Cache] --hit--> Return cached result    [HybridRetriever.retrieve()]
    |                                              |
    | miss                                         |
    v                                              v
[Weight Assignment] --> [Parallel Retrieval] --> [Confidence Adjustment]
    |                   (vector || graph)            |
    v                                               v
[Fusion] --> [Optional Compression] --> [Cache Store] --> Response
```

The agent path and the fast/enhanced path share the same underlying retrieval infrastructure. The agent's `retrieve_vector` and `retrieve_graph` tools call the same `HybridRetriever`, but iteratively and with tool-specific parameters, rather than in a single pass. The LRU cache can benefit both paths: if the agent calls `retrieve_vector` with the same query twice, the second call hits the cache.

---

## Known Limitations (Cross-Cutting)

These limitations span multiple components and represent structural constraints of the RAG approach, not implementation bugs.

**Aggregate and exhaustive queries**: Questions like "how many X are mentioned?", "list all Y", or "enumerate every Z" require corpus-wide coverage. Both the fast path (top-K retrieval) and the agent path (multiple top-K retrievals) only sample a fraction of the corpus. The knowledge graph partially compensates (entities are exhaustively extracted at ingestion), but its coverage depends on extraction quality. There is currently no mechanism to warn the user when a response is based on incomplete coverage.

**Temporal and conditional queries**: Questions like "from what date did X change?" or "above what threshold does Y apply?" depend on the relevant information being in the retrieved chunks. If the answer is implicit (spread across multiple passages that must be cross-referenced), top-K retrieval may miss it. The agent can mitigate this through multiple retrievals, but cannot guarantee it will find dispersed information.

**No end-to-end feedback**: The pipeline has no mechanism to learn from user interactions. Complexity thresholds, fusion weights, and agent behavior are static. A feedback loop (user ratings, follow-up detection, answer quality metrics) could drive adaptive improvements to routing thresholds, weight tables, and agent prompts over time.
