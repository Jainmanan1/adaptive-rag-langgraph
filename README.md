# Adaptive RAG System

An adaptive Retrieval-Augmented Generation pipeline (Python, LangGraph, LangChain, Ollama, ChromaDB, Tavily) that dynamically routes each question to the right handling path — direct answer, local vector store retrieval, or web search — with corrective retries and multi-stage output grading.

Unlike a basic RAG pipeline that always follows a fixed `Question → Retrieve → Generate` path, this system decides *how* a question should be answered based on its content, and self-corrects when a retrieval or generation attempt doesn't hold up.

## 1. Overview

Depending on the query, the system can:

- Answer directly using the LLM (greetings, small talk — no retrieval needed).
- Retrieve information from a local Chroma vector store.
- Search the web when local documents are insufficient or the question is about current events.
- Rewrite the search query when a retrieval attempt fails, instead of blindly repeating it.
- Grade retrieved documents for relevance before using them.
- Detect potential prompt injection in retrieved content.
- Generate an answer using only approved, grounded context.
- Check whether the generated answer is actually grounded in the retrieved information.
- Check whether the final answer actually addresses the user's question.
- Retry failed retrieval/generation operations with exponential backoff and jitter, up to a bounded limit, before falling back gracefully.

## 2. Architecture

```
User Query
    │
    ▼
Input Validation
    │
    ▼
Query Router
    │
    ├──────────────► Direct LLM Answer
    │
    ├──────────────► ChromaDB Retrieval
    │                       │
    │                       ▼
    │               Relevance + Injection
    │                    Grading
    │                       │
    │                       ▼
    │                  Safe Context
    │                       │
    │
    └──────────────► Web Search
                            │
                            ▼
                    Relevance + Injection
                         Grading
                            │
                            ▼
                       Safe Context
                            │
                            ▼
                    Defensive RAG Prompt
                            │
                            ▼
                       LLM Generation
                            │
                            ▼
                    Grounding Grader
                            │
                            ▼
                      Answer Grader
                            │
                            ▼
                       Final Answer
```

If grounding or answer-quality grading fails, the graph rewrites the search query and retries (bounded to 2 attempts) before returning a graceful "couldn't verify a reliable answer" fallback rather than a hallucinated response.

## 3. Key Features

- Query validation (type, length, empty-input checks)
- Intelligent query routing (vectorstore / web search / direct answer)
- Persistent Chroma vector store, built by a separate ingestion script
- Semantic document retrieval
- Document relevance grading
- Web search fallback with query rewriting on retry
- Prompt injection detection on retrieved documents
- Untrusted-document isolation (retrieved content is explicitly treated as data, not instructions)
- Defensive RAG prompting
- Hallucination / grounding grading
- Answer quality grading
- Retry with exponential backoff + jitter, distinguishing retryable vs. non-retryable errors
- Ollama request timeout handling
- Idempotent document ingestion (separate from serving)
- Source tracking and attribution in the final output

## 4. Tech Stack

Python, LangGraph, LangChain, ChromaDB, Ollama (`llama3.1`, `nomic-embed-text`), Tavily Search, Pydantic, httpx.

## 5. Key Findings & Debugging Notes

During development, targeted testing surfaced several non-obvious failure modes:

- **Temporal grounding failure.** Questions like "who won the last Super Bowl" initially returned outdated answers, because neither the LLM nor the search tool had any notion of the current date. Fixed by injecting the current date into the generation prompt per-request (rebuilt fresh on each call, not cached at startup) and biasing web search toward recent content.

- **Time-window false negatives.** A fixed `time_range="year"` filter on web search improved recency for annually-occurring events (Super Bowl) but caused false negatives for less-frequent events — e.g., a Cricket World Cup, held every 2-4 years — since the correct answer fell outside the search window entirely. Removed the fixed time constraint in favor of a `topic="news"` bias combined with date-anchored prompting, which generalizes across events of different frequencies.

- **Partial fabrication in generated lists.** Testing revealed the model would sometimes generate a list where one item was genuinely grounded in retrieved context and several others were plausible-sounding fabrications — a more subtle failure than full hallucination, since the single real item lent unwarranted credibility to the invented ones. Mitigated with an explicit "do not add details not present in the context, even if generally true" instruction in the generation prompt.

- **Routing instability from redundant instructions.** Duplicating routing criteria across both a system preamble and a schema field description caused the router to consistently invert its decisions on test questions. Consolidating routing logic into a single, unambiguous instruction location resolved it.

- **Native library crash during vector store writes.** A silent process crash (no Python traceback, no error message) during batch embedding was traced via Windows Event Viewer to a DLL conflict (`0xc0000005` access violation) between `chromadb` and a `scikit-learn` dependency pulled in by an unrelated experiment. Resolved by removing the conflicting package. This failure mode is invisible to normal Python-level debugging (logging, try/except) since it occurs below the interpreter level.

- **Retry logic scoped to the wrong exception types.** An initial retry wrapper only caught `httpx`-level exceptions, but the Ollama Python client wraps HTTP errors in its own `ResponseError` type — meaning transient Ollama failures were silently treated as non-retryable. Fixed by explicitly catching the Ollama client's error type alongside network-level exceptions.

## 6. Known Limitations / Production Gaps

This project is a working, debugged prototype, not a hardened production deployment. Notable gaps:

- Ollama serves requests from a single local instance; not suitable for concurrent production traffic without moving to a serving solution built for concurrency (vLLM, TGI) or a hosted API.
- No automated test suite — validation was done via targeted manual testing against known-answer questions, not a regression-tested golden set.
- No caching layer; repeated or near-identical questions re-run the full pipeline.
- Document and web-result grading are parallelized via `.batch()`, but end-to-end latency for a single question can still reach 30-90+ seconds depending on retry depth.
- No structured tracing (e.g., LangSmith) — observability is currently log-based with per-request correlation IDs.

## 7. Setup

**Prerequisites**
- [Ollama](https://ollama.com) installed and running locally
- Pull required models: `ollama pull llama3.1` and `ollama pull nomic-embed-text`
- A [Tavily](https://tavily.com) API key

**Install and run**
```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root:
```
TAVILY_API_KEY=your_key_here
```
Build the vector store (run once, or whenever source documents change):
```bash
python ingest.py
```
Run the pipeline:
```bash
python main.py
```

## 8. Example

```
Question: "What are the types of agent memory?"

Answer: Agent memory can be categorized into sensory memory, short-term
(working) memory, and long-term memory, which is further split into
explicit (episodic and semantic) and implicit (procedural) memory.



Sources:
- https://lilianweng.github.io/posts/2023-06-23-agent/
```


## 9. Evaluation (RAGAS)

Built a RAGAS-based evaluation layer (`evals/`) on top of the pipeline, using
llama3.1 (via Ollama) as the judge LLM, testing against a small hand-curated
question set covering all three routing paths.

### Setup note
Hit an unpatched upstream bug in `ragas` (both 0.3.9 and 0.4.3) where an
unconditional import of `ChatVertexAI` from a path removed in current
`langchain-community` versions breaks `import ragas` entirely — see
[ragas#2745](https://github.com/vibrantlabsai/ragas/issues/2745). Worked
around by patching the import in the installed package to fall back
gracefully when Vertex AI isn't installed, matching the fix in the project's
own [open PR #3017](https://github.com/vibrantlabsai/ragas/pull/3017).

### Findings

**Faithfulness returns NaN.** On a question with genuinely weak retrieval
(2 of 4 retrieved chunks were page navigation/boilerplate, not article
content), the Faithfulness metric consistently returned `NaN` rather than a
low score — likely because its statement-decomposition step couldn't extract
verifiable claims from an answer that wasn't well-grounded in the (mostly
irrelevant) context. Notably, this NaN is itself a useful signal: it
correctly flagged a real retrieval problem that the other three metrics
scored past without comment.

**ContextPrecision/ContextRecall scored a suspicious 1.0 on the same weak
context.** On the identical retrieved chunks that produced the Faithfulness
NaN, ContextPrecision and ContextRecall both returned perfect 1.0 scores —
directly contradicting manual inspection, which confirmed half the chunks
were boilerplate with no relevant content. This suggests llama3.1 8B, used
as the judge, is not reliably discriminating context relevance — consistent
with this project's other findings on the model's limits for structured
judgment tasks (see routing instability, fabricated-list findings above).

**Takeaway:** automated eval scores from a small local judge model should not
be trusted without spot-checking against the actual retrieved content — a
"passing" score can mask a real retrieval failure that a stricter metric
(or a human) would catch.

### Latency
A single question with 3 metrics took ~14 minutes end-to-end using llama3.1
as judge — impractical for frequent or large-scale eval runs without a
faster or hosted judge model.
