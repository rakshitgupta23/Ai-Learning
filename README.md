# AI Full Stack Learning Roadmap (Rakshit)

## Layer 0: Mental Model

-   LLM is probabilistic
-   Needs structure and validation

## Layer 1: Controlled LLM

-   FastAPI endpoint
-   Pydantic validation
-   Structured JSON output
-   Parsing & cleaning

## Layer 2: Reliability

-   Error handling
-   Retries
-   Fallback responses

## Layer 3: RAG (Retrieval-Augmented Generation)

-   Keyword vs semantic search
-   Embeddings + cosine similarity
-   Chunking + overlap
-   Ranking + filtering
-   Multi-chunk reasoning
-   Query rewriting
-   Multi-query retrieval (optional)
-   Context optimization
-   Failure analysis

## Layer 4: Memory Systems

### Short-term memory

-   Chat history

### Long-term memory

-   User facts storage
-   Smart extraction using LLM

### Memory Retrieval

-   Embedding-based memory search
-   Only relevant memory passed

## Layer 5: Orchestration (Started)

-   Tool calling
-   Decision layer (LLM decides action)
-   Example: calculator tool

------------------------------------------------------------------------

## Current Level

-   Built full RAG system
-   Added memory + personalization
-   Implemented tool calling basics

------------------------------------------------------------------------

## Next Steps

-   Multi-tool orchestration
-   Agent workflows
-   Evaluation systems
