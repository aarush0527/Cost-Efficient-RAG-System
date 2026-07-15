# Cost-Efficient RAG System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

</p>

<p align="center">
A modular Retrieval-Augmented Generation (RAG) framework built around reproducible evaluation, efficient document indexing, grounded question answering, and low-cost vector search.
</p>

---

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline capable of ingesting heterogeneous document collections, indexing them inside an embedded vector database, retrieving relevant context using semantic search, and generating grounded responses with citations.

Rather than focusing solely on answer generation, the project emphasizes the engineering aspects of building a production-style retrieval system. Every major component—including ingestion, indexing, retrieval, generation, evaluation, logging, and cost analysis—is modular, configurable, and independently testable.

Unlike many demonstration RAG applications that rebuild the index on every execution, this implementation performs **deterministic idempotent ingestion**, allowing unchanged documents to be skipped while only modified content is re-embedded. This significantly reduces unnecessary computation and enables efficient incremental updates.

The project also includes a complete evaluation framework measuring retrieval quality, answer quality, latency, and infrastructure cost, making it suitable not only as a question-answering system but also as a reproducible experimental platform for Retrieval-Augmented Generation research.

---

# Key Features

### Document Processing

- Supports **PDF**, **HTML**, and **Markdown** documents
- Header-aware recursive chunking
- Configurable chunk size and overlap
- Deterministic chunk identifiers
- Incremental document updates
- Automatic removal of stale vectors

---

### Retrieval Engine

- Embedded **Qdrant** vector database
- Configurable Top-K retrieval
- Metadata-aware search
- Multiple embedding backends
    - Sentence Transformers
    - spaCy
    - Deterministic Hashing (testing)
- Configurable retrieval threshold

---

### Grounded Generation

- Groq-powered LLM generation
- Context-aware prompting
- Source citations
- Confidence gate preventing unsupported answers
- Stub mode for offline testing
- Independent generator and evaluation models

---

### Evaluation

Retrieval metrics

- Hit Rate
- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG
- Context Precision

Answer evaluation

- Faithfulness
- Relevance
- Exact Match
- Token F1

Performance

- Retrieval latency
- End-to-end latency
- Token usage
- Infrastructure cost comparison

---

### Engineering

- FastAPI REST API
- CLI interface
- Environment-based configuration
- Structured JSON logging
- Automated evaluation pipeline
- Comprehensive unit and integration tests
- Modular architecture
