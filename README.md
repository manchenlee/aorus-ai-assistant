# AORUS AI Assistant

This repository contains the implementation of the "AORUS AI Assistant," an AI-powered conversational agent designed to answer hardware specification queries for the AORUS MASTER 16 series laptops.

## Environment

* **Python:** 3.11.5
* **Package Manager:** `uv`

## Setup

First, ensure you have `uv` installed. Then, create the virtual environment and install all dependencies defined in the project:

```bash
uv venv --python 3.11.5

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

uv sync
```

### Knowledge Base Parsing & Model Setup

Run the following commands to scrape the hardware specifications from the [GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp) official page and download the necessary models:

```bash
uv run -m scripts.specs_parser
uv run -m scripts.setup_models

```

### Running the AI Assistant

Execute the command below to start the interactive CLI and chat with the AORUS AI Assistant in real-time. Type `exit` to quit the application.

```bash
uv run -m src.main

```

### Synthetic Data Generation

Use the `gemini-2.5-flash` model to generate synthetic evaluation data. Before running this script, please make sure to insert your Gemini API key into the `.env` file (e.g., `GEMINI_API_KEY=your_api_key_here`).

```bash
uv run -m scripts.gen_test_data

```

### Evaluation

Run the evaluation pipeline. The test is divided into two stages: Stage 1 (Quantitative tests) and Stage 2 (Qualitative Non-LLM NLP metrics tests).

You can choose to run the entire pipeline (`all`) or start from a specific stage (`st1` or `st2`). If you are starting from Stage 2, you must provide the JSONL result file generated from Stage 1.

```bash
uv run -m src.evaluate --eval_mode nonllm --stage <stage_to_start> --res <st1_result_jsonl_file>

```

*(Example: `uv run -m src.evaluate --eval_mode nonllm --stage st2 --res aorus_test_results_20260307.jsonl`)*

---

## Methodology & Evaluation

### Inference Engine Selection
I chose `llama-cpp-python` over `vLLM` to satisfy the project's stringent VRAM constraints. Its support for GGUF-formatted models ensures high-performance inference within a limited memory footprint.

### Chunking

The GIGABYTE AORUS MASTER 16 AM6H dataset covers 3 specific models and 17 different hardware specifications. I structured this knowledge into 21 distinct chunks: 17 chunks comparing specific specs across models, 3 chunks providing a comprehensive overview for each model, and 1 final chunk summarizing the differences between all models.

To handle mixed-language inputs (Chinese and English) and the use of colloquialisms or synonyms in user queries, I implemented a `synonyms.json` mapping. During the chunking phase, I perform a keyword substitution to replace terms with standardized, mixed-language terminology.

However, since this substitution can negatively impact the naturalness and formatting of the generated response, I prepared two parallel sets of chunks. The retriever performs its search on the "synonym-substituted" chunks to find the correct index. Once the best match is identified, the corresponding "unaltered" chunk (raw text) is retrieved by that index and passed to the inference model for generation.

### Retrieval

Considering that user queries will frequently contain a mix of Chinese and English, I chose `paraphrase-multilingual-MiniLM-L12-v2` as the embedding model. It excels in multilingual semantic understanding and operates with high efficiency, even in CPU-only environments.

A specialized hardware AI assistant must be able to instantly grasp the semantic intent of a question while precisely capturing exact keywords (such as model numbers or specific metrics). To achieve this, I implemented a **Hybrid Search** pipeline combining Dense Vector Search and Sparse Keyword Search.  

The retrieval workflow operates as follows:  

1. **Query Normalization:** The user's query is first processed using the `synonyms.json` dictionary to map colloquialisms or alternative names to standardized, mixed-language terminology.
2. **Semantic Search (FAISS):** The normalized query is encoded into embeddings. I utilized **FAISS** to calculate the L2 distance between the query and all chunks, generating a comprehensive ranking based on semantic similarity.
3. **Keyword Search (BM25):** Simultaneously, the normalized query is tokenized using `jieba`. The **BM25** algorithm then scores and ranks all chunks based on exact keyword frequencies, ensuring critical hardware terms are not missed.
4. **Reciprocal Rank Fusion (RRF):** To integrate the strengths of both semantic and keyword searches, I applied the RRF algorithm. The final score for each chunk is calculated using the formula $\frac{1}{k + Rank_{Vector}} + \frac{1}{k + Rank_{BM25}}$ (where the smoothing constant `k=60`). This generates the final, highly accurate sorted ranking.  

Finally, the top 3 highest-scoring chunks are retrieved and passed to the LLM (generator) as the authoritative knowledge base context for generating the response.

To prevent the AI assistant from hallucinating or forcibly answering irrelevant questions (Out-of-Scope queries) using unrelated knowledge base data, I implemented a **strict safeguard**. During the final RRF extraction phase, I evaluate the original FAISS vector distance of the top chunks against a predefined Distance Threshold.
If a chunk's semantic distance exceeds this threshold, it is filtered out. If no chunks meet the threshold criteria, the retriever simply returns an empty array. This intentionally triggers the LLM's refusal guardrails, allowing the assistant to gracefully decline the question.

### Generation

Considering the strict **4GB VRAM** constraint, the system utilizes `Qwen3-4B-Instruct` in `Q4_K_M GGUF` quantization format. This setup maintains a memory footprint of approximately 2.5 GB while delivering robust bilingual reasoning capabilities and high instruction-following precision.

To ensure the assistant provides concise and deterministic technical specifications, the generation parameters are strictly tuned:

* **Deterministic Sampling:** Set `temperature=0.1` and `top-p=0.5` to minimize hallucination and variance.
* **Strict Grounding:** The System Prompt mandates that the assistant must rely exclusively on the provided Knowledge Base for fact-based responses.
* **Safety Guardrails:** Input length validation is performed pre-inference to prevent token overflow and system instability.

#### 1. Dual-Tag Reasoning & Stream Processing

To leverage the model's reasoning potential, a structured prompt forces the LLM to generate responses using `<Draft>` and `<Answer>` tags:

1. **Thinking Process:** The LLM first outlines key points in the `<Draft>` section.
2. **Final Response:** The actual reply is synthesized within the `<Answer>` section.
3. **Stream Handling:** Since streaming returns text character-by-character, it is impossible to split tags or perform **OpenCC** conversion instantly. I implemented a **Sliding Window + Buffer** mechanism that temporarily stores incoming characters to detect tag boundaries and batch-convert segments before flushing them to the CLI, ensuring seamless real-time interaction.

####  2. Robustness & Abnormal Query Handling

The system handles non-standard queries through context-aware prompt injection and dynamic prefixing. Before generation, the system detects the query's language and injects specific instructions based on the query category:

* **Categories:** Toxic (Abuse), Chitchat, Out of Scope, Missing Model (Ambiguity), Competitor (Comparison), and No Data (Knowledge Gaps).
* **Dynamic Response:** By enforcing specific response prefixes for each category (e.g., "I'm good, thank you" for Refusals or "Searching for specifications..." for specific lookups), the assistant maintains a consistent persona and avoids "answering for the sake of answering."

### Synthetic Data Generation

To rigorously evaluate the system, I utilized the `gemini-2.5-flash` model to synthesize a comprehensive testing dataset of 100 queries. This dataset is evenly split into **50 Normal QA pairs** (testing factual retrieval and reasoning) and **50 Abnormal QA pairs** (testing guardrails and robustness).

The queries are strictly grounded in the official specifications and disclaimer documents, and are distributed across the following 10 specific categories:

**Normal Queries (Factual & Reasoning - 50 items)**

* **Fact Extraction (15):** Direct inquiries targeting specific hardware metrics (e.g., "What is the refresh rate of the BXH?").
* **Correction (15):** Queries containing false premises to verify that the assistant corrects the user rather than hallucinating agreement (e.g., "I heard the BZH only has a 60Hz screen?").
* **Summarize & Compare (10):** Complex questions requiring cross-referencing and comparing specifications across the three different models.
* **Positive Yes/No (10):** Validating correct assumptions to test the model's confirmation accuracy.

**Abnormal Queries (Robustness & Guardrails - 50 items)**

* **Missing Model (10):** Ambiguous questions lacking a specific model identifier, designed to trigger the assistant to ask for clarification.
* **Competitor (10):** Bait questions involving competing brands (e.g., ROG) to test the assistant's ability to politely refuse comparisons and pivot back to AORUS products.
* **No Data (10):** Inquiries about details absent from the knowledge base (such as warranty or pricing) to ensure the system admits a lack of information instead of inventing answers.
* **Out of Scope (7):** Completely irrelevant topics (e.g., weather inquiries) to trigger the standard out-of-scope refusal guardrails.
* **Chitchat (8) & Toxic (5):** Social greetings, conversational closures, and abusive or provocative language to evaluate the assistant's persona consistency, politeness, and safety filters.

To accurately simulate real-world user behavior, the dataset incorporates mixed-language inputs (Traditional Chinese, English, and "Chinglish") and intentionally includes 20% of queries using colloquial or fuzzy hardware terminology (e.g., using "RAM" instead of "Memory Specification").  

### Evaluation

*(Explain your LLM-free evaluation approach, specifically how you use NLI for Faithfulness and SpaCy/Regex for Entity Overlap to ensure zero hallucination on hardware specs.)*