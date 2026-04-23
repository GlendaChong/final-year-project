# Final-Year-Project: News Environment Perception-Assisted Scientific Journalism with LLMs

This repository contains the implementation of an advanced **Retrieval-Augmented Generation (RAG)** framework designed to transform dense scientific papers into accessible, real-world grounded news reports.

## 🌟 Core Innovation: Evolving-Rubric RAG Pipeline

The primary technical contribution of this project is the **Evolving-Rubric** architecture. Unlike static RAG pipelines, this system utilises a dynamic **Rubric Ledger** to manage the relationship between technical accuracy and journalistic engagement across three distinct phases:

### Phase 1: Search Query Synthesis
* **Technical Anchor Extraction**: Analyses the research paper's Abstract and Introduction to extract a Technical Anchor—the core breakthrough that serves as the foundation for the news story, and generate a search query for RAG. 

### Phase 2: Search-Guided Discovery Loop
* **Anchor Event Discovery**: Performs targeted searches for real-world "Anchor Events," such as industry news, policy changes, or societal impacts.
* **Pivot Query Logic**: If initial searches fail to find a relevant connection, the system generates a **Pivot Query** to search from a different industry angle, ensuring a "Missing Link" is established.

### Phase 3: Adversarial Drafting Loop
* **Journalistic Structure**: Generates a news article following the **Inverted Pyramid** structure.
* **Adversarial Critique**: An **Adversarial Technical Editor** (LLM Judge) critiques the draft against specific ledger constraints.
* **Iterative Evolution**: Failing drafts trigger **Corrective Rubrics**, leading to iterative rewrites until technical fidelity and engagement hooks are perfectly balanced.

---

## 📂 Project Structure

* **`/rag_pipelines`**: Core logic for various RAG strategies:
    * `generate_rag_one_shot_retrieval.py`: The baseline one-shot RAG approach.
    * `generate_rag_iterative_retrieval.py`: Iterative retrieval approach.
    * `generate_rag_utility_aware.py`: Utility-aware RAG approach.
    * `generate_rag_evolving_rubric.py`: Proposed Evolving-Rubric RAG approach.
* **`/nonrag` and `/nonrag_persona`**: Core logic for Non-RAG baseline pipelines. 
* **`rag_final`**: Contains outputs for generated outcomes and evaluation scripts used. 
* **`/prompts`**: Storage for "LLM-as-a-Judge" metric templates and generation personas.
* **`/ablation_results`**: Quantitative data and logs from architectural ablation studies.
* **`/nonrag`**: Baseline news reports generated without external context for comparative analysis.
* **`extracted_papers_summary_5.json`**: Dataset containing human-written news articles and their corresponding research paper's abstract and introduction. 

---

## 📊 Evaluation Framework

The project utilises a multi-dimensional evaluation suite consisting of **9 specialised sub-metrics**:

* **Fidelity**: Accuracy (1A) and Scientific Nuance/Distortion (1B).
* **Impact Framing**: Novelty Emphasis (2A) and Scientific Significance (2B).
* **Journalistic Quality**: Engagement Hook (3A), Logical Attractiveness (3B), and Call to Action (3C).
* **RAG Utility**: Contextual Relevance (4A) and Contextual Coherence (4B).

---

## 🛠️ Setup & Usage
Create a `.env` file in the root directory and configure your API keys: 
```
OPENROUTER_API_KEY=your_openrouter_key
SERPER_API_KEY=your_serper_key
JINA_API_KEY=your_jina_key
```

---

## Running the Pipeline 
To generate a news report using the Evolving-Rubric architecture: run 
```python rag_pipelines/generate_rag_evolving_rubric.py```.

You can also play around using the `demo.py` by running ```streamlit run demo.py```, a simple interface designed for rapid prototyping and live demonstrations of the Evolving Rubric architecture and the generated outputs. 
