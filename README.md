# Verbose ListOps Evaluation Benchmark

Evaluating Large Language Models' (LLMs) ability to perform complex, multi-step reasoning within long, narratively structured contexts is crucial for advancing applications like extracting predictive signals from large volumes of unstructured text (e.g., qualification signals from sales transcripts), yet existing benchmarks often focus on simpler retrieval tasks or lack coherent distractors and independent controls for context length versus task complexity. 

This deeper reasoning is challenging as current LLMs struggle to track hierarchical dependencies, filter semantically related but computationally irrelevant narrative details, and follow embedded computational logic over extended sequences, even when capable of the core task in isolation; the lack of benchmarks combining hierarchical computation, coherent narrative distractors, dual difficulty controls across context length and problem difficulty, and deterministic verification hinders progress.

We introduce Verbose ListOps, a novel benchmark designed to address this gap. It programmatically transforms a ListOps hierarchical computation task into lengthy, coherent narratives generated via LLM-driven post-order Abstract Syntax Tree (AST) traversal, featuring strict agentic validation of numerical content at each step. This provides independent control over context length (via padding) and intrinsic task complexity (via AST structure), alongside deterministic ground truth evaluation. 

Initial experiments demonstrate that current state-of-the-art LLMs (e.g., GPT-4.1) universally fail Verbose ListOps problems at modest (~10,000 token) context lengths, despite easily solving their identical standard ListOps representations, empirically verifying the benchmark's difficulty and the significant challenge posed by narrative-embedded reasoning.

The benchmark's design, grounded in the deterministic ListOps task and enforced by strict agentic number validation during narrative generation, provides a verifiable measure of an LLM's ability to perform structured computation embedded within natural language. This isolates and tests core capabilities—such as identifying relevant operands, tracking conceptual results of sub-tasks, and resisting distraction—required for downstream applications like automated signal extraction from large text corpora. Code and generated data will be made publicly available.


## Overview

Welcome to the Verbose ListOps Evaluation Benchmark repository. This project provides tools to agentically generate synthetic multi-hop QA evaluation tasks of variable difficulty for Large Language Models (LLMs). The benchmark assesses reasoning capabilities when faced with tasks embedded within extremely long, narratively structured, and potentially distracting contexts.

It builds upon the [2018 ListOps](https://arxiv.org/abs/1804.06028) benchmark by first, **reversing the traversal order** of a ListOps problem, and then transforming its concise, symbolic problem representation into lengthy, natural language narratives generated dynamically using an LLM. The core Python script, `verbose-listops.py`, orchestrates this process.

Currently, only GPT-4.5 and Gemini 2.5 Pro are able to generate this benchmark. OpenAI's o(x), GPT, and Anthropic's family of models will have <50% success rates in generating samples.

Despite sharing the exact core task of the ListOps benchmark, current SOTA models universally fail verbose-listops narratives—even with multiple-shot prompting—at just 10,000 tokens, whilst successfully solving their identical standard ListOps representations. However, on easier problems, few-shot prompting has a significant effect on improving success rate.

Standard long‑context benchmarks typically emphasize “needle‑in‑a‑haystack” retrieval—locating a specific fact buried in a long document. In contrast, Verbose ListOps deterministically weaves explicit multi‑step operations (e.g., SUM, MIN, MAX) and their required operands into a narrative. Each story interleaves these operations with semantically related filler sentences, so a model must first identify and sequence the specified operations correctly before performing the computation. Though every narrative is dynamically generated with an LLM, strict deterministic tests guarantee reproducibility and soundness of semantic reasoning.

Due to the nature of this computational task and its simulation of real-world human behaviours, this dataset is especially well-suited to understanding whether frontier models can excel at extracting and scoring arbitrarily defined predictive signals from large corpora of unstructured text, such as qualification signals from sales transcripts. Today's LLMs already excel at LLM-as-Judge tasks, and this benchmark attempts to understand whether they are able to synthesize multiple simple single-output generation tasks across long contexts.

## Comparison with Similar Benchmarks

While Verbose ListOps has a unique combination of features (ListOps core task, LLM-generated narrative context, specific padding mechanism), several other benchmarks evaluate related aspects of long-context reasoning. Here's a comparison:

| Feature            | verbose‑listops             | LongReason (2025) | BABILong (2024) |
| :----------------- | :-------------------------- | :---------------- | :-------------- |
| **Core Task** | ListOps (Hierarchical list operations) | Diverse (Reading Comp, Logic, Math) | bAbI (Simple reasoning tasks) |
| **Context Gen.** | LLM-generated narrative around ListOps AST | LLM expansion of seed tasks | bAbI facts hidden in PG19 text |
| **Difficulty Knobs** | 1. Context Length (via padding) <br> 2. ListOps Complexity (depth, branching) | Context Length Only | Context Length Only |
| **Coherence** | High (LLM generates related story beats & padding) | Moderate (Risk of drift in expansion) | Low (Random book sentences as distractors) |
| **Reasoning Type** | Hierarchical, sequential computation within narrative | Varies by seed task | Simple retrieval, deduction (bAbI patterns) |
| **Scalability** | Configurable contexts (e.g., > 50K tokens) | Controlled lengths (8K-128K) | Extreme scalability (Millions of tokens) |
| **Verification** | Deterministic AST evaluation; Strict number validation in narrative via agentic checks | LLM auto-verification | Based on known bAbI answers |
| **Pros** | Dual difficulty knobs, Coherent expansion, Deterministic & Validated, Hierarchical logic | Diverse tasks, Controlled lengths, Auto-verified | Extreme scale, Deterministic seeds, Multi-task coverage |
| **Cons** | Domain-specific (ListOps), Pipeline complexity (LLM calls), Potential narrative noise | Length-only tuning, Fixed complexity, Synthetic coherence risk | Simplistic reasoning, Unrealistic distractors, Single knob |

Among these benchmarks, verbose‑listops most closely mirrors LongReason’s approach. Unlike LongReason—which only varies difficulty by context length—verbose‑listops offers two levers: context length and the intrinsic complexity of the ListOps problem it narrativizes.

## The Core ListOps Task

The underlying computational task remains identical to the original ListOps benchmark. Models processing the generated narrative must effectively:

1.  **Identify and parse nested operations** on lists embedded within the story.
    *   Supported operations: `MAX`, `MIN`, `MED` (median), `SUM`, `SM` (sum modulo 10), `AVG` (integer average, floored).
    *   Operands are integers (configurable range, e.g., 1-100), described textually and numerically within the narrative *only when they are direct inputs to the current operation*.
2.  **Evaluate these nested operations hierarchically** according to the narrative structure, which follows a post-order (inside-out) evaluation of the underlying ListOps problem.
3.  **Produce a single integer** as the final result based on the root operation.

ListOps fundamentally tests a model's ability to understand hierarchical structures and perform sequential reasoning.

## Design Goal: Testing Large Context Windows with Narrative Complexity

The primary motivation remains to rigorously evaluate model performance on tasks involving extremely large context windows. This implementation focuses on achieving verbosity and complexity through LLM-driven narrative generation that logically follows the computational structure. The **post-order generation** ensures that the narrative complexity unfolds naturally, mirroring how nested tasks are often approached.

Standard benchmarks may not sufficiently stress a model's ability to:

*   Maintain coherence and track dependencies over tens of thousands of tokens within a continuous story.
*   Extract relevant operational information (operators, direct operands) from narrative descriptions interwoven with characters, setting, and plot elements, while correctly handling the *conceptual results* of previously completed sub-tasks.
*   Perform multi-step reasoning when instructions unfold sequentially across a vast input sequence.
*   Resist distraction from contextually relevant but computationally irrelevant "padding" generated by the LLM.

This benchmark aims to address this by programmatically generating long narratives where the ListOps task is the hidden "logic puzzle" the model must solve. Target context lengths can reach configurable limits (e.g., 10,000 tokens or more, adjustable via parameters).

## Methodology: From ListOps AST to Generated Narrative

The `verbose-listops.py` script implements the following process:

1.  **Random AST Generation:** A random ListOps Abstract Syntax Tree (AST) is constructed (`build_random_ast`) defining the nested structure and operations. Parameters control complexity.
2.  **AST Evaluation:** The AST is evaluated (`eval_node`) to determine the ground truth answer and pre-calculate intermediate results for all nodes.
3.  **World Building:** An initial LLM call (`generate_world`) creates a fictional context (characters, genre, setting, primary object).
4.  **Thematic Naming (Optional):** If enabled (`USE_OWNERSHIP_NARRATIVE`), the script traverses the AST post-order, using LLM calls (`generate_owner_name_with_llm`) or fallback logic to assign thematic names (e.g., "Kaelen's Risky Bet", "Sector Gamma Scan Results") to the conceptual outcome of each operation node. This aids narrative coherence.
5.  **Narrative Generation (`generate_narrative` using `_generate_narrative_recursive`):**
    *   The script performs a **post-order traversal** of the AST.
    *   For each operator node, it first recursively processes all its children, generating their corresponding narrative scenes.
    *   After processing children, it prompts the LLM to write a "story beat" for the *current* operator node. This prompt includes:
        *   The world context.
        *   The specific ListOps operation (`op`) and its thematic name (`owner_name`).
        *   Its *direct* atomic operands (numbers introduced specifically for this operation).
        *   References to the *conceptual results* of its child operations (using their thematic names).
        *   The intermediate result (`value`) for *this* node (for certain ops like SUM, AVG, SM where the result itself needs narrative grounding).
        *   **Crucially:** Instructions to narrate the application of the current operation using the direct operands and the *outcomes* (referred to by name) of the already-narrated child operations, ensuring a forward narrative flow.
        *   **Strict Number Validation Rules:** The prompt explicitly lists numbers that MUST be included (direct atoms/result for this beat), numbers that MUST BE AVOIDED (forbidden numbers from previous steps), and MAYBE allowed numbers (like '1' or an operand count if not forbidden), stating NO OTHER numbers are allowed in this specific scene.
    *   A `GenerationContext` dataclass manages shared state (like tokens used, previously introduced numbers, owner map, and scene snippets) across recursive calls.
6.  **Narrative Padding:** After generating a story beat, the script may make additional LLM calls to generate "padding" paragraphs. Padding is prompted to continue the last scene descriptively *without* introducing any numbers or advancing the core calculation plot.
7.  **Token Management:** The process respects configurable token limits (total, beat, padding).
8.  **Question Generation:** A final prompt asks for the result of the top-level operation, referencing the primary object from the world context.

The result is a single block of text containing the full narrative followed by the target question and judging instructions.

## Why Verbose ListOps (LLM-Generated Version)?

This approach provides a challenging test case where the core task is simple, but requires the evaluated model to:

*   Read and understand a long, coherent narrative generated by another LLM.
*   Identify and isolate the embedded computational steps, respecting their sequential, inside-out resolution within the story.
*   Track the hierarchy and dependencies of the operations as presented narratively, using thematic names or context to link steps.
*   Correctly identify and use direct numerical inputs for each step while ignoring numerical results from prior steps mentioned conceptually.
*   Answer a direct question about the final result of the embedded ListOps task.

## Code Implementation Details

The `verbose-listops.py` script includes:

*   **Dependencies:** `openai` (or potentially `anthropic`), `tiktoken`, `inflect`.
*   **Core Logic:** AST generation, evaluation, world generation, thematic naming (optional), post-order narrative beat/padding generation via LLM calls, number validation (`make_number_validator`, `extract_numbers_from_text`), question generation.
*   **`GenerationContext`:** Dataclass managing state during recursive post-order generation.
*   **Configuration:** Constants/`Config` dataclass control API model, token limits, AST structure, retry behavior, naming options, etc.
*   **Logging:** Detailed logs (`llm_turns.log`, `verbose_listops.log`) capture prompts, generations, validation steps, and errors.
*   **API Key:** Requires `OPENAI_API_KEY` (or equivalent) environment variable.
*   **Error Handling:** Includes retry logic (`with_retry`, `retry_api_call`) for API calls.

## Usage

1.  **Setup:**
    *   Ensure Python 3.x is installed.
    *   Install required libraries: `pip install openai tiktoken inflect` (adjust `openai` if using a different provider).
    *   Set the API key environment variable: `export OPENAI_API_KEY='your-api-key'`
2.  **Run the Script:** Execute `python verbose-listops.py`.
3.  **Output:**
    *   Generated samples (narrative + question + metadata) are appended to the JSONL file specified by `OUTPUT_FILENAME` (e.g., `gpt4.5-verbose_listops_dataset_ultra_strict_v4.jsonl`).
    *   Detailed logs are saved in the `logs/` subdirectory.
4.  **Evaluation:**
    *   Provide the full `narrative_prompt` text from a generated sample as input to the LLM you want to evaluate.
    *   Parse the model's response to extract the predicted single-digit answer.
    *   Compare the prediction against the `ground_truth` value in the sample data. Calculate accuracy or other relevant metrics.

## Example Conceptual Flow (Post-Order)

Consider the standard ListOps problem: `SUM(4, 1, (MAX 1, 2), (MIN 1, 2))` (Answer: 8)

The script executes steps like this:

1.  **AST:** Generate an AST for `SUM(4, 1, MAX(1, 2), MIN(1, 2))`. Evaluate it to confirm the answer is 8. Pre-calculate all node values (MAX=2, MIN=1, SUM=8).
2.  **World:** Call LLM -> World: Sci-Fi, Space Station, Capt. Eva Rostova, Eng. Jax, Alien Zorp, seeking 'Data Fragments'.
3.  **Naming (Post-Order):**
    *   Name `MAX(1, 2)` -> LLM generates 'Critical Sensor Reading'.
    *   Name `MIN(1, 2)` -> LLM generates 'Auxiliary Power Cell'.
    *   Name `SUM(...)` -> LLM generates 'Final Data Synthesis'.
4.  **Narrative Generation (Post-Order):**
    *   **Beat 1 (Inner MAX):** Process `MAX(1, 2)`. Prompt LLM: "Scene: find 1 and 2 Data Fragments, rule is take MAX (2). Name: 'Critical Sensor Reading'. Required: {1, 2}. Forbidden: {}. Previous Snippet: Intro." -> LLM generates scene about getting 2 fragments, establishing 'Critical Sensor Reading'. `context.introduced_atoms = {1, 2}`. `context.last_scene_text` updated.
    *   *(Optional Padding)*
    *   **Beat 2 (Inner MIN):** Process `MIN(1, 2)`. Prompt LLM: "Scene: find 1 and 2 Data Fragments, rule is take MIN (1). Name: 'Auxiliary Power Cell'. Required: {1}. Forbidden: {2}. Previous Snippet: End of MAX scene." -> LLM generates scene about getting 1 fragment, establishing 'Auxiliary Power Cell'. `context.introduced_atoms = {1, 2}`. `context.last_scene_text` updated.
    *   *(Optional Padding)*
    *   **Beat 3 (Outer SUM):** Process `SUM(4, 1, MAX_node, MIN_node)`. Prompt LLM: "Scene: Apply SUM rule to newly found 4 and 1 fragments AND conceptual results of 'Critical Sensor Reading' and 'Auxiliary Power Cell'. Result is 8. Name: 'Final Data Synthesis'. Required: {4, 1, 8}. Forbidden: {2}. Previous Snippet: End of MIN scene." -> LLM generates scene describing combining the 4 and 1 fragments with the *concepts* of the previous steps, resulting in 8 fragments for the 'Final Data Synthesis'. `context.introduced_atoms = {1, 2, 4, 8}`. `context.last_scene_text` updated.
5.  **Question:** Prompt LLM: "Based on the entire narrative ending with the 'Final Data Synthesis', what was the final quantity of Data Fragments?" -> LLM generates the final question.

(Note: The actual generated text will be far more verbose and variable due to the LLM's creative generation and padding.)
