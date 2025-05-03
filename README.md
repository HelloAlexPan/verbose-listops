# Verbose ListOps Evaluation Benchmark

## HIGHLY EXPERIMENTAL — WIP

## Overview

Welcome to the Verbose ListOps Evaluation Benchmark repository. This project provides tools to synthetically generate multi-hop qa evaluation tasks of variable difficulity for Large Language Models (LLMs). The benchmark assesses reasoning capabilities when faced with tasks embedded within extremely long, narratively structured, and potentially distracting contexts.

It builds upon the 2018 [ListOps benchmark](https://arxiv.org/abs/1804.06028) by transforming its concise, symbolic problem representation into lengthy, natural language narratives generated dynamically using an LLM (currently Anthropic's Claude). The core Python script, `verbose-listops.py`, orchestrates this process.

Despite sharing the exact core task of the ListOps benchmark, current SOTA models universally fail verbose-listops narratives—even with multiple-shot prompting—at just 10,000 tokens, whilst successfully solving their identical standard ListOps representations. However, on easier problems, few-shot prompting has a significant effect on improving success rate.

Standard long‑context benchmarks typically emphasize “needle‑in‑a‑haystack” retrieval—locating a specific fact buried in a long document. In contrast, Verbose ListOps deterministically weaves explicit multi‑step operations (e.g., SUM, MIN, MAX) and their required operands into a narrative. Each story interleaves these operations with semantically related filler sentences, so a model must first identify and sequence the specified operations correctly before performing the computation. Though every narrative is dynamically generated with an LLM, strict deterministic tests enforced through agentic techniques guarantees both reproducibility and soundness of semantic reasoning.

Due to the nature of this computational task and its simulation of real-world human behaviours, this dataset is especially well-suited to understanding whether frontier models _can_ excel at extracting and scoring arbitrarily defined predictive signals from large corpora of unstructured text, such as qualification signals from sales transcripts. Today's LLM's already excel at LLM-as-Judge tasks, and this benchmark attempts to understand whether they are able to synthesize multiple simple single-output generation tasks across long contexts.

## Comparison with Similar Benchmarks

While Verbose ListOps has a unique combination of features (ListOps core task, LLM-generated narrative context, specific padding mechanism), several other benchmarks evaluate related aspects of long-context reasoning. Here's a comparison:

|          | **verbose‑listops** | **LongReason** | **BABILong** |
|----------|---------------------|----------------|--------------|
| **Pros** | - **Dual Difficulty Knobs**: Varies both context length and inherent ListOps problem complexity.<br>- **Deterministic Narrative**: Reproducible, agent‑enforced semantic reasoning.<br>- **Hierarchical Logic**: Embeds nested operations.<br>- **Scalable Contexts**: Configurable up to > 50 K tokens. | - **Diverse Seed Tasks**: Synthesizes from reading comprehension, logical inference, and math problems.<br>- **Coherent Expansion**: LLM‑generated paragraphs yield realistic long contexts.<br>- **Controlled Lengths**: Splits from 8 K to 128 K tokens.<br>- **Auto‑Verified**: LLM checks ensure reasoning correctness. | - **Needle‑in‑a‑Haystack**: Hides bAbI facts among PG19 text for realistic distraction.<br>- **Extreme Scalability**: Contexts up to millions of tokens.<br>- **Deterministic Seeds**: Known bAbI tasks enable provable difficulty.<br>- **Multi‑Task Coverage**: 20 reasoning types. |
| **Cons** | - **Domain‑Specific**: Limited to ListOps tasks.<br>- **Pipeline Complexity**: Requires external LLM calls for world/narrative generation.<br>- **Narrative Variability**: Prose may introduce unintended noise. | - **Length‑Only Tuning**: No knobs for distractor density or hop count.<br>- **Fixed Complexity**: Seed question difficulty cannot be varied.<br>- **Synthetic Coherence Risk**: Expanded text may drift or feel unnatural. | - **Simplistic Reasoning**: Restricted to bAbI‑style patterns.<br>- **Unrealistic Distractors**: Random book sentences may not mimic real‑world noise.<br>- **Single Knob**: Difficulty varies only by context length. |

Among these benchmarks, verbose‑listops most closely mirrors LongReason’s approach. Unlike LongReason—which only varies difficulty by context length—verbose‑listops offers two levers: context length and the intrinsic complexity of the ListOps problem it narrativizes.

## The Core ListOps Task

The underlying computational task remains identical to the original ListOps benchmark. Models processing the generated narrative must effectively:

1.  Identify and parse nested operations on lists embedded within the story.
2.  The supported operations are:
    * `MAX` (maximum)
    * `MIN` (minimum)
    * `MED` (median)
    * `SUM` (sum)
    * `SM` (sum modulo 10)
    * `AVG` (integer average, floored)
3.  Operands are single-digit integers (0-9), also described textually within the narrative.
4.  Evaluate these nested operations hierarchically according to the narrative structure.
5.  Produce a single-digit integer as the final result based on the root operation.

ListOps fundamentally tests a model's ability to understand hierarchical structures and perform sequential reasoning.

## Design Goal: Testing Large Context Windows with Narrative Complexity

The primary motivation remains to rigorously evaluate model performance on tasks involving extremely large context windows. However, this implementation focuses on achieving verbosity and complexity through **LLM-driven narrative generation** rather than just injecting unrelated filler text.

Standard benchmarks may not sufficiently stress a model's ability to:

* Maintain coherence and track dependencies over tens of thousands of tokens within a continuous story.
* Extract relevant operational information (operators, operands) from narrative descriptions interwoven with characters, setting, and plot elements.
* Perform multi-step reasoning when instructions are presented as part of an unfolding story across a vast input sequence.
* Resist distraction from contextually relevant but computationally irrelevant "padding" generated by the LLM (e.g., side-quests, character interactions).

This benchmark aims to address this by programmatically generating long narratives where the ListOps task is the hidden "logic puzzle" the model must solve. Target context lengths can reach configurable limits (e.g., 10,000 tokens or more, adjustable via parameters).

## Methodology: From ListOps AST to Generated Narrative

The `verbose-listops.py` script implements the following process:

1.  **Random AST Generation:** A random ListOps Abstract Syntax Tree (AST) is constructed (`build_random_ast`) defining the nested structure and operations of the problem. Parameters control the maximum depth and branching factor. The AST is evaluated (`eval_node`) to determine the ground truth answer.
2.  **World Building:** An initial call is made to the Anthropic API (`generate_world`) to create a fictional context, including characters, genre, and setting. This provides a consistent backdrop for the narrative.
3.  **Narrative Generation (`generate_narrative`):**
    * The script traverses the AST (pre-order).
    * For each **operator node** in the AST, it prompts the LLM (Anthropic API via `_client_create`) to write a short "story beat" or scene. This prompt includes:
        * The generated world context (characters, setting, genre).
        * The specific ListOps operation (`op`), its operands (`operands`), and the intermediate result (`value`) for that node.
        * Instructions to weave this logic into the narrative involving the characters, *without explicitly revealing the numeric answer as a meta-clue*.
    * **Narrative Padding:** After generating a story beat for an operation, the script may make additional calls to the LLM to generate "padding" paragraphs. This padding is prompted to continue the *last scene* with side-quests, mysteries, or random asides, explicitly instructed *not* to change established facts or operator logic. This increases verbosity using contextually relevant, LLM-generated text.
    * **Token Management:** The generation process respects configurable token limits for individual beats (`MAX_BEAT_TOKENS`), padding sections (`MAX_PAD_TOKENS`), and the overall narrative (`MAX_TOTAL_TOKENS`), ensuring the output stays within desired bounds.
4.  **Question Generation:** After generating the full narrative based on the AST, a final prompt is sent to the LLM to formulate a question specifically asking for the result of the *top-level* operation performed in the story, using few-shot examples for guidance.

The result is a single block of text containing the full narrative followed by the target question.

## Why Verbose ListOps (LLM-Generated Version)?

This approach provides a challenging test case where the core task is simple, but requires the evaluated model to:

* Read and understand a long, coherent narrative generated by another LLM.
* Identify and isolate the embedded computational steps from the story elements.
* Track the hierarchy and dependencies of the operations as presented narratively.
* Answer a direct question about the final result of the embedded ListOps task.

It tests whether models can truly *use* their long context for complex reasoning within noisy, narratively structured data, going beyond simple filler injection.

## Code Implementation Details

The `verbose-listops.py` script includes:

* **Dependencies:** `anthropic` (for API calls), `tiktoken` (for token counting).
* **Core Logic:** AST generation, evaluation, world generation, narrative beat/padding generation via LLM calls, question generation.
* **Configuration:** Constants at the top of the script control:
    * API Model (`MODEL`)
    * Token limits (`MAX_TOTAL_TOKENS`, `MAX_BEAT_TOKENS`, `MAX_PAD_TOKENS`)
    * AST structure (`DEFAULT_MAX_BRANCH`, `MIN_ARITY`, `ATOM_MIN_VALUE`, `ATOM_MAX_VALUE`)
    * Retry behavior (`RETRY_MAX_ATTEMPTS`, `RETRY_INITIAL_DELAY`)
    * Logging (`LOG_DIR`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`)
* **API Key:** Requires the `ANTHROPIC_API_KEY` environment variable to be set.
* **Error Handling:** Includes retry logic (`retry_api_call`) for API calls with exponential backoff.
* **Logging:** Detailed logs are written to `~/verbose_listops_logs/verbose_listops.log`, including prompts sent to the LLM. World metadata is saved to `world.json` in the same directory.

## Usage

1.  **Setup:**
    * Ensure you have Python installed.
    * Install required libraries: `pip install anthropic tiktoken`
    * Set the `ANTHROPIC_API_KEY` environment variable: `export ANTHROPIC_API_KEY='your-api-key'`
2.  **Run the Script:** Execute `python verbose-listops.py`.
3.  **Output:**
    * The script will print the final generated narrative followed by the question to the standard output.
    * Detailed logs and the generated `world.json` will be saved in `~/verbose_listops_logs/`.
4.  **Evaluation:**
    * Provide the full narrative text (including the final question) as input to the LLM you want to evaluate.
    * Parse the model's response to extract the predicted single-digit answer.
    * Compare the prediction against the ground truth answer (which can be found by running the script or inspecting the logs/AST evaluation logic if needed, though ideally the generation script is trusted). Calculate accuracy or other relevant metrics.

## Example Conceptual Flow

Consider the standard ListOps problem: `[SM 8 1 4 [MAX 9 2 7]]` (Answer: 2)

The script would execute steps like this:

1.  **AST:** Generate an AST representing this structure. Evaluate it to confirm the answer is 2.
2.  **World:** Call the LLM to create a world (e.g., Genre: Sci-Fi, Setting: Space Station, Characters: Captain Eva Rostova, Engineer Jax, Alien Diplomat Zorp).
3.  **Narrative (Simplified):**
    * **Beat 1 (MAX):** Prompt LLM: "Write a scene where Eva, Jax, and Zorp encounter numerical readings 9, 2, 7, and focus on the *largest* one (9) as being critical." -> LLM generates a paragraph or two about finding the critical reading 9.
    * *Padding:* Prompt LLM: "Continue the scene, maybe Jax has a side-task." -> LLM adds paragraphs about Jax fixing a panel.
    * **Beat 2 (SM):** Prompt LLM: "Write a scene where the primary task involves combining readings 8, 1, 4, and the previous critical result (9), then finding the *sum modulo 10* (resulting in 2) to unlock a door." -> LLM generates paragraphs about entering the sequence and the door unlocking with code 2.
    * *Padding:* Prompt LLM: "Continue the scene, maybe Zorp comments on human inefficiency." -> LLM adds Zorp's dialogue.
4.  **Question:** Prompt LLM: "Based on the story ending with the sum modulo 10 operation, formulate a question asking for that final result." -> LLM generates: "Following the procedures described in the narrative, what was the final code generated by the sum modulo 10 operation at the main level?"

*(Note: The actual generated text will be far more verbose and variable due to the LLM's creative generation.)*
