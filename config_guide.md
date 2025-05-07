# Verbose ListOps Configuration (`config = Config()`)

This document outlines the configuration parameters for `verbose-listops.py`. These settings allow fine-grained control over the generation of long-context narrative reasoning problems designed to test Large Language Models (LLMs) beyond simple fact retrieval. The benchmark embeds ListOps (2018) computations within lengthy, coherent narratives, introducing realistic distractors and enabling independent control over context length and reasoning complexity.

## Core Benchmark Variables: Context Length & Reasoning Complexity

These parameters are central to Verbose ListOps's goal of evaluating LLM performance on narrative-embedded reasoning tasks.

### 1. Context Window Size Control

These settings primarily influence the length of the generated narrative, thereby controlling the size of the context window the LLM must process.

| Parameter              | Type  | Default Value | Description & Benchmark Relevance                                                                                                                                                              |
| :--------------------- | :---- | :------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_TOTAL_TOKENS`     | `int` | `10000`       | Target maximum tokens for a complete sample (narrative + question). **Directly controls the context window size.**                                                                             |
| `MAX_BEAT_TOKENS`      | `int` | `1000`        | Max tokens for an LLM-generated narrative "beat" (scene for one ListOps step). Influences narrative segment granularity and contributes to overall context length. Must allow for reasoning. |
| `MAX_PADDING_TOKENS`   | `int` | `1000`        | Max tokens for an LLM-generated "padding" paragraph (distractor text). Allows insertion of distractors and helps reach `MAX_TOTAL_TOKENS`. Must allow for reasoning.                         |
| `MAX_PAD_PARAGRAPHS`   | `int` | `3`           | Maximum padding segments generated after each non-final narrative beat, budget permitting. Controls distractor density.                                                                        |
| `INTRO_MAX_TOKENS`     | `int` | `500`         | Max tokens for the LLM-generated introductory scene. Contributes to initial context length. Must allow for reasoning.                                                                          |

### 2. Reasoning Complexity Control

These settings determine the difficulty of the underlying ListOps computation embedded within the narrative.

| Parameter                     | Type    | Default Value | Description & Benchmark Relevance                                                                                                |
| :---------------------------- | :------ | :------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_OPS`                     | `int`   | `10`          | Maximum ListOps operations in the problem's AST. **Directly controls reasoning complexity** (number of calculation steps).         |
| `MIN_ARITY`                   | `int`   | `3`           | Minimum operands per ListOps operation. Increases items to consider at each reasoning step.                                      |
| `MAX_BRANCH`                  | `int`   | `8`           | Maximum operands per ListOps operation. Allows more complex individual operations.                                               |
| `MIN_ATOM_VAL`                | `int`   | `1`           | Minimum value for atomic numbers in the ListOps problem. Defines the numerical range.                                            |
| `MAX_ATOM_VAL`                | `int`   | `100`         | Maximum value for atomic numbers. Defines the numerical range.                                                                   |
| `EARLY_TERMINATION_PROBABILITY` | `float` | `0.2`         | Chance an AST branch ends early. Influences average depth/bushiness of the reasoning chain (lower values = more complex paths). |

## Narrative Generation & Distractor Realism

These settings control the generation of the narrative world and style, aiming for coherent yet distracting context.

| Parameter                   | Type    | Default Value | Description & Benchmark Relevance                                                                                                   |
| :-------------------------- | :------ | :------------ | :---------------------------------------------------------------------------------------------------------------------------------- |
| `USE_NARRATIVE_ANCHORS`     | `bool`  | `True`        | If `True`, generates conceptual placeholders for intermediate results, forcing thematic tracking of sub-task results.                 |
| `USE_LLM_NAMING`            | `bool`  | `True`        | If `True` (and `USE_NARRATIVE_ANCHORS` is `True`), an LLM generates creative names for narrative anchors for more natural distractors. |
| `MIN_WORLD_CHARS`           | `int`   | `3`           | Minimum characters generated for the fictional world. Adds to narrative richness/distractors.                                       |
| `MAX_WORLD_CHARS`           | `int`   | `6`           | Maximum characters generated for the fictional world.                                                                               |
| `BEAT_CONTEXT`              | `int`   | `200`         | Max characters of previous scene snippet for context in beat generation. Influences local narrative coherence.                      |
| `PADDING_CONTEXT`           | `int`   | `150`         | Max characters of previous scene snippet for context in padding generation.                                                         |
| `WORLD_GEN_TEMP`            | `float` | `0.9`         | Temperature for world metadata generation. Affects diversity of world details.                                                      |
| `BEAT_GEN_TEMP`             | `float` | `0.2`         | Temperature for narrative beat generation. Low value aims for rule-adherent calculation steps.                                      |
| `CREATIVE_NARRATIVE_TEMP`   | `float` | `0.75`        | Temperature for intro/padding. Allows more imaginative filler/distractor text.                                                      |
| `ANCHOR_GEN_TEMP`           | `float` | `0.75`        | Temperature for narrative anchor naming. Influences creativity of anchor names.                                                     |

*Note: `MIN_WORLD_CONCEPTS` and `MAX_WORLD_CONCEPTS` are present in the `Config` class but not directly used by the current `generate_world` function's primary logic.*
*Note: `PADDING_MAX_TOK_PERCENT` is present but not strictly enforced as a percentage; padding aims to fill available budget.*

## LLM Interaction, Validation & Robustness

Settings related to API calls, ensuring generated data is valid, and script robustness.

| Parameter                   | Type    | Default Value | Description & Benchmark Relevance                                                                                                                               |
| :-------------------------- | :------ | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_TOKENS_BUFFER`         | `int`   | `1000`        | Safety buffer (`SAFETY_MARGIN`) for `MAX_TOTAL_TOKENS` to prevent final output exceeding limit. Ensures adherence to target context length.                       |
| `ANCHOR_MAX_TOKENS`         | `int`   | `100`         | Max tokens for API call generating an anchor name. Must be sufficient for internal reasoning and short output. Actual length validated by `MAX_ANCHOR_WORDS`.      |
| `WORLD_GEN_MAX_TOKENS`      | `int`   | `10000`       | Max tokens for API call generating world metadata JSON. Should be ample for the JSON structure.                                                                 |
| `MAX_ANCHOR_WORDS`          | `int`   | `4`           | Validates conciseness of LLM-generated anchor names. Ensures brief thematic references.                                                                         |
| `FEW_SHOT_EXAMPLES`         | `int`   | `1`           | Number of few-shot examples for narrative beat generation to guide rule-following. Aids in generating valid narrative steps.                                    |
| `MIN_ALLOWED_SMALL_NUMBER`  | `int`   | `0`           | Min value for implicitly allowed small integers (e.g., 0, 1, 2) in narrative beats for naturalness.                                                             |
| `MAX_ALLOWED_SMALL_NUMBER`  | `int`   | `2`           | Max value for implicitly allowed small integers. Prevents validator from flagging common small words unnecessarily.                                               |
| `INVALID_RESULT_PLACEHOLDER`| `int`   | `-999`        | Placeholder for validator in specific error cases/non-applicable results. Internal to validator.                                                                |
| `RETRY_MAX_ATTEMPTS`        | `int`   | `5`           | Max general API call retries. Increases resilience.                                                                                                             |
| `RETRY_INITIAL_DELAY`       | `float` | `0.5`         | Initial delay (seconds) for retries.                                                                                                                            |
| `MAX_BEAT_RETRIES`          | `int`   | `5`           | Max retries for generating a narrative beat.                                                                                                                    |
| `MAX_PAD_RETRIES`           | `int`   | `5`           | Max retries for generating padding.                                                                                                                             |
| `INTRO_MAX_RETRIES`         | `int`   | `3`           | Max retries for generating intro scene.                                                                                                                         |
| `WORLDGEN_MAX_RETRIES`      | `int`   | `3`           | Max retries for generating world metadata.                                                                                                                      |
| `INITIAL_WORLD_RETRY_DELAY` | `float` | `0.5`         | Specific initial retry delay for world generation.                                                                                                              |
| `MAX_REQUESTS_PER_SECOND`   | `float` | `500.0`       | Target max API requests/sec for rate limiter. May be adjusted by OpenRouter limits.                                                                             |
| `MIN_REQUEST_INTERVAL`      | `float` | `0.015`       | Minimum time (seconds) between API requests.                                                                                                                    |
| `LOG_MAX_BYTES`             | `int`   | `5242880`     | Max log file size (5MB) before rotation. Manages disk usage.                                                                                                    |
| `LOG_BACKUP_COUNT`          | `int`   | `3`           | Number of backup log files. Preserves historical logs.                                                                                                          |
| `CLEAR_LOGS_ON_START`       | `bool`  | `True`        | If `True`, deletes `logs` directory on start for a clean logging state.                                                                                         |

*Note: `FALLBACK_MIN_NUM_WORD` and `FALLBACK_MAX_NUM_WORD` are for `inflect` library failure, less critical if `inflect` is stable.*

---

## Global Constants (Affecting Batch Generation)

These are defined outside the `Config` class but control the overall script execution for generating datasets.

| Constant                | Type  | Default Value                      | Description & Benchmark Relevance                                                                                               |
| :---------------------- | :---- | :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| `NUM_SAMPLES_TO_GENERATE` | `int` | `5`                                | Default number of Verbose ListOps samples per run. Controls the size of the generated dataset.                                  |
| `DEFAULT_MAX_WORKERS`   | `int` | `20`                               | Default parallel threads for batch generation. Speeds up dataset creation.                                                      |
| `MODEL`                 | `str` | `"google/gemini-2.5-pro-preview-03-25"` | The OpenRouter model identifier for all LLM calls. The LLM being evaluated or used to generate evaluation data for other LLMs. |

This configuration allows researchers to systematically vary context length and reasoning complexity to probe LLM capabilities in understanding and reasoning over long, distracting narratives, a key challenge for advancing AI towards more complex real-world applications.