# Verbose ListOps Configuration (`config = Config()`)

This document outlines the configuration parameters available in `verbose-listops.py` through the `Config` dataclass. These settings control various aspects of the ListOps problem generation, narrative creation, LLM interaction, and data validation.

## Core Experiment Variables

These variables directly influence the difficulty of the generated problems and the style of the narrative.

### 1. ListOps Problem Difficulty

Controls the complexity of the underlying mathematical problems.

*   **`MAX_OPS: int = 10`**
    *   Description: The maximum number of ListOps operations (e.g., SUM, MIN, MAX) in a single Abstract Syntax Tree (AST).
    *   Impact: Higher values lead to more complex and longer calculation chains.

*   **`MIN_ARITY: int = 3`**
    *   Description: The minimum number of operands (atomic numbers or results of sub-operations) for any single ListOps operation.
    *   Impact: Higher values mean each operation involves more inputs.

*   **`MAX_BRANCH: int = 8`**
    *   Description: The maximum number of operands (atomic numbers or results of sub-operations) for any single ListOps operation.
    *   Impact: Higher values allow operations to take many inputs, potentially increasing complexity.

*   **`MIN_ATOM_VAL: int = 1`**
    *   Description: The minimum value for an atomic number (leaf node in the AST).
    *   Impact: Sets the lower bound for numbers directly used in calculations.

*   **`MAX_ATOM_VAL: int = 100`**
    *   Description: The maximum value for an atomic number.
    *   Impact: Sets the upper bound for numbers directly used in calculations. Affects the range of numbers the LLM needs to handle.

*   **`MAX_TOTAL_TOKENS: int = 10000`**
    *   Description: The target maximum number of tokens for a complete generated sample (narrative + question). The script aims to stay under this budget.
    *   Impact: Directly controls the overall length of the generated story.

*   **`MAX_BEAT_TOKENS: int = 1000`**
    *   Description: The maximum number of tokens the LLM is allowed to generate for a single narrative "beat" (a scene corresponding to one ListOps calculation step). This is the `max_tokens` sent to the API for beat generation.
    *   Impact: Limits the length of individual story scenes. Must be sufficient for internal reasoning and desired output.

*   **`MAX_PADDING_TOKENS: int = 1000`**
    *   Description: The maximum number of tokens the LLM is allowed to generate for a single "padding" paragraph (descriptive text between beats). This is the `max_tokens` sent to the API for padding generation.
    *   Impact: Limits the length of filler text. Must be sufficient for internal reasoning and desired output.

*   **`EARLY_TERMINATION_PROBABILITY: float = 0.2`**
    *   Description: The probability (0.0 to 1.0) that an AST branch will terminate early and become an atomic number, rather than continuing to add more operations (up to `MAX_OPS`).
    *   Impact: Higher values lead to shallower, less complex ASTs on average.

### 2. Narrative Context Generation & Style

Controls how the fictional world is created and how the narrative is styled.

*   **`USE_NARRATIVE_ANCHORS: bool = True`**
    *   Description: If `True`, the system will generate conceptual placeholders (narrative anchors) for intermediate results of operations, allowing the story to refer to them thematically rather than numerically.
    *   Impact: Enables more sophisticated narrative construction where intermediate results are not explicitly stated until potentially much later.

*   **`USE_LLM_NAMING: bool = True`**
    *   Description: If `True` (and `USE_NARRATIVE_ANCHORS` is also `True`), an LLM will be used to generate creative names for the narrative anchors. If `False`, a simpler template-based naming scheme is used.
    *   Impact: Affects the creativity and thematic relevance of the anchor names.

*   **`MIN_WORLD_CHARS: int = 3`**
    *   Description: The minimum number of characters to be generated for the fictional world metadata.
    *   Impact: Influences the richness of the world's cast.

*   **`MAX_WORLD_CHARS: int = 6`**
    *   Description: The maximum number of characters to be generated for the fictional world metadata.
    *   Impact: Influences the richness of the world's cast.

*   **`MIN_WORLD_CONCEPTS: int = 5`**
    *   Description: (Currently not directly used by `generate_world` which takes `num_characters` and `num_concepts` for other things, but kept for potential future use or if `generate_world`'s signature changes).
    *   Impact: Potentially influences the thematic depth of the world.

*   **`MAX_WORLD_CONCEPTS: int = 10`**
    *   Description: (Similar to `MIN_WORLD_CONCEPTS`).
    *   Impact: Potentially influences the thematic depth of the world.

*   **`BEAT_CONTEXT: int = 200`**
    *   Description: The maximum number of characters from the end of the previous scene to provide as context when prompting the LLM for the next narrative beat.
    *   Impact: Affects how much of the immediate past the LLM "sees" when writing the next scene.

*   **`PADDING_CONTEXT: int = 150`**
    *   Description: The maximum number of characters from the end of the previous scene to provide as context when prompting the LLM for a padding paragraph.
    *   Impact: Affects context for filler text generation.

*   **`MAX_PAD_PARAGRAPHS: int = 3`**
    *   Description: The maximum number of padding segments (paragraphs) that can be generated after each non-root narrative beat, subject to token budget.
    *   Impact: Controls how much filler text can be added to reach `MAX_TOTAL_TOKENS`.

*   **`PADDING_MAX_TOK_PERCENT: float = 0.60`**
    *   Description: (Currently not directly enforced as a strict percentage of `MAX_TOTAL_TOKENS` for padding, but the padding loop tries to fill available space). This could be a target for future refinement.
    *   Impact: Conceptually, how much of the total story could be padding.

### 3. Temperature

Controls the randomness/creativity of LLM responses. Higher values (e.g., 0.9) are more random, lower values (e.g., 0.2) are more deterministic.

*   **`WORLD_GEN_TEMP: float = 0.9`**
    *   Description: Temperature for LLM calls generating the fictional world metadata (characters, setting, etc.).
    *   Impact: Higher values lead to more diverse and potentially unusual world details.

*   **`BEAT_GEN_TEMP: float = 0.2`**
    *   Description: Temperature for LLM calls generating narrative beats (story scenes for calculations).
    *   Impact: Lower values aim for more focused and rule-adherent scene generation.

*   **`CREATIVE_NARRATIVE_TEMP: float = 0.75`**
    *   Description: Temperature for more creative narrative parts, such as the introductory scene and padding paragraphs.
    *   Impact: Allows for more varied and imaginative introductory and filler text.

*   **`ANCHOR_GEN_TEMP: float = 0.75`**
    *   Description: Temperature for LLM calls generating names for narrative anchors.
    *   Impact: Influences the creativity of anchor names.

## Other Configuration Settings

### Token & Budget Management

*   **`MAX_TOKENS_BUFFER: int = 1000`** (Referred to as `SAFETY_MARGIN` in code)
    *   Description: A safety buffer subtracted from `MAX_TOTAL_TOKENS` when checking if adding new content would exceed the budget. This helps prevent the final assembled sample from going over `MAX_TOTAL_TOKENS` after adding the question or due to slight tokenization differences.
    *   Impact: Makes budget checks more conservative, reducing the chance of exceeding the overall token limit.

*   **`INTRO_MAX_TOKENS: int = 500`**
    *   Description: The maximum number of tokens the LLM is allowed to generate for the introductory scene. This is the `max_tokens` sent to the API for intro generation.
    *   Impact: Limits the length of the story's opening. Must be sufficient for internal reasoning and desired output.

*   **`WORLD_GEN_MAX_TOKENS: int = 10000`**
    *   Description: The maximum number of tokens the LLM is allowed to generate when creating the world metadata JSON. This is the `max_tokens` sent to the API for world generation.
    *   Impact: Should be large enough to accommodate the JSON structure for the specified number of characters and concepts.

*   **`ANCHOR_MAX_TOKENS: int = 100`** (Adjusted from original 10 based on reasoning token discussion)
    *   Description: The maximum number of tokens the LLM is allowed to generate when creating a narrative anchor name. This is the `max_tokens` sent to the API.
    *   Impact: Limits the API response length for an anchor. Must be sufficient for internal reasoning and the short anchor phrase. The actual anchor length is validated by `MAX_ANCHOR_WORDS`.

### LLM Interaction & Prompting

*   **`MAX_ANCHOR_WORDS: int = 4`**
    *   Description: The maximum number of words allowed in a generated narrative anchor name. This is validated *after* the LLM generates the anchor.
    *   Impact: Ensures anchor names are concise.

*   **`FEW_SHOT_EXAMPLES: int = 1`**
    *   Description: The number of few-shot examples to include in the prompt for narrative beat generation, demonstrating adherence to strict number rules.
    *   Impact: Can guide the LLM towards better rule-following.

### Data Validation & Fallbacks

*   **`FALLBACK_MIN_NUM_WORD: int = 0`**
    *   Description: If the `inflect` library fails, this is the minimum number for a basic fallback number-to-word conversion.
    *   Impact: Affects number word extraction if `inflect` is unavailable.

*   **`FALLBACK_MAX_NUM_WORD: int = 20`**
    *   Description: If the `inflect` library fails, this is the maximum number for a basic fallback number-to-word conversion.
    *   Impact: Affects number word extraction if `inflect` is unavailable.

*   **`MIN_ALLOWED_SMALL_NUMBER: int = 0`**
    *   Description: The minimum value for a range of small integers (e.g., 0, 1, 2) that are implicitly allowed in a narrative beat by the validator, even if not part of the direct calculation, for narrative naturalness.
    *   Impact: Makes validation slightly more lenient for common small numbers.

*   **`MAX_ALLOWED_SMALL_NUMBER: int = 2`**
    *   Description: The maximum value for the range of implicitly allowed small integers.
    *   Impact: Defines the upper end of the leniently validated small numbers (e.g., 0, 1, 2 if min is 0 and max is 2).

*   **`INVALID_RESULT_PLACEHOLDER: int = -999`**
    *   Description: A placeholder value used in the number validator for specific error cases or when a result isn't applicable (e.g., for strict zero validation in padding).
    *   Impact: Internal to the validator logic.

### API Configuration & Retries

*   **`RETRY_MAX_ATTEMPTS: int = 5`**
    *   Description: The maximum number of retry attempts for general API calls that fail (e.g., network issues, transient errors), using exponential backoff.
    *   Impact: Increases resilience to temporary API problems.

*   **`RETRY_INITIAL_DELAY: float = 0.5`**
    *   Description: The initial delay (in seconds) before the first retry attempt. Subsequent retries use exponentially increasing delays.
    *   Impact: Controls the starting point of the backoff strategy.

*   **`MAX_BEAT_RETRIES: int = 5`**
    *   Description: The maximum number of retries specifically for generating a single narrative beat if the initial attempts fail validation or result in an API error.
    *   Impact: Determines how many chances the script gets to generate a valid beat.

*   **`MAX_PAD_RETRIES: int = 5`**
    *   Description: The maximum number of retries specifically for generating a single padding paragraph.
    *   Impact: Determines how many chances for valid padding.

*   **`INTRO_MAX_RETRIES: int = 3`**
    *   Description: The maximum number of retries for generating the introductory scene.
    *   Impact: Determines chances for a valid intro.

*   **`WORLDGEN_MAX_RETRIES: int = 3`**
    *   Description: The maximum number of retries for generating the world metadata JSON.
    *   Impact: Determines chances for valid world JSON.

*   **`INITIAL_WORLD_RETRY_DELAY: float = 0.5`**
    *   Description: The initial delay for retrying world generation, separate from the general `RETRY_INITIAL_DELAY`.
    *   Impact: Specific backoff for world generation.

*   **`MAX_REQUESTS_PER_SECOND: float = 500.0`**
    *   Description: Target maximum requests per second for the API rate limiter. The actual rate may be adjusted based on OpenRouter account limits.
    *   Impact: Controls the overall speed of API calls to avoid hitting rate limits.

*   **`MIN_REQUEST_INTERVAL: float = 0.015`**
    *   Description: The minimum time (in seconds) that must pass between consecutive API requests, enforced by the rate limiter.
    *   Impact: Provides a hard floor for request frequency.

### Logging Configuration

*   **`LOG_MAX_BYTES: int = 5 * 1024 * 1024`** (5MB)
    *   Description: The maximum size of a single log file before it is rotated.
    *   Impact: Manages log file disk usage.

*   **`LOG_BACKUP_COUNT: int = 3`**
    *   Description: The number of backup log files to keep after rotation.
    *   Impact: Controls how much historical log data is preserved.

*   **`CLEAR_LOGS_ON_START: bool = True`**
    *   Description: If `True`, the entire `logs` directory (including main logs and LLM turn logs) will be deleted and recreated at the start of the script.
    *   Impact: Ensures a clean logging state for each run. Set to `False` to append to existing logs.

---

## Global Constants (Not part of `Config` class but affect execution)

*   **`NUM_SAMPLES_TO_GENERATE: int = 5`**
    *   Description: The default number of verbose ListOps samples to generate in a single run of the script.
    *   Impact: Controls batch size.

*   **`DEFAULT_MAX_WORKERS: int = 20`**
    *   Description: The default number of parallel threads to use for batch generation of samples.
    *   Impact: Affects the speed of generating multiple samples.

*   **`MODEL: str = "google/gemini-2.5-pro-preview-03-25"`**
    *   Description: The OpenRouter model identifier to be used for all LLM calls.
    *   Impact: Determines which AI model generates the narrative and other creative content.