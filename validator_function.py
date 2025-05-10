# --- NEW FUNCTION: _generate_and_llm_validate_beat (Iterative LLM Validation Loop) ---
def _generate_and_llm_validate_beat(
    original_user_message_for_generator: str, # The initial prompt for the beat generator
    system_prompt_for_generator: str,
    world_info: dict, # For validator prompt
    current_op_node: OpNode, # For validator prompt
    conceptual_inputs_str: str, # For validator prompt
    atomic_inputs_words_str: str, # For validator prompt
    action_description: str, # For validator prompt
    expected_beat_result_words: str | None, # For validator prompt
    current_max_beat_completion_tokens: int,
    sample_index: int,
    context_config: Config, # Pass the main config object
    logger_obj: logging.Logger,
    encoder_obj: any # Pass the tokenizer
) -> str | None: # Returns validated beat text or None
    
    # --- Accumulators for history ---
    history_of_attempts = [] # List of strings (generated beats)
    history_of_critiques = [] # List of dicts (parsed JSON feedback from validator)
    # ---

    for iteration in range(1, context_config.MAX_LLM_VALIDATION_ITERATIONS + 1):
        logger_obj.info(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validation Loop Iteration: {iteration}/{context_config.MAX_LLM_VALIDATION_ITERATIONS}")

        # 1. Prepare Generator Prompt with Full History
        current_generator_user_prompt = original_user_message_for_generator # Base prompt

        if iteration > 1:
            history_prompt_addition = "\n\n--- PREVIOUS ATTEMPTS AND FEEDBACK HISTORY ---\n"
            for i in range(len(history_of_attempts)):
                history_prompt_addition += f"\n**--- Round {i+1} ---**\n"
                attempt_text = history_of_attempts[i]
                if attempt_text.startswith("ERROR_DURING_GENERATION:"): # Handle cases where generation itself failed
                    history_prompt_addition += f"**Your Attempt {i+1} (failed during generation):**\n{attempt_text}\n\n"
                else:
                    history_prompt_addition += f"**Your Attempt {i+1}:**\n{attempt_text}\n\n"
                
                history_prompt_addition += f"**Validator Feedback for Attempt {i+1}:**\n"
                critique = history_of_critiques[i] # Should always exist if attempt exists
                history_prompt_addition += f"  Explanation: {critique.get('explanation_for_generator', 'N/A')}\n"
                history_prompt_addition += f"  Overall Summary for Revision: {critique.get('overall_revision_summary_for_generator_prompt', 'N/A')}\n"
                
                # Add suggested revisions if available and in a useful format
                if critique.get("suggested_revisions") and isinstance(critique.get("suggested_revisions"), list):
                    suggested_revisions = critique.get("suggested_revisions")
                    if suggested_revisions:
                        history_prompt_addition += f"  Suggested Revisions:\n"
                        for rev in suggested_revisions:
                            if isinstance(rev, dict):
                                rev_type = rev.get("type", "change")
                                rev_reason = rev.get("reason", "improve the narrative")
                                history_prompt_addition += f"    - {rev_type}: {rev_reason}\n"

            # Add the most recent critique again clearly for the current revision task
            most_recent_critique = history_of_critiques[-1]
            last_attempt_number = iteration - 1
            history_prompt_addition += (
                f"\n--- CURRENT TASK: REVISE ATTEMPT {last_attempt_number} (FROM ROUND {last_attempt_number} ABOVE) ---\n"
                f"**Based on Validator Feedback for Attempt {last_attempt_number}:** {most_recent_critique.get('overall_revision_summary_for_generator_prompt', 'Please revise based on issues found.')}\n\n"
                f"Please provide a new version (Attempt {iteration}) of the scene, incorporating ALL feedback received so far (especially the most recent) "
                f"to meet the original task requirements and number rules. Ensure the new scene continues from the 'Prior Scene Snippet' if one was part of the original task. "
                f"Output ONLY the revised narrative text."
            )
            current_generator_user_prompt += history_prompt_addition
            generator_temp = context_config.BEAT_REVISION_TEMP
        else:
            generator_temp = context_config.BEAT_GEN_TEMP
        
        # --- Token Count Check for Generator Prompt (Crucial with Full History) ---
        generator_prompt_tokens = 0
        if encoder_obj: # Check if encoder_obj is available
            try:
                generator_prompt_tokens = len(encoder_obj.encode(system_prompt_for_generator + current_generator_user_prompt))
            except Exception as e_encode:
                logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error encoding generator prompt for token count: {e_encode}. Assuming high token count.")
                generator_prompt_tokens = context_config.MODEL_MAX_CONTEXT_TOKENS // 2 # Assume half of max context to be safe
        else:
            logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Tokenizer (encoder_obj) not available. Cannot check prompt token count. Proceeding with caution.")
            # If no encoder, we can't accurately check, so we might skip or assume it fits.
            # For safety, if no encoder, we could potentially restrict history depth, but for now, it proceeds.

        # Using MAX_API_TOKEN_LIMIT for max_completion_tokens to avoid truncation due to reasoning/tool use,
        # but this is for the *API call itself*. The *actual content tokens* are what we budget for.
        # Let's use a more descriptive name for what the generator is expected to output.
        EXPECTED_MAX_BEAT_OUTPUT_TOKENS = context_config.BEAT_MAX_TOKENS 
        
        # Check against configured model context limit
        if generator_prompt_tokens + EXPECTED_MAX_BEAT_OUTPUT_TOKENS > context_config.MODEL_MAX_CONTEXT_TOKENS:
            logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Iteration {iteration}: Combined prompt ({generator_prompt_tokens} toks) "
                             f"and max expected output ({EXPECTED_MAX_BEAT_OUTPUT_TOKENS} toks) = {generator_prompt_tokens + EXPECTED_MAX_BEAT_OUTPUT_TOKENS} toks, "
                             f"which would exceed model context limit ({context_config.MODEL_MAX_CONTEXT_TOKENS}). Failing beat.")
            # If history causes overflow, we might need a strategy to shorten it. For now, it fails.
            # One simple strategy: if history_of_attempts has items, pop the oldest and retry this iteration's prompt construction?
            # This would be complex to manage within this loop without causing infinite sub-loops.
            # For now, if it's too long, it fails the beat.
            return None 
        # ---

        # 2. Generate Beat (or revision)
        try:
            log_prompt(
                header=f"LLM Beat Gen (Iterative Loop Attempt {iteration}) Op: {current_op_node.op}",
                prompt=f"System: {system_prompt_for_generator}\nUser: {current_generator_user_prompt}",
                sample_index=sample_index
            )
            resp_gen = _chat_completion_call(
                model=MODEL, # Main generator model
                messages=[
                    {"role": "system", "content": system_prompt_for_generator},
                    {"role": "user", "content": current_generator_user_prompt},
                ],
                max_completion_tokens=current_max_beat_completion_tokens,
                temperature=generator_temp,
                reasoning={"exclude": True} 
            )
            
            generated_text_raw = ""
            if resp_gen and resp_gen.choices and len(resp_gen.choices) > 0 and resp_gen.choices[0].message:
                _content = resp_gen.choices[0].message.content
                generated_text_raw = _content if _content is not None else ""
            
            # Apply cleaning logic (adapted from _generate_narrative_recursive)
            lines = generated_text_raw.splitlines()
            filtered_lines = []
            prompt_echoing_fragments = (
                "Imply the sum", "reference to the previous", "Narrate comparing", "This scene resolves",
                "The outcome MUST be", "Narrate an action", "Narrate an event", "The final quantity MUST become"
            )
            analysis_patterns_to_remove = [
                r"^\s*Critique:.*$", r"^\s*Checklist:.*$", r"^\s*Analysis:.*$", r"^\s*Rules check:.*$",
                r"^\s*MUST INCLUDE:.*$", r"^\s*MUST AVOID:.*$", r"^\s*Reasoning:.*$", r"^\s*Validation:.*$",
                r"^\s*Following the rules:.*$", r"^\s*Based on the instructions:.*$", r"^\s*The task is to.*$",
                r"^\s*The prompt asks.*$", r"^\s*Scene instructions:.*$", r"^\s*Instructions:.*$",
                r"^\s*Number rules:.*$", r"^\s*Rules:.*$", r"^\s*Confidence Score:.*$", r"^\s*Mental Sandbox:.*$",
                r"^\s*\d+\.\s*STRICT:.*$", r"^\s*\d+\.\s*MAY USE.*$", r"^\s*Output ONLY.*$", r"^\s*REMINDER:.*$",
                r"^\s*Okay.*$", r"^\s*Certainly.*$", r"^\s*```json.*$", r"^\s*```.*$", r"^\s*Generation:\s*",
                r"^\s*Narrative:\s*", r"^\s*Scene \d+:\s*", r"^\s*Beat \d+:\s*", r"^\s*Refinement \d+.*$",
                r"^\s*Yes\s*$", r"^\s*N/A\s*$", r"^\s*\[.*?\]\s*$", r"^\s*-\s.*$", r"^\s*\*\s.*$",
                r"^\s*Outcome is.*$", r"^\s*System:.*$", r"^\s*User:.*$", r"^\s*Check\..*$", r"^\s*Task:.*$",
                r"^\s*\?.*$"
            ]
            for line in lines:
                stripped_line = line.strip()
                is_analysis_or_echo = False
                for pattern in analysis_patterns_to_remove:
                    if re.match(pattern, stripped_line, re.IGNORECASE):
                        is_analysis_or_echo = True
                        # logger_obj.debug(f"Cleaning (iterative beat loop): Removing line matching pattern '{pattern}': '{line}'")
                        break
                if is_analysis_or_echo:
                    continue
                if any(stripped_line.lower().startswith(fragment.lower()) for fragment in prompt_echoing_fragments):
                    # logger_obj.debug(f"Cleaning (iterative beat loop): Removing line starting with prompt fragment: '{line}'")
                    is_analysis_or_echo = True
                if not is_analysis_or_echo:
                    filtered_lines.append(line)
            generated_text_cleaned = "\n".join(filtered_lines).strip()

            # Check for API refusals or empty content
            if not generated_text_cleaned or generated_text_cleaned.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
                logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Generator produced API refusal or empty text in iteration {iteration}.")
                generated_text_cleaned = ""  # Ensure it's empty for consistent handling
            
            log_prompt(
                header=f"LLM Beat Gen (Iterative Loop Attempt {iteration}, Cleaned) Op: {current_op_node.op}",
                prompt=f"Cleaned Generation:\n{generated_text_cleaned}",
                sample_index=sample_index
            )

            if not generated_text_cleaned:
                # Add a placeholder critique if generation is empty, so history structure is maintained
                history_of_attempts.append("") 
                history_of_critiques.append({
                    "is_valid": False,
                    "explanation_for_generator": "The generation was empty or contained only removable boilerplate.",
                    "overall_revision_summary_for_generator_prompt": "The previous generation was empty or boilerplate. Please generate the narrative scene as requested by the original task."
                })
                continue

            history_of_attempts.append(generated_text_cleaned) # Add successful generation to history

        except Exception as e_gen:
            logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error generating beat in LLM loop iteration {iteration}: {e_gen}")
            history_of_attempts.append(f"ERROR_DURING_GENERATION: {str(e_gen)[:500]}") # Log error as attempt, truncated
            history_of_critiques.append({
                "is_valid": False,
                "explanation_for_generator": f"An error occurred during generation: {str(e_gen)[:500]}",
                "overall_revision_summary_for_generator_prompt": "An error occurred during the previous generation attempt. Please try generating the scene again, focusing on the original task."
            })
            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(1)
                continue
            else:
                return None

        # 3. LLM Validate Beat
        # Construct validator user prompt
        validator_system_prompt = "You are an expert AI literary critic and logic checker. Your task is to evaluate a short story segment (a 'beat') that is supposed to narrate a specific mathematical operation. You must determine if the narrative accurately and coherently portrays this operation using the given inputs. Provide your feedback in a structured JSON format."
        
        validator_user_prompt = f"""World Context:
- Genre: {world_info.get("genre", "N/A")}
- Setting: {world_info.get("setting", "N/A")}
- Primary Object: {world_info.get("object", "items")}

Beat Task:
- Operation: {current_op_node.op} ({OP_LABELS.get(current_op_node.op, current_op_node.op)})
- Conceptual Inputs (from previous steps): {conceptual_inputs_str}
- New Atomic Inputs for this beat: {atomic_inputs_words_str}
- Intended Narrative Action: {action_description}
- Expected Numerical Result (for this beat, if not root): {expected_beat_result_words}

Generated Beat Text to Evaluate:
---
{generated_text_cleaned}
---

Validation Criteria:
1. Task Adherence: Does the narrative accurately reflect the specified 'Operation' and 'Intended Narrative Action'? (e.g., if SUM, does it describe combining things?)
2. Input Fidelity: Does the narrative correctly incorporate ALL 'Conceptual Inputs' and 'New Atomic Inputs'? Are any missing or ignored? Are any unmentioned inputs invented?
3. Narrative Coherence: Is the story segment logical and easy to understand in the context of the task?
4. Result Alignment (for non-root nodes if explicit mention is expected, or implicit alignment otherwise): Does the narrative lead to a situation consistent with the 'Expected Numerical Result'? (For root nodes, the result should NOT be stated).

Output Instructions:
Respond ONLY with a single JSON object.
If the beat is valid according to the criteria:
{{
  "is_valid": true,
  "explanation_for_audit": "Concise reason why it's valid."
}}
If the beat is NOT valid:
{{
  "is_valid": false,
  "explanation_for_generator": "Detailed explanation of the primary flaws for the generator LLM.",
  "overall_revision_summary_for_generator_prompt": "A concise (1-2 sentence) instruction for the generator on what to fix. E.g., 'Revise to ensure the narrative describes summing all inputs, including X and Y, instead of selecting the maximum.'",
  "suggested_revisions": [
    // Optional: array of specific suggestions if you can make them very concrete
    // {{ "type": "change_focus", "from": "problematic phrase", "to": "suggested focus/phrase", "reason": "..." }},
    // {{ "type": "ensure_mention", "item": "missing item", "reason": "..." }}
  ]
}}
"""  # Note: Double curly braces for JSON examples in f-string
        
        try:
            log_prompt(
                header=f"LLM Beat Validator Prompt (Iterative Loop Attempt {iteration}) Op: {current_op_node.op}",
                prompt=f"System: {validator_system_prompt}\nUser: {validator_user_prompt}",
                sample_index=sample_index
            )
            resp_val = _chat_completion_call(
                model=context_config.LLM_VALIDATOR_MODEL,
                messages=[
                    {"role": "system", "content": validator_system_prompt},
                    {"role": "user", "content": validator_user_prompt},
                ],
                max_completion_tokens=700, # Increased for potentially larger JSON feedback
                temperature=context_config.LLM_VALIDATOR_TEMP,
                reasoning={"exclude": True}
            )
            validator_raw_output = ""
            if resp_val and resp_val.choices and len(resp_val.choices) > 0 and resp_val.choices[0].message:
                 _content_val = resp_val.choices[0].message.content
                 validator_raw_output = _content_val if _content_val is not None else ""
            
            log_prompt(
                header=f"LLM Beat Validator Raw Response (Iterative Loop Attempt {iteration}) Op: {current_op_node.op}",
                prompt=f"Raw Output:\n{validator_raw_output}",
                sample_index=sample_index
            )

            try:
                # Using existing clean_and_parse_json_block for robust parsing
                validation_result = clean_and_parse_json_block(validator_raw_output) 
            except json.JSONDecodeError as e_json:
                logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Failed to parse LLM validator JSON in iteration {iteration}. Error: {e_json}. Raw: {validator_raw_output[:300]}...")
                validation_result = {
                    "is_valid": False,
                    "explanation_for_generator": "Validator response was not valid JSON or was empty.",
                    "overall_revision_summary_for_generator_prompt": "The validator's response was unparsable or empty. Please try generating the scene again, focusing on the original task and ensuring clear narrative structure."
                }
            
            history_of_critiques.append(validation_result) 

            if validation_result.get("is_valid"):
                logger_obj.info(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validator PASSED beat in iteration {iteration}. Audit: {validation_result.get('explanation_for_audit')}")
                return generated_text_cleaned # Success!
            else:
                logger_obj.warning(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] LLM Validator FAILED beat in iteration {iteration}. Feedback: {validation_result.get('explanation_for_generator')}")
                # Loop continues for next iteration with this feedback
        
        except Exception as e_val:
            logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error during LLM validation call/processing in iteration {iteration}: {e_val}")
            # Ensure history_of_critiques has a corresponding entry if an exception occurred before it was added.
            # Check if len(history_of_critiques) is less than len(history_of_attempts)
            if len(history_of_critiques) < len(history_of_attempts):
                history_of_critiques.append({
                    "is_valid": False,
                    "explanation_for_generator": f"An error occurred during validation: {str(e_val)[:500]}",
                    "overall_revision_summary_for_generator_prompt": "An error occurred during the validation step. Please try generating the scene again, focusing on the original task."
                })
            elif history_of_critiques: # If it exists, update the last one's reason if it was a generic one before this error.
                 # This case might be less common if an exception during call means no prior critique was added for *this specific validation attempt*.
                 # The more likely scenario is handled by the len check above.
                 pass

            if iteration < context_config.MAX_LLM_VALIDATION_ITERATIONS:
                time.sleep(1)
                continue
            else:
                return None # Max iterations reached due to validation error

    logger_obj.error(f"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Beat failed LLM validation after {context_config.MAX_LLM_VALIDATION_ITERATIONS} iterations.")
    return None # Failed all iterations 