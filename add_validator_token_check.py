#!/usr/bin/env python3

def main():
    with open('verbose-listops.py', 'r') as f:
        lines = f.readlines()
    
    # Find the location to insert the validator token check
    validator_prompt_end = None
    
    for i, line in enumerate(lines):
        if '"""  # Note: Double curly braces for JSON examples in f-string' in line:
            validator_prompt_end = i
            break
    
    if validator_prompt_end is not None:
        # Create the token check code to insert
        token_check_code = [
            "        \n",
            "        # --- Token Count Check for Validator Prompt ---\n",
            "        validator_prompt_tokens = 0\n",
            "        if encoder_obj: # Check if encoder_obj is available\n",
            "            try:\n",
            "                validator_prompt_tokens = len(encoder_obj.encode(validator_system_prompt + validator_user_prompt))\n",
            "                # Add extra buffer for potentially long generated_text_cleaned\n",
            "                generated_text_tokens = len(encoder_obj.encode(generated_text_cleaned))\n",
            "                validator_prompt_tokens += generated_text_tokens\n",
            "                \n",
            "                logger_obj.debug(\n",
            "                    f\"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Validator prompt tokens: {validator_prompt_tokens} \"\n",
            "                    f\"(includes {generated_text_tokens} tokens from generated beat text)\"\n",
            "                )\n",
            "                \n",
            "                # Use a percentage of MODEL_MAX_CONTEXT_TOKENS for the validator (e.g., 80%)\n",
            "                VALIDATOR_MAX_CONTEXT_LIMIT = int(context_config.MODEL_MAX_CONTEXT_TOKENS * 0.8)\n",
            "                \n",
            "                # Check if validator prompt would exceed limit\n",
            "                if validator_prompt_tokens > VALIDATOR_MAX_CONTEXT_LIMIT:\n",
            "                    logger_obj.warning(\n",
            "                        f\"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Validator prompt too large: {validator_prompt_tokens} tokens \"\n",
            "                        f\"exceeds {VALIDATOR_MAX_CONTEXT_LIMIT} tokens (80% of max context). Truncating generated text.\"\n",
            "                    )\n",
            "                    \n",
            "                    # If too long, truncate the generated text to fit within limits\n",
            "                    # Calculate max tokens for generated text\n",
            "                    base_prompt_tokens = len(encoder_obj.encode(validator_system_prompt + validator_user_prompt.replace(generated_text_cleaned, \"\")))\n",
            "                    max_generated_text_tokens = VALIDATOR_MAX_CONTEXT_LIMIT - base_prompt_tokens - 500  # 500 token buffer\n",
            "                    \n",
            "                    if max_generated_text_tokens > 500:  # Ensure we keep at least something meaningful\n",
            "                        # Truncate the generated text\n",
            "                        truncated_text = generated_text_cleaned\n",
            "                        while len(encoder_obj.encode(truncated_text)) > max_generated_text_tokens and len(truncated_text) > 100:\n",
            "                            # Cut off ~20% from the end each time\n",
            "                            truncate_point = int(len(truncated_text) * 0.8)\n",
            "                            truncated_text = truncated_text[:truncate_point] + \"\\n[... text truncated for validation ...]\"\n",
            "                        \n",
            "                        # Update the prompt with truncated text\n",
            "                        validator_user_prompt = validator_user_prompt.replace(generated_text_cleaned, truncated_text)\n",
            "                        logger_obj.info(\n",
            "                            f\"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Truncated generated text for validator \"\n",
            "                            f\"from {generated_text_tokens} to ~{len(encoder_obj.encode(truncated_text))} tokens\"\n",
            "                        )\n",
            "            except Exception as e_encode:\n",
            "                logger_obj.error(\n",
            "                    f\"[Sample {sample_index+1}, Beat Op: {current_op_node.op}] Error encoding validator prompt for token count: {e_encode}. \"\n",
            "                    f\"Proceeding without token check.\"\n",
            "                )\n",
            "        # --- End Token Count Check ---\n",
        ]
        
        # Insert the token check code
        lines.insert(validator_prompt_end + 1, ''.join(token_check_code))
        
        # Write the modified content back to the file
        with open('verbose-listops.py', 'w') as f:
            f.writelines(lines)
        
        print(f"Added validator token check after line {validator_prompt_end + 1}")
    else:
        print("Could not find location to insert validator token check")

if __name__ == "__main__":
    main() 