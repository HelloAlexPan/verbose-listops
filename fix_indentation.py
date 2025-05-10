#!/usr/bin/env python3

def main():
    with open('verbose-listops.py', 'r') as f:
        lines = f.readlines()
    
    # Find the problematic section
    target_start = None
    target_end = None
    
    for i, line in enumerate(lines):
        if "# After the outer loop:" in line:
            target_start = i
            break
    
    if target_start is not None:
        # Look for the end of the block
        for i in range(target_start, len(lines)):
            if "# --- END OF MODIFIED BEAT GENERATION LOOP ---" in lines[i]:
                target_end = i
                break
    
    if target_start is not None and target_end is not None:
        # Create the corrected block
        fixed_block = [
            "    # After the outer loop:\n",
            "    if beat_text_final_validated:\n",
            "        beat_text = beat_text_final_validated  # This is the successfully generated and validated beat\n",
            "        # ... (rest of your existing logic: append to scenes, update tokens, etc.)\n",
            "    else:\n",
            "        logger.error(\n",
            "            f\"Operator {node.op} ({narrative_anchor}) failed after {config.MAX_BEAT_RETRIES} outer attempts (incl. LLM validation loops). Aborting narrative generation. {'(ROOT NODE)' if is_root else ''}\"\n",
            "        )\n",
            "        raise BeatGenerationError(\n",
            "            f\"Failed to generate narrative beat for operator {node.op} ({narrative_anchor}) after all outer retries.\"\n",
            "        )\n",
            "    # --- END OF MODIFIED BEAT GENERATION LOOP ---\n"
        ]
        
        # Replace the problematic section
        lines[target_start:target_end+1] = fixed_block
        
        # Write the fixed file
        with open('verbose-listops.py', 'w') as f:
            f.writelines(lines)
        
        print(f"Fixed indentation issue in section from line {target_start+1} to {target_end+1}")
    else:
        print("Could not find the target section to fix")

if __name__ == "__main__":
    main() 