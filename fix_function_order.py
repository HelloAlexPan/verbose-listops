#!/usr/bin/env python3

import re

def main():
    # Read the file
    with open('verbose-listops.py', 'r') as f:
        content = f.read()
    
    # Define the patterns to find the functions
    generate_and_validate_pattern = r'# --- NEW FUNCTION: _generate_and_llm_validate_beat \(Iterative LLM Validation Loop\) ---\s*\ndef _generate_and_llm_validate_beat\('
    narrative_recursive_pattern = r'# --- Narrative Generation with Parent Operator Prompting ---\s*\ndef _generate_narrative_recursive\('
    
    # Find the start and end of the _generate_and_llm_validate_beat function
    generate_and_validate_match = re.search(generate_and_validate_pattern, content)
    if not generate_and_validate_match:
        print("Could not find _generate_and_llm_validate_beat function")
        return
    
    # Find where the function starts and ends
    func_start = generate_and_validate_match.start()
    
    # Find the narrative function where we want to insert before
    narrative_recursive_match = re.search(narrative_recursive_pattern, content)
    if not narrative_recursive_match:
        print("Could not find _generate_narrative_recursive function")
        return
    
    # Find where to insert (right before _generate_narrative_recursive)
    insert_position = narrative_recursive_match.start()
    
    # Find where the function ends (searching from start position to the start of _generate_narrative_recursive)
    # Look for the next "def " or "# ---" pattern after our function starts
    func_content_pattern = re.compile(r'(def\s+\w+|# ---)', re.MULTILINE)
    next_func_match = None
    for match in func_content_pattern.finditer(content[func_start+1:]):
        if match.group() != "def _generate_and_llm_validate_beat":
            next_func_match = match
            break
    
    if not next_func_match:
        print("Could not find the end of _generate_and_llm_validate_beat function")
        return
    
    func_end = func_start + 1 + next_func_match.start()
    
    # Extract the function content
    function_content = content[func_start:func_end]
    
    # Remove the function from its original position
    if func_start < insert_position:
        # If function is before the insert position, removing it will shift the insert position
        new_content = content[:func_start] + content[func_end:]
        # Adjust insert position to account for removed content
        insert_position -= (func_end - func_start)
    else:
        # If function is after insert position (unlikely in this case), no adjustment needed
        new_content = content[:func_start] + content[func_end:]
    
    # Insert the function at the right position (before _generate_narrative_recursive)
    final_content = new_content[:insert_position] + function_content + new_content[insert_position:]
    
    # Write the modified content back to the file
    with open('verbose-listops.py', 'w') as f:
        f.write(final_content)
    
    print("Successfully moved _generate_and_llm_validate_beat function to before _generate_narrative_recursive")

if __name__ == "__main__":
    main() 