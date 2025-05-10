#!/usr/bin/env python3

def main():
    # Read the file
    with open('verbose-listops.py', 'r') as f:
        content = f.readlines()
    
    # Find function line numbers
    narrative_line = None
    validate_line = None
    
    for i, line in enumerate(content):
        if "def _generate_narrative_recursive" in line:
            narrative_line = i
        elif "def _generate_and_llm_validate_beat" in line:
            validate_line = i
    
    if narrative_line is None or validate_line is None:
        print(f"Could not find both functions: narrative_line={narrative_line}, validate_line={validate_line}")
        return
    
    print(f"Found functions at: narrative_line={narrative_line}, validate_line={validate_line}")
    
    # Determine function boundaries
    # For _generate_narrative_recursive
    narrative_end = validate_line
    for i in range(narrative_line + 1, len(content)):
        if "def " in content[i] and i != validate_line:
            narrative_end = i
            break
    
    # For _generate_and_llm_validate_beat
    validate_end = len(content)
    for i in range(validate_line + 1, len(content)):
        if "def " in content[i]:
            validate_end = i
            break
    
    print(f"Function spans: narrative={narrative_line}:{narrative_end}, validate={validate_line}:{validate_end}")
    
    # Extract both function contents
    narrative_func = content[narrative_line:narrative_end]
    validate_func = content[validate_line:validate_end]
    
    # Create new content with functions swapped
    new_content = (
        content[:narrative_line] +  # Content before first function
        validate_func +             # The validate function now comes first
        narrative_func +            # The narrative function now comes second
        content[validate_end:]      # Content after both functions
    )
    
    # Write the modified content back to the file
    with open('verbose-listops.py', 'w') as f:
        f.writelines(new_content)
    
    print("Successfully swapped function positions")

if __name__ == "__main__":
    main() 