#!/usr/bin/env python3

# Script to move _generate_and_llm_validate_beat function
# from its current location to before _generate_narrative_recursive


def main():
    # Read the file
    with open("verbose-listops.py", "r") as f:
        lines = f.readlines()

    # Extract the function (from the start of the func to right before _generate_narrative_recursive)
    # Line numbers are 0-indexed in Python but grep shows 1-indexed
    func_lines = lines[2868:3183]

    # Remove the function from its current location
    del lines[2868:3183]

    # Insert the function just before _generate_narrative_recursive
    lines = lines[:3183] + func_lines + lines[3183:]

    # Write the file back
    with open("verbose-listops.py", "w") as f:
        f.writelines(lines)

    print(
        "Successfully moved function _generate_and_llm_validate_beat to before _generate_narrative_recursive"
    )


if __name__ == "__main__":
    main()
