import random
import textwrap

def generate_filler(paragraph_count=200, max_sentences=5):
    """Generate filler text with repetitive boilerplate and random sentences."""
    boilerplate = (
        "This paragraph is included for verbosity and does not affect the computation. "
        "Please ignore these details in your final reasoning."
    )
    filler_sentences = [
        "The system logs indicate archival of compliance records.",
        "This line serves as a redundant procedural note.",
        "Refer to the extended glossary for additional explanations.",
        "Fictional stakeholder commentary appears here.",
        "Mock legal disclaimers are inserted for length."
    ]
    paragraphs = []
    for _ in range(paragraph_count):
        sentences = [boilerplate] + random.sample(filler_sentences, k=random.randint(1, max_sentences))
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)

def generate_verbose_listops(problem_id, standard_expr, target_tokens=30000):
    """Generate a verbose ListOps narrative of at least target_tokens tokens."""
    base_template = textwrap.dedent(f"""
    Problem ID: {problem_id}

    **Overview & Introduction**
    This document is the Verbose ListOps Evaluation Benchmark problem identified as {problem_id}. 
    It contains extensive narrative sections, compliance boilerplate, fictional logs, and filler text 
    designed to stress-test long-context reasoning. Only sections explicitly labeled "Computation" 
    are relevant to the underlying ListOps task.

    **Standard ListOps Expression**
    `{standard_expr}`

    **Computation Sections**
    Computation Step 1: Parse the provided nested operations.
    Computation Step 2: Identify all sub-operations (`MAX`, `MIN`, `MED`, `SM`) and their operands.
    Computation Step 3: Execute each sub-operation in hierarchical order.
    Computation Step 4: Produce the final single-digit result.

    **Detailed Instructions**
    1. Review all narrative paragraphs and extract the numeric directives.
    2. Disregard any compliance logs, fictional stakeholder notes, mock change requests, 
       or repeated definitions outside the "Computation Sections".
    3. For `MAX`, find the highest integer; for `MIN`, find the lowest; for `MED`, find 
       the median using ListOps conventions; for `SM`, compute sum modulo 10.
    4. Maintain hierarchical order to ensure sub-operations are resolved before parent operations.

    """)

    narrative = base_template
    # Append filler until we reach desired token length
    filler_block = generate_filler(paragraph_count=250)
    narrative += "\n\n" + filler_block

    # Estimate tokens by simple split
    while len(narrative.split()) < target_tokens:
        narrative += "\n\n" + generate_filler(paragraph_count=50)
    
    # Append answer placeholder
    answer_placeholder = textwrap.dedent("""
    **Answer:** <compute the final single-digit result here>
    """)
    narrative += "\n\n" + answer_placeholder
    return narrative

# Example usage: generate a single verbose eval
standard_expressions = [
    "[SM 8 1 4 [MAX 9 2 7]]",
    "[MIN [SM 3 4] 7 5]",
    "[MED 5 2 [MIN 8 6 3] 9]",
    # ... up to 20 expressions
]

for i, expr in enumerate(standard_expressions, start=1):
    problem_id = f"VLO-{i:03d}"
    verbose_problem = generate_verbose_listops(problem_id, expr, target_tokens=30000)
    print(verbose_problem[:1000] + "\n\n... [truncated] ...\n")