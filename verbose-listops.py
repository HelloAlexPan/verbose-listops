#!/usr/bin/env python3
print("verbose-listops: script started")
"""
verbose-listops.py

Hybrid DSL + Anthropic Claude narrative generator for ListOps puzzles.
Parses a ListOps expression, generates core narrative fragments via Jinja2,
then expands each fragment into rich, filler‑laden paragraphs with Claude.
"""

import os
import re
import random
import argparse
from jinja2 import Template
import anthropic
import sys
import threading
import time

class Spinner:
    spinner_cycle = ['-', '\\', '|', '/']
    def __init__(self, message="Loading"):
        self.message = message
        self.stop_running = False
        self.thread = threading.Thread(target=self._spin)

    def _spin(self):
        idx = 0
        while not self.stop_running:
            sys.stdout.write(f"\r{self.message} {self.spinner_cycle[idx]}")
            sys.stdout.flush()
            idx = (idx + 1) % len(self.spinner_cycle)
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_running = True
        self.thread.join()

# 1. DSL Parsing ------------------------------------------------------------
TOKEN_REGEX = r"\[|\]|[^\s\[\]]+"

def tokenize(expr: str):
    return re.findall(TOKEN_REGEX, expr)

def parse_tokens(tokens):
    token = tokens.pop(0)
    if token == '[':
        lst = []
        while tokens[0] != ']':
            lst.append(parse_tokens(tokens))
        tokens.pop(0)
        return lst
    else:
        return int(token) if token.isdigit() else token

def parse_expr(expr: str):
    return parse_tokens(tokenize(expr))


# 2. Template Fragment ------------------------------------------------------
TEMPLATE = """
{% if node is iterable and node[0] is string %}
The operator "{{ node[0] }}" is applied to operands {% for child in node[1:] %}{{ child }}{% if not loop.last %}, {% endif %}{% endfor %}.
{% else %}
The value "{{ node }}" appears in the sequence.
{% endif %}
"""

def generate_fragment_with_template(node):
    tpl = Template(TEMPLATE)
    return tpl.render(node=node)


# 3. Anthropic Expansion ----------------------------------------------------
def expand_with_llm(fragment: str, seed: int, model: str = None, max_tokens: int = 512) -> str:
    # Add API key check
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        exit(1)
    spinner = Spinner("Requesting Claude API")
    spinner.start()
    random.seed(seed)
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model or os.getenv("LLM_MODEL", "claude-3-7-sonnet-20250219"),
            messages=[{
                "role": "user",
                "content": (
                    f"Expand the following fragment into a rich, coherent paragraph "
                    f"embedding logical filler: '{fragment}'"
                )
            }],
            max_tokens=max_tokens
        )
    finally:
        spinner.stop()
    # Extract text from response content blocks
    # If the SDK returns a legacy response with parse()
    if hasattr(response, "parse"):
        message = response.parse()
        return message.content
    # Otherwise, response.content may be a list of dicts
    blocks = getattr(response, "content", None) or getattr(response, "completion", None)
    if isinstance(blocks, list):
        text = ""
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                text += block["text"]
            else:
                text += getattr(block, "text", "")
        return text
    # Fallback to string
    return str(blocks)


# 4. Traverse & Generate ----------------------------------------------------
def traverse_and_generate(ast_node, seed_start=0, model=None, target_tokens=40000):
    fragments = []
    def recurse(node):
        frag = generate_fragment_with_template(node)
        fragments.append((frag, seed_start + len(fragments)))
        if isinstance(node, list) and len(node) > 1:
            for child in node[1:]:
                recurse(child)
    recurse(ast_node)
    print(f"Fragments to expand: {len(fragments)}")
    per_fragment_tokens = max(1, target_tokens // len(fragments))
    print(f"Per-fragment token budget: {per_fragment_tokens}")
    paragraphs = [
        expand_with_llm(f, sd, model, max_tokens=per_fragment_tokens)
        for f, sd in fragments
    ]
    return "\n\n".join(paragraphs)


# 5. CLI ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a verbose ListOps narrative via Anthropic Claude"
    )
    parser.add_argument(
        "expression",
        help="ListOps DSL expression, e.g. [SM 8 1 4 [MAX 9 2 7]]"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Anthropic Claude model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.txt",
        help="Output narrative file"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=40000,
        help="Total desired token count for the narrative"
    )
    args = parser.parse_args()
    print(f"Expression: {args.expression}, seed: {args.seed}, model: {args.model}, output: {args.output}")

    try:
        ast_tree = parse_expr(args.expression)
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return

    print("Parsing complete, invoking generation...")
    narrative = traverse_and_generate(
        ast_tree,
        seed_start=args.seed,
        model=args.model,
        target_tokens=args.target_tokens
    )
    print(f"Generated narrative length: {len(narrative)} characters")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(narrative)
    print(f"Narrative written to {args.output}")


if __name__ == "__main__":
    main()