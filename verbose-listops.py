# README: Things you can edit:
# - max_ops: size of your ListOps problem
# - max_branch: maximum branching factor for AST nodes
# - MAX_BEAT_TOKENS / MAX_PAD_TOKENS: per-call length
# - MAX_TOTAL_TOKENS: overall token cap

import os
import json
import random
from dataclasses import dataclass, field

import tiktoken
import anthropic
from anthropic import Anthropic

# ─── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
MODEL   = "claude-3-7-sonnet-latest"
MAX_TOTAL_TOKENS   = 10_000
SAFETY_MARGIN      = 1_000  # keep a cushion
MAX_BEAT_TOKENS    = 1_000  # per operator-scene
MAX_PAD_TOKENS     = 1_000  # per padding-scene

# ─── Anthropic client & tokenizer ─────────────────────────────────────────────

client   = Anthropic(api_key=API_KEY)
encoder  = tiktoken.get_encoding("cl100k_base")

# ─── AST definitions ────────────────────────────────────────────────────────────

@dataclass
class Node:
    op: str
    children: list = field(default_factory=list)
    value: int = None

@dataclass
class Atom(Node):
    n: int = None
    def __init__(self, n: int):
        super().__init__(op="ATOM", children=[])
        self.n = n
        self.value = n

@dataclass
class OpNode(Node):
    def __init__(self, op: str, children: list):
        super().__init__(op=op, children=children)
        self.value = None

# ─── 1) Random AST generator ──────────────────────────────────────────────────

def build_random_ast(max_ops: int, max_branch: int = 3):
    """Top-down build of up to max_ops operator nodes."""
    ops = ["MAX","MIN","MED","SUM","SM","AVG"]
    count = 0

    def helper():
        nonlocal count
        # If we've used up our operators budget, emit an Atom
        if count >= max_ops:
            return Atom(random.randint(0, 9))
        # Otherwise create an operator node
        count += 1
        op = random.choice(ops)
        arity = random.randint(2, max_branch)
        children = [helper() for _ in range(arity)]
        return OpNode(op, children)

    return helper()

# ─── 2) AST evaluator ─────────────────────────────────────────────────────────

def eval_node(node: Node) -> int:
    if isinstance(node, Atom):
        return node.n
    vals = [eval_node(c) for c in node.children]
    func_map = {
        "MAX": max,
        "MIN": min,
        "MED": lambda v: sorted(v)[len(v)//2],
        "SUM": sum,
        "SM":  lambda v: sum(v) % 10,
        "AVG": lambda v: sum(v)//len(v),
    }
    node.value = func_map[node.op](vals)
    return node.value

# ─── 3) Post-order traversal ──────────────────────────────────────────────────

def postorder(node: Node):
    for c in node.children:
        yield from postorder(c)
    yield node

def preorder(node: Node):
    """Yield nodes in pre-order (node before children)."""
    yield node
    for c in node.children:
        yield from preorder(c)

# ─── 4) World-builder (one-time) ──────────────────────────────────────────────

def generate_world(num_characters: int = 5) -> dict:
    prompt = (
        "You are a world-builder.\n"
        f"Create {num_characters} vivid characters (name, role, quirk), choose a genre and setting.\n"
        "Output as JSON in exactly this shape. Respond with *only* the JSON object, no extra text or markdown:\n"
        "{\n"
        '  "characters": [ { "name": "...", "role": "...", "quirk": "..." }, … ],\n'
        '  "genre": "...",\n'
        '  "setting": "..." \n'
        "}\n"
    )
    resp = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    # parse the world-builder response, allowing for stray characters
    text = resp.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # attempt to extract JSON substring
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        # fallback: show raw response for debugging
        print("Failed to parse JSON from world-builder response:", text)
        raise

# ─── 5) Narrative generator ───────────────────────────────────────────────────

def generate_narrative(ast: Node, world: dict) -> str:
    scenes = []
    tokens_used = 0

    # Prepare operator list and show total count for progress
    # Use pre-order so the root (top-level operator) is processed first
    operator_nodes = [n for n in preorder(ast) if not isinstance(n, Atom)]
    max_pad_paragraphs = 2  # limit padding per operator
    total_ops = len(operator_nodes)
    print(f"Starting narrative generation: {total_ops} operator beats to process")

    for idx, node in enumerate(operator_nodes, start=1):
        print(f"Processing operator {idx}/{total_ops}: {node.op} with operands {[c.value for c in node.children]}")

        # Only generate for operator nodes
        if isinstance(node, Atom):
            continue

        # 5a) Operator beat
        operands = [c.value for c in node.children]
        beat_prompt = (
            f"You are a {world['genre']} storyteller.\n"
            f"Characters: {json.dumps(world['characters'])}\n"
            f"Setting: {world['setting']}\n\n"
            "Here is an operation:\n"
            f"Operator: {node.op}\n"
            f"Operands: {operands}\n"
            f"Result: {node.value}\n\n"
            "Write 1 scene (2–5 short paragraphs) showing how this operation's logic "
            "plays out among the characters. Do NOT reveal the numeric answer as a meta clue."
        )
        resp = client.messages.create(
            model=MODEL,
            system="You are a storyteller. Write plain narrative paragraphs without any markdown headings or section titles.",
            messages=[{"role":"user","content":beat_prompt}],
            max_tokens=MAX_BEAT_TOKENS
        )
        beat = resp.content[0].text
        last_scene = beat
        btoks = len(encoder.encode(beat))
        scenes.append(beat)
        tokens_used += btoks

        # 5b) Padding loop
        pad_prompt = (
            "You are the same storyteller.\n"
            "Continue the last scene below in 2–3 short paragraphs—introduce side-quests, mysteries, or random asides.\n"
            "Last scene:\n"
            f"{last_scene}\n\n"
            "Do NOT change any established facts or operator logic. This is pure padding."
        )
        pad_count = 0
        while tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN and pad_count < max_pad_paragraphs:
            pad_resp = client.messages.create(
                model=MODEL,
                system="Continue the narrative without adding any headings or titles, just plain paragraphs.",
                messages=[{"role":"user","content":pad_prompt}],
                max_tokens=MAX_PAD_TOKENS
            )
            pad = pad_resp.content[0].text
            # break if model refuses due to missing context
            if pad.strip().lower().startswith("i do not have enough context") or pad.strip().lower().startswith("i'm sorry"):
                print(f"Padding refused at operator {idx}, stopping padding.")
                break
            ptoks = len(encoder.encode(pad))
            if tokens_used + ptoks > MAX_TOTAL_TOKENS:
                break
            scenes.append(pad)
            tokens_used += ptoks
            pad_count += 1

        if tokens_used >= MAX_TOTAL_TOKENS:
            break

    # 5c) Final question
    top_op = ast.op
    question = f"Question: What was the {top_op} result at the top level of the story above?\nAnswer:"
    scenes.append(question)

    return "\n\n".join(scenes)

# ─── 6) Main flow ──────────────────────────────────────────────────────────────

def main():
    print("Building random AST...")
    # 1) Build & evaluate AST
    ast = build_random_ast(max_ops=10, max_branch=5)  # tweak as you like
    print("Evaluating AST...")
    eval_node(ast)
    print("AST evaluation complete.")

    print("Generating world metadata...")
    # 2) Generate world
    world = generate_world(num_characters=5)
    print("World metadata generated.")
    print("Starting narrative rendering...")

    # 3) Render narrative
    narrative = generate_narrative(ast, world)
    print("Narrative rendering complete.")

    # 4) Emit
    print(narrative)

if __name__ == "__main__":
    main()