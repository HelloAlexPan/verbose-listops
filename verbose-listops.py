import os
import json
import random
from dataclasses import dataclass, field

import tiktoken
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# ─── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
MODEL   = "claude-2"
MAX_TOTAL_TOKENS   = 100_000
SAFETY_MARGIN      = 1_000  # keep a cushion
MAX_BEAT_TOKENS    = 2_000  # per operator-scene
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

# ─── 4) World-builder (one-time) ──────────────────────────────────────────────

def generate_world(num_characters: int = 5) -> dict:
    prompt = (
        f"{HUMAN_PROMPT}"
        "You are a world-builder.\n"
        f"Create {num_characters} vivid characters (name, role, quirk), choose a genre and setting.\n"
        "Output as JSON in exactly this shape:\n"
        "{\n"
        '  "characters": [ { "name": "...", "role": "...", "quirk": "..." }, … ],\n'
        '  "genre": "...",\n'
        '  "setting": "..." \n'
        "}\n"
        f"{AI_PROMPT}"
    )
    resp = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens_to_sample=2_000,
        stop_sequences=[HUMAN_PROMPT]
    )
    return json.loads(resp.completion)

# ─── 5) Narrative generator ───────────────────────────────────────────────────

def generate_narrative(ast: Node, world: dict) -> str:
    scenes = []
    tokens_used = 0

    for node in postorder(ast):
        # Only generate for operator nodes
        if isinstance(node, Atom):
            continue

        # 5a) Operator beat
        operands = [c.value for c in node.children]
        beat_prompt = (
            f"{HUMAN_PROMPT}"
            f"You are a {world['genre']} storyteller.\n"
            f"Characters: {json.dumps(world['characters'])}\n"
            f"Setting: {world['setting']}\n\n"
            "Here is an operation:\n"
            f"Operator: {node.op}\n"
            f"Operands: {operands}\n"
            f"Result: {node.value}\n\n"
            "Write 1 scene (2–5 short paragraphs) showing how this operation’s logic "
            "plays out among the characters. Do NOT reveal the numeric answer as a meta clue.\n"
            f"{AI_PROMPT}"
        )
        resp = client.completions.create(
            model=MODEL,
            prompt=beat_prompt,
            max_tokens_to_sample=MAX_BEAT_TOKENS,
            stop_sequences=[HUMAN_PROMPT]
        )
        beat = resp.completion
        btoks = len(encoder.encode(beat))
        if tokens_used + btoks > MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            break
        scenes.append(beat)
        tokens_used += btoks

        # 5b) Padding loop
        pad_prompt = (
            f"{HUMAN_PROMPT}"
            "You are the same storyteller.\n"
            "Continue the last scene’s style in 2–3 short paragraphs—introduce side-quests, "
            "mysteries, or random asides. DO NOT change any established facts or operator logic. "
            "This is pure padding.\n"
            f"{AI_PROMPT}"
        )
        while tokens_used < MAX_TOTAL_TOKENS - SAFETY_MARGIN:
            pad_resp = client.completions.create(
                model=MODEL,
                prompt=pad_prompt,
                max_tokens_to_sample=MAX_PAD_TOKENS,
                stop_sequences=[HUMAN_PROMPT]
            )
            pad = pad_resp.completion
            ptoks = len(encoder.encode(pad))
            if tokens_used + ptoks > MAX_TOTAL_TOKENS:
                break
            scenes.append(pad)
            tokens_used += ptoks

        if tokens_used >= MAX_TOTAL_TOKENS:
            break

    # 5c) Final question
    top_op = ast.op
    question = (
        f"{HUMAN_PROMPT}"
        f"Question: What was the {top_op} result at the top level of the story above?\n"
        f"Answer:{AI_PROMPT}"
    )
    scenes.append(question)

    return "\n\n".join(scenes)

# ─── 6) Main flow ──────────────────────────────────────────────────────────────

def main():
    # 1) Build & evaluate AST
    ast = build_random_ast(max_ops=10, max_branch=3)  # tweak size as you like
    eval_node(ast)

    # 2) Generate world
    world = generate_world(num_characters=5)

    # 3) Render narrative
    narrative = generate_narrative(ast, world)

    # 4) Emit
    print(narrative)

if __name__ == "__main__":
    main()