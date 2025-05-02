"""
verbose-listops.py

1. Generates a complex ListOps problem as an Abstract Syntax Tree (AST).
2. Evaluates the AST to get the ground truth result.
3. Generates fictional world metadata (characters, genre, setting) via the Anthropic API
   (optionally in a single Batch API job, controlled by USE_BATCH_API_FOR_WORLDGEN).
4. Renders a narrative where each step of the ListOps calculation is a story 'beat',
   interspersed with optional 'padding' paragraphs, using the Anthropic API.
5. Constructs a final prompt including the narrative and instructions for a 'judge' LLM
   to solve the original ListOps problem based *only* on the narrative.
6. Supports batch generation and saving samples to a JSONL file.

Logging, API retry logic with exponential back‑off, and AST validation included.
Configuration via constants.
"""


import os
import json
import random
import datetime
import logging
import logging.handlers
import time
import tempfile                               # NEW – used for batch input files
from typing import Callable, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import tiktoken
import anthropic
from anthropic import Anthropic
from anthropic.types import Message           # NEW – type hints
from anthropic.types.beta.batches import Batch  # NEW – type hints
import concurrent.futures

# ─── Configuration Constants ────────────────────────────────────────────────────────────────────────────

NUM_SAMPLES_TO_GENERATE = 4                   # How many samples to generate
OUTPUT_FILENAME = "verbose_listops_dataset.jsonl"
DEFAULT_MAX_WORKERS = 8                       # ThreadPool workers

# --- Output configuration ---
LOG_DIR = os.path.expanduser("~/verbose_listops_logs")
DEFAULT_MAX_TOTAL_TOKENS = 10_000
DEFAULT_MAX_BEAT_TOKENS = 1_000
DEFAULT_MAX_PAD_TOKENS  = 1_000
MAX_TOKENS_BUFFER       = 1_000
PROMPT_SHOT_COUNT       = 3

SHOT_EXAMPLES = {
    0: "",
    1: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts: one paying 9 silver pieces, the other only 4. "
        "Kaelen chose the lower‑paying contract to avoid scrutiny. He then received a standard 5 silver piece bonus for completing the task quickly.\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n\n"
        "</Prompt Shot>\n"
    ),
    2: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts: one paying 9 silver pieces, the other only 4 …\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n\n"
        "Example 2:\n"
        "Narrative: “To unlock the ancient vault, the combined energy signature of four power crystals "
        "(reading 1, 1, 1, and 1) was required …”\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Mod 10 = 4.\n"
        "Answer: 4\n\n"
        "</Prompt Shot>\n"
    ),
    3: (
        "<Prompt Shot>\n"
        "Example 1:\n"
        "Narrative: \"The guild offered two contracts …\"\n"
        "Implicit Calculation: MIN(9, 4) = 4. Then SUM(4, 5) = 9.\n"
        "Answer: 9\n\n"
        "Example 2:\n"
        "Narrative: “To unlock the ancient vault …”\n"
        "Implicit Calculation: SUM(1, 1, 1, 1) = 4. Mod 10 = 4.\n"
        "Answer: 4\n\n"
        "Example 3:\n"
        "Narrative: “Three scouts reported patrol durations of 5, 5, and 5 hours …”\n"
        "Implicit Calculation: AVG = 5.\n"
        "Answer: 5\n"
        "</Prompt Shot>\n"
    ),
}

# --- AST generation params ---
DEFAULT_MAX_BRANCH = 3
ATOM_MIN_VALUE     = 0
ATOM_MAX_VALUE     = 9
MIN_ARITY          = 2
DEFAULT_MAX_OPS    = 10

# --- API / logging ---
RETRY_MAX_ATTEMPTS = 5
RETRY_INITIAL_DELAY = 1
LOG_MAX_BYTES      = 5 * 1024 * 1024
LOG_BACKUP_COUNT   = 3

API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "YOUR_API_KEY_HERE"

# Model note: use Haiku for cheap batch testing; switch to Sonnet for prod
MODEL = "claude-3-haiku-20240307"

MAX_TOTAL_TOKENS = DEFAULT_MAX_TOTAL_TOKENS
SAFETY_MARGIN    = MAX_TOKENS_BUFFER
MAX_BEAT_TOKENS  = DEFAULT_MAX_BEAT_TOKENS
MAX_PAD_TOKENS   = DEFAULT_MAX_PAD_TOKENS

# --- Batch mode flags ---
USE_BATCH_API_FOR_WORLDGEN       = False      # ← flip to True to enable batch
BATCH_POLLING_INTERVAL_SECONDS   = 15
BATCH_COMPLETION_TIMEOUT_SECONDS = 1_800      # 30 min

# ─── Logging setup ────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("verbose_listops")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "verbose_listops.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

# ─── Anthropic client / tokenizer ─────────────────────────────────────────────
try:
    client  = Anthropic(api_key=API_KEY, timeout=60.0, connect_timeout=15.0)
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.error(f"Failed to init Anthropic client / tokenizer: {e}")
    client = None
    encoder = None


def retry_api_call(func: Callable):
    def wrapper(*args, **kwargs):
        if client is None:
            raise RuntimeError("Anthropic client not initialized.")
        delay = RETRY_INITIAL_DELAY
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except (anthropic.APIConnectionError,
                    anthropic.RateLimitError) as e:
                logger.warning(f"{func.__name__} attempt {attempt}/{RETRY_MAX_ATTEMPTS} failed: {e}")
            except anthropic.APIStatusError as e:
                logger.warning(f"{func.__name__} status error {e.status_code}: {e.response}")
            except Exception as e:
                logger.warning(f"{func.__name__} generic error: {e}")
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            time.sleep(delay)
            delay *= 2
    return wrapper

@retry_api_call
def _client_create(**kwargs) -> Message:
    return client.messages.create(**kwargs)


@dataclass
class Node:
    op: str
    children: list = field(default_factory=list)
    value: int = None

@dataclass
class Atom(Node):
    n: int = None
    def __init__(self, n: int):
        super().__init__("ATOM", [])
        self.n = n
        self.value = n

@dataclass
class OpNode(Node):
    def __init__(self, op: str, children: list):
        super().__init__(op, children)

# ─── AST helpers ──────────────────────────────────────────────────────────────
def build_random_ast(max_ops: int, max_branch: int = DEFAULT_MAX_BRANCH) -> Node:
    if max_ops < 1:
        raise ValueError("max_ops must be positive")
    ops   = ["MAX", "MIN", "MED", "SUM", "SM", "AVG"]
    count = 0
    def helper():
        nonlocal count
        if count >= max_ops or (count > 0 and random.random() < 0.1):
            return Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE))
        count += 1
        op = random.choice(ops)
        arity = random.randint(MIN_ARITY, max_branch)
        return OpNode(op, [helper() for _ in range(arity)])
    root = helper()
    if isinstance(root, Atom):    # guarantee at least 1 op
        root = OpNode(random.choice(ops), [root, Atom(random.randint(ATOM_MIN_VALUE, ATOM_MAX_VALUE))])
    return root

def validate_ast(node: Node):
    if node.op not in {"ATOM", "MAX","MIN","MED","SUM","SM","AVG"}:
        raise ValueError(f"Unsupported op {node.op}")
    for c in node.children:
        validate_ast(c)

def eval_node(node: Node) -> int:
    if isinstance(node, Atom):
        return node.value
    vals = [eval_node(c) for c in node.children]
    func = {
        "MAX": max,
        "MIN": min,
        "MED": lambda v: sorted(v)[len(v)//2],
        "SUM": sum,
        "SM" : lambda v: sum(v) % 10,
        "AVG": lambda v: sum(v)//len(v),
    }[node.op]
    node.value = func(vals)
    return node.value

def postorder(n: Node):
    for c in n.children: yield from postorder(c)
    yield n


# ─── World‑building helpers ───────────────────────────────────────────────────
def _get_world_gen_prompt(num_chars: int) -> str:
    return (
        "You are a creative world‑builder.\n"
        f"Generate {num_chars} vivid characters …\n"
        "Output only JSON of the form { \"characters\": […], \"genre\": …, \"setting\": … }\n"
    )

def _parse_world_response(text: str, num_chars: int) -> Dict[str, Any]:
    if text.startswith("```json"): text = text[7:].strip()
    if text.startswith("```"):     text = text[3:].strip()
    if text.endswith("```"):       text = text[:-3].strip()
    data = json.loads(text)
    if ("characters" not in data or "genre" not in data or "setting" not in data
        or not isinstance(data["characters"], list)
        or len(data["characters"]) != num_chars):
        raise RuntimeError("Invalid world JSON")
    return data

# ─── World generation (standard) ──────────────────────────────────────────────
def generate_world(num_characters: int = 5) -> dict:
    prompt = _get_world_gen_prompt(num_characters)
    resp   = _client_create(model=MODEL,
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=2_000,
                            temperature=0.8)
    text = resp.content[0].text.strip()
    return _parse_world_response(text, num_characters)

# Batch helpers ---------------------------------------------------------------
def prepare_batch_input_file(reqs: List[Dict[str,Any]]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in reqs: f.write(json.dumps(r)+"\n")
    logger.info(f"Batch input file: {f.name}")
    return f.name

def run_world_generation_batch_job(input_file: str) -> Optional[Batch]:
    with open(input_file, "rb") as f:
        uploaded = client.beta.batches.upload(file=f)
    batch = client.beta.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/messages",
        completion_window="24h"
    )
    return batch

def poll_batch_job(batch_id: str) -> Optional[Batch]:
    start = time.time()
    while time.time()-start < BATCH_COMPLETION_TIMEOUT_SECONDS:
        bj = client.beta.batches.retrieve(batch_id=batch_id)
        if bj.status == "completed":
            return bj
        if bj.status in {"failed","cancelled","expired"}:
            logger.error(f"Batch {batch_id} ended with status {bj.status}")
            return None
        time.sleep(BATCH_POLLING_INTERVAL_SECONDS)
    logger.error("Batch job timed out")
    return None

def process_batch_results(bj: Batch, expected: int) -> Dict[str, Dict[str,Any]]:
    out_id = bj.output_file_id
    content = client.files.content(file_id=out_id)
    mapping: Dict[str, Dict[str,Any]] = {}
    for line in content.strip().splitlines():
        item = json.loads(line)
        cid  = item["custom_id"]
        body = item["response"]["body"]["content"][0]["text"]
        num_chars = int(cid.split("_")[3])     # sample_{i}_chars_{n}
        mapping[cid] = _parse_world_response(body, num_chars)
    if len(mapping)!=expected:
        logger.warning("Mismatch processed vs expected results")
    return mapping

# ─── Narrative generation (unchanged except for max_pad_paragraphs fix) ───────
OP_LABELS = {
    "MAX":"largest value","MIN":"smallest value","SUM":"sum",
    "MED":"median value","AVG":"average (floored)","SM":"sum mod 10"
}

def generate_narrative(ast: Node, world: dict) -> str:
    if encoder is None: raise RuntimeError("Tokenizer not init")
    scenes, tok_used = [], 0
    log_path = os.path.join(LOG_DIR,"verbose_listops_prompts.log")
    operator_nodes = [n for n in postorder(ast) if not isinstance(n, Atom)]
    if not operator_nodes and isinstance(ast, Atom):
        prompt = (
            f"You are a {world['genre']} storyteller.\nCharacters: {json.dumps(world['characters'])}\n"
            f"Setting: {world['setting']}\n\nWrite a short scene featuring the number {ast.value}."
        )
        resp = _client_create(model=MODEL,
                              system="You are a storyteller.",
                              messages=[{"role":"user","content":prompt}],
                              max_tokens=MAX_BEAT_TOKENS)
        scenes.append(resp.content[0].text)
        tok_used += len(encoder.encode(scenes[-1]))

    max_pad_paragraphs = 2
    last_scene = "The story begins..."
    for idx,node in enumerate(operator_nodes,1):
        is_final = node is ast
        operands = [c.value for c in node.children]
        result   = node.value
        beat_prompt = (
            f"You are a creative {world['genre']} storyteller.\nCharacters: {json.dumps(world['characters'])}\n"
            f"Setting: {world['setting']}\nPrevious: \"...{last_scene[-150:]}\"\n\n"
            "--- Final Task ---\n" if is_final else "--- Current Task ---\n"
        )
        beat_prompt += (
            f"* Operation: {node.op} ({OP_LABELS[node.op]})\n* Inputs: {operands}\n"
            f"* Result: {result} (do NOT reveal)\n\n"
            "Write 2‑5 short paragraphs continuing the narrative without exposing the result number."
        )
        resp = _client_create(model=MODEL,
                              system="You are a storyteller.",
                              messages=[{"role":"user","content":beat_prompt}],
                              max_tokens=MAX_BEAT_TOKENS, temperature=0.7)
        txt = resp.content[0].text.strip()
        scenes.append(txt)
        tok_used += len(encoder.encode(txt))
        last_scene = txt
        # optional padding
        pad_cnt = 0
        while not is_final and pad_cnt < max_pad_paragraphs and tok_used < MAX_TOTAL_TOKENS-SAFETY_MARGIN:
            pad_prompt = (
                f"Continue the narrative with 1‑3 paragraphs of atmosphere or character reflection. "
                f"Do not add numbers or calculations.\nPrevious: \"...{last_scene[-150:]}\""
            )
            pad_resp = _client_create(model=MODEL,
                                      system="You are a storyteller.",
                                      messages=[{"role":"user","content":pad_prompt}],
                                      max_tokens=MAX_PAD_TOKENS, temperature=0.6)
            ptxt = pad_resp.content[0].text.strip()
            scenes.append(ptxt)
            tok_used += len(encoder.encode(ptxt))
            last_scene = ptxt
            pad_cnt += 1
            if tok_used >= MAX_TOTAL_TOKENS-SAFETY_MARGIN: break

    body = "\n\n".join(scenes)
    question = "\n\n---\n\nAnalyze the narrative and provide the single final numerical result."
    judge = "\n\n**Final Answer:**"
    few = SHOT_EXAMPLES[PROMPT_SHOT_COUNT]
    return (few + body + question + judge).strip()

# ─── Sample generation helpers ────────────────────────────────────────────────
def generate_narrative_for_sample(idx:int, ast:Node, gt:int, ast_prefix:str, world:Dict[str,Any]) -> Optional[Dict[str,Any]]:
    try:
        prompt = generate_narrative(ast, world)
        return {
            "id": f"verbose_listop_{datetime.datetime.now():%Y%m%d%H%M%S}_{idx+1}",
            "ast_prefix": ast_prefix,
            "ground_truth": gt,
            "world_info": world,
            "narrative_prompt": prompt,
            "metadata":{
                "generation_timestamp": datetime.datetime.now().isoformat(),
                "model_used": MODEL,
                "max_ops": DEFAULT_MAX_OPS,
                "max_branch": DEFAULT_MAX_BRANCH,
                "prompt_shot_count": PROMPT_SHOT_COUNT,
                "generation_mode": "batch_worldgen" if USE_BATCH_API_FOR_WORLDGEN else "standard"
            }
        }
    except Exception as e:
        logger.error(f"[Sample {idx+1}] narrative error: {e}")
        return None

def generate_single_sample(idx:int) -> Optional[Dict[str,Any]]:
    try:
        ast = build_random_ast(DEFAULT_MAX_OPS)
        validate_ast(ast)
        gt  = eval_node(ast)
        ast_prefix = ast_to_prefix(ast)
        world = generate_world(random.randint(3,6))
        return generate_narrative_for_sample(idx, ast, gt, ast_prefix, world)
    except Exception as e:
        logger.error(f"[Sample {idx+1}] generation error: {e}")
        return None

def ast_to_prefix(n: Node) -> str:
    if isinstance(n, Atom):
        return str(n.n)
    return "(" + " ".join([n.op] + [ast_to_prefix(c) for c in n.children]) + ")"

# ─── Main orchestration ───────────────────────────────────────────────────────
def main(num_samples:int=NUM_SAMPLES_TO_GENERATE, output_file:str=OUTPUT_FILENAME, max_workers:int=DEFAULT_MAX_WORKERS):
    if client is None or encoder is None:
        logger.critical("Client or tokenizer not ready.")
        return
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results: List[Dict[str,Any]] = []
    if not USE_BATCH_API_FOR_WORLDGEN:
        logger.info("Standard mode generation.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(generate_single_sample,i) for i in range(num_samples)]
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r: results.append(r)
    else:
        logger.info("Batch mode generation (worldgen).")
        pre: List[Tuple[int,Node,int,str,int]] = []
        for i in range(num_samples):
            ast = build_random_ast(DEFAULT_MAX_OPS)
            validate_ast(ast)
            gt  = eval_node(ast)
            ast_prefix = ast_to_prefix(ast)
            num_chars = random.randint(3,6)
            pre.append((i,ast,gt,ast_prefix,num_chars))
        requests = []
        for i,_,_,_,nchars in pre:
            prompt = _get_world_gen_prompt(nchars)
            cid = f"sample_{i}_chars_{nchars}"
            requests.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/messages",
                "body": {
                    "model": MODEL,
                    "messages":[{"role":"user","content":prompt}],
                    "max_tokens":2000,
                    "temperature":0.8
                }
            })
        tmp = prepare_batch_input_file(requests)
        batch = run_world_generation_batch_job(tmp)
        completed = poll_batch_job(batch.id) if batch else None
        worlds = process_batch_results(completed, len(pre)) if completed else {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs=[]
            for i,ast,gt,ast_prefix,nchars in pre:
                cid = f"sample_{i}_chars_{nchars}"
                world = worlds.get(cid)
                if world:
                    futs.append(ex.submit(generate_narrative_for_sample,i,ast,gt,ast_prefix,world))
                else:
                    logger.error(f"No world for sample {i+1}")
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r: results.append(r)
        try:
            os.remove(tmp)
        except OSError:
            pass

    # write output
    with open(output_file,"a",encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec,ensure_ascii=False)+"\n")
    logger.info(f"Wrote {len(results)} samples to {output_file}")

if __name__ == "__main__":
    main()