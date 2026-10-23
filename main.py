"""MiniLanguageModel v0.1.0: a locally trained GPT-style Transformer, from scratch.

Python 3.11+; install requirements.txt, then run this file or use --test.
No pretrained weights, ready-made Transformer modules, or inference services.
The numbered sections form a small executable textbook. All application code
and tests intentionally live here; checkpoints contain weights, NOT a corpus.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, replace
import hashlib
from html import escape
import io
import json
import math
import re
import os
from pathlib import Path
from pickle import UnpicklingError  # Exception type only; never unrestricted load.
import shlex
import shutil
import sys
import sqlite3
import codecs
import tempfile
import time
from typing import Callable, Iterable, Iterator
import unittest
from unittest.mock import patch
import uuid

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except (ImportError, OSError) as error:
    raise SystemExit(f"PyTorch is required. Run this interpreter with -m pip install -r requirements.txt.\n{error}") from error

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 3


# ============================================================================
# 1. TOKENIZATION AND STREAMING: fixed bytes, stable IDs, explicit documents
# ============================================================================
class ByteTokenizer:
    """A deliberately simple, reversible UTF-8 byte vocabulary.

    IDs 0..255 are actual bytes; 256/257/258 are BOS/EOS/PAD. A byte is NOT a
    word and not always a full character: Finnish 'ä' encodes as [195, 164].
    The vocabulary covers new words without any embedding/output-layer growth.
    It preserves case, whitespace, punctuation and Unicode exactly. It does not
    normalize text, stem words, or silently throw away unfamiliar characters.

    BPE could merge frequent byte sequences later, but that would change IDs
    and the prediction alphabet. Checkpoints record this exact tokenizer spec;
    changing tokenization requires a new model or an explicit migration, never
    silently applying old embeddings to a different vocabulary.
    """
    BOS, EOS, PAD, size = 256, 257, 258, 259

    @staticmethod
    def specification() -> dict:
        return {"type": "utf8-bytes", "version": 1, "size": 259,
                "bos": 256, "eos": 257, "pad": 258, "normalization": "none"}

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids: Iterable[int], errors: str = "replace") -> str:
        """Strip special markers and decode; incomplete/invalid UTF-8 becomes �.

        The network predicts bytes independently and may produce invalid UTF-8,
        especially early in training, or stop in the middle of a character.
        We report replacement characters instead of crashing or pretending
        that every predicted byte is a word. Use errors='strict' to audit bytes.
        No sampling probabilities are secretly modified to repair the output.
        """
        return bytes(token for token in ids if 0 <= token < 256).decode("utf-8", errors=errors)

    @staticmethod
    def label(token: int) -> str:
        if token >= 256:
            return {256: "<BOS>", 257: "<EOS>", 258: "<PAD>"}[token]
        if 32 <= token < 127:
            return repr(chr(token))
        return {10: r"\n", 13: r"\r", 9: r"\t"}.get(token, f"0x{token:02X}")


TOK = ByteTokenizer()


def text_chunks(path: Path, chunk_size: int = 65536) -> Iterator[str]:
    """Strict incremental UTF-8 decoding, stripping an optional leading BOM.

    Python's text decoder handles multibyte characters split across disk reads.
    newline='' preserves CR/LF instead of silently normalizing the document.
    A chunk boundary is transport, never an EOS or an artificial training pair.
    Invalid UTF-8 raises an error; import receipts then remain incomplete.
    """
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive.")
    with path.open(encoding="utf-8-sig", newline="") as source:
        while chunk := source.read(chunk_size):
            yield chunk


def document_windows(chunks: Iterable[str], context: int) -> Iterator[list[int]]:
    """Bounded next-token examples for ONE document, with no lost targets.

    For context=2: BOS a b c EOS -> [BOS,a,b], [b,c,EOS].
    Each example yields input=example[:-1], target=example[1:]. The overlap is
    one INPUT token; no TARGET is counted twice. Empty documents yield nothing.
    A final short window is retained, including EOS after an exact-size block.
    Separate calls reset BOS, so unrelated files can never form fake sequences.

    This nonoverlapping target policy is simple and streaming-friendly. Near
    a block boundary the model sees less history; it is a training tradeoff,
    not a claim of full-document context. Long documents are never buffered.
    """
    if context < 1:
        raise ValueError("Context must be positive.")
    window = [TOK.BOS]
    nonempty = False
    for chunk in chunks:
        for token in chunk.encode("utf-8"):
            nonempty = True
            window.append(token)
            if len(window) == context + 1:
                yield window
                window = [token]
    if nonempty:
        window.append(TOK.EOS)
        yield window


def batches(windows: Iterable[list[int]], batch_size: int) -> Iterator[list[list[int]]]:
    """At most batch_size*(context+1) token references are retained here."""
    batch: list[list[int]] = []
    for window in windows:
        batch.append(window)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ============================================================================
# 2. CONFIGURATION: architecture is distinct from mutable training/sampling knobs
# ============================================================================
def positive_integer(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def finite_number(value: object, name: str, minimum: float = 0, inclusive: bool = False) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        raise ValueError(f"Invalid finite {name}: {value!r}")


@dataclass
class Architecture:
    embedding: int = 128
    heads: int = 4
    layers: int = 4
    context: int = 128
    feedforward: int = 512
    dropout: float = .1

    def validate(self) -> None:
        for name in ("embedding", "heads", "layers", "context", "feedforward"):
            positive_integer(getattr(self, name), name)
        if self.embedding % self.heads:
            raise ValueError("Embedding dimension must be divisible by attention heads.")
        finite_number(self.dropout, "dropout", inclusive=True)
        if self.dropout >= 1:
            raise ValueError("Dropout must be in [0,1).")

    @classmethod
    def preset(cls, name: str) -> Architecture:
        presets = {"tiny": dict(embedding=32, heads=4, layers=2, context=64, feedforward=128),
                   "small": {},
                   "medium": dict(embedding=256, heads=8, layers=6, context=256, feedforward=1024)}
        if name not in presets:
            raise ValueError("Choose tiny, small, or medium.")
        return cls(**presets[name])


@dataclass
class Training:
    batch_size: int = 16
    learning_rate: float = .0003
    weight_decay: float = .01
    gradient_clip: float = 1.0
    epochs: int = 1  # Full passes for /train; bounded-batch repeats only in legacy mode.
    max_steps: int = 0  # 0 = no per-document limit; hitting it means INCOMPLETE.
    warmup_steps: int = 100
    device: str = "auto"
    efficient_attention: bool = True

    def validate(self) -> None:
        for name in ("batch_size", "epochs"):
            positive_integer(getattr(self, name), name)
        for name in ("max_steps", "warmup_steps"):
            positive_integer(getattr(self, name), name, 0)
        finite_number(self.learning_rate, "learning rate")
        finite_number(self.gradient_clip, "gradient clip")
        finite_number(self.weight_decay, "weight decay", inclusive=True)
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("Device must be auto, cpu, or mps.")
        if type(self.efficient_attention) is not bool:
            raise ValueError("efficient_attention must be boolean.")


@dataclass
class Sampling:
    temperature: float = .8
    top_k: int = 40
    top_p: float = .95
    max_tokens: int = 200
    seed: int = 42
    debug: bool = False
    auto_train: bool = False

    def validate(self) -> None:
        finite_number(self.temperature, "temperature")
        finite_number(self.top_p, "top-p")
        if self.top_p > 1:
            raise ValueError("Top-p must be in (0,1].")
        positive_integer(self.top_k, "top-k", 0)
        positive_integer(self.max_tokens, "maximum tokens")
        positive_integer(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("Seed must be less than 2**63.")
        if type(self.debug) is not bool or type(self.auto_train) is not bool:
            raise ValueError("Debug and auto_train must be booleans.")


# ============================================================================
# TERMINAL PROGRESS: presentation only; never changes batches or optimizer steps
# ============================================================================
def document_step_total(byte_count: int, context: int, batch_size: int, epochs: int) -> int:
    """Exact updates for this trainer's nonoverlapping windows and batch repeats.

    Each nonempty document predicts every UTF-8 byte once plus EOS. Thus:
      windows = ceil((bytes + 1) / context)
      updates = ceil(windows / batch_size) * epochs
    Padding has no target and contributes no extra optimizer update. Empty
    documents produce no windows. This counts work, not language quality.
    """
    windows = (byte_count + 1 + context - 1) // context if byte_count else 0
    return ((windows + batch_size - 1) // batch_size) * epochs


class TrainingProgress:
    """A dependency-free, single-line equivalent of a small terminal progress bar.

    Carriage return rewrites one line; spaces erase its previous contents. No
    ANSI cursor-up sequences or nested bars are required by the PyCharm console.
    Ordinary redirected output gets a start/end summary rather than CR noise.
    Refreshes are throttled to avoid thousands of expensive console writes.

    total=None means the iterable's remaining work is unknown: never fabricate
    a percentage or ETA. A step budget is NOT the full document size. Stopping
    early keeps the actual completed count and never fills the bar to 100%.
    """
    def __init__(self, callback: Callable = print, *, name: str = 'Input text',
                 total: int | None = None, epochs: int = 1,
                 dataset_position: tuple[int, int] | None = None,
                 validation: dict | None = None, verbose: bool = False,
                 path: Path | None = None, stream=None, live: bool | None = None,
                 clock: Callable = time.perf_counter, phase: str = 'legacy') -> None:
        self.phase = phase
        self.callback, self.name, self.total, self.epochs = callback, name, total, epochs
        self.dataset_position, self.validation, self.verbose = dataset_position, validation, verbose
        self.stream = stream if stream is not None else sys.stdout
        self.live = (callback is print and (self.stream.isatty() or os.environ.get('PYCHARM_HOSTED') == '1')) if live is None else live
        self.clock, self.started = clock, clock()
        self.last_refresh, self.width = -float('inf'), 0
        self.completed, self.batch, self.epoch = 0, 0, 0
        self.loss, self.rate = None, None
        self.path, self.signature = path, self.file_signature(path)
        self.closed = False
        explanation = {'legacy': 'epochs repeat each bounded batch; steps are optimizer updates.',
                       'corpus': 'shuffled corpus epoch; steps are optimizer updates.',
                       'validation': 'held-out batches; no gradients or optimizer updates.'}
        self(name + ' | ' + explanation[phase])
        self.render(force=True)

    @staticmethod
    def file_signature(path: Path | None):
        if path is None:
            return None
        try:
            stat = path.stat()
            return stat.st_size, stat.st_mtime_ns, stat.st_ino
        except OSError:
            return None

    @classmethod
    def for_file(cls, path: Path, learner, callback: Callable = print, **kwargs):
        # UTF-8 byte tokens let us determine the work from file size, subtracting
        # only the optional BOM that text_chunks strips. No extra corpus read or
        # retained text is needed. Invalid UTF-8 still fails in the real reader.
        total = None
        try:
            signature = cls.file_signature(path)
            with path.open('rb') as source:
                bom = source.read(3) == b'\xef\xbb\xbf'
            if signature is not None and cls.file_signature(path) == signature:
                total = document_step_total(max(0, signature[0] - (3 if bom else 0)),
                                            learner.architecture.context,
                                            learner.training.batch_size, learner.training.epochs)
        except OSError:
            pass  # The ingestion path reports its own actionable read error.
        return cls(callback, name=path.name, total=total, epochs=learner.training.epochs,
                   verbose=learner.sampling.debug, path=path, **kwargs)

    def clear(self) -> None:
        if self.live and self.width:
            self.stream.write('\r' + ' ' * self.width + '\r')
            self.stream.flush()
            self.width = 0

    def __call__(self, message: str) -> None:
        self.clear()
        self.callback(message)

    def update(self, *, loss: float, rate: float, batch: int, epoch: int) -> None:
        self.completed += 1
        self.loss, self.rate, self.batch, self.epoch = loss, rate, batch, epoch
        # A changed source or an underestimated dynamic iterator invalidates the
        # denominator. Do not clamp to 100% and pretend unexpected work is done.
        if ((self.path is not None and self.file_signature(self.path) != self.signature)
                or (self.total is not None and self.completed > self.total)):
            self.total = None
        self.render()

    @staticmethod
    def duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'

    def line(self) -> str:
        elapsed = max(0, self.clock() - self.started)
        if self.total is None:
            work = f'{self.completed}/? steps | ETA ?'
        else:
            fraction = self.completed/self.total if self.total else 0
            filled = min(8, int(fraction*8))
            eta = self.duration(elapsed/self.completed*(self.total-self.completed)) if self.completed else '?'
            work = f'[{"="*filled}{"."*(8-filled)}] {fraction:.0%} {self.completed}/{self.total} steps | ETA {eta}'
        dataset = ''
        if self.dataset_position:
            index, total = self.dataset_position
            dataset = f'Data {index-1}/{total} files ({(index-1)/total:.0%}) | '
        loss = f'{self.loss:.3f}' if self.loss is not None else '--'
        rate = f'{self.rate:.3g}' if self.rate is not None else '--'
        validation = '--'
        if self.validation and self.validation.get('nll') is not None:
            validation = f'{self.validation["nll"]:.3f}@{self.validation["step"]}'
        if self.phase == 'validation':
            return f'{self.name} | {work.replace("steps", "batches")} | held-out loss {loss}'
        return (f'{dataset}{self.name} | batch {self.batch} epoch {self.epoch}/{self.epochs} '
                f'({self.epoch/self.epochs:.0%}) | {work} | loss {loss} val {validation} lr {rate}')

    def compact_line(self) -> str:
        """Keep all key metrics visible on ordinary 80-column terminals.

        D=processed dataset files, B=batch, E=epoch within that batch, L=latest
        training loss, V=last measured validation loss. The full summary labels
        validation with its measurement step so an old value is not called new.
        """
        dataset = ''
        if self.dataset_position:
            index, total = self.dataset_position
            dataset = f'D{index-1}/{total} '
        work = f'{self.completed}/?'
        eta = '?'
        if self.total is not None:
            work = f'{self.completed/self.total:.0%} {self.completed}/{self.total}' if self.total else '0/0'
            if self.completed:
                eta = self.duration((self.clock()-self.started)/self.completed*(self.total-self.completed))
        loss = f'{self.loss:.2f}' if self.loss is not None else '--'
        validation = f'{self.validation["nll"]:.2f}' if self.validation and self.validation.get('nll') is not None else '--'
        rate = f'{self.rate:.1g}' if self.rate is not None else '--'
        return (f'{dataset}{self.name[:8]} B{self.batch} E{self.epoch}/{self.epochs} '
                f'{work} L{loss} V{validation} lr{rate} ETA {eta}')

    def render(self, force: bool = False) -> None:
        now = self.clock()
        if not self.live or self.closed or (not force and now-self.last_refresh < .2):
            return
        self.last_refresh = now
        line = self.line()
        # Avoid wrapping into several terminal rows, which a carriage return
        # cannot erase. The full metrics remain in the completion summary and
        # /debug output; narrow terminals use compact metric labels.
        columns = shutil.get_terminal_size((180, 24)).columns
        if len(line) >= columns:
            line = self.compact_line()
        line = line[:max(1, columns-1)]
        self.stream.write('\r' + line + ' ' * max(0, self.width-len(line)))
        self.stream.flush()
        self.width = len(line)

    def finish(self, status: str, *, average: float | None, targets: int) -> None:
        if self.closed:
            return
        self.clear()
        self.closed = True
        mean = f'{average:.4f}' if average is not None else 'n/a'
        # Dataset completion counts files whose ingestion loop has finished,
        # not an invented step-weighted estimate across differently sized files.
        unit = 'evaluation batches' if self.phase == 'validation' else 'updates'
        metric = 'held-out loss' if self.phase == 'validation' else 'mean online loss'
        self(f'{status}: {self.name} | {self.completed} {unit}, {targets:,} targets | '
             f'{metric} {mean} | elapsed {self.duration(self.clock()-self.started)}')
        if self.phase == 'validation': return
        validation = (f'{self.validation["nll"]:.4f} (measured at step {self.validation["step"]})'
                      if self.validation and self.validation.get('nll') is not None else 'not measured')
        self(f'  Steps: {self.completed}/{self.total if self.total is not None else "unknown"} | '
             f'latest loss: {self.loss} | validation: {validation} | learning rate: {self.rate}')


# ============================================================================
# 3. A GPT-STYLE DECODER: explicit Q/K/V, causal attention, pre-norm residuals
# ============================================================================
class CausalSelfAttention(nn.Module):
    """Implement attention ourselves; the fast path uses the same Q/K/V weights.

    A token vector x_i contains information at position i. Learned linear maps
    produce a query ('what information to seek'), a key ('what to match'), and
    a value ('what information to send'). These are analogies, not programmed
    linguistic rules. Parameters are optimized numerical tensors, not facts.
    """
    def __init__(self, config: Architecture) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_size = config.embedding // config.heads
        self.qkv = nn.Linear(config.embedding, 3 * config.embedding)
        self.projection = nn.Linear(config.embedding, config.embedding)
        self.dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)
        # A registered buffer follows CPU/MPS moves but is not a trainable
        # parameter. Rebuildable masks need not occupy checkpoint files.
        self.register_buffer("causal", torch.ones(config.context, config.context, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, *, inspect: bool = False, efficient: bool = True):
        batch, length, width = x.shape  # [B,T,C], e.g. [16,128,128].
        # One matrix multiply yields three projections [B,T,3C]. Split into
        # heads, each with dimension D=C/H, then transpose to [B,H,T,D].
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_size)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = None
        if efficient and not inspect:
            # Fundamental PyTorch attention kernel, NOT nn.Transformer or GPT.
            # It may avoid materializing the full T*T score matrix. Dropout
            # must explicitly become zero in eval mode (SDPA does not infer it).
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                      dropout_p=self.dropout if self.training else 0.0)
        else:
            # Attention(Q,K,V) = softmax(Q K^T / sqrt(D) + mask) V.
            # Q @ K^T: [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T].
            # Entry (i,j) compares query i with key j; sqrt(D) controls score
            # magnitude so softmax does not saturate merely because D is large.
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
            # At position i only j<=i is visible. Mask BEFORE softmax using
            # -infinity, whose exponential is zero. Without this, position i
            # could read target i+1 and cheat during next-token training.
            scores = scores.masked_fill(~self.causal[:length, :length], -float("inf"))
            weights = F.softmax(scores, dim=-1)
            # Each row sums to 1 BEFORE training-time dropout. In inspection
            # we use eval mode, so the displayed rows are actual distributions.
            attended = F.dropout(weights, p=self.dropout, training=self.training) @ value
        # Heads learn different weighted mixtures. Concatenate [B,H,T,D] back
        # into [B,T,C], then mix head outputs through another learned map.
        joined = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.residual_dropout(self.projection(joined)), weights


class TransformerBlock(nn.Module):
    """Pre-layer-normalized attention and feed-forward network with residuals.

    LayerNorm normalizes features at EACH position independently, not across
    sequence positions (which would leak future information). Its scale and
    shift are learned. A residual x + f(x) preserves a direct signal/gradient
    path across depth, making optimization less fragile.

    x -> x + Attention(LayerNorm(x)) -> x + MLP(LayerNorm(x)).
    The position-wise MLP expands C to F, applies GELU, then projects back to C.
    Attention exchanges information between positions; the MLP transforms each
    position's features. Neither operation contains hardcoded language rules.
    """
    def __init__(self, config: Architecture) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.embedding)
        self.attention = CausalSelfAttention(config)
        self.norm_feedforward = nn.LayerNorm(config.embedding)
        self.feedforward = nn.Sequential(nn.Linear(config.embedding, config.feedforward), nn.GELU(),
                                         nn.Linear(config.feedforward, config.embedding), nn.Dropout(config.dropout))

    def forward(self, x, *, inspect: bool = False, efficient: bool = True):
        attended, weights = self.attention(self.norm_attention(x), inspect=inspect, efficient=efficient)
        x = x + attended
        x = x + self.feedforward(self.norm_feedforward(x))
        return x, weights


class GPT(nn.Module):
    """A genuine decoder-only Transformer built from basic PyTorch layers.

    Tokens [B,T] -> embeddings [B,T,C] + learned positions [T,C] -> N blocks
    -> final normalization -> logits [B,T,V]. Here V=259 fixed byte/special IDs.
    Position embeddings provide order: attention without position information
    does not itself identify how far apart two tokens occur.
    """
    def __init__(self, config: Architecture) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.tokens = nn.Embedding(TOK.size, config.embedding, padding_idx=TOK.PAD)
        self.positions = nn.Embedding(config.context, config.embedding)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.embedding)
        self.output = nn.Linear(config.embedding, TOK.size, bias=False)
        self.apply(self._initialize)
        # Weight tying shares input/output representations and reduces parameter
        # count. PyTorch's parameters() deduplicates this shared Parameter, so
        # AdamW does not update it twice. Checkpoint validation checks the tie.
        self.output.weight = self.tokens.weight
        # Smaller residual projections limit signal growth when many blocks are
        # added together; sqrt(2L) accounts for two residual branches per layer.
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, std=.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.feedforward[2].weight, std=.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _initialize(module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, token_ids, *, inspect: bool = False, efficient: bool = True):
        if token_ids.ndim != 2 or not 1 <= token_ids.shape[1] <= self.config.context:
            raise ValueError("Input must be [batch,time] within the architecture context limit.")
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)
        x = self.embedding_dropout(self.tokens(token_ids) + self.positions(positions))
        attention = []
        for block in self.blocks:
            x, weights = block(x, inspect=inspect, efficient=efficient)
            if inspect:
                attention.append(weights)
        logits = self.output(self.final_norm(x))
        # BOS and PAD are inputs, never targets. Identical masks are applied in
        # training, prediction, and evaluation; EOS remains a valid next token.
        logits = logits.clone()
        logits[..., TOK.BOS] = -float("inf")
        logits[..., TOK.PAD] = -float("inf")
        return logits, attention


# ============================================================================
# 4. DEVICES AND SAMPLING: stable distributions, honest MPS fallback
# ============================================================================
def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def choose_device(request: str, progress: Callable = print) -> str:
    """Probe real attention/backward/AdamW before selecting MPS; CPU is safe.

    Availability alone does not establish support for every operation. A failed
    probe is reported. Automatic fallback changes the device, never the model.
    """
    if request not in {"auto", "cpu", "mps"}:
        raise ValueError("Device must be auto, cpu, or mps.")
    if request == "cpu":
        return "cpu"
    if not torch.backends.mps.is_available():
        progress("MPS unavailable; using CPU.")
        return "cpu"
    try:
        with torch.random.fork_rng(devices=[]):
            probe = nn.Linear(8, 24).to("mps")
            optimizer = torch.optim.AdamW(probe.parameters(), foreach=False)
            q, k, v = probe(torch.zeros(1, 4, 8, device="mps")).chunk(3, dim=-1)
            F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True).sum().backward()
            optimizer.step()
            synchronize("mps")
        return "mps"
    except (RuntimeError, NotImplementedError) as error:
        progress(f"MPS probe failed; using CPU: {error}")
        return "cpu"


def sampling_probabilities(logits, settings: Sampling):
    """Return CPU float64 P(sample), after temperature, top-k, then top-p.

    Softmax(z/T) sharpens at T<1 and flattens at T>1. Subtract max(z) BEFORE
    dividing by T: all numbers are then <=0, avoiding positive overflow even
    for tiny T. Underflow of unlikely candidates to zero is harmless.

    Top-p (nucleus) keeps the SMALLEST descending-probability prefix whose sum
    reaches p. The token that crosses the threshold must be KEPT. For example
    [.60,.25,.15], p=.80 keeps the first TWO, then renormalizes [.706,.294].
    Top-k limits candidates first; top-p operates on that restricted softmax.
    Ties are broken by stable token-ID order for reproducible experiments.
    """
    settings.validate()
    scores = logits.detach().cpu().double().clone()
    if scores.shape != (TOK.size,):
        raise ValueError("Expected one vocabulary-sized logit vector.")
    valid = torch.cat((scores[:256], scores[TOK.EOS:TOK.EOS+1]))
    if not torch.isfinite(valid).all():
        raise ValueError("Invalid/non-finite next-token logits.")
    scores[TOK.BOS] = scores[TOK.PAD] = -float("inf")
    scores = (scores - scores.max()) / settings.temperature
    if settings.top_k:
        indices = torch.argsort(scores, descending=True, stable=True)[:settings.top_k]
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[indices] = True
        scores[~keep] = -float("inf")
    probabilities = torch.softmax(scores, dim=-1)
    if settings.top_p < 1:
        order = torch.argsort(probabilities, descending=True, stable=True)
        sorted_p = probabilities[order]
        cumulative_before = torch.cumsum(sorted_p, dim=0) - sorted_p
        remove = cumulative_before >= settings.top_p
        remove[0] = False
        probabilities[order[remove]] = 0
        probabilities /= probabilities.sum()
    return probabilities


class TrainingLimit(RuntimeError):
    """A bounded training run stopped before all source targets were processed."""


# ============================================================================
# 5. LEARNING: actual cross-entropy, autograd, AdamW, and bounded input batches
# ============================================================================
class Learner:
    """Own one Transformer, optimizer, RNG streams, metrics, and import receipts.

    No dataset/text field exists here. Temporary window lists and graphs become
    unreachable after updates. Metrics retain only the last 1000 scalar records.
    Continuous learning can forget earlier patterns; explicit source-file replay
    is necessary to revisit them. Batch order affects optimization.
    """
    def __init__(self, architecture: Architecture | None = None, training: Training | None = None,
                 sampling: Sampling | None = None, progress: Callable = print) -> None:
        self.architecture = architecture or Architecture()
        self.training = training or Training()
        self.sampling = sampling or Sampling()
        self.architecture.validate()
        self.training.validate()
        self.sampling.validate()
        self.device = choose_device(self.training.device, progress)
        # Keep this model's training RNG independent from tests, probes, and UI.
        # The saved RNG starts AFTER weight initialization, so resumed dropout
        # follows the same stream on a matching CPU/PyTorch configuration.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.sampling.seed)
            self.net = GPT(self.architecture).to(self.device)
            self.cpu_rng = torch.get_rng_state().clone()
        self.mps_rng = None
        if self.device == "mps":
            previous = torch.mps.get_rng_state()
            torch.mps.manual_seed(self.sampling.seed)
            self.mps_rng = torch.mps.get_rng_state().clone()
            torch.mps.set_rng_state(previous)
        self.generator = torch.Generator(device="cpu").manual_seed(self.sampling.seed)
        # Decouple weight decay from gradients (AdamW). Apply decay to matrix
        # weights, not LayerNorm gains or biases. Each tied parameter appears once.
        matrices, vectors = [], []
        for parameter in self.net.parameters():
            (matrices if parameter.ndim >= 2 else vectors).append(parameter)
        self.optimizer = torch.optim.AdamW([
            {"params": matrices, "weight_decay": self.training.weight_decay},
            {"params": vectors, "weight_decay": 0.0}], lr=self.training.learning_rate, foreach=False)
        self.steps = 0
        self.bytes_read = 0
        self.targets_optimized = 0
        self.documents_completed = 0
        self.latest_loss: float | None = None
        self.training_seconds = 0.0
        self.history: deque[dict] = deque(maxlen=1000)
        self.imports: dict[str, dict] = {}
        # Path -> most recently completed content hash, stored WITH the weights.
        # Historical receipts remain available: replacing this manifest entry
        # does not subtract the old document's influence from neural parameters.
        self.dataset: dict[str, str] = {}
        self.corpus_state: dict = {}
        self.identity = uuid.uuid4().hex
        self.last_checkpoint: str | None = None
        self.needs_restore = False

    @contextmanager
    def training_randomness(self):
        """Install/capture only this learner's dropout RNG around training work."""
        cpu_before = torch.get_rng_state()
        mps_before = torch.mps.get_rng_state() if self.device == "mps" else None
        torch.set_rng_state(self.cpu_rng)
        if self.device == "mps" and self.mps_rng is not None:
            torch.mps.set_rng_state(self.mps_rng)
        try:
            yield
        finally:
            self.cpu_rng = torch.get_rng_state().clone()
            if self.device == "mps":
                self.mps_rng = torch.mps.get_rng_state().clone()
            torch.set_rng_state(cpu_before)
            if mps_before is not None:
                torch.mps.set_rng_state(mps_before)

    def move_to(self, device: str) -> None:
        """Move weights and Adam moments together; leave scalar step counters CPU."""
        self.net.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu" if key == "step" else device)
        self.device = device
        if device == "mps" and self.mps_rng is None:
            self.mps_rng = torch.mps.get_rng_state().clone()

    def batch_tensors(self, windows: list[list[int]]):
        """Inputs/targets [B,T], right-padding inputs and IGNORING padded targets.

        Example [BOS,97,98,EOS] -> input [BOS,97,98], target [97,98,EOS].
        A causal model cannot read right-padding from earlier real positions;
        targets=-100 remove padding losses from both the numerator and mean.
        """
        width = max(len(window)-1 for window in windows)
        x = torch.full((len(windows), width), TOK.PAD, dtype=torch.long)
        y = torch.full_like(x, -100)
        for row, window in enumerate(windows):
            x[row, :len(window)-1] = torch.tensor(window[:-1])
            y[row, :len(window)-1] = torch.tensor(window[1:])
        return x.to(self.device), y.to(self.device)

    def learning_rate(self) -> float:
        # Linear warmup avoids an abrupt large initial update. After warmup the
        # configured base rate is constant; no misleading hidden decay schedule.
        warmup = self.training.warmup_steps
        return self.training.learning_rate * min(1.0, (self.steps + 1) / warmup) if warmup else self.training.learning_rate

    def optimize_batch(self, windows: list[list[int]], progress: Callable = print) -> tuple[float, int]:
        """Perform ONE optimizer step, returning mean NLL and true target count.

        Cross-entropy = -log P(correct next token). P=.25 -> loss=1.386 nats;
        P=.80 -> loss=.223 nats. F.cross_entropy computes stable log-softmax
        directly from logits; passing already-softmaxed probabilities is wrong.

        Autograd builds a graph of tensor operations during the forward pass.
        backward() applies the chain rule through output, MLPs, attention, and
        embeddings. parameter.grad holds d(loss)/d(parameter). Gradient descent
        moves theta in the negative-gradient direction; AdamW adapts that step
        using moving first/second moments plus separate weight decay.
        """
        if self.needs_restore:
            raise ValueError("A previous optimizer update failed; load a known-good checkpoint before more training.")
        self.net.train()
        x, y = self.batch_tensors(windows)
        count = sum(len(window)-1 for window in windows)
        rate = self.learning_rate()
        for index, group in enumerate(self.optimizer.param_groups):
            group['lr'] = rate
            group['weight_decay'] = self.training.weight_decay if index == 0 else 0.0
        cpu_before = torch.get_rng_state().clone()
        try:
            loss = self._backward(x, y)
            nn.utils.clip_grad_norm_(self.net.parameters(), self.training.gradient_clip, error_if_nonfinite=True)
        except (RuntimeError, NotImplementedError) as error:
            # Retrying is safe ONLY before optimizer.step(): weights and moments
            # have not changed. Do not retry OOM/NaN failures as unsupported ops.
            unsupported = "not implemented" in str(error).lower() or "not supported" in str(error).lower()
            if self.device != "mps" or not unsupported:
                raise
            progress(f"MPS operation unsupported; retrying this not-yet-updated batch on CPU: {error}")
            self.move_to("cpu")
            torch.set_rng_state(cpu_before)
            loss = self._backward(x.cpu(), y.cpu())
            nn.utils.clip_grad_norm_(self.net.parameters(), self.training.gradient_clip, error_if_nonfinite=True)
        # Clipping above bounds the aggregate gradient norm, not model output.
        try:
            self.optimizer.step()
        except BaseException:
            # A partial optimizer update cannot be rolled back by zero_grad().
            # Refuse to claim it is resumable; the import receipt stays incomplete.
            self.needs_restore = True
            raise
        self.steps += 1
        self.targets_optimized += count
        self.latest_loss = float(loss.detach().item())
        self.history.append({"step": self.steps, "loss": self.latest_loss, "learning_rate": rate,
                             "targets": self.targets_optimized})
        return self.latest_loss, count

    def _backward(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)
        logits, _ = self.net(x, efficient=self.training.efficient_attention)
        loss = F.cross_entropy(logits.reshape(-1, TOK.size), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss: no optimizer step applied.")
        loss.backward()
        return loss

    def train_chunks(self, chunks: Iterable[str], progress: Callable = print) -> dict:
        """Train bounded batches, release input, and report only actual operations.

        epochs means repeated passes on the current batch before discarding it.
        It is not a full corpus epoch. max_steps is a per-document budget; if
        the budget prevents consuming all data/passes, raise TrainingLimit so
        imports CANNOT mark the file complete. Already learned updates remain.
        """
        if self.corpus_state: raise ValueError("Use /corpus to preserve this model's validation split.")
        before_steps, before_bytes = self.steps, self.bytes_read
        started = time.perf_counter()
        loss_sum, targets = 0.0, 0
        display = progress if isinstance(progress, TrainingProgress) else TrainingProgress(
            progress, epochs=self.training.epochs, verbose=self.sampling.debug)
        status = 'Stopped'
        def counted_chunks() -> Iterator[str]:
            for chunk in chunks:
                self.bytes_read += len(chunk.encode("utf-8"))
                yield chunk
        try:
            with self.training_randomness():
                for batch_number, batch in enumerate(batches(document_windows(counted_chunks(), self.architecture.context), self.training.batch_size), 1):
                    for epoch in range(1, self.training.epochs + 1):
                        if self.training.max_steps and self.steps-before_steps >= self.training.max_steps:
                            raise TrainingLimit("Step budget reached before the document/passes completed. Partial learning remains; receipt is incomplete.")
                        loss, count = self.optimize_batch(batch, display)
                        loss_sum += loss * count
                        targets += count
                        display.update(loss=loss, rate=self.optimizer.param_groups[0]['lr'],
                                       batch=batch_number, epoch=epoch)
                        if self.sampling.debug:
                            display(f'  step={self.steps:,} | batch={batch_number} epoch={epoch}/{self.training.epochs} | loss={loss:.4f} | lr={self.optimizer.param_groups[0]["lr"]:.6g} | optimized targets={self.targets_optimized:,} | bytes read={self.bytes_read:,}')
            status = 'Training finished'
            if self.bytes_read > before_bytes:
                self.documents_completed += 1
        finally:
            synchronize(self.device)
            self.training_seconds += time.perf_counter()-started
            display.finish(status, average=loss_sum/targets if targets else None, targets=targets)
        average = loss_sum/targets if targets else None
        result = {"steps": self.steps-before_steps, "bytes_read": self.bytes_read-before_bytes,
                  "targets_optimized": targets, "online_nll": average}
        return result

    def train_text(self, text: str, progress: Callable = print) -> dict:
        if not isinstance(progress, TrainingProgress):
            progress = TrainingProgress(progress, total=document_step_total(len(text.encode('utf-8')),
                self.architecture.context, self.training.batch_size, self.training.epochs),
                epochs=self.training.epochs, verbose=self.sampling.debug)
        return self.train_chunks([text], progress)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    @contextmanager
    def inference(self):
        """Temporarily disable dropout/autograd and restore the previous mode."""
        previous = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.net.train(previous)

    def context_ids(self, prompt: str) -> list[int]:
        return ([TOK.BOS] + TOK.encode(prompt))[-self.architecture.context:]

    def inference_forward(self, x, *, inspect: bool = False):
        """Retry unsupported MPS inference on CPU, without changing learned values.

        Only the execution device changes; no optimizer step or RNG draw occurs
        in evaluation mode. Other errors (including OOM) remain visible.
        """
        try:
            return self.net(x, inspect=inspect, efficient=self.training.efficient_attention)
        except (RuntimeError, NotImplementedError) as error:
            unsupported = "not implemented" in str(error).lower() or "not supported" in str(error).lower()
            if self.device != 'mps' or not unsupported:
                raise
            print(f'MPS inference unsupported; using the same Transformer on CPU: {error}')
            self.move_to('cpu')
            return self.net(x.cpu(), inspect=inspect, efficient=self.training.efficient_attention)

    def predict_ids(self, ids: list[int], inspect: bool = False):
        if self.needs_restore:
            raise ValueError("Restore a known-good checkpoint after the failed optimizer update.")
        with self.inference():
            x = torch.tensor([ids[-self.architecture.context:] or [TOK.BOS]], dtype=torch.long, device=self.device)
            logits, attention = self.inference_forward(x, inspect=inspect)
            return logits[0, -1].detach().cpu(), [matrix[0].detach().cpu() for matrix in attention]

    def generate_steps(self, prompt: str) -> Iterator[dict]:
        """Autoregression: forward -> sample ONE byte/EOS -> append -> repeat.

        Recompute a sliding context for clarity, rather than implementing a KV
        cache prematurely. Learned positions restart at zero within each window.
        Long prompts are left-truncated in byte units, possibly mid-character;
        the model accepts every byte ID, and diagnostics label bytes honestly.
        """
        if not self.steps:
            raise ValueError("Transformer is untrained. Enter training text or /import first; no other model will answer.")
        ids = self.context_ids(prompt)
        for _ in range(self.sampling.max_tokens):
            logits, _ = self.predict_ids(ids)
            raw = torch.softmax(logits.double(), dim=-1)
            sampled = sampling_probabilities(logits, self.sampling)
            selected = int(torch.multinomial(sampled, 1, generator=self.generator).item())
            yield {"ids": ids.copy(), "raw": raw, "sampling": sampled, "selected": selected}
            if selected == TOK.EOS:
                break
            ids = (ids + [selected])[-self.architecture.context:]

    def generate(self, prompt: str) -> str:
        return TOK.decode(step['selected'] for step in self.generate_steps(prompt))

    def evaluate(self, chunks: Iterable[str]) -> dict:
        """Score held-out bytes/EOS without changing weights, counters or RNG.

        Use the same nonoverlapping context-window policy as training. Count
        EVERY non-padding target once, including the last partial window.
        Report NLL/perplexity over bytes+EOS, plus bits/byte excluding EOS:
            BPB = sum_byte_targets[-log P(byte)] / (byte_count * log(2)).
        This is not word-level perplexity. EOS loss is reported separately.
        No temperature/top-k/top-p is applied to model likelihood evaluation.
        """
        if self.needs_restore:
            raise ValueError("Restore a checkpoint after the failed optimizer update.")
        total_loss, byte_loss, end_loss = 0.0, 0.0, 0.0
        targets, byte_count, ends = 0, 0, 0
        with self.inference():
            for batch in batches(document_windows(chunks, self.architecture.context), self.training.batch_size):
                x, y = self.batch_tensors(batch)
                logits, _ = self.inference_forward(x)
                y = y.to(logits.device)  # May have fallen back from MPS to CPU.
                losses = F.cross_entropy(logits.reshape(-1, TOK.size), y.reshape(-1), ignore_index=-100, reduction='none').reshape_as(y)
                mask = y != -100
                bytes_mask = (y >= 0) & (y < 256)
                end_mask = y == TOK.EOS
                total_loss += float(losses[mask].sum().item())
                byte_loss += float(losses[bytes_mask].sum().item())
                end_loss += float(losses[end_mask].sum().item())
                targets += int(mask.sum().item())
                byte_count += int(bytes_mask.sum().item())
                ends += int(end_mask.sum().item())
        nll = total_loss/targets if targets else None
        return {"byte_targets": byte_count, "eos_targets": ends, "targets": targets,
                "nll_bytes_and_eos": nll, "perplexity_bytes_and_eos": safe_exp(nll),
                "bits_per_byte_excluding_eos": byte_loss/(byte_count*math.log(2)) if byte_count else None,
                "eos_nll": end_loss/ends if ends else None,
                "context_bytes": self.architecture.context, "window_policy": "nonoverlapping targets, one input overlap"}


def corpus_document_chunks(path: Path) -> Iterator[bytes | None]:
    """Yield bounded byte blocks; None ends a complete story/chapter/document.

    Split BEFORE creating windows: overlapping windows must never send part of
    the same chapter to training and another part to validation. Chapters from
    one novel are still related; their validation loss measures within-book
    generalization, not performance on independent books or conversations.

    TinyStories' explicit separator takes precedence. Otherwise, two or more
    standalone CHAPTER I / CHAPTER 1 headings activate chapter mode. A bounded
    first scan detects the format. readline's size limit handles enormous lines
    without loading a whole document into RAM. Only true line starts can match.
    Gutenberg front matter and license remain intact in the distributed file;
    chapter mode excludes them from learning, never edits the user's source.
    """
    heading = re.compile(rb'CHAPTER[ \t]+(?:[IVXLCDM]+|[0-9]+)[ \t]*\r?\n?$', re.I)
    delimiter = b'<|endoftext|>'
    chapters, explicit, tail, line_start = 0, False, b'', True
    with path.open('rb') as source:
        while block := source.readline(65536):
            explicit |= delimiter in tail + block
            tail = block[-len(delimiter):]
            if line_start and heading.fullmatch(block.lstrip(b'\xef\xbb\xbf')): chapters += 1
            line_start = block.endswith(b'\n')
    with path.open('rb') as source:
        if chapters >= 2 and not explicit:
            active, line_start = False, True
            while block := source.readline(65536):
                if line_start and block.startswith(b'*** END OF THE PROJECT GUTENBERG'):
                    break
                if line_start and heading.fullmatch(block.lstrip(b'\xef\xbb\xbf')):
                    if active: yield None
                    active = True
                if active: yield block
                line_start = block.endswith(b'\n')
            yield None
        else:
            pending = b''
            while block := source.read(65536):
                pending += block
                while delimiter in pending:
                    before, pending = pending.split(delimiter, 1)
                    yield before
                    yield None
                keep = len(delimiter) - 1
                if len(pending) > keep:
                    yield pending[:-keep]
                    pending = pending[-keep:]
            yield pending
            yield None


class CorpusWindows:
    """Temporary disk-backed windows; complete stories decide the split FIRST.

    TinyStories uses <|endoftext|>; novels use standalone chapter headings.
    Other files are individual documents (never randomly split their windows).
    SHA-256 of complete story bytes assigns duplicates to the same partition.
    SQLite keeps window payloads and shuffle sorting on disk, not in RAM. The
    temporary database contains source-derived tokens and is deleted on close.
    """
    def __init__(self, paths, context, seed=42, validation_fraction=.2):
        if not 0 < validation_fraction < 1: raise ValueError('Validation fraction must be between zero and one.')
        paths = sorted(map(Path, paths))
        self.manifest = {str(path.resolve()): content_hash(path) for path in paths}
        self.specification = dict(files=self.manifest, split_seed=seed, fraction=validation_fraction,
                                  context=context, tokenizer=TOK.specification(), delimiter='<|endoftext|>',
                                  segmentation='stories-or-chapters-v1')
        self.folder = tempfile.TemporaryDirectory()
        self.db = sqlite3.connect(str(Path(self.folder.name)/'windows.sqlite'))
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA cache_size=-4096')
        self.db.execute('CREATE TABLE stories (digest TEXT PRIMARY KEY)')
        self.db.execute('CREATE TABLE windows (id INTEGER PRIMARY KEY, split TEXT, payload TEXT, ordering TEXT)')
        self.seed = seed
        try:
            for path in paths:
                with tempfile.TemporaryFile() as story:
                    def flush_story():
                        size = story.tell()
                        if not size: return
                        story.seek(0); digest = hashlib.file_digest(story, 'sha256').hexdigest()
                        if self.db.execute('INSERT OR IGNORE INTO stories VALUES (?)',(digest,)).rowcount:
                            split = 'validation' if int(hashlib.sha256(f'{seed}:{digest}'.encode()).hexdigest(),16)/2**256 < validation_fraction else 'train'
                            story.seek(0)
                            def chunks():
                                decoder = codecs.getincrementaldecoder('utf-8-sig')('strict')
                                while block := story.read(65536): yield decoder.decode(block)
                                yield decoder.decode(b'',final=True)
                            for window in document_windows(chunks(), context):
                                self.db.execute('INSERT INTO windows(split,payload) VALUES (?,?)',(split,json.dumps(window)))
                        story.seek(0); story.truncate()
                    for block in corpus_document_chunks(path):
                        if block is None: flush_story()
                        else: story.write(block)
            if any(content_hash(Path(path)) != digest for path,digest in self.manifest.items()):
                raise ValueError('Corpus changed during preparation.')
            self.db.commit()
        except BaseException:
            self.close(); raise

    def count(self, split):
        return self.db.execute('SELECT count(*) FROM windows WHERE split=?',(split,)).fetchone()[0]

    def windows(self, split, epoch=None):
        if epoch is not None:
            # A keyed hash gives a deterministic permutation without collecting
            # all windows or indices in a Python list. SQLite sorts on disk.
            self.db.create_function('shuffle_key',1,lambda i:hashlib.sha256(f'{self.seed}:{epoch}:{i}'.encode()).hexdigest())
            self.db.execute('UPDATE windows SET ordering=shuffle_key(id) WHERE split=?',(split,))
        order = 'ordering,id' if epoch is not None else 'id'
        for row in self.db.execute(f'SELECT payload FROM windows WHERE split=? ORDER BY {order}',(split,)):
            yield json.loads(row[0])

    def close(self):
        self.db.close(); self.folder.cleanup()


def corpus_loss(learner, corpus, split, progress=None):
    """Target-weighted held-out cross entropy, with dropout and autograd OFF."""
    loss_sum, count = 0., 0
    display = TrainingProgress(progress, name=f'Evaluate {split}', phase='validation', total=math.ceil(corpus.count(split)/learner.training.batch_size)) if progress else None
    completed = False
    try:
        with learner.inference():
            for batch in batches(corpus.windows(split),learner.training.batch_size):
                x,y=learner.batch_tensors(batch)
                logits,_=learner.inference_forward(x)
                loss_sum += float(F.cross_entropy(logits.reshape(-1,TOK.size),y.reshape(-1),ignore_index=-100,reduction='sum').item())
                count += sum(len(w)-1 for w in batch)
                if display: display.update(loss=loss_sum/count,rate=0,batch=display.completed+1,epoch=1)
        completed = True
    finally:
        if display: display.finish('Evaluation finished' if completed else 'Evaluation stopped',average=loss_sum/count if count else None,targets=count)
    return loss_sum/count if count else None


def train_corpus(learner, corpus, output, *, patience=3, min_delta=.001, schedule="constant", progress=print):
    """True shuffled corpus epochs, cosine decay, validation and best-state restore.

    Best checkpoints include Adam moments and RNG, not only weights. Restoring
    them makes subsequent updates consistent with the chosen model. Total work
    actually performed is reported separately from the restored step counter.
    This operation never overwrites an existing checkpoint path.
    """
    if Path(output).suffix.lower() != '.pt' or not Path(output).parent.is_dir():
        raise ValueError('Choose a .pt destination in an existing directory before training.')
    if schedule not in {'constant','cosine'}: raise ValueError('Use constant or cosine schedule.')
    prior = learner.corpus_state.get('specification', {})
    policy = lambda spec: {k:v for k,v in spec.items() if k != 'files'}
    if learner.steps and policy(prior) != policy(corpus.specification):
        raise ValueError('Independent validation cannot be established for these weights. Use /train --rebuild, or explicitly /train --legacy (no held-out claim).')
    if prior and prior.get('files') != corpus.specification['files']:
        progress('Corpus changed: story-hash split assignments remain fixed. New validation scores are not directly comparable to the previous corpus.')
    if Path(output).exists(): raise ValueError('Choose a new checkpoint path; existing models are preserved.')
    if not corpus.count('train') or not corpus.count('validation'):
        raise ValueError('Need nonempty training and validation partitions. Add several independent documents, CHAPTER headings, or <|endoftext|>-separated stories.')
    positive_integer(patience,'patience'); finite_number(min_delta,'minimum improvement',inclusive=True)
    per_epoch=math.ceil(corpus.count('train')/learner.training.batch_size)
    planned=per_epoch*learner.training.epochs
    records=[]; best=float('inf'); significant=float('inf'); stale=0; updates=0; targets=0
    base=learner.training.learning_rate
    started=learner.steps
    original_checkpoint=learner.last_checkpoint
    previous_epochs=learner.corpus_state.get("epochs_completed",0)
    # The schedule is local to this explicitly requested corpus run. Existing
    # training settings and the legacy baseline procedure remain unchanged.
    learner.learning_rate=lambda: base*min(1.,(updates+1)/max(1,learner.training.warmup_steps))*(1 if schedule=="constant" else (.1+.9*.5*(1+math.cos(math.pi*updates/max(1,planned)))) )
    try:
        with tempfile.TemporaryDirectory() as temporary:
            best_path=Path(temporary)/'best.pt'
            save_checkpoint(learner,Path(temporary)/'before.pt')
            run_started=time.perf_counter()
            try:
                for epoch in range(learner.training.epochs):
                    display=TrainingProgress(progress,name=f'Train epoch {epoch+1}/{learner.training.epochs}',total=per_epoch,phase='corpus',epochs=learner.training.epochs)
                    epoch_started=time.perf_counter()
                    total_loss=0.; epoch_targets=0
                    try:
                        with learner.training_randomness():
                            for batch in batches(corpus.windows('train',previous_epochs+epoch),learner.training.batch_size):
                                loss,n=learner.optimize_batch(batch,display)
                                updates+=1; targets+=n; total_loss+=loss*n; epoch_targets+=n
                                display.update(loss=loss,rate=learner.optimizer.param_groups[0]['lr'],batch=updates,epoch=epoch+1)
                    finally:
                        learner.training_seconds += time.perf_counter()-epoch_started
                        display.finish('Epoch stopped' if display.completed<per_epoch else 'Epoch finished',average=total_loss/epoch_targets if epoch_targets else None,targets=epoch_targets)
                    progress('Validation: evaluating held-out documents/chapters (no gradients).')
                    validation=corpus_loss(learner,corpus,'validation',progress)
                    records.append(dict(epoch=epoch+1,train_loss=total_loss/epoch_targets,validation_loss=validation,steps=updates,targets=targets))
                    progress(str(records[-1]))
                    learner.corpus_state = dict(specification=corpus.specification,epochs_completed=previous_epochs+epoch+1,
                        history=records.copy(),schedule=schedule,patience=patience,min_delta=min_delta,
                        engine='V4 Corpus Trainer',continuation='epoch boundary; a new invocation restarts its LR schedule')
                    if validation < significant-min_delta:
                        significant=validation; stale=0
                    else: stale+=1
                    # Minimum delta controls patience, NOT which weights are best.
                    if validation < best:
                        best=validation; save_checkpoint(learner,best_path)
                    if stale>=patience: break
                chosen=load_checkpoint(best_path,learner.device,progress=lambda _:None)
                chosen.corpus_state['run_seconds']=time.perf_counter()-run_started
                chosen.corpus_state['run_history']=records
                chosen.corpus_state['actual_steps']=updates
                chosen.corpus_state['actual_targets']=targets
                save_checkpoint(chosen,Path(output))
                return chosen,records
            except BaseException:
                # No partially trained model replaces the caller's active state.
                restored=load_checkpoint(Path(temporary)/'before.pt',learner.device,progress=lambda _:None)
                learner.__dict__.update(restored.__dict__)
                learner.last_checkpoint=original_checkpoint
                raise
    finally:
        learner.__dict__.pop('learning_rate',None)


def safe_exp(value: float | None) -> float | None:
    if value is None:
        return None
    return math.exp(value) if value < 709 else float('inf')


# ============================================================================
# 6. CHECKPOINTS: one atomic full-state file, restricted loading
# ============================================================================
def content_hash(path: Path) -> str:
    """Content identity in bounded byte chunks; not encryption or privacy."""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_copy(value):
    """Copy tensor storage to CPU, preserving ordinary containers for torch.save."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def save_checkpoint(learner: Learner, path: Path) -> None:
    """Atomically replace one checkpoint, without accumulating backup copies.

    Weights, Adam moments, RNG and provenance commit together. A temporary
    sibling is flushed before replacement; failure before replacement leaves
    the previous checkpoint untouched. finally removes any unfinished sibling.
    There is deliberately no permanent undo history after a successful save.
    """
    if path.suffix.lower() != '.pt':
        raise ValueError('Use a .pt checkpoint path to avoid overwriting source/text files.')
    if learner.needs_restore:
        raise ValueError('Optimizer update was interrupted; restore a known-good checkpoint. Refusing to save uncertain training state.')
    state = {
        'format': 'mini-gpt', 'version': CHECKPOINT_VERSION, 'tokenizer': TOK.specification(),
        'architecture': asdict(learner.architecture), 'training': asdict(learner.training), 'sampling': asdict(learner.sampling),
        'model': cpu_copy(learner.net.state_dict()), 'optimizer': cpu_copy(learner.optimizer.state_dict()),
        'rng': {'cpu': learner.cpu_rng.clone(), 'mps': cpu_copy(learner.mps_rng), 'sampling': learner.generator.get_state()},
        'metadata': {'identity': learner.identity, 'steps': learner.steps, 'bytes_read': learner.bytes_read,
                     'targets_optimized': learner.targets_optimized, 'documents_completed': learner.documents_completed,
                     'latest_loss': learner.latest_loss, 'training_seconds': learner.training_seconds,
                     'history': list(learner.history), 'imports': learner.imports, 'dataset': learner.dataset, 'corpus_state': learner.corpus_state, 'saved_device': learner.device}}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        learner.last_checkpoint = str(path.resolve())
        return None
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def validate_receipts(receipts: object) -> None:
    if not isinstance(receipts, dict):
        raise ValueError('Invalid import receipts.')
    for digest, receipt in receipts.items():
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or not isinstance(receipt, dict) or receipt.get('status') not in {'complete', 'incomplete'}
                or not isinstance(receipt.get('paths'), list) or not receipt['paths']
                or any(not isinstance(path, str) for path in receipt['paths'])):
            raise ValueError('Invalid import receipt entry.')
        for key in ('bytes_read', 'steps', 'attempts'):
            positive_integer(receipt.get(key), 'receipt '+key, 0)


def load_checkpoint(path: Path, device: str | None = None, progress: Callable = print) -> Learner:
    """Build and validate a replacement learner BEFORE the app swaps live state.

    weights_only=True restricts deserialization to tensors and primitive data;
    NEVER retry with unrestricted pickle. Restriction is not a RAM sandbox:
    an enormous checkpoint can still exhaust memory, so use trusted sources.
    Architecture, tokenizer, parameter shapes, Adam states, RNG, and metadata
    must agree. Incompatible V1/V2 files are rejected without being modified.
    """
    try:
        data = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(data, dict) or data.get('format') != 'mini-gpt'
                or type(data.get('version')) is not int or data['version'] != CHECKPOINT_VERSION):
            raise ValueError('Not a V3 Transformer checkpoint. Older architectures are incompatible and remain untouched.')
        if data.get('tokenizer') != TOK.specification():
            raise ValueError('Tokenizer identity mismatch: byte IDs cannot be silently replaced.')
        architecture = Architecture(**data['architecture'])
        training = Training(**data['training'])
        sampling = Sampling(**data['sampling'])
        architecture.validate(); training.validate(); sampling.validate()
        # Validate the original training config before applying an explicit
        # destination-device override. Cross-device resume is approximate.
        if device is not None:
            training.device = device
        meta = data['metadata']
        for key in ('steps', 'bytes_read', 'targets_optimized', 'documents_completed'):
            positive_integer(meta[key], key, 0)
        if meta['latest_loss'] is not None:
            finite_number(meta['latest_loss'], 'saved loss', inclusive=True)
        finite_number(meta['training_seconds'], 'training time', inclusive=True)
        if not isinstance(meta['identity'], str) or len(meta['identity']) != 32:
            raise ValueError('Invalid model identity.')
        validate_receipts(meta['imports'])
        # Older V3 checkpoints already contain content-linked receipts. Recover
        # their manifest without trusting any external JSON audit or file dates.
        dataset = meta.get('dataset')
        if dataset is None:
            dataset = {path: digest for digest, receipt in meta['imports'].items()
                       if receipt['status'] == 'complete' for path in receipt['paths']}
        if not isinstance(dataset, dict) or any(
                not isinstance(path, str) or not isinstance(digest, str)
                or digest not in meta['imports']
                or path not in meta['imports'][digest]['paths']
                for path, digest in dataset.items()):
            raise ValueError('Dataset manifest must reference checkpoint receipts.')
        if not isinstance(meta['history'], list) or len(meta['history']) > 1000:
            raise ValueError('Invalid scalar metric history.')
        last_step = 0
        for record in meta['history']:
            positive_integer(record['step'], 'history step')
            if not last_step < record['step'] <= meta['steps']:
                raise ValueError('Non-monotonic metric history.')
            last_step = record['step']
            positive_integer(record['targets'], 'history targets')
            finite_number(record['loss'], 'history loss', inclusive=True)
            finite_number(record['learning_rate'], 'history learning rate')
        learner = Learner(architecture, training, sampling, progress)
        expected = learner.net.state_dict()
        model_state = data['model']
        if not isinstance(model_state, dict) or set(model_state) != set(expected):
            raise ValueError('Invalid parameter names.')
        for name, value in model_state.items():
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid parameter tensor: {name}')
        if not torch.equal(model_state['tokens.weight'], model_state['output.weight']):
            raise ValueError('Tied embedding/output weights disagree.')
        optimizer_state = data['optimizer']
        template = learner.optimizer.state_dict()
        if set(optimizer_state) != {'state', 'param_groups'} or len(optimizer_state['param_groups']) != len(template['param_groups']):
            raise ValueError('Invalid AdamW state structure.')
        shapes = {}
        for saved_group, expected_group, live_group in zip(optimizer_state['param_groups'], template['param_groups'], learner.optimizer.param_groups):
            if set(saved_group) != set(expected_group) or saved_group['params'] != expected_group['params']:
                raise ValueError('Optimizer parameter ordering mismatch.')
            for key, value in expected_group.items():
                if key not in {'params', 'lr', 'weight_decay'} and saved_group[key] != value:
                    raise ValueError(f'Unsupported optimizer option: {key}')
            finite_number(saved_group['lr'], 'saved Adam learning rate')
            finite_number(saved_group['weight_decay'], 'saved weight decay', inclusive=True)
            shapes.update(zip(saved_group['params'], live_group['params']))
        states = optimizer_state['state']
        if not isinstance(states, dict) or set(states) != (set(shapes) if meta['steps'] else set()):
            raise ValueError('Optimizer moments do not match completed updates.')
        for index, state in states.items():
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Invalid Adam moment fields.')
            for name, value in state.items():
                shape = torch.Size([]) if name == 'step' else shapes[index].shape
                if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                        or value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                    raise ValueError('Invalid optimizer tensor.')
            if state['step'].item() != meta['steps'] or (state['exp_avg_sq'] < 0).any():
                raise ValueError('Adam steps or second moments are inconsistent.')
        for name in ('cpu', 'sampling'):
            value = data['rng'][name]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError('Invalid RNG state.')
            torch.Generator(device='cpu').set_state(value)
        mps_rng = data['rng']['mps']
        if mps_rng is not None and (not isinstance(mps_rng, torch.Tensor) or mps_rng.dtype != torch.uint8 or mps_rng.ndim != 1):
            raise ValueError('Invalid MPS RNG state.')
        learner.net.load_state_dict(model_state, strict=True)
        learner.optimizer.load_state_dict(optimizer_state)
        learner.move_to(learner.device)
        learner.cpu_rng = data['rng']['cpu'].clone()
        if mps_rng is not None:
            learner.mps_rng = mps_rng.clone()
        learner.generator.set_state(data['rng']['sampling'])
        for key in ('identity', 'steps', 'bytes_read', 'targets_optimized', 'documents_completed', 'latest_loss', 'training_seconds', 'imports'):
            setattr(learner, key, meta[key])
        learner.dataset = dataset
        corpus_state = meta.get('corpus_state', {})
        if not isinstance(corpus_state, dict): raise ValueError('Invalid corpus metadata.')
        # JSON-compatible numerical metadata only, never executable objects.
        json.dumps(corpus_state, allow_nan=False)
        learner.corpus_state = corpus_state
        learner.history = deque(meta['history'], maxlen=1000)
        learner.last_checkpoint = str(path.resolve())
        if meta.get('saved_device') != learner.device:
            progress('Checkpoint moved between devices; numerical/RNG continuation may differ from uninterrupted training.')
        return learner
    except (KeyError, IndexError, TypeError, AttributeError, RuntimeError, EOFError, UnpicklingError, OverflowError) as error:
        raise ValueError(f'Invalid or incompatible checkpoint (original file unchanged): {error}') from error


# ============================================================================
# 7. VISUAL INSPECTION: actual attention and scalar metrics, no plotting packages
# ============================================================================
def attention_svg(matrix, ids: list[int], layer: int, head: int, path: Path) -> None:
    """Render a standalone heatmap from real eval-mode attention probabilities.

    Rows are queries; columns are keys. Upper-triangle zeros demonstrate the
    causal mask. Attention weights describe one mixing operation, not a complete
    explanation of reasoning. An explicit inspection prompt IS present in this
    optional artifact; no training document is retrieved or archived here.
    """
    values = matrix.tolist()
    cell, left, top = 22, 135, 170
    length = len(ids)
    width, height = max(620, left+cell*length+30), top+cell*length+70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="30" font-family="sans-serif" font-size="18">Transformer attention · layer {layer} · head {head}</text>',
             '<text x="20" y="55" font-family="sans-serif" font-size="12">Rows = queries; columns = keys. Hover cells for probabilities. Darker = larger weight.</text>',
             '<text x="20" y="76" font-family="sans-serif" font-size="12">Actual evaluation-mode weights. Attention is not a full explanation of model behavior.</text>']
    for index, token in enumerate(ids):
        label = escape(f'{index}: {TOK.label(token)}')
        y = top+index*cell
        x = left+index*cell
        parts.append(f'<text x="{left-8}" y="{y+15}" text-anchor="end" font-family="monospace" font-size="10">{label}</text>')
        parts.append(f'<text transform="translate({x+15},{top-8}) rotate(-60)" font-family="monospace" font-size="10">{label}</text>')
        for column, probability in enumerate(values[index]):
            # Linear color mapping across [0,1]; no per-row rescaling that could
            # misrepresent weak diffuse attention as a near-certain selection.
            red = round(245-225*probability)
            green = round(248-168*probability)
            blue = round(255-75*probability)
            parts.append(f'<rect x="{left+column*cell}" y="{y}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="#ddd" stroke-width=".4"><title>query {index}, key {column}: {probability:.8f}</title></rect>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def training_svg(learner: Learner, validation: list[dict], path: Path) -> None:
    """Six lightweight metric charts made from bounded scalar history only."""
    records = list(learner.history)
    if not records:
        raise ValueError('No completed training steps to plot.')
    series = [
        ('Training loss (nats, online batches)', [(r['step'], r['loss']) for r in records]),
        ('Perplexity (same online batches)', [(r['step'], safe_exp(r['loss'])) for r in records]),
        ('Learning rate', [(r['step'], r['learning_rate']) for r in records]),
        ('Optimized targets (includes repeated passes)', [(r['step'], r['targets']) for r in records]),
        ('Parameter count', [(r['step'], learner.parameter_count()) for r in records]),
        ('Held-out NLL (if evaluated)', [(r['step'], r['nll']) for r in validation if r['nll'] is not None])]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800" viewBox="0 0 1000 800">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<text x="24" y="28" font-family="sans-serif" font-size="18">Transformer training · actual metrics · last 1000 updates</text>']
    for index, (title, points) in enumerate(series):
        x0, y0 = 75+(index % 2)*490, 90+(index//2)*250
        points = [(x, y) for x, y in points if y is not None and math.isfinite(y)]
        parts.append(f'<text x="{x0}" y="{y0-20}" font-family="sans-serif" font-size="13">{escape(title)}</text>')
        parts.append(f'<path d="M{x0},{y0} v160 h380" fill="none" stroke="#64748b"/>')
        if not points:
            parts.append(f'<text x="{x0+20}" y="{y0+75}" font-family="sans-serif" font-size="13">No measurements</text>')
            continue
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin, ymax = min(y for _, y in points), max(y for _, y in points)
        coords = [(x0+(x-xmin)/max(1,xmax-xmin)*380, y0+160-(y-ymin)/(ymax-ymin if ymax>ymin else 1)*150) for x,y in points]
        polyline = ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>')
        for x,y in coords[-1:]:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2563eb"/>')
        parts.append(f'<text x="{x0}" y="{y0+184}" font-family="monospace" font-size="10">step {xmin} → {xmax}; values {ymin:.4g} → {ymax:.4g}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def show_prediction(learner: Learner, ids: list[int], raw, sampled, selected: int | None = None) -> None:
    print(f'TRANSFORMER PREDICTION | context length={len(ids)} bytes/special tokens')
    print(f'Decoded context (may start/end mid-UTF-8): {TOK.decode(ids)!r}')
    print(f'Token IDs: {ids}')
    print(f'Temperature={learner.sampling.temperature}; top-k={learner.sampling.top_k}; top-p={learner.sampling.top_p}')
    print('Byte/special token (not words) | raw model P -> final sampling P')
    for index in torch.argsort(raw, descending=True, stable=True)[:10].tolist():
        print(f'  {index:3} {TOK.label(index):12} {float(raw[index]):8.3%} -> {float(sampled[index]):8.3%}')
    if selected is not None:
        print(f'Selected: ID {selected}, {TOK.label(selected)} (P={float(sampled[selected]):.3%})')


# ============================================================================
# 8. APPLICATION AND IMPORTS: one model, model-bound duplicate detection
# ============================================================================
HELP = """
Normal /train uses V4 corpus epochs and held-out validation. /train --legacy opens deprecated manual teaching.
  /train [--epochs N] [--rebuild]  Train folder; rebuild requires RESET
  /train --rebuild              Fresh model; replaces active checkpoint ONLY after success
  /data                         Content changes and checkpoint training history
  /corpus [patience] [delta] [constant|cosine]  Corpus epochs; canonical checkpoint
  /import [PATH] [--recursive] [--force]
      Default: training_data/; only UTF-8 .txt files.
  /architecture                 Show architecture; /preset tiny|small|medium
  /config KEY VALUE             Set architecture BEFORE initialization
  /new tiny|small|medium         Explicit new model; requires RESET confirmation
  /learningrate FLOAT           Training base rate (default .0003)
  /batchsize N /epochs N         Batch size; full corpus passes (/train)
  /steps N /warmup N             Per-document step budget (0 unlimited); warmup
  /device auto|cpu|mps           Device selection; reported CPU fallback
  /attentionmode manual|fast    Same learned projections, different kernel
  /stats /trainstats            Actual counts, parameters, loss, optimizer stats
  /temperature FLOAT /topk N /topp FLOAT /max N /seed INTEGER
  /predict TEXT /debug on|off /step TEXT
  /attention [--layer N] [--head N] TEXT    Actual attention matrix + SVG
  /plot                         Scalar training/validation charts as SVG
  /evaluate PATH                Held-out byte NLL, perplexity, bits/byte
  /benchmark TEXT               Measured inference speed (no training)
  /save [PATH.pt] /load [PATH.pt] Default models/transformer.pt, full state
  /autotrain on|off              Learn chat prompts AFTER reply (default off)
  /help /quit
Plain chat is continuation, not instruction-following. Generated text never trains.
No model-selection or old-architecture comparison commands remain.
"""


class Application:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.data_dir = root / 'training_data'
        self.models_dir = root / 'models'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.architecture = Architecture()
        self.training = Training()
        self.sampling = Sampling()
        self.learner: Learner | None = None
        self.validation: deque[dict] = deque(maxlen=100)
        self.saved_fingerprint: str | None = None
        self.rebuilding = False

    def fingerprint(self) -> str | None:
        """Detect persistent changes without copying/hashing millions of weights.

        Every successful optimizer update increments steps. Settings, receipts,
        counters and RNG also matter: generating text changes the next random
        sample even though it does not train weights. A failed optimizer update
        is separately guarded by needs_restore; it must never be saved.
        """
        learner = self.learner
        if learner is None:
            return None
        state = [learner.identity, learner.steps, learner.bytes_read,
                 learner.targets_optimized, learner.documents_completed,
                 learner.training_seconds, learner.imports, learner.dataset, learner.corpus_state, asdict(self.architecture),
                 asdict(self.training), asdict(self.sampling),
                 learner.generator.get_state().tolist()]
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def autosave(self, progress: Callable = print) -> bool:
        """Save changed weights, settings and provenance to the canonical file."""
        if self.rebuilding:
            return True  # Transaction commits only after ALL selected files succeed.
        if self.learner is None or self.fingerprint() == self.saved_fingerprint:
            return True
        try:
            path = self.models_dir/'transformer.pt'
            save_checkpoint(self.learner, path)
            self.saved_fingerprint = self.fingerprint()
            progress(f'Automatically saved: {path}')
            return True
        except (OSError, ValueError, RuntimeError) as error:
            progress(f'AUTOSAVE FAILED: {error}. Changes remain unsaved; keep this session open and use /save or reload the last valid checkpoint.')
            return False

    def train_text(self, text: str) -> None:
        """Manual teaching and optional chat training share the same save policy."""
        try:
            learner = self.ensure_model()
            display = TrainingProgress(total=document_step_total(len(text.encode('utf-8')),
                self.architecture.context, self.training.batch_size, self.training.epochs),
                epochs=self.training.epochs, verbose=self.sampling.debug,
                validation=self.validation[-1] if self.validation else None)
            learner.train_text(text, display)
        finally:
            # Safe partial updates can be preserved too. An interrupted AdamW
            # update is refused by save_checkpoint, protecting the last good file.
            self.autosave()

    def pending_files(self) -> tuple[list[Path], int, int]:
        """Read hashes, never train. Return new content, partial files, duplicates.

        Two differently named files with identical bytes represent one proposed
        import. Incomplete receipts require deliberate recovery, not Enter.
        Only direct .txt children are offered; /import --recursive stays available.
        """
        learner = self.ensure_model()
        pending, seen, incomplete, duplicates = [], set(), 0, 0
        for path in sorted(self.data_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != '.txt':
                continue
            try:
                digest = content_hash(path)
            except OSError as error:
                print(f'Cannot inspect {path.name}: {error}')
                continue
            receipt = learner.imports.get(digest)
            if receipt and receipt['status'] == 'incomplete':
                incomplete += 1
            elif receipt or digest in seen:
                duplicates += 1
            else:
                pending.append(path)
            seen.add(digest)
        return pending, incomplete, duplicates

    def dataset_status(self) -> dict:
        """Compare content identities against the active checkpoint, read-only.

        Filenames identify replacements; SHA-256 identifies the actual bytes.
        A renamed copy is still the same learned content, so it need not trigger
        retraining or rebuilding. Removed duplicate copies likewise do not erase
        a content identity if another current file still contains those bytes.
        """
        learner = self.ensure_model()
        current, errors = {}, []
        for path in sorted(self.data_dir.iterdir()):
            if path.is_file() and path.suffix.lower() == '.txt':
                try:
                    current[str(path.resolve())] = content_hash(path)
                except OSError as error:
                    errors.append(f'{path}: {error}')
        # Only the managed folder participates. External /import documents are
        # visible in /data, but disappearing external paths are not folder edits.
        root = self.data_dir.resolve()
        source_manifest = learner.corpus_state.get('specification', {}).get('files', learner.dataset)
        previous = {path: digest for path, digest in source_manifest.items()
                    if Path(path).parent == root}
        removed = sorted(set(previous) - set(current))
        modified = sorted(path for path in current if path in previous and current[path] != previous[path])
        added = sorted(set(current) - set(previous))
        unchanged = sorted(path for path in current if previous.get(path) == current[path])
        groups = {}
        for path, digest in current.items():
            groups.setdefault(digest, []).append(path)
        pending, incomplete = [], []
        for digest, paths in groups.items():
            receipt = learner.imports.get(digest)
            if receipt is None:
                pending.append(Path(paths[0]))
            elif receipt['status'] == 'incomplete':
                incomplete.extend(paths)
        # Training a replacement adds gradients, never mathematically unlearns
        # previous gradients. Ask about a rebuild when learned bytes disappear.
        stale = any(previous[path] not in groups for path in removed + modified)
        return dict(current=current, removed=removed, modified=modified, added=added,
                    unchanged=unchanged, duplicates=[paths for paths in groups.values() if len(paths)>1],
                    pending=pending, incomplete=incomplete, stale=stale, errors=errors)

    def show_data(self, status: dict | None = None, *, detailed: bool = True) -> None:
        status = status if status is not None else self.dataset_status()
        if self.ensure_model().corpus_state:
            print('Corpus source pool includes held-out stories; it is NOT a claim every source byte was trained.')
            print(json.dumps(self.learner.corpus_state['specification'], indent=2))
        print('===================================\n TRAINING DATA CHANGED / STATUS\n===================================')
        for label, key in [('Removed', 'removed'), ('Modified', 'modified'), ('New', 'added'), ('Unchanged', 'unchanged')]:
            print(f'{label} files: {len(status[key])}')
            if detailed:
                for path in status[key]:
                    print(f'  {Path(path).name}')
        print(f'Duplicate copies: {sum(len(group)-1 for group in status["duplicates"])}')
        if detailed:
            for group in status['duplicates']:
                print('  Identical content: ' + ', '.join(Path(path).name for path in group))
        for error in status['errors']:
            print(f'Cannot inspect: {error}')
        if detailed:
            print('Files used by the active checkpoint (including historical versions):')
            for digest, receipt in self.ensure_model().imports.items():
                print(f'  {receipt["status"]}: {digest[:12]} — ' + ', '.join(receipt['paths']))
        else:
            print('Use /data for filenames and complete checkpoint training history.')
        print('Deleting/replacing files does not remove their influence from existing weights.')
        print('Manual text inputs are not archived; their influence also remains until a fresh rebuild.')

    def rebuild(self, status: dict | None = None) -> bool:
        """Train a provisional fresh learner, committing only on total success.

        This is a transaction at model granularity, not gradient subtraction.
        The old Python learner and checkpoint stay intact while a second model
        trains. This temporarily uses memory for both models. No partial new
        checkpoint is published, so automatic startup cannot select a failed
        rebuild. A crash loses provisional work, not the last committed model.
        """
        status = status if status is not None else self.dataset_status()
        if status['errors']:
            print('Rebuild cancelled: fix unreadable training files first.')
            return False
        if not status['current']:
            print('Rebuild cancelled: no training files. Existing model preserved.')
            return False
        old = self.ensure_model()
        if old.steps and not self.confirm_reset():
            return False
        old_fingerprint, old_validation = self.saved_fingerprint, self.validation.copy()
        self.checkpoint_current()
        committed = False
        self.rebuilding = True
        try:
            # Independent initialization, not a copy of old weights/Adam state.
            fresh_sampling = replace(self.sampling, seed=uuid.uuid4().int % (2**31))
            self.use_model(Learner(replace(self.architecture), replace(self.training), fresh_sampling))
            for file_index, path in enumerate(status['current'], 1):
                if content_hash(Path(path)) != status['current'][path]:
                    raise ValueError('Dataset changed after selection; scan again before rebuilding.')
                result = self.import_files(Path(path), dataset_position=(file_index, len(status['current'])))
                if result['incomplete']:
                    raise ValueError('A selected document did not finish training.')
            final = self.dataset_status()
            if final['errors'] or final['current'] != status['current']:
                raise ValueError('Dataset changed during rebuild; replacement was not committed.')
            if not self.learner.steps:
                raise ValueError('No trainable text found; replacement was not committed.')
            save_checkpoint(self.learner, self.models_dir/'transformer.pt')
            committed = True
            self.saved_fingerprint = self.fingerprint()
            print('Rebuild complete and saved. Fresh weights learned only the selected current files.')
            return True
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            print(f'Rebuild not committed: {error}. Previous model preserved; provisional learning discarded.')
            return False
        finally:
            self.rebuilding = False
            if not committed:
                self.use_model(old)
                self.saved_fingerprint = old_fingerprint
                self.validation = old_validation

    @property
    def training_engine(self) -> str:
        # Resolve from the callable selected by the normal /train workflow.
        return self.train_dataset.__func__.engine_name

    def train_dataset(self, *, rebuild=False, epochs=None) -> bool:
        """One transactional entry point for startup and the normal /train command.

        Old, untracked weights cannot claim independent validation. Explicit
        legacy teaching stays available, while a confirmed fresh rebuild makes
        a trustworthy deterministic split. Original state remains available in
        a temporary full checkpoint until the validated candidate is committed.
        """
        old=self.ensure_model()
        if rebuild and old.steps and not self.confirm_reset(): return False
        if old.steps and not old.corpus_state and not rebuild:
            print('This checkpoint has Unknown training provenance. Use /train --rebuild for held-out validation, or /train --legacy to retain its knowledge without that claim.')
            return False
        corpus=None
        pointer,fingerprint=old.last_checkpoint,self.saved_fingerprint
        with tempfile.TemporaryDirectory() as folder:
            before=Path(folder)/'before.pt';save_checkpoint(old,before);old.last_checkpoint=pointer
            try:
                candidate= (Learner(replace(self.architecture),replace(self.training),replace(self.sampling,seed=uuid.uuid4().int%(2**31))) if rebuild else old)
                candidate.training=replace(candidate.training,epochs=epochs or candidate.training.epochs)
                corpus=CorpusWindows(sorted(self.data_dir.glob('*.txt')),candidate.architecture.context,42)
                replacement,_=train_corpus(candidate,corpus,Path(folder)/'candidate.pt')
                # Validate serialization before publishing the replacement.
                replacement=load_checkpoint(Path(folder)/'candidate.pt',candidate.device,progress=lambda _:None)
                save_checkpoint(replacement,self.models_dir/'transformer.pt')
                self.use_model(replacement)
                print('Validated V4 state saved. Ready to chat.')
                return True
            except (OSError,ValueError,RuntimeError,KeyboardInterrupt) as error:
                restored=load_checkpoint(before,old.device,progress=lambda _:None)
                restored.last_checkpoint=pointer
                self.use_model(restored);self.saved_fingerprint=fingerprint
                print(f'Training not committed: {error}. Previous model preserved.')
                return False
            finally:
                if corpus is not None:corpus.close()
    train_dataset.engine_name = 'V4 Corpus Trainer'

    def startup(self, *, fresh=False, checkpoint=None):
        canonical=self.models_dir/'transformer.pt'
        if fresh and canonical.exists() and not self.confirm_reset():fresh=False
        selected=checkpoint or canonical
        if not fresh and selected.exists():self.use_model(load_checkpoint(selected))
        elif checkpoint is not None and not fresh:raise ValueError('Checkpoint does not exist.')
        model=self.ensure_model()
        print('=======================================\n        MINI LANGUAGE MODEL\n=======================================')
        print(f'Architecture: GPT Transformer\nTraining engine: {self.training_engine}')
        print(f'Checkpoint training engine: {model.corpus_state.get("engine", "V4 Corpus Trainer" if model.corpus_state else "Unknown")}')
        print(f'Active checkpoint: {Path(model.last_checkpoint).name if model.last_checkpoint else "none (new model)"}\nParameters: {model.parameter_count():,}\nTraining steps: {model.steps:,}\nDevice: {model.device}')
        status=self.dataset_status()
        current=set(status['current'].values())
        previous=set(model.corpus_state.get('specification',{}).get('files',model.dataset).values())
        if status['errors']:
            print('Training files unreadable; use /data to inspect. Training skipped.')
        elif current != previous and current:
            stale=status['stale']
            print('TRAINING DATA HAS CHANGED' if stale else 'NEW TRAINING DATA DETECTED')
            print(f'New: {len(status["added"])} | Removed: {len(status["removed"])} | Modified: {len(status["modified"])}')
            for filename in status['current']:
                print('  ' + Path(filename).name)
            print('Training from scratch takes time; CPU training may be slow. You can skip and explore /help.')
            new_model = model.steps == 0
            if new_model:
                print('No trained model found.\n[1] Train a new AI from scratch\n[2] Start without training\n[3] Help')
            else:
                print('[1] '+('Rebuild from scratch' if stale else 'Train updated dataset')+'\n[2] '+('Continue existing model' if stale else 'Rebuild from scratch')+'\n[3] Skip and chat')
            while True:
                choice=input('Choose [1]: ').strip() or '1'
                if new_model and choice == '3':
                    self.command('/help', 'chat')
                    continue
                if choice in {'1','2','3','s','S'}:break
            if choice == '1' or (choice == '2' and not new_model):
                rebuild=False if new_model else ((choice=='1') if stale else (choice=='2'))
                if not rebuild and stale:print('Removed data still influences existing weights; continuing does not forget it.')
                self.train_dataset(rebuild=rebuild)
        elif previous and not current:
            print('No training files remain. Existing model preserved; add files before rebuilding.')
        elif not current:
            print('Add UTF-8 .txt files to training_data, then use /train. TinyStories uses <|endoftext|> story boundaries.')
        else: print('Training data: unchanged')
        return 'chat'

    def legacy_startup(self, *, fresh: bool = False, checkpoint: Path | None = None) -> str:
        """Load only the canonical file; invalid checkpoints fail closed.

        Explicit --checkpoint can inspect an external export. --fresh requires
        RESET confirmation if it would replace an existing canonical model.
        """
        print('===================================\n  MINI LANGUAGE MODEL — GPT (legacy workflow)\n===================================')
        canonical = self.models_dir/'transformer.pt'
        # Only files with our private temporary-save prefix can be cleaned here.
        # The app is single-process; do not run concurrent writers to this folder.
        for temporary in self.models_dir.glob('.checkpoint-*.tmp'):
            temporary.unlink(missing_ok=True)
        if fresh and canonical.exists() and not self.confirm_reset():
            fresh = False
        if not fresh:
            selected = checkpoint or canonical
            if selected.exists():
                # Fail closed on corrupt files: never replace learned state with
                # a random model merely because loading failed.
                self.use_model(load_checkpoint(selected))
            elif checkpoint is not None:
                raise ValueError(f'Checkpoint does not exist: {selected}')
        learner = self.ensure_model()
        print('Transformer loaded successfully!' if learner.last_checkpoint else 'New Transformer created; it has not learned any text yet.')
        print(f'Model: {Path(learner.last_checkpoint).name if learner.last_checkpoint else "Transformer " + learner.identity[:8]}')
        print(f'Parameters: {learner.parameter_count():,}\nTraining steps: {learner.steps:,}')
        print(f'Device: {"Apple MPS" if learner.device == "mps" else "CPU"}')
        if learner.corpus_state:
            changes = self.dataset_status()
            print(f'Corpus files: {len(changes["added"])} added, {len(changes["modified"])} modified, {len(changes["removed"])} removed. Use /data to inspect.')
            print('Corpus checkpoint: automatic legacy imports are disabled to protect the held-out split. Use /corpus to continue matching data.')
            return 'chat' if learner.steps else 'training'
        status = self.dataset_status()
        pending = status['pending']
        if status['errors']:
            for error in status['errors']:
                print(f'Cannot inspect: {error}')
            print('Automatic training disabled until the folder can be scanned completely.')
        elif status['stale']:
            self.show_data(status, detailed=False)
            print('Your current Transformer was trained on data that is no longer present.')
            print('[1] Rebuild AI from scratch (Recommended)\n[2] Continue training existing AI\n[3] Skip training and start chatting')
            while True:
                choice = input('Choose [1]: ').strip() or '1'
                if choice in {'1', '2', '3'}:
                    break
            if choice == '1':
                self.rebuild(status)
            elif choice == '2':
                print('Continuing retains information from removed files and previous versions. Nothing has been forgotten.')
                if pending:
                    self.checkpoint_current()
                    for file_index, path in enumerate(pending, 1):
                        self.import_files(path, dataset_position=(file_index, len(pending)))
                        if self.ensure_model().needs_restore:
                            break
        elif pending:
            print(f'New training content: {len(pending)} files. Training can take time.')
            for path in pending:
                print(f'  {path.name}')
            while True:
                answer = input('Press Enter to train or S to skip: ').strip().lower()
                if answer in {'', 's'}:
                    break
            if not answer:
                self.checkpoint_current()
                for file_index, path in enumerate(pending, 1):
                    self.import_files(path, dataset_position=(file_index, len(pending)))
                    if self.ensure_model().needs_restore:
                        break
        if status['incomplete']:
            print(f'Incomplete imports: {len(status["incomplete"])}; not automatically replayed. Use /train --rebuild or deliberate /import --force.')
        learner = self.ensure_model()
        print('Advanced commands are available with /help; /train returns to teaching mode.')
        if learner.steps:
            return 'chat'
        print('Initial training: type a short English or Finnish text below, or add .txt files to training_data/ and use /import. Your first successful training opens chat.')
        return 'training'

    def ensure_model(self) -> Learner:
        if self.learner is None:
            self.learner = Learner(self.architecture, self.training, self.sampling)
        return self.learner

    def use_model(self, learner: Learner) -> None:
        self.learner = learner
        self.architecture, self.training, self.sampling = learner.architecture, learner.training, learner.sampling
        self.validation.clear()
        self.saved_fingerprint = self.fingerprint()

    def checkpoint_current(self) -> Path:
        """Commit current state to the one active file, without retaining backups."""
        path = self.models_dir/'transformer.pt'
        save_checkpoint(self.ensure_model(), path)
        self.saved_fingerprint = self.fingerprint()
        return path

    def confirm_reset(self) -> bool:
        print('WARNING: this replaces your trained AI. No permanent backup will be retained.')
        return input('Type RESET to confirm, or Enter to cancel: ').strip() == 'RESET'

    def import_files(self, source: Path | None = None, *, recursive: bool = False, force: bool = False,
                     progress: Callable = print,
                     dataset_position: tuple[int, int] | None = None) -> dict:
        """Hash, stream, learn, verify, then mark completion on this model only.

        No auto-replay of incomplete files: a partial optimizer history cannot
        be inferred from a file hash or undone by decrementing a word counter.
        --force is an explicit full replay and can repeat a learned prefix.
        No automatic historical checkpoint is retained. Modified content is a new document, not a subtraction/diff.
        """
        source = source or self.data_dir
        if source.is_dir():
            paths = source.rglob('*') if recursive else source.iterdir()
            paths = sorted((p for p in paths if p.is_file() and p.suffix.lower()=='.txt'), key=str)
        elif source.is_file() and source.suffix.lower() == '.txt':
            paths = [source]
        else:
            raise ValueError('Provide a .txt file or directory.')
        learner = self.ensure_model()
        if learner.corpus_state:
            raise ValueError('Use /corpus for this checkpoint; legacy imports could train on its validation stories.')
        result = {'found': len(paths), 'complete': 0, 'skipped': 0, 'incomplete': 0, 'bytes_read': 0, 'steps': 0}
        progress(f'IMPORTING TRAINING DATA | Found: {len(paths)} text files | Transformer {learner.identity[:8]}')
        for number, path in enumerate(paths, 1):
            progress(f'[{number}/{len(paths)}] {path}')
            try:
                digest = content_hash(path)
            except OSError as error:
                progress(f'Failed before training: {error}')
                result['incomplete'] += 1
                continue
            canonical = str(path.resolve())
            previous = learner.imports.get(digest)
            if previous and not force:
                if canonical not in previous['paths']:
                    previous['paths'].append(canonical)
                if previous['status'] == 'complete':
                    learner.dataset[canonical] = digest
                    progress('Skipped: this content is complete on the ACTIVE Transformer.')
                    result['skipped'] += 1
                else:
                    progress('INCOMPLETE previous attempt. Deliberately --force replay or confirm a rebuild; partial learning may repeat.')
                    result['incomplete'] += 1
                self.autosave(progress)
                continue
            if any(canonical in receipt['paths'] and key != digest for key,receipt in learner.imports.items()):
                progress('Modified file: treating its FULL contents as new data; older learning is not subtracted.')
            before_steps, before_bytes = learner.steps, learner.bytes_read
            receipt = {'status': 'incomplete', 'paths': list(dict.fromkeys([*(previous['paths'] if previous else []), canonical])), 'bytes_read': 0, 'steps': 0,
                       'attempts': previous['attempts']+1 if previous else 1}
            learner.imports[digest] = receipt
            # Receipt intent is in memory until the file finishes or stops.
            # A hard crash reloads the previous atomic state, without new updates.
            try:
                if content_hash(path) != digest:
                    raise ValueError('File changed before ingestion; retry after it stops changing.')
                display = TrainingProgress.for_file(path, learner, progress,
                    dataset_position=dataset_position or (number, len(paths)),
                    validation=self.validation[-1] if self.validation else None)
                try:
                    learner.train_chunks(text_chunks(path), display)
                finally:
                    # Also clear the UI if an extension fails before entering the
                    # trainer. The real trainer normally closes its own display.
                    if not display.closed:
                        display.finish('Stopped', average=None, targets=0)
                if content_hash(path) != digest:
                    raise ValueError('File changed while importing. Updates may reflect mixed content.')
                receipt['status'] = 'complete'
                learner.dataset[canonical] = digest
                result['complete'] += 1
                index, total = dataset_position or (number, len(paths))
                progress(f'File verified complete | Dataset processed: {index}/{total} files ({index/total:.0%})')
            except (OSError, UnicodeError, ValueError, RuntimeError, KeyboardInterrupt) as error:
                result['incomplete'] += 1
                progress(f'Status: INCOMPLETE ({error}). Earlier updates remain; no automatic historical rollback is retained.')
                if isinstance(error, KeyboardInterrupt):
                    raise
                if learner.needs_restore:
                    progress('Optimizer update failed; stopping imports until a checkpoint is restored.')
                    break
            finally:
                receipt['bytes_read'] = learner.bytes_read-before_bytes
                receipt['steps'] = learner.steps-before_steps
                result['bytes_read'] += receipt['bytes_read']
                result['steps'] += receipt['steps']
                self.autosave(progress)
        progress(f'IMPORT FINISHED: {result}')
        return result

    def stats(self) -> None:
        learner = self.ensure_model()
        print(f'Architecture: GPT-style decoder-only | {asdict(self.architecture)}')
        print(f'Tokenizer: {TOK.specification()}')
        print(f'Parameters: {learner.parameter_count():,}; parameter tensor bytes: {sum(p.numel()*p.element_size() for p in learner.net.parameters()):,}')
        print('Tensor bytes exclude gradients, optimizer moments, activations, and process overhead.')
        print(f'Device: {learner.device}; optimizer: AdamW; settings: {asdict(self.training)}')
        print(f'Current optimizer learning rate: {learner.optimizer.param_groups[0]["lr"]:.6g}')
        print(f'Steps: {learner.steps:,}; bytes read: {learner.bytes_read:,}; targets optimized across passes: {learner.targets_optimized:,}')
        print(f'Complete nonempty document runs: {learner.documents_completed}; latest batch NLL: {learner.latest_loss}; batch perplexity: {safe_exp(learner.latest_loss)}')
        if learner.training_seconds:
            print(f'Measured cumulative training throughput (includes data reads): {learner.targets_optimized/learner.training_seconds:.1f} optimized targets/s')
        print(f'Sampling settings: {asdict(self.sampling)}; incomplete receipts: {sum(r["status"]=="incomplete" for r in learner.imports.values())}')

    def respond(self, prompt: str, step: bool = False) -> None:
        learner = self.ensure_model()
        output = []
        iterator = learner.generate_steps(prompt)
        while True:
            if step and input('Enter = sample one byte/EOS; q = stop > ').strip().lower() == 'q':
                break
            try:
                decision = next(iterator)
            except StopIteration:
                break
            if step or self.sampling.debug:
                show_prediction(learner, decision['ids'], decision['raw'], decision['sampling'], decision['selected'])
            output.append(decision['selected'])
            if decision['selected'] == TOK.EOS or len(output) == self.sampling.max_tokens:
                break
        print('AI  > ' + TOK.decode(output))

    def benchmark(self, prompt: str) -> dict:
        learner = self.ensure_model()
        ids = learner.context_ids(prompt)
        learner.predict_ids(ids)  # Warmup excluded from measured inference.
        synchronize(learner.device)
        start = time.perf_counter()
        for _ in range(10):
            learner.predict_ids(ids)
        synchronize(learner.device)
        elapsed = time.perf_counter()-start
        result = {'forward_passes': 10, 'context_tokens': len(ids), 'device': learner.device,
                  'seconds': elapsed, 'milliseconds_per_forward': elapsed*100,
                  'parameters': learner.parameter_count(), 'byte_vocabulary': TOK.size}
        if learner.last_checkpoint and Path(learner.last_checkpoint).is_file():
            result['last_checkpoint_bytes_on_disk'] = Path(learner.last_checkpoint).stat().st_size
        print(json.dumps(result, indent=2))
        print('Measured forward passes, not promised generation or training throughput; includes device-to-CPU logits transfer.')
        return result

    def command(self, line: str, mode: str) -> str:
        name, _, argument = line.partition(' ')
        argument = argument.strip()
        if name == '/corpus':
            options=shlex.split(argument)
            if len(options)>3: raise ValueError('Use /corpus [patience] [min_delta] [constant|cosine]')
            corpus=CorpusWindows(sorted(self.data_dir.glob('*.txt')),self.architecture.context,42)
            try:
                with tempfile.TemporaryDirectory() as folder:
                    old=self.ensure_model()
                    pointer, fingerprint=old.last_checkpoint,self.saved_fingerprint
                    before=Path(folder)/'before.pt'
                    save_checkpoint(old,before)
                    old.last_checkpoint=pointer
                    try:
                        replacement,records=train_corpus(old,corpus,Path(folder)/'candidate.pt',
                            patience=int(options[0]) if options else 3,
                            min_delta=float(options[1]) if len(options)>1 else .001,
                            schedule=options[2] if len(options)>2 else 'constant')
                        save_checkpoint(replacement,self.models_dir/'transformer.pt')
                    except BaseException:
                        restored=load_checkpoint(before,old.device,progress=lambda _:None)
                        restored.last_checkpoint=pointer
                        self.use_model(restored)
                        self.saved_fingerprint=fingerprint
                        raise
                    self.use_model(replacement)
                    print('Best validation state saved to models/transformer.pt.')
            finally: corpus.close()
            return 'chat'
        if name == '/train':
            options=shlex.split(argument)
            if options == ['--legacy']:
                print('Legacy incremental teaching: no independent held-out validation claim.')
                return 'training'
            rebuild='--rebuild' in options
            options=[item for item in options if item != '--rebuild']
            epochs=None
            if options:
                if len(options)!=2 or options[0]!='--epochs':
                    raise ValueError('Use /train [--epochs N] [--rebuild], or /train --legacy.')
                epochs=int(options[1]);positive_integer(epochs,'epochs')
            return 'chat' if self.train_dataset(rebuild=rebuild,epochs=epochs) else mode
        if name == '/data':
            self.show_data()
            return mode
        if name in {'/done', '/train', '/quit'}:
            return {'/done':'chat', '/train':'training', '/quit':'quit'}[name]
        if name == '/help':
            print(HELP)
        elif name == '/architecture':
            print(json.dumps(asdict(self.architecture), indent=2))
            print('Architecture changes require a new model. /new requires RESET confirmation; no backup is retained.')
        elif name in {'/preset', '/config'}:
            if self.learner is not None and (self.learner.steps or self.learner.imports):
                raise ValueError('Architecture is already initialized. Use /new PRESET explicitly and confirm RESET.')
            if name == '/preset':
                replacement = Architecture.preset(argument)
            else:
                key, _, value = argument.partition(' ')
                if key not in asdict(self.architecture):
                    raise ValueError('Unknown architecture field.')
                replacement = replace(self.architecture, **{key:float(value) if key=='dropout' else int(value)})
            replacement.validate()
            # An explicit architecture command may replace a pristine random
            # initialization, but never discard learning or import receipts.
            self.learner = None
            self.architecture = replacement
        elif name == '/new':
            replacement = Architecture.preset(argument)
            if self.learner is not None:
                if self.learner.steps and not self.confirm_reset():
                    return mode
                self.checkpoint_current()
            self.learner = None
            self.architecture = replacement
            self.validation.clear()
            print('New untrained Transformer configured; no old weights are reused.')
        elif name in {'/learningrate','/batchsize','/epochs','/steps','/warmup','/device','/attentionmode'}:
            key = {'/learningrate':'learning_rate','/batchsize':'batch_size','/epochs':'epochs','/steps':'max_steps',
                   '/warmup':'warmup_steps','/device':'device','/attentionmode':'efficient_attention'}[name]
            if name == '/attentionmode':
                if argument not in {'manual','fast'}:
                    raise ValueError('Use manual or fast.')
                value = argument == 'fast'
            else:
                value = argument if key == 'device' else float(argument) if key=='learning_rate' else int(argument)
            replacement = replace(self.training, **{key:value})
            replacement.validate()
            if self.learner is not None:
                if key == 'device':
                    self.learner.move_to(choose_device(value))
                self.learner.training = replacement
            self.training = replacement
            print(f'Training setting {key} = {value}')
        elif name in {'/temperature','/topk','/topp','/max','/seed','/debug','/autotrain'}:
            key = {'/temperature':'temperature','/topk':'top_k','/topp':'top_p','/max':'max_tokens',
                   '/seed':'seed','/debug':'debug','/autotrain':'auto_train'}[name]
            if key in {'debug','auto_train'}:
                if argument not in {'on','off'}:
                    raise ValueError('Use on or off.')
                value = argument == 'on'
            else:
                value = float(argument) if key in {'temperature','top_p'} else int(argument)
            replacement = replace(self.sampling, **{key:value})
            replacement.validate()
            if self.learner:
                self.learner.sampling = replacement
                if key == 'seed':
                    self.learner.generator.manual_seed(value)
                    print('Sampling RNG reset; existing weights/training-dropout RNG unchanged.')
            self.sampling = replacement
        elif name in {'/stats','/trainstats'}:
            self.stats()
        elif name == '/import':
            options = shlex.split(argument)
            path_parts, recursive, force = [], False, False
            for option in options:
                if option == '--recursive': recursive = True
                elif option == '--force': force = True
                elif option.startswith('--'): raise ValueError('Unknown import option; Transformer is the only model. See /help.')
                else: path_parts.append(option)
            self.import_files(Path(' '.join(path_parts)) if path_parts else None, recursive=recursive, force=force)
        elif name == '/predict':
            learner = self.ensure_model()
            if not learner.steps:
                print('UNTRAINED network: these are random-initialization probabilities, not learned language.')
            ids = learner.context_ids(argument)
            logits, _ = learner.predict_ids(ids)
            show_prediction(learner, ids, torch.softmax(logits.double(), -1), sampling_probabilities(logits,self.sampling))
        elif name == '/step':
            self.respond(argument, step=True)
        elif name == '/attention':
            layer, head = 0, 0
            prompt = argument
            while prompt.startswith('--'):
                parts = prompt.split(None, 2)
                if len(parts) < 2 or parts[0] not in {'--layer', '--head'}:
                    raise ValueError('Use --layer N and --head N before the attention prompt.')
                number = int(parts[1])
                if parts[0] == '--layer': layer = number
                else: head = number
                prompt = parts[2] if len(parts) == 3 else ''
            learner = self.ensure_model()
            if not 0 <= layer < self.architecture.layers or not 0 <= head < self.architecture.heads:
                raise ValueError('Layer/head out of range (zero-based indices).')
            ids = learner.context_ids(prompt)
            _, matrices = learner.predict_ids(ids, inspect=True)
            matrix = matrices[layer][head]
            (self.root/'outputs').mkdir(exist_ok=True)
            path = self.root/'outputs'/f'attention-layer{layer}-head{head}.svg'
            attention_svg(matrix, ids, layer, head, path)
            print(f'Actual attention heatmap: {path}; untrained={learner.steps==0}')
            print('Last query attends to (position, token, probability):')
            for index in torch.argsort(matrix[-1],descending=True)[:10].tolist():
                print(index, TOK.label(ids[index]), f'{float(matrix[-1,index]):.3%}')
        elif name == '/plot':
            (self.root/'outputs').mkdir(exist_ok=True)
            path = self.root/'outputs'/'training.svg'
            training_svg(self.ensure_model(), list(self.validation), path)
            print(f'Training metric charts: {path}')
        elif name == '/evaluate':
            if not argument: raise ValueError('Specify a separate held-out UTF-8 .txt file.')
            path = Path(argument)
            learner = self.ensure_model()
            digest = content_hash(path)
            if digest in learner.imports or digest in learner.corpus_state.get('specification', {}).get('files', {}).values():
                print('WARNING: this content was imported or partly imported; it is not held-out data.')
            result = learner.evaluate(text_chunks(path))
            # Display-only validation history belongs to the UI, not learner
            # weights/RNG/counters. It is cleared when a different model loads.
            self.validation.append({'step':learner.steps,'nll':result['nll_bytes_and_eos']})
            print(json.dumps(result, indent=2))
        elif name == '/benchmark':
            self.benchmark(argument)
        elif name == '/save':
            path = Path(argument) if argument else self.models_dir/'transformer.pt'
            if path.resolve().parent == self.models_dir.resolve() and path.name != 'transformer.pt':
                raise ValueError('models/ holds only transformer.pt; export elsewhere explicitly.')
            save_checkpoint(self.ensure_model(), path)
            self.saved_fingerprint = self.fingerprint()
            print(f'Saved full Transformer training state: {path}')
        elif name == '/load':
            path = Path(argument) if argument else self.models_dir/'transformer.pt'
            replacement = load_checkpoint(path, self.training.device)
            self.use_model(replacement)
            self.autosave()
            print(f'Restored weights, AdamW, RNG, and import receipts at step {replacement.steps}.')
        else:
            raise ValueError('Unknown command. Use /help. This version has only the Transformer.')
        return mode


def interactive(*, fresh: bool = False, checkpoint: Path | None = None,
                root: Path = ROOT) -> None:
    """Friendly default flow; all advanced slash commands remain available.

    The finally block also runs on EOF and Ctrl-C. It cannot run after power
    loss or a forced process kill, the last atomic commit is the only on-disk recovery state. Corpus training
    commits only after a successful run; legacy imports commit per file.
    """
    app = Application(root)
    try:
        mode = app.startup(fresh=fresh, checkpoint=checkpoint)
        guided = not app.ensure_model().steps
        while mode != 'quit':
            if mode == 'training':
                print('TRAINING MODE — each input is an independent document.')
            while mode == 'training':
                line = input('TRAIN > ')
                try:
                    if line.startswith('/'):
                        mode = app.command(line, mode)
                    elif line.strip():
                        app.train_text(line)
                    if guided and mode == 'training' and app.ensure_model().steps:
                        mode, guided = 'chat', False
                except KeyboardInterrupt:
                    print('\nTraining interrupted; partial progress may remain. Check recovery status before retrying.')
                except (ValueError, OSError, RuntimeError) as error:
                    print(f'Error: {error}')
            if mode == 'chat':
                print('===================================\n  CHAT WITH YOUR AI\n===================================')
                print('Text continuation; messages are not learned unless /autotrain on.')
            while mode == 'chat':
                line = input('YOU > ')
                try:
                    if line.startswith('/'):
                        mode = app.command(line, mode)
                    else:
                        app.respond(line)
                        if app.sampling.auto_train and line:
                            app.train_text(line)
                except KeyboardInterrupt:
                    print('\nOperation interrupted; return to commands or /quit.')
                except (ValueError, OSError, RuntimeError) as error:
                    print(f'Error: {error}')
    except (EOFError, KeyboardInterrupt):
        print('\nClosing session.')
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Startup failed: {error}')
    finally:
        app.autosave()



# ============================================================================
# 9. EXECUTABLE TESTS: causality, actual learning, continuation, and provenance
# ============================================================================
class TransformerTests(unittest.TestCase):
    """Small CPU tests inspect mathematical properties, not just code structure."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def learner(self, *, dropout: float = 0, efficient: bool = False, **changes) -> Learner:
        config = Architecture(embedding=16, heads=2, layers=2, context=16, feedforward=32, dropout=dropout)
        training = Training(device='cpu', batch_size=4, learning_rate=.01, warmup_steps=0, efficient_attention=efficient)
        for key, value in changes.items():
            setattr(training, key, value)
        return Learner(config, training, Sampling(seed=12, top_k=1, top_p=1, max_tokens=10), progress=lambda _:None)

    def test_utf8_roundtrip_and_fixed_vocabulary(self) -> None:
        text = 'Äiti sanoi: “Yö, hyvää!”\n你好 👋'
        self.assertEqual(TOK.decode(TOK.encode(text), errors='strict'), text)
        self.assertEqual(TOK.encode('ä'), [195,164])
        self.assertEqual(TOK.size, 259)
        self.assertEqual(TOK.decode([TOK.BOS,*TOK.encode(text),TOK.EOS,TOK.PAD]), text)
        self.assertEqual(TOK.decode([195]), '�')
        self.assertEqual(TOK.label(195), '0xC3')
        self.assertEqual(TOK.label(TOK.BOS), '<BOS>')

    def test_streaming_utf8_chunk_sizes(self) -> None:
        text = 'Finnish ääkköset.\r\nKissa! 👋'
        whole = list(document_windows([text], 5))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'utf8.txt'
            path.write_bytes(text.encode('utf-8'))
            for size in (1,2,3,5,65536):
                self.assertEqual(list(document_windows(text_chunks(path,size),5)),whole)
            path.write_bytes(b'\xef\xbb\xbf'+text.encode('utf-8'))
            self.assertEqual(list(document_windows(text_chunks(path,1),5)),whole)

    def test_document_windows_empty_boundary_and_no_duplicate_targets(self) -> None:
        for context in (1,2,3,8):
            windows = list(document_windows(['ab','cd'],context))
            self.assertTrue(all(len(w)<=context+1 for w in windows))
            self.assertEqual([t for w in windows for t in w[1:]], [97,98,99,100,TOK.EOS])
        self.assertEqual(list(document_windows([''],3)), [])
        windows = list(document_windows(['ab'],2))+list(document_windows(['cd'],2))
        self.assertEqual(sum(w[0]==TOK.BOS for w in windows),2)
        self.assertFalse(any(TOK.EOS in w[:-1] for w in windows))

    def test_shapes_tied_parameters_and_attention_rows(self) -> None:
        learner = self.learner()
        x = torch.tensor([[TOK.BOS,97,98,99],[TOK.BOS,100,101,102]])
        learner.net.eval()
        logits, matrices = learner.net(x,inspect=True)
        self.assertEqual(tuple(logits.shape),(2,4,259))
        self.assertEqual(len(matrices),2)
        for matrix in matrices:
            self.assertEqual(tuple(matrix.shape),(2,2,4,4))
            torch.testing.assert_close(matrix.sum(-1), torch.ones(2,2,4))
            self.assertEqual(torch.count_nonzero(matrix.triu(1)).item(),0)
        self.assertIs(learner.net.tokens.weight,learner.net.output.weight)
        self.assertTrue(torch.isneginf(logits[...,TOK.BOS]).all())
        self.assertTrue(torch.isneginf(logits[...,TOK.PAD]).all())

    def test_future_token_cannot_change_past_logits_or_gradients(self) -> None:
        learner = self.learner()
        learner.net.eval()
        first = torch.tensor([[TOK.BOS,97,98,99,100]])
        second = torch.tensor([[TOK.BOS,97,98,120,121]])
        for efficient in (False,True):
            a,_ = learner.net(first,efficient=efficient)
            b,_ = learner.net(second,efficient=efficient)
            torch.testing.assert_close(a[:,:3,:256], b[:,:3,:256], atol=1e-6,rtol=1e-6)
        block = learner.net.blocks[0]
        embeddings = torch.randn(1,5,16,requires_grad=True)
        output,_ = block(embeddings,efficient=False)
        output[0,1].square().sum().backward()
        self.assertEqual(torch.count_nonzero(embeddings.grad[:,2:]).item(),0)

    def test_manual_attention_matches_fast_path(self) -> None:
        learner=self.learner(dropout=.2)
        learner.net.eval()
        x=torch.tensor([[TOK.BOS,97,98,99]])
        manual,_=learner.net(x,inspect=True)
        fast,_=learner.net(x,efficient=True)
        torch.testing.assert_close(manual[:,:,:256],fast[:,:,:256],atol=2e-6,rtol=2e-5)

    def test_residual_identity_when_branches_zero(self) -> None:
        learner=self.learner()
        block=learner.net.blocks[0]
        with torch.no_grad():
            for parameter in block.attention.parameters(): parameter.zero_()
            for parameter in block.feedforward.parameters(): parameter.zero_()
        x=torch.randn(2,4,16)
        output,_=block(x,efficient=False)
        torch.testing.assert_close(output,x,atol=0,rtol=0)

    def test_parameter_gradients_and_adam_updates(self) -> None:
        learner=self.learner()
        before={k:v.clone() for k,v in learner.net.state_dict().items()}
        learner.train_text('abc',progress=lambda _:None)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in learner.net.parameters()))
        self.assertTrue(any(not torch.equal(v,learner.net.state_dict()[k]) for k,v in before.items()))
        self.assertEqual(learner.steps,1)
        self.assertEqual(learner.targets_optimized,4)
        self.assertEqual(learner.bytes_read,3)
        self.assertEqual(len(learner.optimizer.state),len(list(learner.net.parameters())))

    def test_repetitive_loss_and_memorized_generation(self) -> None:
        learner=self.learner()
        initial=learner.evaluate(['abcabc'])['nll_bytes_and_eos']
        for _ in range(70): learner.train_text('abcabc',progress=lambda _:None)
        final=learner.evaluate(['abcabc'])['nll_bytes_and_eos']
        self.assertLess(final, initial*.05)
        self.assertEqual(learner.generate('ab'),'cabc')
        decisions=list(learner.generate_steps('ab'))
        self.assertEqual(decisions[-1]['selected'],TOK.EOS)

    def test_generation_limit_empty_unknown_and_context_crop(self) -> None:
        learner=self.learner()
        with self.assertRaises(ValueError): learner.generate('untrained')
        self.assertEqual(learner.train_text('',progress=lambda _:None)['steps'],0)
        learner.train_text('a',progress=lambda _:None)
        learner.sampling.max_tokens=3
        logits=torch.full((TOK.size,),-100.)
        logits[97]=100.
        with patch.object(learner,'predict_ids',return_value=(logits,[])):
            for prompt in ('','未知'*50,'a'):
                decisions=list(learner.generate_steps(prompt))
                self.assertEqual(len(decisions),3)
                self.assertTrue(all(len(d['ids'])<=learner.architecture.context for d in decisions))
        logits[TOK.EOS]=200.
        with patch.object(learner,'predict_ids',return_value=(logits,[])):
            self.assertEqual(len(list(learner.generate_steps(''))),1)

    def test_temperature_topk_and_crossing_token_top_p(self) -> None:
        logits=torch.full((TOK.size,),-1000.,dtype=torch.float64)
        logits[:3]=torch.log(torch.tensor([.6,.25,.15],dtype=torch.float64))
        p=sampling_probabilities(logits,Sampling(temperature=1,top_k=0,top_p=.8))
        torch.testing.assert_close(p[:3],torch.tensor([.6/.85,.25/.85,0],dtype=torch.float64))
        p=sampling_probabilities(logits,Sampling(temperature=1,top_k=1,top_p=1))
        self.assertEqual(float(p[0]),1)
        base=sampling_probabilities(logits,Sampling(temperature=1,top_k=0,top_p=1))
        cold=sampling_probabilities(logits,Sampling(temperature=.5,top_k=0,top_p=1))
        hot=sampling_probabilities(logits,Sampling(temperature=2,top_k=0,top_p=1))
        self.assertGreater(cold[0],base[0]); self.assertLess(hot[0],base[0])
        tiny=sampling_probabilities(logits,Sampling(temperature=1e-320,top_k=0,top_p=1))
        self.assertEqual(float(tiny[0]),1)
        self.assertEqual(float(base[TOK.BOS]+base[TOK.PAD]),0)

    def test_seed_reproducible_sampling(self) -> None:
        learner=self.learner()
        learner.train_text('abc',progress=lambda _:None)
        learner.sampling.top_k=0
        learner.sampling.top_p=1
        learner.generator.manual_seed(123)
        one=[d['selected'] for d in learner.generate_steps('a')]
        learner.generator.manual_seed(123)
        self.assertEqual(one,[d['selected'] for d in learner.generate_steps('a')])

    def test_save_load_exact_cpu_optimizer_dropout_continuation(self) -> None:
        original=self.learner(dropout=.2,efficient=True)
        original.train_text('First text.',progress=lambda _:None)
        original.sampling.top_k=0
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'transformer.pt'
            save_checkpoint(original,path)
            loaded=load_checkpoint(path,'cpu')
            for learner in (original,loaded): learner.train_text('Same second text.',progress=lambda _:None)
            for name,value in original.net.state_dict().items():
                torch.testing.assert_close(value,loaded.net.state_dict()[name],atol=0,rtol=0)
            self.assertEqual(original.generate('S'),loaded.generate('S'))
            self.assertEqual(original.steps,loaded.steps)
            self.assertEqual(original.targets_optimized,loaded.targets_optimized)
            self.assertTrue(torch.equal(original.cpu_rng,loaded.cpu_rng))
            for a,b in zip(original.optimizer.state.values(),loaded.optimizer.state.values()):
                for key in a: torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

    def test_atomic_replacement_retains_only_one_file(self) -> None:
        learner=self.learner()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            path.write_bytes(b'legacy file must survive')
            backup=save_checkpoint(learner,path)
            self.assertIsNone(backup)
            self.assertEqual(list(Path(folder).iterdir()),[path])
            self.assertEqual(load_checkpoint(path,'cpu').steps,0)
            with self.assertRaises(ValueError): save_checkpoint(learner,Path(folder)/'data.txt')

    def test_malformed_checkpoints_and_tokenizer_rejected(self) -> None:
        learner=self.learner()
        learner.train_text('abc',progress=lambda _:None)
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/'good.pt';bad=Path(folder)/'bad.pt'
            save_checkpoint(learner,original)
            modifications=[lambda d:d.update(version=2),
                           lambda d:d['tokenizer'].update(type='bpe'),
                           lambda d:d['architecture'].update(heads=3),
                           lambda d:d['metadata'].update(steps=-1),
                           lambda d:d['model'].update({'positions.weight':torch.zeros(1)}),
                           lambda d:d['optimizer']['state'][0].update(exp_avg=torch.zeros(1)),
                           lambda d:d['rng'].update(cpu=torch.zeros(2))]
            for modify in modifications:
                data=torch.load(original,weights_only=True);modify(data);torch.save(data,bad)
                with self.assertRaises(ValueError): load_checkpoint(bad,'cpu')
            bad.write_bytes(b'not a checkpoint')
            with self.assertRaises(ValueError):load_checkpoint(bad,'cpu')

    def test_evaluation_is_read_only_and_correct_units(self) -> None:
        learner=self.learner(dropout=.2)
        learner.train_text('Äiti',progress=lambda _:None)
        before={k:v.clone() for k,v in learner.net.state_dict().items()}
        moments=cpu_copy(learner.optimizer.state_dict())
        rng=learner.generator.get_state().clone();cpu=learner.cpu_rng.clone()
        global_rng=torch.get_rng_state().clone()
        state=(learner.steps,learner.bytes_read,learner.targets_optimized,list(learner.history),learner.net.training)
        result=learner.evaluate(['Kissa!'])
        self.assertEqual(result['byte_targets'],6)
        self.assertEqual(result['eos_targets'],1)
        self.assertAlmostEqual(result['perplexity_bytes_and_eos'],math.exp(result['nll_bytes_and_eos']))
        self.assertTrue(math.isfinite(result['bits_per_byte_excluding_eos']))
        self.assertIsNone(learner.evaluate([''])['nll_bytes_and_eos'])
        self.assertEqual(state,(learner.steps,learner.bytes_read,learner.targets_optimized,list(learner.history),learner.net.training))
        self.assertTrue(torch.equal(rng,learner.generator.get_state()) and torch.equal(cpu,learner.cpu_rng))
        self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,learner.net.state_dict()[k]) for k,v in before.items()))
        for index,values in moments['state'].items():
            for key,value in values.items(): self.assertTrue(torch.equal(value,learner.optimizer.state_dict()['state'][index][key]))
        # A constant zero-logit model has 257 equally probable targets (bytes+EOS).
        with patch.object(learner.net,'forward') as forward:
            def uniform(x,**kwargs):
                logits=torch.zeros(*x.shape,TOK.size)
                logits[...,TOK.BOS]=logits[...,TOK.PAD]=-float('inf')
                return logits,[]
            forward.side_effect=uniform
            metrics=learner.evaluate(['ä'])
            self.assertAlmostEqual(metrics['nll_bytes_and_eos'],math.log(257),places=5)
            self.assertAlmostEqual(metrics['bits_per_byte_excluding_eos'],math.log2(257),places=5)

    def test_read_chunking_does_not_change_training_and_batches_bounded(self) -> None:
        whole,streamed=self.learner(),self.learner()
        text='Kissa syö kalaa. '*20
        whole.train_text(text,progress=lambda _:None)
        original=streamed.optimize_batch
        sizes=[]
        def checked(windows,progress):
            sizes.append(len(windows))
            self.assertLessEqual(len(windows),streamed.training.batch_size)
            self.assertTrue(all(len(w)<=streamed.architecture.context+1 for w in windows))
            return original(windows,progress)
        with patch.object(streamed,'optimize_batch',side_effect=checked):
            streamed.train_chunks(iter(text),progress=lambda _:None)
        self.assertTrue(sizes)
        self.assertEqual(whole.targets_optimized,streamed.targets_optimized)
        for name,value in whole.net.state_dict().items():torch.testing.assert_close(value,streamed.net.state_dict()[name],atol=0,rtol=0)

    def test_nonfinite_loss_and_failed_optimizer_are_not_success(self) -> None:
        learner=self.learner()
        before={k:v.clone() for k,v in learner.net.state_dict().items()}
        with patch.object(learner.net,'forward',return_value=(torch.full((1,4,TOK.size),float('nan'),requires_grad=True),[])):
            with self.assertRaises(ValueError): learner.train_text('abc',progress=lambda _:None)
        self.assertEqual(learner.steps,0)
        self.assertTrue(all(torch.equal(v,learner.net.state_dict()[k]) for k,v in before.items()))
        with patch.object(learner.optimizer,'step',side_effect=RuntimeError('partial optimizer failure')):
            with self.assertRaises(RuntimeError): learner.train_text('abc',progress=lambda _:None)
        self.assertTrue(learner.needs_restore)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):save_checkpoint(learner,Path(folder)/'bad.pt')

    def test_device_probe_unavailable_falls_back(self) -> None:
        messages=[]
        with patch.object(torch.backends.mps,'is_available',return_value=False):
            self.assertEqual(choose_device('auto',messages.append),'cpu')
            self.assertEqual(choose_device('mps',messages.append),'cpu')
        self.assertTrue(messages)

    def test_unsupported_mps_preupdate_retry_does_not_double_update(self) -> None:
        learner = self.learner()
        windows = [[TOK.BOS, 97, 98, TOK.EOS]]
        cpu_inputs = learner.batch_tensors(windows)
        original_forward = learner.net.forward
        calls = 0
        def unsupported_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise NotImplementedError('operation not implemented for MPS')
            return original_forward(*args, **kwargs)
        # Simulate the failure boundary without requiring actual MPS hardware.
        learner.device = 'mps'
        with patch.object(learner, 'batch_tensors', return_value=cpu_inputs), patch.object(learner.net, 'forward', side_effect=unsupported_once):
            learner.optimize_batch(windows, progress=lambda _: None)
        self.assertEqual(learner.device, 'cpu')
        self.assertEqual(learner.steps, 1)
        self.assertEqual(calls, 2)

    def test_warmup_and_configuration_validation(self) -> None:
        learner=self.learner(warmup_steps=4)
        self.assertAlmostEqual(learner.learning_rate(),.0025)
        learner.train_text('abc',progress=lambda _:None)
        self.assertAlmostEqual(learner.learning_rate(),.005)
        for config in (Architecture(embedding=17,heads=4),Architecture(dropout=1),Architecture(context=0)):
            with self.assertRaises(ValueError):config.validate()
        for config in (Sampling(temperature=0),Sampling(top_p=0),Sampling(top_p=1.1),Sampling(top_k=-1),Sampling(seed=-1)):
            with self.assertRaises(ValueError):config.validate()
        for config in (Training(batch_size=0),Training(learning_rate=float('nan')),Training(max_steps=-1)):
            with self.assertRaises(ValueError):config.validate()

    def test_attention_and_training_visualizations(self) -> None:
        learner=self.learner()
        learner.train_text('ab',progress=lambda _:None)
        ids=learner.context_ids('ä')
        _,matrices=learner.predict_ids(ids,inspect=True)
        with tempfile.TemporaryDirectory() as folder:
            attention=Path(folder)/'attention.svg';training=Path(folder)/'training.svg'
            attention_svg(matrices[0][0],ids,0,0,attention)
            training_svg(learner,[{'step':1,'nll':2.0}],training)
            import xml.etree.ElementTree as ET
            ET.parse(attention); ET.parse(training)
            self.assertIn('query 0, key 1: 0.00000000',attention.read_text(encoding='utf-8'))
            self.assertIn('Perplexity',training.read_text(encoding='utf-8'))


class ImportAndCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def setUp(self) -> None:
        consent=patch('builtins.input',return_value='RESET');consent.start();self.addCleanup(consent.stop)
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.app=Application(Path(folder.name))
        self.app.architecture=Architecture(embedding=16,heads=2,layers=1,context=16,feedforward=32,dropout=0)
        self.app.training=Training(device='cpu',batch_size=2,warmup_steps=0)
        self.source=self.app.data_dir/'example.txt'
        self.source.write_text('Kissa syö.',encoding='utf-8')

    def run_import(self, **kwargs) -> dict:
        return self.app.import_files(progress=lambda _:None, **kwargs)

    def test_folder_content_duplicates_modified_alias_and_force(self) -> None:
        copy=self.app.data_dir/'copy.txt';copy.write_bytes(self.source.read_bytes())
        (self.app.data_dir/'ignore.md').write_text('ignored')
        result=self.run_import()
        self.assertEqual((result['found'],result['complete'],result['skipped']),(2,1,1))
        self.assertEqual(self.run_import()['skipped'],2)
        self.source.write_text('New contents.',encoding='utf-8')
        messages=[]
        result=self.app.import_files(progress=messages.append)
        self.assertEqual(result['complete'],1)
        self.assertTrue(any('Modified file' in text for text in messages))
        self.assertEqual(self.run_import(force=True)['complete'],2)

    def test_recursive_filter_and_independent_document_eos(self) -> None:
        nested=self.app.data_dir/'nested';nested.mkdir()
        (nested/'second.TXT').write_text('Other document.')
        result=self.run_import();self.assertEqual(result['found'],1)
        result=self.run_import(recursive=True);self.assertEqual(result['complete'],1)
        learner=self.app.learner
        expected=len(self.source.read_text(encoding='utf-8').encode('utf-8'))+len('Other document.'.encode())+2
        self.assertEqual(learner.targets_optimized,expected)
        self.assertEqual(learner.documents_completed,2)

    def test_empty_files_and_invalid_utf8(self) -> None:
        self.source.write_bytes(b'')
        self.assertEqual(self.run_import()['complete'],1)
        self.assertEqual(self.app.learner.steps,0)
        self.source.write_bytes(b'\xff')
        self.assertEqual(self.run_import()['incomplete'],1)
        self.assertEqual(self.app.learner.steps,0)

    def test_partial_failure_requires_explicit_recovery(self) -> None:
        learner=self.app.ensure_model()
        def broken(*args,**kwargs):
            yield 'a'*80
            raise OSError('simulated stream failure')
        with patch.dict(globals(),{'text_chunks':broken}):
            result=self.run_import()
        self.assertEqual(result['incomplete'],1)
        self.assertGreater(learner.steps,0)
        before=learner.steps
        self.assertEqual(self.run_import()['incomplete'],1)
        self.assertEqual(learner.steps,before)
        self.assertEqual(self.run_import(force=True)['complete'],1)
        self.assertEqual(learner.imports[content_hash(self.source)]['status'],'complete')

    def test_step_budget_never_claims_full_import(self) -> None:
        self.app.training.max_steps=1
        self.source.write_text('abcdef'*50)
        result=self.run_import()
        self.assertEqual(result['complete'],0)
        self.assertEqual(result['incomplete'],1)
        self.assertEqual(self.app.learner.steps,1)
        self.assertEqual(self.app.learner.documents_completed,0)
        self.assertEqual(self.app.learner.imports[content_hash(self.source)]['status'],'incomplete')

    def test_history_restores_only_with_actual_checkpoint(self) -> None:
        self.run_import()
        path=self.app.models_dir/'model.pt';save_checkpoint(self.app.learner,path)
        fresh=Application(self.app.root)
        fresh.architecture=self.app.architecture;fresh.training=self.app.training
        self.assertEqual(fresh.import_files(progress=lambda _:None)['complete'],1)
        fresh.use_model(load_checkpoint(path,'cpu'))
        self.assertEqual(fresh.import_files(progress=lambda _:None)['skipped'],1)

    def test_import_commits_only_canonical_checkpoint(self):
        with redirect_stdout(io.StringIO()): self.run_import()
        self.assertEqual([p.name for p in self.app.models_dir.iterdir()],['transformer.pt'])
        self.assertGreater(load_checkpoint(self.app.models_dir/'transformer.pt').steps,0)

    def test_file_changed_mid_import_remains_incomplete(self) -> None:
        learner=self.app.ensure_model();original=learner.train_chunks
        def changed(chunks,progress):
            result=original(chunks,progress)
            self.source.write_text('Changed file')
            return result
        with patch.object(learner,'train_chunks',side_effect=changed):self.assertEqual(self.run_import()['incomplete'],1)
        self.assertEqual(next(iter(learner.imports.values()))['status'],'incomplete')

    def test_existing_data_and_legacy_history_untouched(self) -> None:
        old=self.app.models_dir/'import_history.json';old.write_text('{"old":"journal"}')
        checkpoint=self.app.models_dir/'legacy.pt';checkpoint.write_bytes(b'old checkpoint')
        original=self.source.read_bytes()
        self.run_import()
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual(old.read_text(),'{"old":"journal"}')
        self.assertEqual(checkpoint.read_bytes(),b'old checkpoint')

    def test_commands_sampling_inspection_save_load_and_no_model_selection(self) -> None:
        output=io.StringIO()
        with redirect_stdout(output):
            for command in ('/import','/architecture','/stats','/learningrate .001','/batchsize 1','/steps 0','/epochs 1',
                            '/temperature .7','/topk 5','/topp .9','/max 4','/seed 10','/debug on','/predict Kissa',
                            '/attention --layer 0 --head 1 Kissa','/plot','/save','/load','/benchmark Kissa'):
                self.app.command(command,'training')
            self.assertEqual(self.app.command('/done','training'),'chat')
            self.assertEqual(self.app.command('/train --legacy','chat'),'training')
        self.assertIn('Restored weights, AdamW',output.getvalue())
        self.assertTrue((self.app.root/'outputs'/'attention-layer0-head1.svg').exists())
        self.assertEqual(self.app.sampling.top_p,.9)
        self.assertFalse(self.app.sampling.auto_train)
        with self.assertRaises(ValueError):self.app.command('/model ngram','chat')
        with self.assertRaises(ValueError):self.app.command('/config context 100','training')

    def test_bad_load_leaves_active_state_intact(self) -> None:
        learner=self.app.ensure_model()
        bad=self.app.models_dir/'bad.pt';bad.write_bytes(b'wrong format')
        with self.assertRaises(ValueError):self.app.command(f'/load {bad}','training')
        self.assertIs(self.app.learner,learner)

    def test_explicit_new_preserves_old_checkpoint(self) -> None:
        self.run_import()
        identity=self.app.learner.identity
        with redirect_stdout(io.StringIO()):self.app.command('/new tiny','training')
        self.assertIsNone(self.app.learner)
        self.assertEqual(load_checkpoint(self.app.models_dir/'transformer.pt','cpu').identity,identity)
        self.assertFalse(list(self.app.models_dir.glob('recovery-*.pt')))


class StartupTests(unittest.TestCase):
    """Exercise real disk checkpoints and real updates without user datasets."""
    def setUp(self) -> None:
        consent=patch('builtins.input',return_value='RESET');consent.start();self.addCleanup(consent.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.app = Application(self.root)
        self.app.architecture = Architecture(embedding=16, heads=2, layers=1,
                                             context=16, feedforward=32, dropout=0)
        self.app.training = Training(device='cpu', batch_size=2, warmup_steps=0)
        self.app.sampling.max_tokens = 2

    def trained(self) -> Learner:
        learner = self.app.ensure_model()
        learner.train_text('Hei kissa.', progress=lambda _: None)
        return learner

    def startup(self, app: Application | None = None, answers: tuple = (), **kwargs) -> str:
        with redirect_stdout(io.StringIO()), patch('builtins.input', side_effect=answers):
            return (app or self.app).legacy_startup(**kwargs)

    def test_canonical_checkpoint_ignores_newer_unrelated_files(self) -> None:
        learner = self.trained()
        old = self.app.models_dir/'old.pt'
        save_checkpoint(learner, old)
        learner.sampling.temperature = .43
        learner.train_text('More text', progress=lambda _: None)
        latest = self.app.models_dir/'transformer.pt'
        save_checkpoint(learner, latest)
        invalid = self.app.models_dir/'broken.pt'; invalid.write_bytes(b'broken')
        empty = Learner(self.app.architecture, self.app.training, self.app.sampling)
        empty_path = self.app.models_dir/'empty.pt'; save_checkpoint(empty, empty_path)
        for index, path in enumerate((old, latest, invalid, empty_path), 1):
            os.utime(path, (index, index))
        app = Application(self.root)
        self.assertEqual(self.startup(app), 'chat')
        self.assertEqual(app.learner.last_checkpoint, str(latest.resolve()))
        self.assertEqual(app.learner.steps, learner.steps)
        self.assertEqual(app.sampling.temperature, .43)
        self.assertEqual(app.training.device, 'cpu')
        self.assertEqual(app.architecture, learner.architecture)
        self.assertEqual(invalid.read_bytes(), b'broken')

    def test_enter_trains_and_restart_does_not_prompt_or_duplicate(self) -> None:
        source = self.app.data_dir/'finnish.txt'; source.write_text('Hyvää yötä!',encoding='utf-8')
        self.assertEqual(self.startup(answers=('',)), 'chat')
        steps = self.app.learner.steps
        self.assertGreater(steps, 0)
        self.assertFalse(list(self.app.models_dir.glob('recovery-*.pt')))
        app = Application(self.root)
        self.assertEqual(self.startup(app), 'chat')
        self.assertEqual(app.learner.steps, steps)
        self.assertEqual(app.learner.imports[content_hash(source)]['status'], 'complete')
        self.assertEqual(app.fingerprint(), app.saved_fingerprint)

    def test_skip_and_empty_install_do_not_train(self) -> None:
        source = self.app.data_dir/'sample.txt'; source.write_text('abc')
        self.assertEqual(self.startup(answers=('S',)), 'training')
        self.assertEqual(self.app.learner.steps, 0)
        self.assertEqual(self.app.learner.imports, {})
        source.unlink()
        self.assertEqual(self.startup(fresh=True), 'training')
        self.assertEqual(self.app.learner.steps, 0)
        self.app.command('/preset tiny', 'training')
        self.assertIsNone(self.app.learner)

    def test_modified_alias_and_partial_detection(self) -> None:
        source = self.app.data_dir/'a.txt'; source.write_text('abc')
        self.app.import_files(progress=lambda _: None)
        alias = self.app.data_dir/'alias.txt'; alias.write_text('abc')
        source.write_text('changed')
        with redirect_stdout(io.StringIO()):
            pending, partial, duplicate = self.app.pending_files()
        self.assertEqual((pending, partial, duplicate), ([source], 0, 1))
        learner = self.app.learner
        learner.imports[content_hash(source)] = dict(status='incomplete', paths=[str(source.resolve())],
                                                   bytes_read=1, steps=0, attempts=1)
        self.app.autosave(lambda _: None)
        app = Application(self.root)
        self.assertEqual(self.startup(app), 'chat')  # No input() call is permitted.
        self.assertEqual(app.pending_files(), ([], 1, 1))

    def test_partial_import_is_saved_and_not_replayed(self) -> None:
        source = self.app.data_dir/'long.txt'; source.write_text('abc ' * 100)
        self.app.training.max_steps = 1
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.startup(answers=('',)), 'chat')
        app = Application(self.root)
        self.assertEqual(self.startup(app), 'chat')
        self.assertEqual(app.learner.steps, 1)
        self.assertEqual(app.learner.imports[content_hash(source)]['status'], 'incomplete')

    def test_save_failure_retains_dirty_state_and_last_good_file(self) -> None:
        self.trained(); self.app.autosave(lambda _: None)
        original = (self.app.models_dir/'transformer.pt').read_bytes()
        self.app.command('/temperature 0.5', 'chat')
        with patch(__name__+'.save_checkpoint', side_effect=OSError('disk full')):
            self.assertFalse(self.app.autosave(lambda _: None))
        self.assertNotEqual(self.app.fingerprint(), self.app.saved_fingerprint)
        self.assertEqual((self.app.models_dir/'transformer.pt').read_bytes(), original)
        self.app.learner.needs_restore = True
        self.assertFalse(self.app.autosave(lambda _: None))
        self.assertEqual((self.app.models_dir/'transformer.pt').read_bytes(), original)

    def test_chat_quit_eof_and_interrupt_save_settings(self) -> None:
        self.trained(); self.app.autosave(lambda _: None)
        for end in ('/quit', EOFError(), KeyboardInterrupt()):
            output = io.StringIO()
            with redirect_stdout(output), patch('builtins.input', side_effect=['/temperature 0.6', 'Hei', end]):
                interactive(root=self.root)
            loaded = load_checkpoint(self.app.models_dir/'transformer.pt')
            self.assertEqual(loaded.sampling.temperature, .6)
            self.assertIn('CHAT WITH YOUR AI', output.getvalue())
            self.assertIn('AI  >', output.getvalue())
            self.assertIn('Automatically saved', output.getvalue())
            self.assertEqual(loaded.steps, self.app.learner.steps)

    def test_initial_manual_training_opens_chat_and_saves(self) -> None:
        # Patch only application construction to keep this end-to-end test tiny.
        with redirect_stdout(io.StringIO()) as output, patch(__name__+'.Application', return_value=self.app), \
                patch('builtins.input', side_effect=['/train --legacy', 'Hyvää päivää.', '/done', 'Hei', '/quit']):
            interactive(root=self.root, fresh=True)
        self.assertIn('CHAT WITH YOUR AI', output.getvalue())
        loaded = load_checkpoint(self.app.models_dir/'transformer.pt')
        self.assertGreater(loaded.steps, 0)
        self.assertEqual(loaded.documents_completed, 1)

    def test_clean_exit_avoids_redundant_checkpoint(self) -> None:
        self.trained(); self.app.autosave(lambda _: None)
        path = self.app.models_dir/'transformer.pt'
        stamp = path.stat().st_mtime_ns
        with redirect_stdout(io.StringIO()), patch('builtins.input', return_value='/quit'):
            interactive(root=self.root)
        self.assertEqual(path.stat().st_mtime_ns, stamp)


class DatasetChangeTests(unittest.TestCase):
    setUp = StartupTests.setUp
    startup = StartupTests.startup

    def teach_file(self, name: str = 'old.txt', text: str = 'Old Finnish kissa.') -> Path:
        path = self.app.data_dir/name
        path.write_text(text, encoding='utf-8')
        self.app.import_files(path, progress=lambda _: None)
        return path

    def test_new_only_continues_same_identity(self) -> None:
        self.teach_file()
        old = self.app.learner
        identity, steps = old.identity, old.steps
        new = self.app.data_dir/'new.txt'; new.write_text('New koira.')
        status = self.app.dataset_status()
        self.assertEqual(status['added'], [str(new.resolve())])
        self.assertFalse(status['stale'])
        self.assertEqual(self.startup(answers=('',)), 'chat')
        self.assertEqual(self.app.learner.identity, identity)
        self.assertGreater(self.app.learner.steps, steps)

    def test_remove_modify_replace_and_unchanged(self) -> None:
        first = self.teach_file('one.txt', 'one')
        second = self.teach_file('two.txt', 'two')
        third = self.teach_file('three.txt', 'three')
        first.unlink(); second.write_text('replacement')
        new = self.app.data_dir/'new.txt'; new.write_text('new')
        status = self.app.dataset_status()
        for key, paths in [('removed',[first]), ('modified',[second]), ('added',[new]), ('unchanged',[third])]:
            self.assertEqual(status[key], [str(path.resolve()) for path in paths])
        self.assertTrue(status['stale'])
        second.unlink(); third.unlink()
        status = self.app.dataset_status()
        self.assertEqual(len(status['removed']), 3)
        self.assertEqual(len(status['added']), 1)

    def test_identical_renamed_content_needs_no_prompt(self) -> None:
        original = self.teach_file()
        duplicate = self.app.data_dir/'copy.txt'; duplicate.write_bytes(original.read_bytes())
        status = self.app.dataset_status()
        self.assertEqual(len(status['duplicates']), 1)
        self.assertEqual(status['pending'], [])
        original.unlink()
        self.assertFalse(self.app.dataset_status()['stale'])
        self.assertEqual(self.startup(), 'chat')
        self.assertEqual(len(self.app.learner.imports), 1)

    def test_default_rebuild_resets_weights_optimizer_and_history(self) -> None:
        original = self.teach_file()
        old = self.app.learner
        old_weights = {k: v.clone() for k,v in old.net.state_dict().items()}
        original.unlink()
        replacement = self.app.data_dir/'new.txt'; replacement.write_text('Uusi koira.')
        original_train = Learner.train_chunks
        inspected = []
        def inspect_new(learner, chunks, progress=print):
            self.assertNotEqual(learner.identity, old.identity)
            self.assertEqual(learner.steps, 0)
            self.assertEqual(learner.documents_completed, 0)
            self.assertEqual(len(learner.optimizer.state), 0)
            self.assertEqual(learner.dataset, {})
            self.assertNotIn(content_hash(replacement), old.imports)
            self.assertFalse(torch.equal(learner.net.tokens.weight, old_weights['tokens.weight']))
            inspected.append(True)
            return original_train(learner, chunks, progress)
        with patch.object(Learner, 'train_chunks', new=inspect_new):
            self.assertEqual(self.startup(answers=('', 'RESET')), 'chat')
        self.assertTrue(inspected)
        active = load_checkpoint(self.app.models_dir/'transformer.pt')
        self.assertNotEqual(active.identity, old.identity)
        self.assertEqual(set(active.imports), {content_hash(replacement)})
        self.assertEqual(active.dataset, {str(replacement.resolve()):content_hash(replacement)})
        self.assertFalse(list(self.app.models_dir.glob('recovery-*.pt')))
        self.assertTrue(all(torch.equal(value,old.net.state_dict()[key]) for key,value in old_weights.items()))
        self.assertEqual(self.startup(Application(self.root)), 'chat')  # No menu after a clean rebuild.

    def test_continue_warns_and_skip_does_not_train(self) -> None:
        source = self.teach_file()
        old_identity = self.app.learner.identity
        source.write_text('Replacement Finnish.')
        output = io.StringIO()
        with redirect_stdout(output), patch('builtins.input', return_value='2'):
            self.assertEqual(self.app.legacy_startup(), 'chat')
        self.assertIn('Nothing has been forgotten', output.getvalue())
        self.assertEqual(self.app.learner.identity, old_identity)
        self.assertEqual(len(self.app.learner.imports), 2)
        source.unlink()
        steps = self.app.learner.steps
        self.assertEqual(self.startup(answers=('3',)), 'chat')
        self.assertEqual(self.app.learner.steps, steps)

    def test_failed_or_interrupted_rebuild_preserves_committed_checkpoint(self) -> None:
        source = self.teach_file()
        source.write_text('Replacement')
        old = self.app.learner
        canonical = self.app.models_dir/'transformer.pt'
        before = canonical.read_bytes()
        for failure in (RuntimeError('training failed'), KeyboardInterrupt()):
            with redirect_stdout(io.StringIO()), patch.object(Learner, 'train_chunks', side_effect=failure):
                self.assertFalse(self.app.rebuild())
            self.assertIs(self.app.learner, old)
            self.assertEqual(canonical.read_bytes(), before)
            self.assertEqual(load_checkpoint(canonical).identity, old.identity)
            self.assertFalse(self.app.rebuilding)

    def test_final_save_failure_rolls_back_rebuild(self) -> None:
        source = self.teach_file(); source.write_text('New version')
        old = self.app.learner
        canonical = self.app.models_dir/'transformer.pt'; before = canonical.read_bytes()
        real_save = save_checkpoint
        def fail_new(learner, path):
            if learner.identity != old.identity:
                raise OSError('disk full')
            return real_save(learner, path)
        with redirect_stdout(io.StringIO()), patch(__name__+'.save_checkpoint', side_effect=fail_new):
            self.assertFalse(self.app.rebuild())
        self.assertIs(self.app.learner, old)
        self.assertEqual(canonical.read_bytes(), before)

    def test_manifest_roundtrip_legacy_and_forced_partial_reimport(self) -> None:
        source = self.teach_file()
        canonical = self.app.models_dir/'transformer.pt'
        self.assertEqual(load_checkpoint(canonical).dataset, self.app.learner.dataset)
        data = torch.load(canonical, weights_only=True)
        del data['metadata']['dataset']
        legacy = self.app.models_dir/'legacy.pt'; torch.save(data, legacy)
        self.assertEqual(load_checkpoint(legacy).dataset, self.app.learner.dataset)
        # A failed forced replay does not invalidate an earlier completed manifest.
        with patch.object(self.app.learner, 'train_chunks', side_effect=RuntimeError('failed')):
            self.app.import_files(source, force=True, progress=lambda _: None)
        loaded = load_checkpoint(canonical)
        self.assertEqual(loaded.dataset, self.app.learner.dataset)
        self.assertEqual(loaded.imports[content_hash(source)]['status'], 'incomplete')

    def test_data_command_no_training_and_rebuild_command(self) -> None:
        source = self.teach_file()
        steps = self.app.learner.steps
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.app.command('/data', 'chat'), 'chat')
        self.assertIn(str(source.resolve()), output.getvalue())
        self.assertEqual(self.app.learner.steps, steps)
        identity = self.app.learner.identity
        source.write_text('<|endoftext|>'.join(f'English story {i} about a cat.' for i in range(40)))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.app.command('/train --rebuild', 'chat'), 'chat')
        self.assertNotEqual(self.app.learner.identity, identity)

    def test_empty_folder_rebuild_keeps_previous_model(self) -> None:
        source = self.teach_file(); source.unlink()
        identity = self.app.learner.identity
        self.assertEqual(self.startup(answers=('',)), 'chat')
        self.assertEqual(self.app.learner.identity, identity)


class TrainingProgressTests(unittest.TestCase):
    learner = TransformerTests.learner

    def test_exact_totals_utf8_bom_final_batch_and_epochs(self) -> None:
        learner = self.learner(epochs=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ää.txt'
            for text in ('', 'ä', 'a'*15, 'a'*16, 'a'*64, 'Hei!\r\nÄiti'):
                for bom in (b'', b'\xef\xbb\xbf'):
                    path.write_bytes(bom + text.encode('utf-8'))
                    display = TrainingProgress.for_file(path, learner, lambda _: None)
                    actual_batches = list(batches(document_windows(text_chunks(path, 1),
                        learner.architecture.context), learner.training.batch_size))
                    self.assertEqual(display.total, len(actual_batches)*3)
                    before = learner.steps
                    learner.train_chunks(text_chunks(path, 1), display)
                    self.assertEqual(display.completed, learner.steps-before)
                    self.assertEqual(display.completed, display.total)
                    self.assertTrue(display.closed)

    def test_live_line_refresh_throttling_and_narrow_console(self) -> None:
        stream, messages, now = io.StringIO(), [], [0.0]
        with patch('shutil.get_terminal_size', return_value=os.terminal_size((80,24))):
            display = TrainingProgress(messages.append, stream=stream, live=True,
                total=4, name='training.txt', epochs=2, clock=lambda:now[0], dataset_position=(1,3))
            first_length = len(stream.getvalue())
            display.update(loss=2.5,rate=.0003,batch=1,epoch=1)
            self.assertEqual(len(stream.getvalue()), first_length)  # throttled
            now[0]=2
            display.update(loss=2,rate=.0003,batch=1,epoch=2)
            self.assertIn('50%', display.line())
            self.assertIn('ETA 00:00:02', display.line())
            rendered = stream.getvalue().split('\r')[-1]
            for label in ('D0/3', 'E2/2', 'L2.00', 'lr', 'ETA'):
                self.assertIn(label, rendered)
            self.assertLessEqual(len(rendered), 79)
            display.finish('Stopped',average=2.25,targets=10)
        self.assertNotIn('\x1b', stream.getvalue())
        self.assertNotIn('\n', stream.getvalue())  # callback handles only summaries
        self.assertIn('Steps: 2/4', '\n'.join(messages))
        self.assertTrue(display.closed)

    def test_unknown_and_changed_lengths_have_no_file_percentage_or_eta(self) -> None:
        display = TrainingProgress(lambda _:None, live=False)
        display.update(loss=1,rate=.01,batch=1,epoch=1)
        self.assertIn('1/? steps | ETA ?', display.line())
        display.total = 1
        display.update(loss=1,rate=.01,batch=2,epoch=1)
        self.assertIsNone(display.total)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.txt'; path.write_text('abc')
            file_display=TrainingProgress.for_file(path,self.learner(),lambda _:None)
            path.write_text('changed contents')
            file_display.update(loss=1,rate=.01,batch=1,epoch=1)
            self.assertIsNone(file_display.total)

    def test_step_budget_and_interruption_do_not_claim_completion(self) -> None:
        learner = self.learner(epochs=3, max_steps=1)
        messages=[]
        display = TrainingProgress(messages.append,total=3,epochs=3)
        with self.assertRaises(TrainingLimit):
            learner.train_chunks(['abc'],display)
        self.assertEqual(display.completed,1)
        self.assertEqual(display.total,3)
        self.assertTrue(display.closed)
        self.assertIn('Stopped', '\n'.join(messages))
        self.assertNotIn('Training finished', '\n'.join(messages))
        other=self.learner()
        display=TrainingProgress(messages.append,total=2)
        with patch.object(other,'optimize_batch',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                other.train_chunks(['abc'],display)
        self.assertEqual(display.completed,0)
        self.assertTrue(display.closed)

    def test_debug_logs_do_not_change_weights_or_sampling(self) -> None:
        quiet, detailed = self.learner(epochs=2), self.learner(epochs=2)
        detailed.sampling.debug = True
        quiet_messages, detailed_messages = [], []
        quiet.train_text('Kissa syö kalaa.',quiet_messages.append)
        detailed.train_text('Kissa syö kalaa.',detailed_messages.append)
        self.assertFalse(any('step=' in line for line in quiet_messages))
        self.assertEqual(sum('step=' in line for line in detailed_messages),detailed.steps)
        for name, value in quiet.net.state_dict().items():
            torch.testing.assert_close(value,detailed.net.state_dict()[name],atol=0,rtol=0)
        self.assertEqual(quiet.generate('Kissa'),detailed.generate('Kissa'))

    def test_pycharm_live_and_redirected_summary(self) -> None:
        stream=io.StringIO()
        with patch.dict(os.environ, {'PYCHARM_HOSTED':'1'}), redirect_stdout(stream):
            display=TrainingProgress(total=1)
            self.assertTrue(display.live)
            display.update(loss=1,rate=.001,batch=1,epoch=1)
            display.finish('Training finished',average=1,targets=3)
        self.assertIn('\r',stream.getvalue())
        redirected=io.StringIO()
        with patch.dict(os.environ, {'PYCHARM_HOSTED':'0'}), redirect_stdout(redirected):
            display=TrainingProgress(total=1)
            self.assertFalse(display.live)
            display.update(loss=1,rate=.001,batch=1,epoch=1)
            display.finish('Training finished',average=1,targets=3)
        self.assertNotIn('\r',redirected.getvalue())
        self.assertIn('Training finished',redirected.getvalue())

    def test_multi_file_multi_epoch_and_measured_validation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app=Application(Path(folder))
            app.use_model(self.learner(epochs=2))
            app.validation.append({'step':0,'nll':4.2})
            for name in ('one.txt','two.txt'):
                (app.data_dir/name).write_text('Hyvää päivää! ' + name,encoding='utf-8')
            events=[]; original=TrainingProgress.update
            def observe(display, **kwargs):
                original(display,**kwargs)
                events.append((display.name,display.dataset_position,display.epoch,display.line()))
            with patch.object(TrainingProgress,'update',new=observe):
                result=app.import_files(progress=lambda _:None)
            self.assertEqual(result['complete'],2)
            self.assertEqual({event[1] for event in events},{(1,2),(2,2)})
            self.assertEqual({event[2] for event in events},{1,2})
            self.assertTrue(all('val 4.200@0' in event[3] for event in events))
            self.assertEqual(app.validation[-1],{'step':0,'nll':4.2})


class CorpusEngineTests(unittest.TestCase):
    def test_coverage_split_duplicates_shuffle_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stories.txt'
            stories=[f'Story number {i}: a small cat went home.' for i in range(40)]
            path.write_text('<|endoftext|>'.join(stories+stories))
            corpus=CorpusWindows([path],16)
            try:
                self.assertEqual(corpus.db.execute('SELECT count(*) FROM stories').fetchone()[0],40)
                ids=lambda split,epoch:list(corpus.windows(split,epoch))
                first=ids('train',0); second=ids('train',1)
                self.assertEqual(sorted(first),sorted(second))
                self.assertNotEqual(first,second)
                self.assertEqual(first,ids('train',0))
                learner=TransformerTests().learner()
                before={k:v.clone() for k,v in learner.net.state_dict().items()}
                self.assertTrue(math.isfinite(corpus_loss(learner,corpus,'validation')))
                self.assertEqual(learner.steps,0)
                self.assertTrue(all(torch.equal(v,learner.net.state_dict()[k]) for k,v in before.items()))
                with patch.object(learner,'optimize_batch',side_effect=KeyboardInterrupt()):
                    with self.assertRaises(KeyboardInterrupt):
                        train_corpus(learner,corpus,Path(folder)/'new.pt',progress=lambda _:None)
                self.assertFalse((Path(folder)/'new.pt').exists())
                self.assertTrue(all(torch.equal(v,learner.net.state_dict()[k]) for k,v in before.items()))
            finally: corpus.close()


class MilestoneTests(unittest.TestCase):
    def test_provenance_best_restore_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'stories.txt'
            source.write_text('<|endoftext|>'.join(f'Child {i} played with a ball.' for i in range(40)))
            corpus=CorpusWindows([source],16)
            try:
                learner=TransformerTests().learner(epochs=3)
                with patch(__name__+'.corpus_loss',side_effect=[4.,3.999]):
                    chosen,records=train_corpus(learner,corpus,root/'best.pt',patience=1,min_delta=.1,progress=lambda _:None)
                self.assertEqual(len(records),2)
                self.assertEqual(chosen.steps,records[-1]['steps'])
                self.assertEqual(chosen.corpus_state['specification'],corpus.specification)
                self.assertEqual(load_checkpoint(root/'best.pt').corpus_state,chosen.corpus_state)
                with self.assertRaises(ValueError): chosen.train_text('held out story')
                chosen.training.epochs=1
                resumed,_=train_corpus(chosen,corpus,root/'resumed.pt',progress=lambda _:None)
                self.assertEqual(resumed.corpus_state['epochs_completed'],3)
                untracked=TransformerTests().learner();untracked.train_text('already learned',lambda _:None)
                with self.assertRaises(ValueError):train_corpus(untracked,corpus,root/'unsafe.pt',progress=lambda _:None)
                self.assertFalse((root/'unsafe.pt').exists())
            finally:corpus.close()

    def test_rollback_preserves_checkpoint_pointer_and_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'stories.txt'
            source.write_text('<|endoftext|>'.join(f'Story {i}.' for i in range(40)))
            corpus=CorpusWindows([source],16)
            try:
                learner=TransformerTests().learner()
                save_checkpoint(learner,root/'original.pt');pointer=learner.last_checkpoint
                original={k:v.clone() for k,v in learner.net.state_dict().items()}
                with patch.object(learner,'optimize_batch',side_effect=KeyboardInterrupt()):
                    with self.assertRaises(KeyboardInterrupt):train_corpus(learner,corpus,root/'new.pt',progress=lambda _:None)
                self.assertEqual(learner.last_checkpoint,pointer)
                self.assertEqual(learner.steps,0)
                self.assertTrue(all(torch.equal(v,learner.net.state_dict()[k]) for k,v in original.items()))
            finally:corpus.close()


class CleanupTests(unittest.TestCase):
    setUp = StartupTests.setUp

    def test_failed_atomic_save_preserves_file_and_cleans_temporary(self):
        learner=self.app.ensure_model()
        path=self.app.models_dir/'transformer.pt'
        save_checkpoint(learner,path);original=path.read_bytes()
        learner.train_text('Temporary test only.',lambda _:None)
        with patch.object(Path,'replace',side_effect=OSError('simulated replace failure')):
            with self.assertRaises(OSError):save_checkpoint(learner,path)
        self.assertEqual(path.read_bytes(),original)
        self.assertEqual(list(self.app.models_dir.iterdir()),[path])
        for _ in range(4):save_checkpoint(learner,path)
        self.assertEqual(list(self.app.models_dir.iterdir()),[path])

    def test_reset_cancel_and_corrupt_startup_fail_closed(self):
        learner=self.app.ensure_model();learner.train_text('Keep my learned words.',lambda _:None)
        self.app.autosave(lambda _:None)
        path=self.app.models_dir/'transformer.pt';original=path.read_bytes()
        with patch('builtins.input',return_value=''),redirect_stdout(io.StringIO()):
            self.assertEqual(self.app.command('/new tiny','chat'),'chat')
        self.assertIs(self.app.learner,learner)
        self.assertEqual(path.read_bytes(),original)
        path.write_bytes(b'invalid checkpoint')
        fresh=Application(self.app.root)
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):fresh.startup()
        self.assertIsNone(fresh.learner)
        self.assertEqual(path.read_bytes(),b'invalid checkpoint')

    def test_corpus_ui_retains_only_canonical_checkpoint(self):
        source=self.app.data_dir/'stories.txt'
        source.write_text('<|endoftext|>'.join(f'The child {i} played.' for i in range(40)))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.app.command('/corpus 2 0 constant','training'),'chat')
        files=list(self.app.models_dir.iterdir())
        self.assertEqual([p.name for p in files],['transformer.pt'])
        restored=load_checkpoint(files[0],'cpu')
        self.assertTrue(restored.corpus_state['run_history'])
        self.assertEqual(restored.identity,self.app.learner.identity)


class V4IntegrationTests(unittest.TestCase):
    setUp=StartupTests.setUp

    def test_complete_workflow_uses_corpus_engine(self):
        source=self.app.data_dir/'english.txt'
        source.write_text('<|endoftext|>'.join(f'English story {i}: a child plays.' for i in range(40)))
        with patch('builtins.input',return_value=''),redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.app.startup(),'chat')
        self.assertIn('Training engine: V4 Corpus Trainer',output.getvalue())
        first=self.app.learner.identity
        self.assertEqual(self.app.learner.corpus_state['engine'],'V4 Corpus Trainer')
        self.assertTrue(self.app.learner.corpus_state['run_history'][0]['validation_loss']>0)
        restart=Application(self.app.root)
        with patch('builtins.input',side_effect=AssertionError('No prompt expected')),redirect_stdout(io.StringIO()):
            self.assertEqual(restart.startup(),'chat')
        added=self.app.data_dir/'more.txt';added.write_text('<|endoftext|>'.join(f'Another story {i}: a dog sleeps.' for i in range(40)))
        with patch('builtins.input',return_value=''),redirect_stdout(io.StringIO()):
            restart.startup()
        self.assertEqual(restart.learner.identity,first)
        self.assertEqual(len(restart.learner.corpus_state['specification']['files']),2)
        source.unlink();added.write_text('<|endoftext|>'.join(f'Replacement story {i}: a bird flies.' for i in range(40)))
        with patch('builtins.input',side_effect=['','RESET']),redirect_stdout(io.StringIO()):
            restart.startup()
        self.assertNotEqual(restart.learner.identity,first)
        self.assertEqual(len(restart.learner.corpus_state['specification']['files']),1)
        self.assertEqual([p.name for p in restart.models_dir.iterdir()],['transformer.pt'])
        restart.learner.sampling.max_tokens=2
        self.assertIsInstance(restart.learner.generate('A child'),str)
        with redirect_stdout(io.StringIO()):
            restart.command('/train --epochs 2','chat')
        self.assertLessEqual(len(restart.learner.corpus_state['run_history']),2)

    def test_unknown_provenance_preserves_weights(self):
        model=self.app.ensure_model();model.train_text('Legacy text',lambda _:None)
        self.app.autosave(lambda _:None)
        path=self.app.models_dir/'transformer.pt';before=path.read_bytes()
        with redirect_stdout(io.StringIO()) as output:
            self.app.command('/train','chat')
        self.assertIn('Unknown training provenance',output.getvalue())
        self.assertEqual(path.read_bytes(),before)


class ReleaseTests(unittest.TestCase):
    """Publication regressions use disposable sources and checkpoints only."""

    def test_chapters_split_before_windows_and_exclude_front_matter(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'novel.txt'
            chapters=[f'CHAPTER {i}\nChapter {i} contains Finnish: kissa syö.\n' for i in range(1,41)]
            path.write_text('CONTENTS AND LICENSE HEADER\n'+''.join(chapters)+'*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLICENSE FOOTER',encoding='utf-8')
            before=path.read_bytes()
            corpus=CorpusWindows([path],16)
            try:
                self.assertEqual(corpus.db.execute('SELECT count(*) FROM stories').fetchone()[0],40)
                self.assertGreater(corpus.count('train'),0)
                self.assertGreater(corpus.count('validation'),0)
                expected={}
                for chapter in chapters:
                    digest=hashlib.sha256(chapter.encode()).hexdigest()
                    split='validation' if int(hashlib.sha256(f'42:{digest}'.encode()).hexdigest(),16)/2**256 < .2 else 'train'
                    expected.setdefault(split,[]).extend(document_windows([chapter],16))
                for split in ('train','validation'):
                    self.assertEqual(list(corpus.windows(split)),expected[split])
                self.assertEqual(path.read_bytes(),before)
            finally: corpus.close()

    def test_delimiters_override_headings_and_large_lines_are_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stories.txt'
            text='CHAPTER I\n'+'x'*150000+'\nCHAPTER II\n<|endoftext|>Another story'
            path.write_text(text,encoding='utf-8')
            parts=list(corpus_document_chunks(path))
            self.assertEqual(sum(p is None for p in parts),2)
            self.assertLessEqual(max(len(p) for p in parts if p is not None),65548)
            self.assertEqual(b''.join(p for p in parts if p is not None),text.replace('<|endoftext|>','').encode())

    def test_progress_distinguishes_validation_from_optimizer_updates(self):
        messages=[]
        display=TrainingProgress(messages.append,name='Validation',phase='validation',total=1)
        display.update(loss=2.,rate=0,batch=1,epoch=1)
        display.finish('Finished',average=2.,targets=8)
        self.assertIn('evaluation batches',messages[-1])
        self.assertNotIn('mean online loss',messages[-1])
        self.assertIn('batches',display.line())

    def test_first_run_skip_and_help_never_train(self):
        with tempfile.TemporaryDirectory() as folder:
            app=Application(Path(folder));(app.data_dir/'demo.txt').write_text('hello')
            with patch('builtins.input',side_effect=['3','2']),patch(__name__+'.train_corpus') as train,redirect_stdout(io.StringIO()) as output:
                self.assertEqual(app.startup(),'chat')
            train.assert_not_called()
            self.assertIn('No trained model found',output.getvalue())
            self.assertIn('demo.txt',output.getvalue())


def run_milestone_experiment(reference=False):
    """Reproducible token-budget-matched comparison; writes only ignored artifacts."""
    import platform
    Path("experiments").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    report={'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','split_seed':42,'results':[],'limitations':'Synthetic templated English stories; not natural-language quality evidence. Same fixed split across model seeds. Baseline local batch repetition vs corpus global epochs.'}
    with tempfile.TemporaryDirectory() as folder:
     p=Path(folder); source=p/'stories.txt'
     stories=[f'One day, {name} found a {item}. The {item} was small. {name} smiled and went home.' for name in ['Tom','Anna','Ben','Lucy','Sam','Rose','Jack','Lily'] for item in ['ball','book','toy','hat']]
     source.write_text('<|endoftext|>'.join(stories));report['dataset_sha256']=content_hash(source)
     corpus=CorpusWindows([source],128 if reference else 32,42)
     config=Architecture() if reference else Architecture(embedding=32,heads=4,layers=1,context=32,feedforward=64,dropout=.1)
     training=Training(batch_size=8,epochs=80,learning_rate=.003,warmup_steps=5,device='cpu')
     report['architecture']=asdict(config);report['training']=asdict(training)
     for seed in ([42] if reference else [12,42,73]):
      for arm in ['baseline','constant','cosine']:
       learner=Learner(config,replace(training),Sampling(seed=seed,max_tokens=100,top_k=10,top_p=.9))
       start=time.perf_counter();curve=[]
       if arm=='baseline':
        best=float('inf'); best_path=p/f'{seed}-baseline.pt'
        with learner.training_randomness():
         for batch in batches(corpus.windows('train'),8):
          for repeat in range(80):
           learner.optimize_batch(batch,lambda _:None)
           if learner.steps%math.ceil(corpus.count("train")/8)==0:
            value=corpus_loss(learner,corpus,'validation');curve.append(dict(steps=learner.steps,validation_loss=value))
            if value<best: best=value;save_checkpoint(learner,best_path)
        actual_steps=learner.steps;actual_tokens=learner.targets_optimized
        learner=load_checkpoint(best_path)
       else:
        learner,curve=train_corpus(learner,corpus,p/f'{seed}-{arm}.pt',patience=80,min_delta=0,schedule=arm,progress=lambda _:None)
        actual_steps=curve[-1]['steps'];actual_tokens=curve[-1]['targets']
       seconds=time.perf_counter()-start
       result=dict(seed=seed,arm=arm,steps=actual_steps,tokens=actual_tokens,selected_step=learner.steps,seconds=seconds,train_loss=corpus_loss(learner,corpus,'train'),validation_loss=corpus_loss(learner,corpus,'validation'),sample=learner.generate('One day, Tom'),curve=curve)
       report['results'].append(result);Path('experiments/reference-comparison.json' if reference else 'experiments/comparison.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in result.items() if k not in {'curve','sample'}},flush=True)
     corpus.close()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',action='store_true',help='Use the unchanged small architecture for the milestone comparison')
    parser.add_argument('--milestone-experiment',action='store_true',help='Run the three-seed, equal-budget training comparison')
    parser.add_argument('--test',action='store_true',help='Run the CPU test suite, including actual tiny-pattern learning')
    parser.add_argument('--fresh',action='store_true',help='Create an independent model without automatically loading checkpoints')
    parser.add_argument('--checkpoint',type=Path,help='Explicit initial Transformer checkpoint')
    arguments=parser.parse_args()
    # Tiny CPU matrices often run slower with excessive worker threads. Limit
    # the default, without claiming a universally optimal setting for all Macs.
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    if arguments.milestone_experiment:
        run_milestone_experiment(reference=arguments.reference)
    elif arguments.test:
        unittest.main(argv=[__file__],verbosity=2)
    else:
        interactive(fresh=arguments.fresh,checkpoint=arguments.checkpoint)
