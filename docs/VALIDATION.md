# v0.1.0 release validation

These are measured mechanics checks, not a claim of conversational quality or
superiority to another training procedure. Run `python main.py --test` to repeat
the isolated CPU regression suite. Test data/checkpoints are temporary.

## Clean installation and real novel training

An isolated release directory had no preexisting environment or checkpoint.
A new Python 3.12.14 virtual environment installed `requirements.txt` successfully
(PyTorch 2.14.1). The documented first-launch workflow detected the official
novel, trained the default 842,880-parameter model, saved it, and opened chat.

| Measurement | Result |
|---|---:|
| Device | CPU, 4 PyTorch threads, Apple Silicon host |
| Model/sampling seed | 42 |
| Split seed | 42 |
| Context / batch size | 128 / 16 |
| Epochs | 1 complete dataset pass |
| Train / validation windows | 2,882 / 565 |
| Optimization steps | 181 |
| Training targets | 367,696 |
| Validation targets | 72,004 |
| Mean online training loss | 3.493989 nats/target |
| Held-out chapter loss | 2.521389 nats/target |
| Measured training + validation run | 14.82 seconds on this host |

Timing is one local measurement, not a promise for other machines. Training loss
is an average during updates, whereas validation is measured after the epoch;
the two are not evaluated at identical weights. Targets include bytes and EOS.

Actual 80-token continuation for `The garden was`, seed 42, temperature 0.8,
top-k 40, top-p 0.95 (line endings rendered normally):

```text
 y.
phe as ay

Is arethe st be an arorer stik ded thay n'e we helldooust athe
```

This deliberately unpolished sample shows the limitations of one-epoch training.
There is no claim of semantic understanding, useful answers, or generalization
to other books. The chapter split avoids overlapping-window leakage, but all
chapters still come from the same novel.

Restart loaded the same model identity and skipped training when data was
unchanged. Adding a temporary text file produced the new-data offer. Cancelling
a rebuild retained the trained checkpoint. Tests additionally cover successful
fresh rebuilds, interrupted training, failed atomic writes, best-state restoration,
and unchanged validation weights/counters.

## Apple MPS

A separate disposable 16-dimensional, one-layer model trained on 30 synthetic
English stories using actual Apple MPS. Seven corpus optimizer updates covered
693 targets; validation covered 107 targets. Training loss was 5.539499 and
validation loss 5.535005. Checkpoint saving/loading and generation also completed
on MPS. This verifies the execution path, not model quality or full-novel speed.
CPU fallback was separately exercised in the restricted test environment and
by regression tests for unavailable/unsupported MPS operations.

## Automated checks

The local CPU suite passes **72 tests**, including causal masking, future-token
leakage, gradients, optimizer updates, tiny-pattern learning, UTF-8, generation,
sampling, full-state continuation, atomic saves, dataset changes, genuine shuffled
epochs, validation separation, early stopping, interruption recovery, and the
new chapter loader and first-launch interface. CI runs the same suite across
Linux, Windows, and macOS with Python 3.11 and 3.12; its badge shows current status.

## Publication safety

The release is a fresh Git history from an explicit file allowlist, independent
of local development objects. Only the official demonstration novel is bundled;
no other training data, trained weights, virtual environments, or local experiment
outputs are included. The original developer checkpoint and datasets were hashed
before and after preparation and remained unchanged. This report contains no
original checkpoint identity, private dataset content, or machine-specific paths.

The exact official demo bytes and license are documented in DATA_LICENSE.md.
The public repository's `models/` contains only `.gitkeep`.
