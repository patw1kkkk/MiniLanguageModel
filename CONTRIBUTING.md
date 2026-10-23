# Contributing

Bug reports, documentation improvements, and focused pull requests are welcome.
Please describe the expected behavior, actual behavior, Python/PyTorch versions,
operating system, and a small public or synthetic reproduction.

1. Create a virtual environment and install `requirements.txt`.
2. Make a focused change; keep educational comments accurate.
3. Run `python main.py --test` (CPU tests, temporary data/checkpoints only).
4. Add a regression test for behavior changes. Tests currently live in `main.py`.
5. Explain validation and any limitations in your pull request.

Do not commit personal text, model weights, credentials, or generated outputs.
Do not run destructive tests against a real user's checkpoint. A small synthetic
fixture is preferable to a large downloaded corpus. Follow the existing simple
single-file design; discuss architectural changes before implementing them.

Training improvements need held-out evidence with recorded seeds and comparable
budgets. Lower training loss alone is not proof of better generation.
