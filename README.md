# edge-maxxing-newdream-sdxl

This holds the baseline for the SDXL Nvidia GeForce RTX 4090 contest, which can be forked freely and optimized

Some recommendations are as follows:
- Installing dependencies should be done in pyproject.toml, including git dependencies
- Compiled models should be included directly in the repository(rather than compiling during loading), loading time matters far more than file sizes
- Avoid changing `src/main.py`, as that includes mostly protocol logic. Most changes should be in `models` and `src/pipeline.py`

For testing, you need a docker container with pytorch and ubuntu 22.04.
You can download your listed dependencies with `uv` installed with:
```bash
pipx install uv
pipx ensurepath
```
You can then relock with `uv lock`, and then run with `uv run start_inference`
