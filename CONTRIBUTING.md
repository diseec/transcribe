# Contributing

Thanks for helping improve whisperx-local.

## Before opening an issue

- Search existing issues and discussions first.
- Reproduce the problem with a small recording or a synthetic test case when possible.
- Never upload private recordings, transcripts, Hugging Face tokens, or model caches.
- Include the operating system, CPU architecture, Python version, backend, and the output of `./cli status` with secrets and personal paths removed.

## Development setup

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3 python3-venv
./cli install
./cli resources
./cli selftest
```

The same repository also supports macOS. Create the virtual environment on the machine
where it will run; do not copy `.venv` between operating systems or CPU architectures.

## Making changes

- Keep changes focused and preserve the canonical raw-transcript contract.
- Add or update tests beside the code they cover.
- Update the README or third-party notices when user-facing behavior, setup, or licensing changes.
- Do not commit recordings, transcripts, model files, `.env` files, `.venv`, or generated output.
- Use the existing standard-library `unittest` suite unless there is a strong reason to add a dependency.

Run the full check before opening a pull request:

```bash
./cli selftest
git diff --check
```

## Pull requests

Describe the user-visible result, the platforms tested, and any model or dependency changes.
Keep unrelated formatting or refactoring out of the same pull request. A maintainer may ask
for a focused regression test or documentation update before merging.
