# Contributing Guide

Thank you for contributing to AgentInfer. Before submitting a change, search the existing issues and pull requests.
For substantial features or API changes, open an issue first to describe the goal, proposed approach, and compatibility
impact.

## Development Environment

AgentInfer requires Python 3.10 or later. Create a virtual environment and install the project and development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pre-commit pytest
pre-commit install
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. To run vLLM integration tests,
use Linux or WSL 2 and install the supported vLLM 0.23.0 release in the same environment as AgentInfer.

## Change Requirements

- Keep each change focused, and add or update tests for behavioral changes.
- Changes under `agentinfer/` must update the authoritative `docs/zh/` content and the corresponding `docs/en/`
 content in the same pull request.
- Place each new document in exactly one category: Tutorial, How-to, Reference, or Explanation. Only Tutorial
 filenames use numeric prefixes.
- Use short, descriptive kebab-case English filenames, and cover only one topic per document.
- Never commit credentials or other sensitive information in code, configuration, test logs, or documentation.

## Local Validation

Run tests directly related to the change first, then run the same checks as CI:

```bash
python -m pytest tests -q
pre-commit run --all-files --hook-stage manual
```

If GPU or external-service tests cannot run locally, list the skipped checks and explain why in the pull request's
Test Result section.

## Commits and Pull Requests

This project follows the [Developer Certificate of Origin](DCO) in the repository root. Every commit must include a
sign-off:

```bash
git commit --signoff -m "type(scope): summary"
```

Use the repository pull request template, complete the Purpose, Changes, Test Plan, and Test Result sections, and link
related issues. Feature changes must include documentation. For a new public API, document its signature, parameters,
return value, and exceptions in the Reference documentation.

By contributing, you agree that your contribution will be released under the project's
[Apache License 2.0](LICENSE).
