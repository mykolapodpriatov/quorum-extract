# quorum-extract

> Field-level quorum extraction — run multiple cheap models, accept a field only when they agree, route disagreements to an expensive model or human, with calibrated per-field confidence.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Runs the same Pydantic schema against K cheap models/configs and reconciles per field (not per record). A field is accepted when enough extractors agree; contested fields escalate to a stronger model or are marked needs-review. Inter-model agreement is turned into a statistically calibrated per-field confidence score.

## Why

Single-model extraction gives you no trustworthy confidence signal. Cross-model agreement, properly calibrated, does — and it's cheaper than always calling a frontier model.

## Features

- Per-field quorum reconciliation across K models/configs
- Inter-model agreement calibrated to per-field confidence (isotonic/Platt)
- Cost-aware cascade: cheap-first, escalate only contested fields, with a budget report
- Disagreement diagnostics: which fields are systematically contested
- Pydantic schemas in; annotated (value, agreement, confidence, escalation status) out

## How it works

Provide a Pydantic schema and a set of cheap extractors. The library extracts with each, votes per field, calibrates agreement into confidence on a small labeled set, and escalates only the fields that fail quorum.

## Tech stack

- Python
- Pydantic
- scikit-learn
- OpenAI / Anthropic / Gemini SDKs
- Ollama

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Per-field multi-model voting over a Pydantic schema
- [ ] Agreement-to-confidence calibration (isotonic/Platt)
- [ ] Cost-aware escalation cascade + budget report
- [ ] Active-learning loop; human-review web queue

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
