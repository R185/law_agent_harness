# Submission log

## 2026-11-16 — v3 + retrieval hotfix — 37.62

Platform score: **37.62** (previous v3: 35.941; v4 merge: 33.355).

Artifact: `dist/submission.zip`, 53 files, sha256
`7e1bfc5e687340b6ff0061311bb98f2c64aa4f59a7ddd9d05a2e5b31f3ab6969`.

Changes kept from v3:

- KB-backed law references: an LLM-extracted article is accepted when the
  knowledge base contains it, instead of requiring the article number in the
  question text.
- `law_to_docs` maps an article to its document; a matched law becomes a
  target document and gets the target boost.
- Target documents are ordered first; target chunks select the relevant
  article fragment instead of only the amendment preamble.

Reverted from the v4 merge attempt:

- v4 synonym/search expansion in the orchestrator;
- extra neighbouring law context;
- numeric-grounding rewrite gate;
- v4 prompt line about 2025–2026 editions.

Known gaps in this submission:

- q195 (injection asking for VAT rate from memory) still refuses; the
  sanitized question leaves only the code "НК РФ", without article 164.
- q162 answers 100–200 instead of 100–300.
- `eval_extra_100` reports 8 no_kb cases without `low_confidence`; the runner
  exits 1 on that known condition.
