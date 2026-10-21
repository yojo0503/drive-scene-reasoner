# DriveLM alignment implementation plan

**Goal:** Repair the five reviewed gaps and prepare verified server assets without training or inference.

**Architecture:** Retain drivescene modules, add preparation/protocol/assets modules for distinct responsibilities, and keep upstream code in an ignored checkout.

**Constraints:** No model execution; no fabricated Graph VQA edges; no annotation leakage into prompts; no credential logging; no data or weights in Git.

- [x] Implement and exercise resumable, revision-pinned, hash-checked asset download tooling; launch model download in tmux.
- [x] Extend QARecord and strict image validation; preserve original metadata and official IDs.
- [x] Split labeled preparation and inference preparation; export metadata, graphs and split provenance.
- [x] Harden local evaluation and add official output conversion/tag-0 scoring.
- [x] Add LoRA loading and image budgets; align prompts without loading a model during tests.
- [x] Add reproducible training recipe, validation input, snapshots and resume; validate in dry-run mode only.
- [x] Test regressions, CLI workflows and real public fixtures; update documentation and server asset inventory.
- [x] Transfer changes to the server, rerun CPU checks, commit reviewed source changes and report any access/download blockers accurately.

Each behavioral change receives regression tests for its failure mode before acceptance. Implementation runs inline in the authorized chat.
