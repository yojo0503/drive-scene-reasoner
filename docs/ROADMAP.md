# Milestone roadmap

## M0: Repository / data compatibility (now)
- [x] Independent source layout + pyproject packaging
- [x] DriveLM nested JSON reader and camera path resolution
- [x] Scene-level holdout split and Swift-format exporter
- [x] CPU tests + synthetic toy dataset generator
- [x] Qwen3-VL inference adapter and *unverified* ms-swift LoRA config
- [x] Local normalized QA evaluator (not leaderboard scoring)
- [x] Local-only API starter

## M1: Verified baseline (first milestone)
- [ ] Download a licensed DriveLM dataset subset and verify images/QA alignment
- [ ] Qwen3-VL-8B base inference on 100+ held-out samples
- [ ] Front-only versus 6-view comparison with OOM/latency logging
- [ ] Add one official DriveLM task metric as a separate evaluator
- [ ] Confirm GPU/driver/memory requirements for number of RTX 3090s available

## M2: Domain adaptation
- [ ] LoRA 8B SFT on DriveLM train scenes
- [ ] Adapter loading in inference engine
- [ ] Seeded experiments, config snapshots and loss curves
- [ ] Failure-case categorization across task families

## M3: Reasoning research
- [ ] Grounded evidence extraction and object tag evaluation
- [ ] Scene graph / staged perception-prediction-planning QA
- [ ] Dynamic per-question camera selection
- [ ] Visual token/accuracy/latency trade-offs

## M4: Demo engineering
- [ ] Scene browsing web frontend
- [ ] Video/multi-camera synchronization viewer
- [ ] Batch inference jobs and experiment dashboard
- [ ] Safe, offline report exports
