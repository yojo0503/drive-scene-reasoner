# Upstream attribution and licenses

DriveLM: https://github.com/OpenDriveLab/DriveLM

Citation: Sima et al., *DriveLM: Driving with Graph Visual Question Answering* (2023; ECCV 2024).

- Original DriveLM code/assets: Apache-2.0 unless otherwise noted in upstream repository.
- DriveLM **language annotations**: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike).
- nuScenes images / source data: separate dataset distribution and usage terms.
- Qwen models, ms-swift, Transformers: separately licensed; check the versions you install.

This repository template does not vend any upstream training data, images, weights or source files. It uses the **documented data structure** and separately written reader code to interoperate with a dataset that the user downloads independently. Do not copy or redistribute upstream materials without checking their terms.

Official challenge validation (`v1_1_val_nus_q_only.json`) may have no answer annotations, so do not report local "validation accuracy" against that file.

## Optional upstream checkout for comparison

For original baseline reproduction, clone upstream **outside** this new repository or into the ignored `third_party/` directory:

```bash
git clone --depth 1 https://github.com/OpenDriveLab/DriveLM.git third_party/DriveLM
```

This keeps the modern project independent of the old baseline environment, while allowing access to upstream conversion/evaluation scripts when you explicitly choose to audit them.
