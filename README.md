# GPDM-PET

Generation-Prior Diffusion Model (GPDM) for accelerated direct attenuation and scatter correction of whole-body 18F-FDG PET.

## Overview

GPDM-PET is a diffusion-model-based framework for direct attenuation and scatter correction (ASC) in PET imaging. The model learns a generation prior from attenuation- and scatter-corrected PET images and directly generates corrected PET images from non-corrected PET inputs.

## Repository Structure

```text
GPDM-PET/
├── models/                 # Network architectures
├── train.py                # Training script
├── infer.py                # Inference script
├── diffusion_model.py      # Diffusion model implementation
├── utils.py                # Utility functions
└── README.md
```

## Training

```bash
python train.py
```

## Inference

```bash
python infer.py
```

## Method

The proposed GPDM framework employs a diffusion-based generative prior to directly generate attenuation- and scatter-corrected PET images from non-corrected PET images. The method is designed for accelerated whole-body PET imaging while maintaining image quality and quantitative accuracy.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{GPDMPET,
  title={Generation-Prior Diffusion Model for Accelerated Direct Attenuation and Scatter Correction of Whole-Body 18F-FDG PET},
  author={Cho, Min Jeong and others},
  year={2026}
}
```

## License

This project is released under the MIT License.
