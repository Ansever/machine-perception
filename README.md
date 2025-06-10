# Machine Perception Project - Video Object Segmentation

Dataset Used - **MOSE Dataset** ([dataset-link](https://mose.video/))

## Models Examined

- Segment Anything Model 2 (SAM2)
- Space-Time Memory Network (STM)
- You Only Look Once v11 (YOLOv11-seg)

## Main parts of the code

- `src/machine_perception/experiments/predictor.py` - unified interface and specific predictors for SAM2, STM and YOLO
- `src/machine_perception/experiments/stm-finetuning.ipynb` - notebook with code for fine-tuning STM
- `src/machine_perception/experiments/mose-evaluation.ipynb` - (quite messy) notebook for predicting and evaluating results on the MOSE dataset using models

## References

- SAM2 Paper: <https://arxiv.org/abs/2408.00714>
- STM Paper: <https://arxiv.org/abs/1904.00607>
- YOLO Paper: <https://arxiv.org/abs/2410.17725>
