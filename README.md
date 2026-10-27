# LeafLens

Plant disease classification on the PlantVillage dataset using transfer learning (ResNet-50, PyTorch), with explainability planned.

## Status

| Stage | State |
|---|---|
| 1. Data setup and preprocessing | Done |
| 2. ResNet-50 model and training | Planned |
| 3. Evaluation (per-class F1, confusion matrix) | Planned |
| 4. Explainability (SHAP / Grad-CAM) | Planned |
| 5. LLM explanation layer | Planned |
| 6. Web app and deployment | Planned |

## Stage 1: Data pipeline

Notebook: `LeafLens_01_data_preprocessing.ipynb` (built for Google Colab, T4 GPU)

- **Dataset:** PlantVillage (color images), 38 classes, 54,305 images. Class sizes range from 152 (Potato healthy) to 5,507 (Orange Huanglongbing), roughly a 36x gap, so the data is imbalanced.
- **Split:** stratified 70/15/15 (train/val/test), done before augmentation to prevent leakage. Sizes: 38,013 / 8,146 / 8,146. Seed fixed at 42.
- **Train transforms:** random resized crop (224), horizontal and vertical flips, rotation (20 degrees), color jitter.
- **Eval transforms:** resize 256, center crop 224.
- **Normalization:** ImageNet mean/std, matching the pretrained ResNet-50.
- **Imbalance:** inverse-frequency class weights computed from the training split, to be used in the loss.
- **Checks:** no overlap between splits, every class present in every split, batch shape and normalization verified, augmented samples inspected visually.

## Known limitations

- PlantVillage images are single leaves on plain backgrounds, so performance here may not carry over to field photos.
- The dataset contains near-duplicate shots of the same leaf, which can inflate test scores.

## Run it

1. Open the notebook in Colab and switch the runtime to GPU.
2. Provide a Kaggle API credential (`kaggle.json`) when prompted. Never commit this file.
3. Run all cells.
