# Skin Cancer Classification Using CNNs

Multi-class classification of 9 skin lesion types using deep learning with transfer learning and class imbalance handling.

## Project Report (IEEE Format)
https://drive.google.com/drive/folders/1urnD3jDn6AQgb09W9nBl1X-OE2pBn3nD

## Pre-recorded Presentation Video
https://www.youtube.com/watch?v=BNObPrvPU5k

## Presentation Slides
https://drive.google.com/drive/folders/1po_sQSeaiDr7B2NjP0lSXOFzpw-hsJGu

## Dataset Link
https://drive.google.com/drive/folders/1kPRBt3u1aDLjx0Yh3LWFpWUbdJGkAp-K

## Demo Video


## Dataset Details
- **Source:** ISIC (International Skin Imaging Collaboration)
- **Classes:** 9 skin lesion types (actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion)
- **Structure:** `Data/Skin_cancer_ISIC_data/Train/` and `Test/` folders

## Models Trained

1. **Baseline CNN** - Simple 3-layer CNN (baseline performance)
2. **DenseNet201** - Transfer learning with fine-tuning (44.92% accuracy)
3. **InceptionV3** - Transfer learning with fine-tuning
4. **MobileNetV2** - Transfer learning with fine-tuning
5. **Binary Classification** - Melanoma vs Non-Melanoma with SMOTE

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)

## Usage

### Training
Open and run `skin_cancer_notebook_updated_6.ipynb` in Jupyter:
- Execute cells sequentially
- Models save automatically during training
- Best models saved as `best_*.keras` files

### Testing Single Images

Run this cell
```
# Load saved model
model = keras.models.load_model('best_densenet201.keras')

# Use the predict_image() function
img_path = "Data/Skin_cancer_ISIC_data/Test/nevus/ISIC_0000007.jpg"

pil = Image.open(img_path)
top_idx, top_probs = predict_image(pil, top_k=5)
```


**What it does:**
- Displays the input image
- Shows top-K predicted classes with confidence scores
- Automatically preprocesses image (resizes to 75×100, applies model-specific preprocessing)

## Saved Models

| Model | Filename | Test Accuracy |
|-------|----------|---------------|
| Baseline CNN | `best_simple_cnn.keras` | ~35% |
| DenseNet201 | `best_densenet201.keras` | 44.92% |
| InceptionV3 | `best_inceptionv3.keras` | ~43% |
| MobileNetV2 | `best_mobilenetv2.keras` | ~42% |
| Binary SMOTE | `best_densenet_binary_smote.keras` | Binary melanoma detection |

## Key Features

- **Transfer Learning:** Leverages ImageNet pre-trained weights
- **Fine-tuning:** Unfreezes last 40% of layers for domain adaptation
- **Class Imbalance Handling:** Class weighting, undersampling, SMOTE
- **Data Augmentation:** Random rotation, zoom, shifts, flips
- **Comprehensive Metrics:** Accuracy, precision, recall, F1, Cohen's Kappa, confusion matrices

## Project Structure

```
Project/
├── skin_cancer_notebook_updated_6.ipynb    # Main notebook
├── Data/
│   └── Skin_cancer_ISIC_data/
│       ├── Train/                          # Training images (9 class folders)
│       └── Test/                           # Test images (9 class folders)
├── best_*.keras                            # Saved models
├── requirements.txt                        # Dependencies
└── Project_Update_*.pdf                    # Final report
```

## Project Report

See `Project_Update_Skin Cancer Classification Using Convolutional Neural Networks.pdf` for complete methodology and results.

