# Solar PV Panel Fault Detection using Deep Learning

An AI-driven image classification pipeline built to automate fault detection on large-scale solar PV plants, developed during a Deep Learning Internship (University of Queensland – TIET Collaboration).

## Overview

Manual inspection of solar panels for dust, bird droppings, cracks, and other faults is slow and doesn't scale for large solar farms. This project uses a CNN-based image classification pipeline to automatically flag panels that need cleaning or maintenance from images, enabling faster and more reliable inspection at scale.

## Key Features

- **Transfer learning with VGG16**: Fine-tuned a pretrained VGG16 CNN for panel fault classification, achieving **81% classification accuracy**.
- **Bird-drop detection model**: A dedicated retrained model (`bird_drop_retrained.h5`) focused on identifying bird-dropping contamination on panels.
- **End-to-end image processing pipeline**: Preprocessing, augmentation, and inference notebooks for classifying panel condition from images.

## Repository Structure

```
.
├── Solar_Panel_3thapar.ipynb            # Solar panel classification - iteration 3
├── Solar_Panel_5thapar.ipynb            # Solar panel classification - iteration 5
├── vgg16CompleteClassification2.ipynb   # Full VGG16 transfer learning pipeline
├── Solar_Bird_Drop.ipynb                # Bird-dropping fault detection notebook
├── bird_drop_retrained.h5               # Saved/retrained bird-drop detection model
├── testing/                             # Testing scripts / sample data
└── README.md
```

## Tech Stack

- **Language**: Python
- **Deep Learning**: TensorFlow / Keras
- **Model Architecture**: VGG16 (transfer learning, fine-tuned)
- **Data Handling**: NumPy, Pandas, OpenCV / PIL for image preprocessing
- **Visualization**: Matplotlib

## Methodology

1. **Data preprocessing** – Solar panel images resized, normalized, and augmented to improve model generalization.
2. **Transfer learning** – Used VGG16 pretrained on ImageNet as a feature extractor, with custom classification layers added on top.
3. **Fine-tuning** – Unfroze selected top layers of VGG16 and retrained on the panel fault dataset.
4. **Evaluation** – Achieved 81% classification accuracy on the fault detection task.
5. **Specialized model** – Trained a separate model to specifically detect bird-dropping contamination as a distinct fault category.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Kanishk-blip/-git_test-.git
   cd -git_test-
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib opencv-python
   ```
3. Open any notebook (e.g. `vgg16CompleteClassification2.ipynb`) in Jupyter/Colab and run all cells.
4. To use the pretrained bird-drop detection model directly:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('bird_drop_retrained.h5')
   ```

## Results

| Model | Task | Accuracy |
|---|---|---|
| VGG16 (fine-tuned) | Solar panel fault classification | 81% |

## Future Improvements

- Expand dataset size and diversity for better generalization.
- Experiment with lighter architectures (MobileNet, EfficientNet) for edge/drone deployment.
- Add object detection (bounding boxes) instead of whole-image classification, to localize the fault on the panel.
- Deploy as a real-time inference pipeline integrated with drone/camera feeds.

## Acknowledgements

Developed as part of a Deep Learning Internship in collaboration with the University of Queensland and Thapar Institute of Engineering & Technology (TIET).
