# Nature Image Classification using Deep Learning


## 📌 Project Overview
A deep learning approach to classify nature images into 6 categories:  
**Sea | Mountain | Glacier | Forest | Street | Building**.  
Compares performance of VGG16, VGG19, ResNet, InceptionV3, and a custom CNN model.

## 🎯 Key Features
- **Multi-model comparison**: VGG16/VGG19, ResNet152V2, InceptionV3, Custom CNN
- **Ensemble techniques** to boost accuracy
- **Real-world applications**: Urban planning, tourism, climate research
- Data augmentation (rotation, flipping, zooming, etc.)

## 📊 Results Summary
| Model          | Accuracy | F1-Score |
|----------------|----------|----------|
| Custom CNN     | 0.82     | 0.8164   |
| VGG16          | 0.90     | 0.9010   |
| VGG19          | 0.90     | 0.90     |
| InceptionV3    | 0.90     | 0.90     |
| ResNet152V2    | 0.89     | 0.8938   |

*(Confusion matrices available in [Technical Report](Technical%20report.pdf))*

## 🛠️ Installation
```bash
git clone https://github.com/AshrafAlham/Nature-image-detection.git
cd Nature-image-detection
pip install -r requirements.txt

🏗️ Project Structure
Nature-image-detection/
├── data/               # Dataset (not included in repo)
├── models/             # Pretrained model weights
├── notebooks/          # Jupyter notebooks for training/evaluation
├── scripts/            # Utility scripts (preprocessing, etc.)
├── results/            # Outputs (graphs, metrics)
├── Technical report.pdf # Full project documentation
└── README.md           # This file

📂 Dataset
Source: Kaggle

Size: 18,000 images (14k train / 3k test / 1k prediction)

Classes: Sea, Mountain, Glacier, Forest, Street, Building

Augmentation: Rotation, scaling, flipping, brightness adjustment

👨💻 Author: Ashraf Alham
📅 Last Updated: 15 May 2025

