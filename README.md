# 🤖 Trustworthy AI: Attack, Explain, Defend Dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-Latest-orange.svg)](https://gradio.app/)

An interactive web dashboard that demonstrates the core concepts of **Trustworthy AI** and **adversarial machine learning**. Upload an image, select a pre-trained model, apply adversarial attacks to fool the model, and then use defense mechanisms to recover the correct prediction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Core Concepts](#core-concepts)
  - [Adversarial Attacks](#adversarial-attacks)
  - [Adversarial Defenses](#adversarial-defenses)
  - [AI Explainability](#ai-explainability)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project provides a hands-on demonstration of key pillars of AI security and trustworthiness. Users can:

1. **Upload an image** or use sample images
2. **Select a pre-trained model** from the model zoo (ResNet50, VGG16, MobileNetV2, DenseNet121, EfficientNet-B0)
3. **Apply adversarial attacks** (FGSM, PGD, C&W) to generate adversarial examples
4. **Visualize** how small perturbations can fool powerful models
5. **Apply defense mechanisms** to mitigate the attacks
6. **Analyze results** through interactive reports and visualizations

The dashboard makes complex AI security concepts accessible and understandable through interactive experimentation.

---

## 🚀 Features

### 🎨 Model Selection
Choose from multiple pre-trained ImageNet models:
- **ResNet50** - Deep residual network
- **VGG16** - Visual Geometry Group network
- **MobileNetV2** - Lightweight mobile architecture
- **DenseNet121** - Densely connected network
- **EfficientNet-B0** - Efficient scaling network

### ⚔️ Adversarial Attacks
Apply three common attacks to fool the selected model:
- **FGSM** (Fast Gradient Sign Method) - Fast one-shot attack
- **PGD** (Projected Gradient Descent) - Iterative powerful attack
- **C&W (L2)** (Carlini-Wagner) - Optimization-based attack

### 🛡️ Defense Mechanisms
Test four different pre-processing defenses:
- **JPEG Compression** - Lossy compression to remove high-frequency noise
- **Bit-Depth Reduction** - Color quantization defense
- **Gaussian Blur** - Smoothing filter defense
- **Ensemble** - Combined defense (JPEG + Blur)

### 📊 Interactive Analysis
Get instant, visual reports showing:
- Original vs. adversarial predictions
- Prediction confidence scores
- Attack success indicators
- Defense recovery status
- Visual perturbation analysis

---

## 🧠 Core Concepts

### Adversarial Attacks

Adversarial attacks demonstrate how small, often imperceptible perturbations to an input can cause machine learning models to make completely wrong predictions.

#### FGSM (Fast Gradient Sign Method)
A simple, fast, and intuitive "one-shot" attack. It calculates the gradient of the loss with respect to the input image and adds a small perturbation in the direction of the sign of that gradient.

**Key characteristics:**
- Single-step attack
- Fast computation
- Easy to implement
- Good for understanding basic concepts

#### PGD (Projected Gradient Descent)
A more powerful, iterative version of FGSM. Takes multiple small steps in the gradient direction, re-calculating the gradient each time, while ensuring the perturbation stays within bounds.

**Key characteristics:**
- Multi-step iterative attack
- More powerful than FGSM
- Better fooling rate
- Often used for adversarial training

#### C&W (Carlini-Wagner L2)
A sophisticated optimization-based attack that finds the smallest possible perturbation (measured by L2 distance) to cause misclassification.

**Key characteristics:**
- Optimization-based approach
- Minimal perturbation
- Highly effective
- Computationally expensive
- Benchmark attack for robustness testing

### Adversarial Defenses

Defenses apply transformations to the input image to "clean" or "purify" it of adversarial perturbations, potentially restoring the model's correct prediction.

#### JPEG Compression
Lossy compression discards high-frequency information that often contains adversarial noise while preserving important image features.

#### Bit-Depth Reduction
Reduces the number of colors in an image through quantization, smoothing out precise perturbations added by attackers.

#### Gaussian Blur
Applies a blur filter that averages pixels in a neighborhood, washing out high-frequency adversarial noise.

#### Ensemble Defense
Combines multiple defenses (JPEG + Blur) to provide layered protection, catching what individual defenses might miss.

### AI Explainability (XAI)

The project includes explainability methods to understand model decisions:
- **GradCAM** - Visualizes which regions the model focuses on
- **LIME** - Local interpretable model-agnostic explanations
- **SHAP** - SHapley Additive exPlanations

These help diagnose why attacks succeed or fail.

---

## 💻 Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or newer** ([Download Python](https://www.python.org/downloads/))
- **pip** (Python package installer, comes with Python)
- **Git** (for cloning the repository)
- **Virtual environment** (recommended for isolation)

**System Requirements:**
- 4GB+ RAM (8GB recommended)
- 2GB+ free disk space
- GPU (optional, but recommended for faster processing)

### Setup Instructions

#### Step 1: Clone the Repository

```bash
git clone https://github.com/lucifer4330k/Trustwothy-Artificial-Intelligence.git
cd Trustwothy-Artificial-Intelligence
```

#### Step 2: Create a Virtual Environment (Recommended)

Using a virtual environment helps avoid conflicts with system-wide Python packages.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
.\venv\Scripts\activate
```

#### Step 3: Install Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- Gradio (web interface)
- OpenCV, PIL, matplotlib (image processing)
- NumPy, scipy (numerical computing)
- LIME, SHAP (explainability)
- Additional dependencies

**Note:** The first run will download pre-trained models (~100-500MB), which may take a few minutes.

---

## 🎮 Usage

### Starting the Application

Once installation is complete, run the main application:

```bash
python app.py
```

The application will:
1. Initialize the device (GPU if available, otherwise CPU)
2. Load ImageNet class labels
3. Load all pre-trained models into memory
4. Launch the Gradio web server

### Accessing the Dashboard

After starting, the terminal will display a local URL:

```
Running on local URL: http://127.0.0.1:7860
```

Open this URL in your web browser to access the interactive dashboard.

### Using the Dashboard

1. **Upload an Image**: Click the upload button or drag-and-drop an image
2. **Select a Model**: Choose from ResNet50, VGG16, MobileNetV2, DenseNet121, or EfficientNet-B0
3. **Choose an Attack**: Select FGSM, PGD, or C&W
4. **Set Parameters**: Adjust epsilon (perturbation magnitude) and other attack parameters
5. **Generate Attack**: Click to create the adversarial example
6. **Apply Defense**: Choose and apply a defense mechanism
7. **Analyze Results**: Review the comprehensive analysis report

### Example Workflow

```python
# The dashboard handles this automatically, but conceptually:
# 1. Load image
# 2. Get original prediction
# 3. Generate adversarial example with chosen attack
# 4. Get adversarial prediction (should be wrong)
# 5. Apply defense
# 6. Get defended prediction (hopefully correct again)
```

---

## 📂 Project Structure

```
Trustwothy-Artificial-Intelligence/
├── app.py                 # Main entry point: loads models and launches web app
├── backend.py             # Core logic: attacks, defenses, Gradio UI
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation (this file)
├── .gradio/               # Gradio cache and temporary files
└── __pycache__/          # Python bytecode cache
```

### File Descriptions

- **`app.py`**: Main script that sets up the device, loads all pre-trained models into a model zoo, loads ImageNet labels, and launches the Gradio interface.

- **`backend.py`**: Contains all core functionality including:
  - Image preprocessing and prediction functions
  - Adversarial attack implementations (FGSM, PGD, C&W)
  - Defense mechanism implementations
  - Visualization utilities
  - GradCAM, LIME, and SHAP explainability methods
  - Gradio interface definition

- **`requirements.txt`**: Lists all Python dependencies with their versions.

---

## 🛠️ Technology Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.10+ |
| **PyTorch** | Deep learning framework | Latest |
| **Gradio** | Web UI framework | Latest |
| **torchvision** | Pre-trained models & transforms | Latest |
| **NumPy** | Numerical computing | Latest |
| **Matplotlib** | Visualization | Latest |
| **OpenCV** | Image processing | Latest |
| **PIL/Pillow** | Image manipulation | Latest |
| **LIME** | Model interpretability | Latest |
| **SHAP** | Explainable AI | Latest |
| **seaborn** | Statistical visualization | Latest |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA out of memory" error
**Solution:** 
- Reduce batch size
- Use CPU instead of GPU by setting `device = torch.device("cpu")`
- Close other GPU-intensive applications

#### Issue: Models downloading slowly or timing out
**Solution:**
- Check your internet connection
- The first run downloads ~500MB of model weights
- Consider downloading models manually if needed

#### Issue: "ModuleNotFoundError" or import errors
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

#### Issue: Gradio interface not loading
**Solution:**
- Check if port 7860 is already in use
- Try a different port: modify `demo.launch(server_port=7861)`
- Check firewall settings

#### Issue: Poor attack success rate
**Solution:**
- Increase epsilon value for stronger attacks
- Try different attack methods (PGD is stronger than FGSM)
- Some models are naturally more robust

#### Issue: Defense not recovering prediction
**Solution:**
- This is expected behavior - not all defenses work on all attacks
- Try different defense mechanisms
- Some attacks are designed to be defense-resistant

### Getting Help

If you encounter issues not listed here:
1. Check the [Issues](https://github.com/lucifer4330k/Trustwothy-Artificial-Intelligence/issues) page
2. Search for similar problems
3. Open a new issue with:
   - Your Python version
   - Error message/traceback
   - Steps to reproduce
   - System information (OS, GPU if applicable)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 **Report bugs** - Open an issue describing the bug
- 💡 **Suggest features** - Open an issue with your idea
- 📝 **Improve documentation** - Submit PRs for documentation updates
- 🔧 **Submit fixes** - Fix bugs or implement features
- ⭐ **Star the repository** - Show your support!

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Ensure code quality and add comments
4. **Test your changes**: Verify everything works
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes clearly

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include comments for complex logic
- Test your changes thoroughly
- Update documentation as needed

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 👥 Team

This project was developed by:

| Name | Roll Number | Role |
|------|-------------|------|
| **Debanjan Maji** | 2023BCS017 | Developer |
| **Avadh Khandelwal** | 2023BMS008 | Developer |
| **Arpit Singh** | 2023BMS002 | Developer |
| **Musfiraa Arif** | 2023BCS078 | Developer |

---

## 🙏 Acknowledgments

- This project is based on work originally developed in the **TAI_Final.ipynb** Jupyter Notebook
- Pre-trained models from [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html)
- ImageNet labels from [imagenet-simple-labels](https://github.com/anishathalye/imagenet-simple-labels)
- Adversarial attack implementations inspired by research papers:
  - FGSM: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
  - PGD: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
  - C&W: [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)
- Explainability methods:
  - GradCAM: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
  - LIME: [Why Should I Trust You?](https://arxiv.org/abs/1602.04938)
  - SHAP: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

---

## 📧 Contact & Support

- **Repository**: [Trustwothy-Artificial-Intelligence](https://github.com/lucifer4330k/Trustwothy-Artificial-Intelligence)
- **Issues**: [Report a bug or request a feature](https://github.com/lucifer4330k/Trustwothy-Artificial-Intelligence/issues)
- **Discussions**: Share ideas and ask questions

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ for Trustworthy AI Research**

</div>
