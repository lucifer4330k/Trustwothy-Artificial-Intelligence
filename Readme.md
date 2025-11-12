🤖 Trustworthy AI: Attack, Explain, Defend Dashboard

This project is an interactive web dashboard that demonstrates the core concepts of Trustworthy AI and adversarial machine learning.

Users can upload an image, select a pre-trained model (like ResNet50), and then apply various adversarial attacks (FGSM, PGD, C&W) to fool the model. The dashboard visualizes the attack's success and then allows the user to apply defense mechanisms (like JPEG compression or Gaussian blur) to try and recover the original, correct prediction.

(A (placeholder) screenshot of the running Gradio application. It shows the image upload panel, model/attack/defense dropdowns, and the analysis report.)

🚀 Features

Model Selection: Choose from multiple pre-trained ImageNet models, including:

ResNet50

VGG16

MobileNetV2

DenseNet121

EfficientNet-B0

Adversarial Attacks: Apply three common attacks to fool the selected model:

FGSM (Fast Gradient Sign Method)

PGD (Projected Gradient Descent)

C&W (L2) (Carlini-Wagner L2)

Defense Mechanisms: Test four different pre-processing defenses to mitigate attacks:

JPEG Compression

Bit-Depth Reduction

Gaussian Blur

Ensemble (JPEG + Blur)

Interactive Analysis: Get an instant, visual report on:

The original and adversarial predictions.

The final prediction after applying a defense.

A status update on whether the attack fooled the model.

A status update on whether the defense recovered the prediction.

🧠 Core Concepts & Mechanisms

This dashboard provides a hands-on demonstration of key pillars of AI security and trustworthiness.

1. Adversarial Attacks

Shows how small, often human-imperceptible perturbations to an input (like an image) can cause a powerful machine learning model to make a completely wrong prediction (e.g., misclassifying a "Labrador" as an "ostrich").

FGSM (Fast Gradient Sign Method): A simple, fast, and intuitive "one-shot" attack. It calculates the gradient of the loss with respect to the input image and then adds a small perturbation in the direction of the sign of that gradient. This is like finding the single "steepest" direction to push the image to make the model wrong, and taking one big step in that direction.

PGD (Projected Gradient Descent): A more powerful, iterative version of FGSM. Instead of one big step, PGD takes multiple small steps in the gradient direction, re-calculating the gradient each time. After each step, it "projects" the image back to ensure the total perturbation doesn't exceed a predefined limit (epsilon), making it a much more potent and deceptive attack.

C&W (L2) (Carlini-Wagner): A very powerful, optimization-based attack. Its goal is to find the smallest possible perturbation (measured by L2 distance) that can cause a misclassification. It is slower but highly effective and often used as a benchmark to test how robust a model truly is.

2. Adversarial Defenses

Demonstrates that by applying simple, "destructive" transformations to the input image, we can sometimes "clean" or "purify" it of the adversarial perturbation and restore the model's correct prediction. These methods work because the tiny, high-frequency adversarial noise is often more fragile than the image's main features.

JPEG Compression: This lossy compression algorithm is excellent at discarding high-frequency information that is "less important" to the human eye. This process often discards the adversarial noise as well, potentially restoring the original prediction.

Bit-Depth Reduction: Reduces the number of colors in an image (e.g., from 16 million colors down to 16). This quantization process forces similar colors to become identical, effectively "smoothing" out the tiny, precise perturbations added by the attacker.

Gaussian Blur: Applies a simple blur filter, which averages the pixels in a small neighborhood. This smoothing operation effectively "washes out" the high-frequency adversarial noise, often at the cost of making the image slightly less sharp.

Ensemble: This combines multiple defenses (JPEG + Blur) in the hope that what one defense misses, the other will catch, providing a more robust layered defense.

3. AI Explainability (XAI)

The original notebook also includes methods like GradCAM, LIME, and SHAP, which are techniques used to understand why a model made a specific decision. This helps in diagnosing why an attack was successful (e.g., "the model was fooled because the attack made it focus on the wrong part of the image").

🛠️ How to Run This Project Locally

Follow these steps to set up and run the web application on your local machine.

Prerequisites

Python 3.10 or newer

pip (Python package installer)

Step 1: Get the Code

Download the project files (app.py, backend.py, requirements.txt) and place them all together in a new folder (e.g., tai-dashboard).

Step 2: Create a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to avoid conflicts with your system-wide Python packages.

Open your terminal or command prompt and navigate to your project folder:

cd path/to/tai-dashboard


Create the virtual environment:

# On macOS or Linux
python3 -m venv venv

# On Windows
python -m venv venv


Activate the virtual environment:

# On macOS or Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate


Step 3: Install Dependencies

With your virtual environment active, install all the required packages from the requirements.txt file:

pip install -r requirements.txt


Step 4: Run the Application

Once the installation is complete, run the main app.py script:

python app.py


The script will start, load the pre-trained models into memory (this may take a moment), and then launch the Gradio web server.

Step 5: Access the Dashboard

The terminal will show a local URL, typically:
Running on local URL: http://127.0.0.1:7860

Open this URL (http://127.0.0.1:7860) in your web browser to access and use the application.

📂 File Structure

.
├── app.py             # Main script: Loads models and launches the web app
├── backend.py         # All core logic: attacks, defenses, and Gradio UI definition
└── requirements.txt   # List of all required Python packages


💻 Technology Stack

Python: Core programming language.

Gradio: For building the interactive web UI.

PyTorch: For loading models and running all ML operations.

NumPy, Matplotlib, OpenCV, PIL: For image processing, transformation, and visualization.

📜 License

This project is licensed under the MIT License.

🙏 Acknowledgements

This project is based on the work and code originally developed in the TAI_Final.ipynb Jupyter Notebook.

👥 Team Members

Debanjan Maji (2023BCS017)

Avadh Khandelwal (2023BMS008)

Arpit Singh (2023BMS002)

Musfiraa Aarif (2023BCS078)