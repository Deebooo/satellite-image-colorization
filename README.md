
<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>
Pokemon-Analysis
</h1>
<h3 align="center">📍 Unleash the Power of GANs with Satellite Image Colorization<</h3>
<h3 align="center">⚙️ Developed with the software and tools below:</h3>

<p align="center">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=for-the-badge&logo=Docker&logoColor=white" alt="Docker" />
</p>
</div>

---

## 📚 Table of Contents
- [📚 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [💫 Features](#-features)
- [📂 Project Structure](#project-structure)
- [🧩 Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [🗺 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---


## 📍 Overview

This project focuses on colorizing satellite images using a Generative Adversarial Network (GAN) model. The project leverages the power of deep learning to convert grayscale satellite images into colorized versions, enhancing the visual information available in satellite data.

---

## 💫 Features


| Feature                     | Description                                                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------------------|
| **🏗 Structure and Organization** | The codebase is organized into directories based on the type of analysis being performed, and each directory contains .py files with a specific focus on colorizing satellite images data. |
| **📝 Code Documentation**        | The code is well-documented with detailed explanations of each step taken in the analysis process.     |
| **🚀 High Performance**          | Leverages GPU acceleration with PyTorch for efficient training and inference.                          |
| **📈 Metrics and Evaluation**    | Comprehensive evaluation metrics including accuracy, precision, recall, F1 score, PSNR, and SSIM.      |
---


<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-github-open.svg" width="80" />

## 📂 Project Structure


```bash
repo
├── README.md
├── data
│   └── __init__.py
│   └── dataset.py
├── models
│   └── __init__.py
│   └── generator.py
│   └── discriminator.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── utils.py
│   └── visualization.py
├── train.py
├── main.py
├── requirements.txt
├── predict
│   └── __init__.py
│   └── predict.py
├── preprocessing
│   └── __init__.py
│   └── image-tiling.py

5 directories, 16 files
```

---

## 🧩 Modules

### Data Module
- `dataset.py`: Contains the `SatelliteImageDataset` class for loading and preprocessing the satellite images.

### Models Module
- `generator.py`: Defines the U-Net inspired Generator model.
- `discriminator.py`: Defines the PatchGAN Discriminator model.

### Utils Module
- `metrics.py`: Functions for calculating evaluation metrics.
- `utils.py`: Utility functions including weight initialization.
- `visualization.py`: Functions for visualizing and saving sample images.

---


<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-src-open.svg" width="80" />

## 🚀 Getting Started

### ✅ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> - Install Python: Make sure you have Python installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org/) and follow the installation instructions specific to your operating system.
> - Set up a virtual environment (optional but recommended): It's a good practice to create a virtual environment to isolate your project dependencies. Open a terminal or command prompt and run the following commands:
    On Windows:
        python -m venv myenv
        myenv\Scripts\activate
    On macOS and Linux:
        python3 -m venv myenv
    source myenv/bin/activate


## 🗺 Roadmap

> - [X] [📌  Task 1: Implement satellite images colorization using an initial dataset]
> - [X] [📌  Task 2: Evaluate the solution using relevent metrics]
> - [ ] [📌  Task 3: Implement satellite images colorization using a more representative and relevent dataset]
> - [ ] [📌  Task 4: Create a web interface for easy interaction.]
> - [ ] [📌  Task 5: Post on LinkedIn.]

---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a pull request to the original repository.
Open a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `[📌  MIT License]` License. See the [LICENSE](https://github.com/Deebooo/satellite-colorization/blob/main/LICENSE) file for additional info.
---
