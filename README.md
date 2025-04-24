# football-player-tracking-yolo

![first-look-at-datasets](readme-assets/result_tracker.gif)

### Data

## ✨ Key Features

- **Modular Design:**
- **Configuration Management:** Uses Hydra for flexible configuration via YAML files and command-line overrides.
- **Pre-trained Weights:** Leverages pre-trained weights for the YolO.

## 🛠️ Technologies Used

- **Python:** Core programming language.
- **Pytorch:** Deep learning framework.
- **YoLO:** Pre-trained
- \*\* \*\*: Tracking Players and ball
- **Jupyter Notebook:** For exploratory data analysis (EDA), model development, and training.
- **Hydra:** Configuration management.
- **Roboflow:** Dataset management and download.
- **Docker:** Containerization for consistent environment.
- **VS Code Dev Containers:** Development environment setup.

## Project Structure

```

```

## 🔧 Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vivekpatel99/football-players-tracking-yolo.git
   cd football-players-tracking-yolo
   ```

2. **Install VS Code Extensions:**

   - Docker
   - Dev Containers

3. **Rebuild and Reopen in Container:**

   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS).
   - Select `Dev Containers: Rebuild and Reopen in Container`.

4. **Set up Environment Variables:**

   - Create a `.env` file in the project root directory.
   - Define the path to your dataset:
     ```dotenv
     # .env
     ROBOFLOW_API_KEY=
     ```
   - The `train.py` script uses `pyrootutils` and `dotenv` to automatically load this.

## Data Configuration

- Ensure the path specified in `DATA_ROOT` within your `.env` file points to the correct location.

## ⚙️ Configuration

- This project uses Hydra for managing configurations.
- The main configuration file is `configs/train.yaml`.
- It composes configurations from subdirectories like `configs/model`, `configs/data`, etc.
- You can modify parameters directly in the YAML files or override them via the command line.

**Example:** Change the model type or learning rate:

## 🏋️ Training

4. Output: Training logs and checkpoints will typically be saved in an outputs/ directory managed by Hydra (check hydra.run.dir in the config). Validation metrics are printed at the end of each epoch.

## Models

## 📈 Results & Visualizations

## 🖥️ Hardware Specifications

This project was developed and tested on the following hardware:

- **CPU:** AMD Ryzen 5900X
- **GPU:** NVIDIA GeForce RTX 3080 (10GB VRAM)
- **RAM:** 32 GB DDR4

While these specifications are recommended for optimal performance, the project can be adapted to run on systems with less powerful hardware.

## 📚 Reference

- [ReCoDE-DeepLearning-Best-Practices](https://imperialcollegelondon.github.io/ReCoDE-DeepLearning-Best-Practices/)
- [Roboflow-tutorial](https://www.youtube.com/watch?v=aBVGKoNZQUw)
- [Roblflow-tutoril-colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb#scrollTo=H1smkPKfYm00)
