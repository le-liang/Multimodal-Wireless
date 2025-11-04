# Multimodal-Wireless Dataset: Extension and Replay Toolkit

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://example.com)
[![Download Dataset](https://img.shields.io/badge/Download-Data-green)](https://example.com/download)
[![Watch Tutorial](https://img.shields.io/badge/YouTube-Tutorial-red)](https://www.youtube.com/watch?v=dxxxx)

This guide provides instructions on how to set up the environment and software needed to extend or replay the Multimodal-Wireless dataset.

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Installation Guide](#-installation-guide)
  - [1. Setting up the Python Environment](#1-setting-up-the-python-environment)
  - [2. Setting up Blender with Mitsuba Renderer](#2-setting-up-blender-with-mitsuba-renderer)
  - [3. Setting up CARLA Simulator](#3-setting-up-carla-simulator)
- [▶️ How to Use](#️-how-to-use)

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:
- **Conda** (either Anaconda or Miniconda)

## 🚀 Installation Guide

Please follow these steps carefully to configure your system.

### 1. Setting up the Python Environment

First, we will create a dedicated Conda environment to manage all Python dependencies.

1.  **Create and activate the Conda environment:**
    Open your terminal and run the following commands. This will create a new environment named `mmw` with Python 3.10.

    ```bash
    conda create -n mmw python==3.10
    conda activate mmw
    ```

2.  **Install required packages:**
    Once the environment is active, install all the necessary Python libraries using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### 2. Setting up Blender with Mitsuba Renderer

This project relies on Blender and the Mitsuba renderer for visualization and data generation.

1.  **Download Blender:**
    Download and install **Blender version 4.3.2** from the official website:
    [blender.org/download](https://www.blender.org/download)

2.  **Download Mitsuba Add-on:**
    Download the `mitsuba-blender.zip` file from the [official Mitsuba-Blender GitHub releases page](https://github.com/mitsuba-renderer/mitsuba-blender/releases).

3.  **Install the Add-on in Blender:**
    - Open Blender.
    - Navigate to `Edit` → `Preferences` → `Add-ons`.
    - Click the `Install...` button and select the `mitsuba-blender.zip` file you just downloaded.
    - Enable the add-on. The installation is successful if you see a **'Mitsuba (.xml)'** option under `File` > `Export`.

4.  **Install PyYAML for Blender:**
    You need to install the `pyyaml` package into Blender's internal Python environment. You may need to do this from the command line using Blender's Python executable.

    > **Note:** The exact command depends on your operating system and where Blender is installed. Find the Python executable inside your Blender installation folder.

### 3. Setting up CARLA Simulator

This project is compatible with a specific version of the CARLA simulator.

1.  **Download CARLA:**
    Download **CARLA version 0.9.15** from the official GitHub release page:
    [CARLA 0.9.15 Releases](https://github.com/carla-simulator/carla/releases/tag/0.9.15)

## ▶️ How to Use

Once your environment and software are fully set up, you can begin the extension or replay process by following the detailed steps in our tutorial video.

➡️ **[Watch the Tutorial on YouTube](https://www.youtube.com/watch?v=dxxxx)**
