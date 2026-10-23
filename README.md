# Multimodal-Wireless Dataset: Extension and Replay Toolkit

This guide provides instructions on how to set up the environment and software needed to extend or replay the Multimodal-Wireless dataset.

## What's New

### 🚀 v1.1.0 — Accelerated Data Generation

- **Upgraded pipeline:** Reuse the static scene and update dynamic vehicle poses frame by frame in Sionna RT, replacing the previous workflow of exporting and loading a complete scene for every frame.
- **20× faster generation:** The pipeline redesign, together with code-level optimizations, reduces total generation time by approximately **95%** for the same workload while preserving the generated results.
- **Lower storage requirements:** Eliminate the need to generate and store a separate full scene for every frame, removing the large intermediate scene-storage overhead of long sequences.
- **Bug fix:** Address a minor issue reported by the community.

> **Note:** The pipeline architecture update itself does not change the channel results. Minor differences in the generated channels between the original version and v1.1.0 arise solely from this issue fix (mean channel NMSE of -19.3 dB).

See the [detailed pipeline comparison](https://le-liang.github.io/mmw/tutorials.html#accelerated-pipeline) for the original and accelerated workflows.

## Table of Contents

- [What's New](#whats-new)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
  - [1. Setting up the Python Environment](#1-setting-up-the-python-environment)
  - [2. Setting up Blender with Mitsuba Renderer](#2-setting-up-blender-with-mitsuba-renderer)
  - [3. Setting up CARLA Simulator](#3-setting-up-carla-simulator)
- [How to Extend/Replay](#how-to-extendreplay)

## Prerequisites

Before you begin, ensure you have the following tasks done:
- Clone the current github repository
- Download the Blender asset from 'Blender_Towns' folder in [Dataset](https://1drv.ms/f/c/e891739d21c98746/EpnYIxEFXAdAmoj7GePie8QB5be53v1TTER4OmjjQbNYNQ?e=ykbmZh) or [中国大陆镜像](https://pan.quark.cn/s/de33117418c4)

If you only want to replay some scenario:
- Download the dataset of the scenario from 'sunny' folder in [Dataset](https://1drv.ms/f/c/e891739d21c98746/EpnYIxEFXAdAmoj7GePie8QB5be53v1TTER4OmjjQbNYNQ?e=ykbmZh) or [中国大陆镜像](https://pan.quark.cn/s/de33117418c4)

## Installation Guide

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

## How to Extend/Replay

Once your environment and software are fully set up, you can begin the extension or replay process by following the detailed steps in our tutorial video.

The tutorial demonstrates the original three-step workflow below. See the [pipeline comparison](https://le-liang.github.io/mmw/tutorials.html#accelerated-pipeline) for the v1.1.0 update:

- CARLA: Define the scenario in CARLA via the config file and implement it via carla_main.py, which produces five sensor data and the yaml file of all frames simultaneously.
- Blender: Reconstruct the whole scenario of the first frame in Blender based on the .yaml file.
- Sionna: Load the scene of the first frame in Sionna. For each new frame, move the vehicle objects in Sionna, compute multi-path components and generate the corresponding channel.

➡️ **Watch the Tutorial on** [YouTube](https://youtu.be/M0qTrLj61aM) or [Bilibili](https://www.bilibili.com/video/BV1r11nBGEaV/?spm_id_from=333.1387.homepage.video_card.click&vd_source=96080c7f8528346ec45df46d67c8d3e1)

Note: ./Sionna_channel_storage/channel_storage_v2i.py collects channels with LOS paths, with the reflection mode on and the scattering mode off. To dump pure NLOS channels, one should refer to ./Sionna_channel_storage/channel_storage_v2v.py which involves the scattering paths calculation when no LOS paths or other reflection paths exists.
