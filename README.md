# Immune Cell Image Classification with Attention-Enhanced ResNet

This project provides a flexible PyTorch implementation of **ResNet18/ResNet34** models enhanced with **attention mechanisms** — designed specifically for classifying **immune cell interactions** from **multi-channel microscopy images**.

The goal is to accurately classify various types of immune cells and their interactions (e.g., T cells, B cells, synapses) based on high-dimensional, multi-modal image data, using techniques like **Spatial Attention** and **CBAM** to improve model performance.

---

## Project Overview

IFC (Imaging Flow Cytometry) images of immune cells are provided in `.h5` (HDF5) format, each containing:
- `image`: raw microscopy input with multiple spectral or functional channels (e.g., `(H, W, 8)`)
- `mask`: cell segmentation masks
and some also containing:
- `label`: one of several immune cell interaction classes

These are combined to form a **16-channel input tensor**, allowing the model to learn both visual and spatial context across multiple bio-markers.

---

## Model Variants

Implemented CNN architectures:

| Model | Description |
|-------|-------------|
| `ResNet18` | Standard 18-layer residual network |
| `ResNet18WithSpatialAttention` | Adds spatial attention at final layer |
| `ResNet18WithRepeatedSpatialAttention` | Spatial attention after every residual block |
| `ResNet18WithCBAM` | Full CBAM (channel + spatial attention) |
| `ResNet34` | Standard 34-layer residual network |
| `ResNet34WithSpatialAttention` | Adds spatial attention at final layer |

All support:
- Custom input channels (default: `16`)
- Custom number of classes (default: `9`)
- Optional dropout

---

## Installation

```bash
git clone https://github.com/roeerozenstein/Project_14_Ruppin_2025
cd Project_14_Ruppin_2025
pip install -r requirements.txt

