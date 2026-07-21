# DriveSeg — Hybrid PDE vs Full Neural Network Instance Segmentation for ADAS

Comparative study of two instance segmentation pipelines for autonomous driving scenarios (ADAS): a hybrid approach combining classical CNN detection with variational Level Set refinement, versus a purely deep learning end-to-end approach.

## Overview

The goal of the project is to build a pipeline for detecting and segmenting road-relevant elements (vehicles, pedestrians, signage) in video sequences, targeting the requirements of an Advanced Driver Assistance System (ADAS):

- Real-time (or near real-time) execution constraints
- Topological accuracy of the segmented contours
- Robustness to visual noise typical of road scenes

Two independent approaches were designed, implemented, and benchmarked against each other to evaluate the trade-off between mathematical/geometric methods and modern deep learning.

## Approach

### Pipeline 1 — Hybrid: CNN + Level Set

1. **Frame sampling** — uniform temporal sampling of frames from the input video, kept both in RGB (for the neural detector) and grayscale (for the PDE stage).
2. **Detection (YOLOv8)** — a pretrained CNN performs object detection restricted to road-relevant COCO classes, producing bounding boxes and confidence scores.
3. **Perona-Malik anisotropic diffusion** — a denoising pre-processing step that smooths homogeneous regions while preserving real object edges, reducing the risk of spurious gradients feeding the next stage.
4. **Level Set segmentation** — each bounding box is converted into a Signed Distance Function (SDF); the zero-level contour is then evolved via a partial differential equation until it locks onto the object's real boundary, guided by an edge-stopping function derived from the image gradient.

This approach is purely geometric in its refinement stage: it does not "understand" what an object is, it only reacts to color/intensity gradients starting from the CNN-provided bounding box.

### Pipeline 2 — Full Neural Network (YOLOv9-seg)

1. **Frame sampling** — same temporal sampling strategy as Pipeline 1.
2. **Single-stage inference** — a YOLOv9 segmentation model (pretrained on COCO, filtered to road classes) performs detection *and* mask generation in a single forward pass.
3. **Mask rescaling** — predicted masks are resized back to the original frame resolution.
4. **Rendering** — instance masks, bounding boxes, class labels and confidence scores are overlaid on the original frames.

Unlike Pipeline 1, there is no separate geometric refinement stage: the network directly learns semantic boundaries from training data.

## Results & Conclusions

| Aspect | Pipeline 1 (Hybrid) | Pipeline 2 (Full-NN) |
|---|---|---|
| Contour quality | Fails on low-contrast edges and shadows — the Level Set has no semantic understanding and can absorb background regions | Semantically aware — the network learns object shape, not just pixel-intensity contrast |
| Performance | Bottlenecked by sequential, CPU-bound PDE iterations; GPU/CPU handoff kills real-time performance | Fully tensor-based, maps efficiently onto GPU; ~30 FPS achieved, ~38x faster than the hybrid approach |
| Best use case | Offline analysis where clean, high-contrast boundaries are guaranteed | Real-time / ADAS scenarios where low latency is a safety requirement |

**Key takeaway:** the hybrid mathematical approach is theoretically elegant and interpretable, but its sequential PDE-solving step cannot exploit GPU parallelism the way tensor-based neural inference can. For real-time ADAS constraints, the full neural network pipeline is the only viable option among the two.

## Repository Structure

```
.
├── README.md
├── requirements.txt
└── notebooks/
    ├── Pipe1_Segmentation.ipynb          # Hybrid CNN + Level Set pipeline
    └── Pipe2b_YOLOseg_Standalone.ipynb    # Full-NN pipeline (YOLOv9-seg)
```

## Requirements & Setup

**⚠️ These notebooks are designed to run on Google Colab.** They rely on `google.colab.files.upload()` for video input and assume a Colab GPU runtime (CUDA) for the benchmark sections. Running them locally requires adapting the video input cell (replace the Colab upload call with a local file path) and, optionally, adjusting device selection if no CUDA GPU is available.

Install dependencies (if adapting for local execution):

```bash
pip install -r requirements.txt
```

## How to Run (on Google Colab)

1. Open the desired notebook in [Google Colab](https://colab.research.google.com/).
2. Make sure the runtime type is set to **GPU** (`Runtime > Change runtime type > GPU`) for realistic benchmark results.
3. Run all cells in order — the first code cell will prompt you to upload a video file.
4. Sampled frames, detections, segmentation masks and benchmark plots will be displayed inline as the notebook executes.

## License

This project was developed for academic purposes. Feel free to explore the code; if you reuse or build upon it, a reference to this repository is appreciated.
