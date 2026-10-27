# ObTopoMem

**Topological Mapping Using Monocular Camera for Object-Level Change Detection**

ObTopoMem is a research project for constructing an object-level representation of an environment from video. It combines image segmentation, feature matching, graph-based grouping, and epipolar geometry to associate object observations across frames and reconstruct missing image-space object centers.

The current command-line workflow processes one video into a cached topological environment (`TopoEnv`) and generates visualizations. This representation supports research into object-level change detection; the command does not produce a separate before/after change report.

## Research background

ObTopoMem originated from my master’s thesis,
[*Topological Mapping Using Monocular Camera for Object-Level Change
Detection*](https://hdl.handle.net/11511/118427)
(M.Sc. in Robotics, Middle East Technical University, December 2025).

This repository **generally follows** the methodology described in the
thesis. Implementation details may differ as the code is refined and
the research develops.

The repository serves as a continuation of this research, with the aim
of extending its methods, capabilities, and experimental evaluation
over time.

## How it works

1. Select keyframes and intermediate frames from the input video.
2. Extract SuperPoint and DINOv2 features through OmniGlue, and segment keyframes with Segment Anything (SAM).
3. Match keyframes to one another and intermediate frames to their neighboring keyframes.
4. Build a segment graph and group observations into object hypotheses, called `AnyThing` objects.
5. Merge and filter object hypotheses, then propagate them to intermediate frames using prompted segmentation.
6. Build the topological environment and estimate missing object centers using fundamental matrices and epipolar-line intersections.
7. Export a sampled segmentation video, graph plots, representative object crops, and diagnostic images.

## Requirements

The supplied setup workflow targets **Linux with an NVIDIA GPU**

- Git, including submodule support.
- Conda available on `PATH`.
- Python 3.11, managed by the setup script in the `obtopomem` Conda environment.
- An NVIDIA driver compatible with the installed GPU frameworks and the GPU's NVENC encoder.
- FFmpeg available on `PATH`, built with the `h264_nvenc` encoder.
- Disk space for the model checkpoints, per-object caches, and generated images.

The setup installs TensorFlow with its CUDA dependencies and PyTorch CUDA 12.6 wheels. Exact python dependency versions are declared in [pyproject.toml](pyproject.toml). Conda provides the interpreter and pip; it does not install the system NVIDIA driver or FFmpeg.

Video export currently requires NVENC even though some model wrappers can select CPU execution. Listing `h264_nvenc` in FFmpeg confirms that the encoder is included in the build, but successful encoding also requires compatible hardware and drivers.

## Installation

Clone this repository with its submodules, then enter the repository directory. If it is already cloned, initialize the pinned dependencies with:

```bash
git submodule update --init --recursive
```

On Debian or Ubuntu, FFmpeg can be installed with:

```bash
sudo apt install ffmpeg
```

Check that the encoder is available:

```bash
ffmpeg -hide_banner -encoders | grep h264_nvenc
```

From the repository root, run:

```bash
bash scripts/setup_env.sh
conda activate obtopomem
```

The script creates the environment if needed, installs the project and both upstream packages in editable mode, checks for python dependencies, and checks FFmpeg availability. If an `obtopomem` environment already exists, it is reused and must contain Python 3.11.

Keep the source checkout in place: the supported workflow uses an editable installation and resolves model paths relative to the repository.

## Model checkpoints

Download the pretrained models from the repository root:

```bash
python scripts/download_models.py --list
python scripts/download_models.py --only omniglue
python scripts/download_models.py --only sam
```

The expected layout is:

```text

models/
├── omniglue/
│   ├── og_export/
│   │   └── saved_model.pb
│   ├── sp_v6/
│   │   └── saved_model.pb
│   └── dinov2_vitb14_pretrain.pth
└── sam/
    └── sam_vit_l_0b3195.pth
```

The SavedModel directories also contain supporting files from their archives: preserve the complete extracted directories. Model files are excluded from Git.

The downloader skips models that pass basic presence checks and refuses to overwrite an existing destination that fails those checks. It prints download SHA-256 hashes, but does not currently verify them against pinned reference hashes.

## Run

```bash
python -m obtopomem.main \
  --video /path/to/video.mp4 \
  --output-dir outputs/example
```

Use a new or empty output directory for each new input or processing configuration. The program records the configuration in `run_config.json` before processing.

For all available options:

```bash
python -m obtopomem.main --help
```

### Processing options

| Option | Default | Meaning |
| --- | --- | --- |
| `--key-frame-interval-seconds` | `0.1` | Keyframe spacing in seconds; `-1` selects every frame. |
| `--max-keypoints` | `4096` | Maximum SuperPoint keypoints per frame; `0` removes the cap. |
| `--max-dim` | `630` | Starting maximum image dimension for feature extraction and calibration. |
| `--no-calibrate-vram` | Calibration enabled | Use `--max-dim` directly instead of attempting smaller dimensions after extraction failures. |
| `--keypoint-density-threshold` | `0.001` | Keypoint-density threshold used in segment processing. |
| `--sam-pred-iou-thresh` | `0.95` | SAM predicted mask-quality threshold. |
| `--sam-stability-score-thresh` | `0.98` | SAM mask-stability threshold. |
| `--sam-box-nms-thresh` | `0.2` | SAM box nonmaximum-suppression threshold. |
| `--sam-crop-nms-thresh` | `0.05` | SAM cross-crop nonmaximum-suppression threshold. |

With calibration enabled, `--max-dim` must be at least 512. Calibration probes feature extraction; it does not guarantee that every later pipeline stage will fit in GPU memory.

### Reuse an existing run

Load the saved video path and processing settings automatically:

```bash
python -m obtopomem.main \
  --output-dir outputs/example \
  --use-cached-config
```

This implies cache reuse and cannot be combined with explicit video or processing options. Alternatively, repeat the original arguments and add `--reuse-cache`:

```bash
python -m obtopomem.main \
  --video /path/to/video.mp4 \
  --output-dir outputs/example \
  --reuse-cache
```

The recorded settings and video path, size, and modification time must match. An existing `TopoEnv` is loaded when available; otherwise, processing runs with the available caches. Visualization outputs are regenerated and may be overwritten.

Keep the original video and output directory at their original locations: cached objects contain paths. Cache files use Python pickle and should only be loaded from trusted runs. The manifest checks inputs and settings, but does not fingerprint the source code or model weights; use a new output directory after changing either.

## Outputs

Results are written beneath `--output-dir`:

| Output | Contents |
| --- | --- |
| `run_config.json` | Input identity and processing settings. |
| `segment_wsam_output.mp4` | Sampled video with object masks overlaid. |
| `ffmpeg_export.log` | FFmpeg diagnostics, including encoder failures. |
| `anything_graph.png` | Object graph visualization. |
| `segment_graph.png` | Segment graph visualization. |
| `representative_segments/` | Representative object crops and the lifespan plot, when objects are available. |
| `keypoint_frames/` | Diagnostic images with masks, keypoints, bounding boxes, and reconstructed centers. |
| `segment_frames/` | Segment visualization images. |
| Class-named cache directories | Serialized frames, segments, keypoints, matches, objects, and the topological environment. |

Additional graph images may be generated while object groups are merged. Image and plot availability depends on the observations retained during processing.

**The video is intentionally a sampled preview.** Frames without computed segmentation are omitted, while the output uses the input FPS. Its duration is therefore generally shorter than the source video, and it does not preserve the source timeline or audio.

## Research scope and limitations

- **Runtime scales with keyframe count.** All keyframe pairs are matched, giving `K (K-1) / 2` pairs for `K` keyframes. Selecting every frame can be expensive.
Per frame images and caches also consume disk space.
- **Short-lived objects are filtered.** Before intermediate-frame propagation, objects must appear in more than 10 keyframes to survive the final object-count filter.
Short clips or sparse keyframe selection can leave no retained objects.
- **Missing centers are not always recoverable.** The default geometric filling stage requires at least 12 common object observations to attempt a frame-pair estimate and at least two usable epipolar lines to fill a missing center. Unresolved entries remain missing. There is no homography fallback for planar scenes or pure rotation.
- **Reconstruction can vary between runs.** RANSAC uses unseeded randomness, and reconstructed geometric results are recomputed when loading the serialized environment. Cached visualization is not guaranteed to be identical across runs.
- **Validation is currently manual.** The repository does not yet include a project-owned automated test suite or CI workflow.

## Troubleshooting

- **Missing checkpoints** run `python scripts/download_models.py` and verify the model directory layout above.
- **Missing upstream imports:** initialize the submodules and rerun `bash scripts/setup_env.sh` inside the supported setup workflow.
- **Configuration mismatch:** use the original settings with `--reuse-cache`, use `--use-cached-config`, or choose a new output directory.
- **GPU memory exhaustion:** reduce feature-extraction dimensions or the keypoint cap, and inspect which stage failed. Feature calibration does not cover SAM or every later operation.
- **Video export failure:** inspect `ffmpeg_export.log` in the output directory and verify FFmpeg, NVENC hardware support, and the NVIDIA driver.
- **No usable segments or retained objects:** inspect the video content, SAM thresholds, sampling interval, and object-retention threshold described above.

## Project layout

```text
scripts/
├── setup_env.sh           # Environment and editable-package setup
└── download_models.py    # Pretrained checkpoint downloads
src/obtopomem/
├── main.py               # CLI and run configuration checks
├── VideoProcessor.py     # Processing pipeline
├── TopoEnv.py            # Environment representation and geometry
├── AnyThing.py           # Object hypotheses and object graph
├── FrameDescriptor*.py   # Frame features and coordinate handling
├── FrameSegmentDescriptor.py
├── Graph.py              # Graph operations and visualization
├── BaseCacheManager.py   # Disk and memory cache management
├── ImagePoint.py         # Keypoints and keypoint collections
├── Matches.py            # Feature correspondences
├── VideoContainer.py     # Video access
├── wOmniGlue.py          # OmniGlue wrapper
├── wSAM.py               # SAM wrapper
├── paths.py              # Repository-relative model locations
└── utils.py              # Visualization and export
third_party/
├── omniglue/             # Pinned Git submodule
└── segment-anything/     # Pinned Git submodule
```

## Upstream projects

ObTopoMem builds on the following projects:

- [OmniGlue](https://github.com/google-research/omniglue), for feature matching with foundation-model guidance.
- [Segment Anything](https://github.com/facebookresearch/segment-anything), for automatic and prompted segmentation.
- [DINOv2](https://github.com/facebookresearch/dinov2), for visual features used through OmniGlue.
- [SuperPoint](https://github.com/rpautrat/SuperPoint), for keypoint detection and descriptors used through OmniGlue.
- [igraph](https://igraph.org/), for graph representation and analysis.
- [Infomap](https://www.mapequation.org/infomap/), for graph community detection.
- [leidenalg](https://github.com/vtraag/leidenalg), for Leiden community detection.

The pinned submodules retain their own license files. Refer to the upstream repositories for their citation information and for terms applying to their code and pretrained models.

## License

Copyright © 2026 Ata Baran Fırat.

The associated master's thesis is published through Middle East Technical University's institutional repository under CC BY-NC-ND 4.0. This license applies to the thesis document, whereas the original source code in this repository is separately licensed under Apache 2.0.

Unless otherwise noted, original ObTopoMem code is licensed under the
[Apache License, Version 2.0](LICENSE).

Third-party code, dependencies, and pretrained models retain their respective licenses and are not relicensed by this repository. The SAM and OmniGlue submodules include their own Apache-2.0 license files.

The current pipeline also depends on GPL-licensed libraries: igraph
(GPL-2.0-or-later) and leidenalg (GPL-3.0 or later). Apache-2.0 is compatible with GPL-3.0,
but it does not replace those libraries' terms. Distribution of a combined work
covered by the GPL must comply with the applicable GPL requirements, including
corresponding-source obligations, as well as preserving applicable upstream
notices. The complete dependency stack should therefore not be treated as
licensed solely under Apache-2.0.
