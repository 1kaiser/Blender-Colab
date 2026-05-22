# Blender-Colab ☁️

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/blender-colab/blob/master/BlenderColab.ipynb)

> 🥳 Thanks to the Google Colaboratory team for making cloud GPU rendering possible!  
> [Colaboratory Release Notes](https://colab.research.google.com/notebooks/relnotes.ipynb)

Render Blender 3.0.1 scenes (with 2.9.x support) on a free GPU using Google Colab.  
The blend file, rendered frames, and FLIP Fluids addon are all hosted on the
[v1.0.0 GitHub Release](https://github.com/1kaiser/Blender-Colab/releases/tag/v1.0.0) —
no manual Google Drive upload required to get started.

---

## Approach

![Pipeline diagram](https://user-images.githubusercontent.com/26379748/154967374-47a122f3-43e1-4bd8-92ef-51b130253567.png)

---

## Release Downloads (v1.0.0)

| Asset | Description | Size | Download |
|---|---|---|---|
| `particles.blend` | Blender 3.0.1 particle / FLIP fluid scene | 12 MB | [⬇ Download](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/particles.blend) |
| `blenderoutput_frames.zip` | 25 rendered frames (frame 0000–0024) | 81 MB | [⬇ Download](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/blenderoutput_frames.zip) |
| `flip_fluids_addon.zip` | FLIP Fluids addon for Blender | — | [⬇ Download](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/flip_fluids_addon.zip) |
| `multiview_april_sun.png` | 4-view April sun path render (combined) | 823 KB | [View](https://github.com/1kaiser/Blender-Colab/blob/master/assets/multiview_april_sun.png) |

---

## Multi-View Render — April Sun Path

The notebook includes a multi-view rendering cell that renders the scene from
**four orthographic camera angles** with the sun placed at an **April noon** solar position:

| Parameter | Value |
|---|---|
| Date | April 15 |
| Latitude | 45 °N |
| Solar elevation | 55 ° |
| Azimuth | 180 ° (due south — noon) |

### Combined view (all four in one row)

![Multi-view April sun path](https://raw.githubusercontent.com/1kaiser/Blender-Colab/master/assets/multiview_april_sun.png)

### Individual views

| Top View | Front View |
|---|---|
| ![Top](https://raw.githubusercontent.com/1kaiser/Blender-Colab/master/assets/view_top.png) | ![Front](https://raw.githubusercontent.com/1kaiser/Blender-Colab/master/assets/view_front.png) |

| Side View | Isometric View |
|---|---|
| ![Side](https://raw.githubusercontent.com/1kaiser/Blender-Colab/master/assets/view_side.png) | ![Isometric](https://raw.githubusercontent.com/1kaiser/Blender-Colab/master/assets/view_isometric.png) |

### Camera settings used

| View | Location | Rotation | Ortho scale |
|---|---|---|---|
| Top | (0.13, 0.39, 20) | 0°, 0°, 0° | 14 |
| Front | (0.13, scene−20, 1.27) | 90°, 0°, 0° | 8 |
| Side | (scene+20, 0.39, 1.27) | 90°, 0°, 90° | 13 |
| Isometric | (12, −12, 9) offset | 60°, 0°, 45° | 14 |

Camera centres are derived from the actual scene mesh bounds (Z centre = 1.27).

---

## Usage

### Input — blend file

The notebook downloads `particles.blend` from the GitHub release via `wget`.
Change `blend_file_url` in the config cell to use your own file:

```python
# Default — example particles/FLIP fluid scene from the release
blend_file_url = 'https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/particles.blend'

# Custom — upload to Google Drive and use the direct download URL
blend_file_url = 'https://drive.google.com/uc?id=<your_file_id>&export=download'
```

### Output — rendered frames

Rendered frames are uploaded to a Google Drive folder during execution.
Set `output_directory_id` to your Drive folder ID:

```
https://drive.google.com/drive/folders/<your_folder_id>
                                        ↑ use this as output_directory_id
```

### Output stages

Each frame renders as a PNG named by frame number:

```
blenderoutput/
  blender-0000.png   (~3.1 MB, frame 1)
  blender-0001.png
  ...
  blender-0024.png   (frame 25)
```

Rendered with Cycles on GPU (CUDA/OpenCL), uploaded to Google Drive via `gshell`
after each frame batch.

### FLIP Fluids Addon

The FLIP Fluids addon is installed automatically from the release:

```python
addon_url = 'https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/flip_fluids_addon.zip'
```

---

## Existing output preview

A sample frame from the rendered particle simulation:

![Sample output](https://user-images.githubusercontent.com/26379748/155858106-f984d774-cc2e-4da3-a2a3-5d41ec1b6e7c.png)

---

## A few notes

1. You must own a Google account for Drive output uploads.
2. One notebook can run for a maximum of 12 hours (24 hours with Colab Pro).
3. EEVEE rendering is not supported in a virtual machine.
4. Do note that GPU access may be limited or blocked after many hours of rendering.
5. This script is intended for those without access to a high-end GPU. Please use responsibly.

---

## FAQ

### An error occurred!
Check which cell failed and identify the error. Re-run the cell with the play button, or go to `Runtime > Restart and run all`. If the issue persists, open a GitHub issue with the error log and your setup details.

Common errors:
- `MessageError: TypeError: Failed to fetch` while downloading — keep the Colab tab open during frame download.

---

## Credits

- **blender-colab skeleton** adapted from [ynshung/blender-colab](https://github.com/ynshung/blender-colab)
- **gshell** library from [wkentaro/gshell](https://github.com/wkentaro/gshell) — navigate Google Drive like a shell
- **google-drive-to-sqlite** from [simonw/google-drive-to-sqlite](https://github.com/simonw/google-drive-to-sqlite) — SQLite metadata from Drive

---

## Disclaimer

Google Colab is optimised for data science and neural network workloads, not 3D rendering.
Free-tier usage limits, idle timeouts, and GPU speed vary.
[Colab Pro](https://colab.research.google.com/signup) offers longer sessions and faster GPUs.
See the [FAQ](https://research.google.com/colaboratory/faq.html) for details.
