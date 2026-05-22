# Blender-Colab☁️
# ✅<a href="https://colab.research.google.com/github/1kaiser/blender-colab/blob/master/BlenderColab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 🥳🥳👏 Thanks To Google Colaboratory Team for making this possible 🥳🥳🥳 ! [Colaboratory Release Notes](https://colab.research.google.com/notebooks/relnotes.ipynb)

This allows you to 😎Render Blender 3.0.1 with 2.9.--- 🙂 supported scene using ☁️Google Colaboratory runtime selected as GPU.
The blend file and rendered output frames are now hosted directly on the [GitHub Release](https://github.com/1kaiser/Blender-Colab/releases/tag/v1.0.0) — no Google Drive upload required to get started.
This script provides basic functionality so you may modify the script to your liking to suit your needs.

### Approach
![abstract](https://user-images.githubusercontent.com/26379748/154967374-47a122f3-43e1-4bd8-92ef-51b130253567.png)

---

## Release Downloads (v1.0.0)

All assets are available at the [v1.0.0 release](https://github.com/1kaiser/Blender-Colab/releases/tag/v1.0.0):

| Asset | Description | Size | Download |
|---|---|---|---|
| `particles.blend` | Blender 3.0.1 particle simulation scene | 12 MB | [Download](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/particles.blend) |
| `blenderoutput_frames.zip` | 25 rendered frames (blender-0000.png → blender-0024.png) | 81 MB | [Download](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/blenderoutput_frames.zip) |
| `flip_fluids_addon.zip` | FLIP Fluids addon for Blender | — | [Download](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/flip_fluids_addon.zip) |

---

## Usage

### Input — blend file

The notebook now downloads `particles.blend` directly from the GitHub release using `wget`.
Set `blend_file_url` in the config cell to point to your own blend file if needed:

```python
# Default — uses the example particles.blend from the release
blend_file_url = 'https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/particles.blend'

# Custom — upload your file to Google Drive and use its sharing URL, or any public URL
blend_file_url = 'https://drive.google.com/uc?id=<your_file_id>&export=download'
```

### Output — rendered frames

Rendered frames are uploaded to a Google Drive folder during execution.
Set `output_directory_id` to your own Drive folder ID:

```
https://drive.google.com/drive/folders/<your_folder_id>
                                        ↑ use this as output_directory_id
```

A pre-rendered example output (25 frames of the particles scene) is available
as [`blenderoutput_frames.zip`](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/blenderoutput_frames.zip).

### Output stages

The render pipeline produces one PNG frame per Blender frame number:

```
blender-0000.png  →  frame 1    (~3.1 MB each)
blender-0001.png  →  frame 2
...
blender-0024.png  →  frame 25
```

Each frame is rendered with Cycles on GPU (CUDA/OpenCL), then uploaded
directly to the configured Google Drive output folder via `gshell`.

### FLIP Fluids Addon

Download the FLIP Fluids addon from the release and install it into Blender:

```python
addon_url = 'https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/flip_fluids_addon.zip'
```

The notebook installs and enables it automatically via `bpy.ops.preferences.addon_install`.

---

## A few notes

1. You must own a Google account for Drive output uploads.
2. One notebook can only run for maximum time of 12 hours (24 hours for Google Colab Pro) but not guaranteed.
3. EEVEE rendering is not supported in a virtual machine.
4. This script is not tested fully yet. Expect some errors.
5. Do note that your access to GPU may be limited or blocked if you render for many hours.
6. This script is intended for those who have no access to high-end GPU for rendering. Please use them responsibly!

---

## FAQ

### An error occurred!
Check which section of the code failed and identify the error (such as misspelled files or path). If you don't understand the error, try re-running the code with the play button at the side. If it still fails, go to `Runtime > Restart and run all` to restart the code or try `Runtime > Factory reset runtime`. If all else fails, open an issue in GitHub with the error log you encountered attached and the details of your setup.

Common errors:
* `MessageError: TypeError: Failed to fetch` while downloading: The tab must be opened so that the frames can be downloaded.

---

## Credits

### The "blender-colab" code skeleton (./blender-colab folder) was adapted from the [ynshung/blender-colab][1] repository.
[1]: https://github.com/ynshung/blender-colab \
    <a href="https://colab.research.google.com/github/ynshung/blender-colab/blob/master/blender_render.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### The "Gshell" library was used from [wkentaro/gshell][2] repository.
[2]: https://github.com/wkentaro/gshell
    gshell = Google Drive + Shell >>> Navigate in Google Drive as you do on shell.

### The "google-drive-to-sqlite" library was used from [simonw/google-drive-to-sqlite][3] repository.
[3]: https://github.com/simonw/google-drive-to-sqlite
    Create a SQLite database containing metadata from Google Drive.

![Screenshot 2022-02-27 003233](https://user-images.githubusercontent.com/26379748/155858106-f984d774-cc2e-4da3-a2a3-5d41ec1b6e7c.png)

---

## Disclaimer
Google Colab is specialized for data centres, neural network etc, not rendering 3D scenes. Because the computing power provided are free, the usage limits, idle timeouts and speed of the rendering may vary. [ColabPro](https://colab.research.google.com/signup) is available for those who want a more powerful GPU and longer session for rendering. See the [FAQ](https://research.google.com/colaboratory/faq.html) for more info.
