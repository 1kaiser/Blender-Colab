# Blender-Colab☁️
# ✅<a href="https://colab.research.google.com/github/1kaiser/blender-colab/blob/master/BlenderColab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 🥳🥳👏 Thanks To Google Colaboratory Team for making this possible 🥳🥳🥳 ! [Colaboratory Release Notes](https://colab.research.google.com/notebooks/relnotes.ipynb)



This allows you to 😎Render Blender 3.0.1 with 2.9.--- 🙂 supported scene using ☁️Google Colaboratory runtime selected as GPU.
You have to upload the blender file to Google Drive to get the **file_id**. Rendered frames are transferred directly to Google Drive.
This script provides basic functionality so you may modify the script to your liking to suit your needs.

### Approach 
![abstract](https://user-images.githubusercontent.com/26379748/154967374-47a122f3-43e1-4bd8-92ef-51b130253567.png)

### Download FLIP Fluids Addon
You can now download the FLIP Fluids addon directly from the [GitHub Release](https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/flip_fluids_addon.zip).

## Usage
### Upload type
* `upload & rename` blender file ( particles.blend ) to google drive
* `blend_file_id` : Google drive File url `https://drive.google.com/file/d/14rU-w4UwEkRoslf6zGqFceu5ade7AU_O/view?usp=sharing` use **14rU-w4UwEkRoslf6zGqFceu5ade7AU_O** as ***file_id***

### Download type
* `output_directory_id` : Google drive folder url `https://drive.google.com/drive/u/0/folders/1ph2v7i3Qm8U6KVPh1DEJ1UCXRKVMYEY0` use **1ph2v7i3Qm8U6KVPh1DEJ1UCXRKVMYEY0** as ***directory_id***

### A few notes
1. You must own a Google account.
2. One notebook can only run for maximum time of 12 hours (24 hours for Google Colab Pro) but not guaranteed.
3. EEVEE rendering is not supported in a virtual machine.
4. This script is not tested fully yet. Expect some errors.
5. Do note that your access to GPU may be limited or blocked if you render for many hours.
6. This script is intended for those who have no access to high-end GPU for rendering. Please use them responsibly!

## FAQ
### An error occured!
Check which section of the code failed and identify the error (such as misspelled files or path). If you don't understand the error, try re-running the code with the play button at the side. If it still fails, go to `Runtime > Restart and run all` to restart the code or try `Runtime > Factory reset runtime`. If all else fails, open an issue in GitHub with the error log you encountered attached and the details of your setup.

Common errors:
* `MessageError: TypeError: Failed to fetch` while downloading: The tab must be opened so that the frames can be downloaded.

## Credits

### The "blender-colab" code skeleton (./blender-colab folder) was adapted from the [ynshung/blender-colab][1] repository.
[1]: https://github.com/ynshung/blender-colab \
    <a href="https://colab.research.google.com/github/ynshung/blender-colab/blob/master/blender_render.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  
### The "Gshell" library was used from [wkentaro/gshell][2] repository.
[2]: https://github.com/wkentaro/gshell 
    gshell = Google Drive + Shell >>> Navigate in Google Drive as you do on shell (gshell = Google Drive + Shell).
    
    
### The "google-drive-to-sqlite" library was used from [simonw/google-drive-to-sqlite][3] repository.
[3]: https://github.com/simonw/google-drive-to-sqlite 
    Create a SQLite database containing metadata from Google Drive.
    If you use Google Drive, and especially if you have shared drives with other people there's a good chance you have hundreds or even thousands of files that you may not be fully aware of.This tool can download metadata about those files - their names, sizes, folders, content types, permissions, creation dates and more - and store them in a SQLite database.
    This lets you use SQL to analyze your Google Drive contents, using Datasette or the SQLite command-line tool or any other SQLite database browsing software.

![Screenshot 2022-02-27 003233](https://user-images.githubusercontent.com/26379748/155858106-f984d774-cc2e-4da3-a2a3-5d41ec1b6e7c.png)


## Disclaimer
Google Colab is specialized for data centres, neural network etc, not rendering 3D scenes. Because the computing power provided are free, the usage limits, idle timeouts and speed of the rendering may varies. [ColabPro](https://colab.research.google.com/signup) is available for those who wanted to have more powerful GPU and longer session for rendering. See the [FAQ](https://research.google.com/colaboratory/faq.html) for more info.
