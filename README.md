# DriveAssist

A system that uses computer vision to alert the driver when it detects traffic lights or road signs, helping them stay attentive and drive safely.

## Inspiration

I wanted to create something with machine learning that would be genuinely useful in my day-to-day life. One of my relatives, who always recommends good project ideas to me, suggested that I build a system that alerts the driver when it detects road signs. I loved this idea and expanded upon it by also including traffic lights.

I will continue expanding on this project by adding support for new types of road signs in the future.

## Features

-   YOLO model for detecting traffic lights and road signs
-   EasyOCR integration to extract speed values from speed limit signs
-   Flexible input options: supports images, video file, and live camera feed
-   Pygame integration for real-time audio output
-   Custom logic for reducing false positive predictions from the model

## Tech Stack

**Languages:** Python

**Libraries & Tools:** Ultralytics (YOLO), EasyOCR, OpenCV, Pygame, NumPy

## Demo

https://drive.google.com/file/d/1akh2-6TfXNgXhZpitiBZOvNwW9qbtRs7/view?usp=drive_link

## Deployment

1. Download the trained model:

    https://drive.google.com/file/d/1O77gxL7PurmNavvM7wI9FtN3UyhwLdOh/view?usp=drive_link

2. Fork this repository.

3. Clone the repository:

    ```bash
    git clone https://github.com/<your_username>/DriveAssist
    ```

4. Navigate to the directory:

    ```bash
    cd DriveAssist
    ```

5. Create a virtual environment:

    ```bash
    python -m venv <env_name>
    ```

6. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

7. Run inference:

    - Single Image:
        ```bash
        python main.py --model <path/to/model> --source <path/to/image>
        ```
    - Image Folder:
        ```bash
        python main.py --model <path/to/model> --source <path/to/folder>
        ```
    - Video File:
        ```bash
        python main.py --model <path/to/model> --source <path/to/video/file>
        ```
    - Camera Feed:

        You need the index of the connected USB camera (e.g. `usb0`).

        ```bash
        python main.py --model <path/to/model> --source <index_of_usb_camera>
        ```

## Train Your Own YOLO Model

To train your own model, you can use the following notebook: https://colab.research.google.com/drive/1lJACvS_HwbCpCmPS5d_K2_c1bcA_c3JR?usp=drive_link

## Acknowledgements

-   [How to train a YOLO model - Edje Electronics](https://www.youtube.com/watch?v=r0RspiLG260)
