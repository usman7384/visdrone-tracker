# Drone object detection and tracking (VisDrone)

A small learning project: train a detector on drone images, shrink it, break it with bad images, then use it to track cars and people in video.

Everything is in two notebooks:

- `01_detector.ipynb` - train YOLOv8n, evaluate, export to ONNX (normal and INT8), time it on CPU, test with blur / noise / low light
- `02_tracking.ipynb` - track objects in video with ByteTrack and score it with TrackEval

## Results

Detector (VisDrone-DET val, 548 images):

| Model | mAP@50-95 | CPU latency (ms) |
|---|---|---|
| PyTorch | 0.170 | 52.5 |
| ONNX | 0.169 | 31.0 |
| ONNX INT8 | 0.158 | 55.0 |

INT8 was *slower* on my CPU (Ryzen 9 5900HS). Chart: `reports/accuracy_vs_latency.png`.

Image damage (mAP@50-95, clean = 0.170): heavy noise 0.032, heavy blur 0.047, heavy low light 0.126. Noise hurts most. Full table: `reports/robustness.csv`.

Tracking (VisDrone-MOT val, 7 videos):

| Class | Detector | HOTA | MOTA | IDF1 |
|---|---|---|---|---|
| car | ONNX | 44.7 | 27.2 | 50.9 |
| car | ONNX INT8 | 43.8 | 27.6 | 51.4 |
| pedestrian | ONNX | 31.5 | 19.0 | 37.9 |
| pedestrian | ONNX INT8 | 30.5 | 18.7 | 36.8 |

INT8 costs about 1 HOTA point. Pedestrians are harder than cars (small and crowded).

## Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
git clone https://github.com/JonathonLuiten/TrackEval
jupyter notebook
```

- Run `01_detector.ipynb` first. VisDrone-DET downloads itself (about 2 GB). Training took about 1.5 hours on an RTX 3060 laptop GPU.
- For `02_tracking.ipynb`, download the VisDrone-MOT **val** set (1.48 GB, link in the [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) repo) and put the zip at `data/mot/VisDrone2019-MOT-val.zip`.
- TrackEval is old and breaks on new NumPy. In its `.py` files, replace `np.float`, `np.int`, `np.bool` with `float`, `int`, `bool`.
- Install the CUDA build of torchvision, not the CPU one, or training fails with an NMS error.

## Limits

One training seed, 640 px images, 50 epochs, batch size 16, two tracked classes. Numbers are from a single run, so treat them as rough.
