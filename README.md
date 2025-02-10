# YOLO: Re-Implementation of  YOLOv9, YOLOv7, YOLO-RD with MIT liscence

This repository will contains the complete codebase, pre-trained models, and  instructions for training and deploying YOLOv9 on small datasets.

## TL;DR

- This is the official YOLO model re-implementation with an MIT License.
- For quick deployment: you can directly install by pip+git:

```shell
pip install git+https://jichengyuan/YOLO-with-MIT-License.git
yolo task.data.source=0 # source could be a single file, video, image folder, webcam ID
```

## Quick-Start

````
## Data Preparation
The final total data structure is shown like this:
```bash
YOLO
├── data
│   ├── custData
│   │   ├── a.jpg
│   │   ├── b.jpg
│   │   ├── c.jpg
│   │   ├── d.jpg
│   │   ├── annotation.json
```
````

To get started using YOLOv9's developer mode, we recommand you clone this repository and install the required dependencies:

```shell
git clone git@github.com:WongKinYiu/YOLO.git
cd YOLO
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

[Tutorial with for training and inference with Juputernotebook](tutorials_detection_yolo.ipynb)

## Citations

```
@inproceedings{wang2022yolov7,
      title={{YOLOv7}: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors},
      author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
      year={2023},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},

}
@inproceedings{wang2024yolov9,
      title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
      year={2024},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
}
@inproceedings{tsui2024yolord,
      author={Tsui, Hao-Tang and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
      title={{YOLO-RD}: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary},
      booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2025},
}

@article{mmdetection,
  title   = {{YOLO}: Official Implementation of YOLOv9, YOLOv7, YOLO-RD},
  author  = {MultimediaTechLab},
  journal= {https://github.com/MultimediaTechLab/YOLO},
  year={2024}
}
```
