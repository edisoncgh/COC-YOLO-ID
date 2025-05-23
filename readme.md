# COC-YOLO-ID Open Source Repo

COC-YOLO-ID is a Data-Model Synergistic Method for Tiny Target Detection in Insulator Defects. Currently, this work has been submitted to ICONIP'25.

### Abstract
Utilizing object detection technology based on UAV filming images is currently the most commonly used method for inspecting insulator defects in transmission lines. However, the training effects of UAV filming images suffer from the tiny target size and complex image backgrounds. To address these challenges, we design a cropping-based image data augmentation algorithm with the COC(Corner Optimal Cover) strategy for high-resolution UAV-filming images. Moreover, we proposed a lightweight, improved YOLO11n model to suit the UAV-based detection scenarios. This model introduces multiple convolution structures and mixed attention mechanisms, which achieve light-weight while maintaining a relatively higher detection accuracy. Our experiments on a real-world inspection image dataset show that our augmentation algorithm can truly enhance the model's ability to perceive tiny targets. Furthermore, our improved model has a size of 2.10M parameters and the computational load is 5~25% lower than the comparison models. Meanwhile, the mAP of this model achieved 90.1%, 10.4% higher than the baseline model. 

### Content
- `COC_Augmentation`
  - `COC_algo.py`
    A cropping & quantity-expansion augmentation algorithm with the COC(Corner Optimal Cover) strategy for tiny targets in high-resolution images.
  - `cropping.py`
    The implementation for executing the algorithm.
- `MCS-YOLO.yaml`
  The structure description of MCS-YOLO model.
- `DC-C3k2.py`
  An improved YOLOv11 C3k2 module: DC-C3k2 and its structural foundation DualConv module.
- `L-GSConv.py`
  A lightened GSConv module and its implementation for embedding into our model. The original repo page: https://github.com/AlanLi1997/slim-neck-by-gsconv
- `MLCA.py`
  The implementation for embedding MLCA into our model. The original repo page: https://github.com/wandahangFY/MLCA