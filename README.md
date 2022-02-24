# Demo Version to use kinect depth info to show a popup alert

## Requirements
- Download `yolov3.h5` pretrained model and place it in the `model_data` directory
- Install Kinect.exe library: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md
- conda create -n py38 python==3.8

***********************
Download Visual Studio Compiler MSVC 2019

First do this:
conda install tensorflow-gpu==2.5.0 keras, (By default, it installed cudatoolkit and cudnn)
////
Installed manually from PC
CUDA(cudatoolkit)==11.2.2, 
cudnn=8.1.1
///
Ref: for Tensorflow + GPU compatibility
https://www.tensorflow.org/install/source_windows

Japanese reference: 
https://qiita.com/nemutas/items/c7d9cca91a7e1bd404b6

***************************************

conda install -c conda-forge kivy py2neo opencv matplotlib pandas openpyxl pyqt pyqtwebengine pathlib
pip install googletrans==4.0.0-rc1

//kivy-deps-glew kivy-deps-sdl2 kivy-garden 


## Dependencies

`Shreesh-module`
- kivy
- py2neo
- pyKinectAzure
- tensorflow (2.3.x ???)
- keras
- yolo3 modules
- cv2
- PIL
- numpy


`Ozaki-module`
- googletrans==4.0.0-rc1
- pandas
