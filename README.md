# License plate recognition

Решение тестового задания для компании ZeBrains.
Система распознавания номеров машин.

Реализована как предобученный детектор на базе yolov5 + распознавание символов через tesseract LSTM.
Скрипт написан под windows.

## Installation
needed:
python >= 3.6.8
tesseract >= 4.0

```
git clone https://github.com/DaniilProgMath/license-plate-recognition.git
cd license-plate-recognition
pip install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd ..
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.0-218-g7539cd7
pip install -r requirements.txt
```


## Testing
```
python main.py --image-path \some_image_path\image.jpg
```

## Work Examples

![Image alt](https://github.com/DaniilProgMath/license-plate-recognition/raw/develop/work_examples/img_3.jpg)
![Image alt](https://github.com/DaniilProgMath/license-plate-recognition/raw/develop/work_examples/img_10.jpg)
![Image alt](https://github.com/DaniilProgMath/license-plate-recognition/raw/develop/work_examples/img_5.jpg)
