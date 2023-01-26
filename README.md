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
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.0-218-g7539cd7
pip install -r requirements.txt
cd ..\license-plate-recognition
pip install -r requirements.txt
```


## Testing

python main.py
