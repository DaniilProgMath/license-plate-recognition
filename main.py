import os
import cv2
import torch
import argparse
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

# Объявление пути до ядра tesseract
pytesseract.pytesseract.tesseract_cmd = \
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_car_plate_detection_model(ckpt_path=r'weights\best.pt'):
    """
    Загрузка модели и её весов yolov5
    """
    yolov5 = torch.hub.load('..\yolov5',
                            'custom',
                            path=ckpt_path,
                            source='local',
                            force_reload=True)
    return yolov5


def run_yolo_plate_detection(img):
    """
    Инференс детектора
    """
    model = load_car_plate_detection_model()
    pred = model(img, size=1280, augment=False)
    bbox_objects = list()
    for i, row in pred.pandas().xyxy[0].iterrows():
        if row['confidence'] < model.conf: break
        bbox_objects.append(row.to_dict())
    return bbox_objects


def filter_wrong_char(tesseract_predicted_text):
    """
    Очистка результата распознования номера от символов
    которых там быть не должно.
    """
    car_key_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    car_key_symbols += "АВЕКМНОРСТУХ"
    filtered_text = ""
    for sb in tesseract_predicted_text.upper():
        if sb in car_key_symbols:
            filtered_text += sb
    return filtered_text


def run_symbol_recognition(img, car_plate_bboxes):
    """
    Применение tesseract для распознавания символов.
    """
    for i, bbox in enumerate(car_plate_bboxes):
        img_cropped = img[int(bbox['ymin']):int(bbox['ymax']),
                      int(bbox['xmin']):int(bbox['xmax'])]
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        tesseract_predicted_text = pytesseract.image_to_string(blackhat,
                                        lang="eng+rus", config="--oem 1")
        car_plate_bboxes[i]["text"] = filter_wrong_char(
            tesseract_predicted_text)
    return car_plate_bboxes


def draw_bboxes_and_plate_number(image, car_plate_data):
    """
        Отрисовка результатов детекции и распознаных символов на изображении.
    """
    image_with_detections = np.copy(image)
    for car_plate in car_plate_data:
        pt1 = [int(car_plate['xmin']), int(car_plate['ymin'])]
        pt2 = [int(car_plate['xmax']), int(car_plate['ymax'])]
        image_with_detections = cv2.rectangle(image_with_detections,
                                              pt1, pt2, (0, 255, 0), 3)
        image_with_detections = print_unicode_symbols_on_image(image_with_detections,
                                                               car_plate["text"], pt1)

    return image_with_detections


def print_unicode_symbols_on_image(image, text, coords):
    """
    Функция для отрисовки русских букв на изображении.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    font = ImageFont.truetype("C:\Windows\Fonts\\arial.ttf", 15)
    draw = ImageDraw.Draw(pil_image)
    coords[1] -= 10
    coords[0] += 20
    draw.text(coords, text, font=font, fill=(255, 0, 0))

    image_extended = np.asarray(pil_image)
    image_extended = cv2.cvtColor(image_extended, cv2.COLOR_RGB2BGR)
    return image_extended


def run_car_plate_recognition(img_path):
    """
    Общая функция системы.
    """
    image = cv2.imread(img_path)
    car_plate_bboxes = run_yolo_plate_detection(image)
    car_plate_bboxes = run_symbol_recognition(image, car_plate_bboxes)
    image = draw_bboxes_and_plate_number(image, car_plate_bboxes)
    img_name = img_path.split("\\")[-1]
    cv2.imwrite(os.path.join("predicted_image", img_name), image)


if not os.path.isdir("predicted_image"):
    os.mkdir("predicted_image")

parser = argparse.ArgumentParser(
                    prog = 'license-plate-recognition',
                    description = 'Решение тестового задания ' + \
                                  'для компании ZeBrains. ' + \
                                  'Система распознавания номеров машин.' + \
                    'Реализована как предобученный ' + \
                    'детектор на базе yolov5 + распознавание ' + \
                    'символов через tesseract LSTM. ' + \
                    'Скрипт написан под windows.')
parser.add_argument('--image-path')
args = parser.parse_args()

run_car_plate_recognition(args.image_path)
