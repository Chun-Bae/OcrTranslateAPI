# ocr_crop.py
import os
from PIL import Image
from surya.detection import batch_text_detection
from surya.model.detection.segformer import load_model, load_processor

def crop_images(image_path, save_dir='saves'):
    try:
        # 이미지를 불러옴
        image = Image.open(image_path)
        model, processor = load_model(), load_processor()

        # predictions is a list of dicts, one per image
        predictions = batch_text_detection([image], model, processor)

        # 저장할 디렉토리가 없으면 생성
        os.makedirs(save_dir, exist_ok=True)

        cropped_image_paths = []

        # 각 TextDetectionResult 객체에서 'bbox' 정보를 추출하고 이미지를 잘라 저장
        for i, result in enumerate(predictions):
            for j, box in enumerate(result.bboxes):
                bbox = box.bbox  # [x1, y1, x2, y2]
                # 이미지를 잘라냄
                cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                cropped_image_path = os.path.join(save_dir, f'cropped_{i}_{j}.jpg')
                cropped_image.save(cropped_image_path)
                cropped_image_paths.append(cropped_image_path)

        print("Images have been cropped and saved successfully.")
        return cropped_image_paths

    except Exception as e:
        print(f"An error occurred in crop_images: {str(e)}")
        return None
