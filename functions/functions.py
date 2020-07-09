
def resize_face_square(image,face_size,bg_coler):
    old_size = image.shape[:2]
    ratio = float(face_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = face_size - new_size[1]
    delta_h = face_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=bg_coler)
    return image

import cv2
from PIL import Image, ImageDraw, ImageFont
def draw_rec(img,face_data,class_index,class_name,class_confidence):
    rec_color = (0, 0, 255) if class_index==0 else (255, 0, 0)
    x,y,w,h = face_data
    img = cv2.rectangle(img, (x,y), (x+w,y+h), color=rec_color, thickness=2)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("simhei.ttf", 15, encoding="utf-8")
    draw.text((x,y-20), class_name, rec_color, font=font)
    draw.text((x,y+h+10), 'Conf:%s' % class_confidence, rec_color, font=font)
    return np.array(img)



import numpy as np
def predict_small(img,model,face_data,config):
    x,y,w,h = face_data.get('box')
    cropImg = img[int(y):int(y+h), int(x):int(x+h)]
    face_size = config.get('face_size') if config.get('face_size') is not None else 128
    normalImg = resize_face_square(cropImg,face_size,(0,0,0))
    preds = model.predict(np.expand_dims(normalImg/255, axis=0))[0]
    class_index = preds.argmax()
    class_name = {0:'有口罩',1:'无口罩'}.get(class_index)
    class_confidence = '%.2f' % round(preds[class_index],2)
    if preds[class_index] >0.8:
        img = draw_rec(img,(x,y,w,h),class_index,class_name,class_confidence)
    return img,class_index