import cv2
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
# import matplotlib.pyplot as plt

def post2json(request):
    try:    
        content_type = request.META.get("CONTENT_TYPE")
        # print(f"content_type = {content_type}")
        body = {}
        
        if content_type == "application/x-www-form-urlencoded":
            pass
        elif content_type == "application/json":
            body_unicode = request.body.decode('utf-8')
            body = json.loads(body_unicode)
        
        return body
    
    except Exception as e:
        print(e)

###
def _cv2_to_base64(image_bgr):
    # image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    base64_str = cv2.imencode(".png",image_bgr)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str

def _base64_to_cv2(base64_str):
    img_str = base64.b64decode(base64_str)
    # np_arr = np.fromstring(img_str, np.uint8)
    np_arr = np.frombuffer(img_str, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr

def _base64_to_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image

## result is more short than cv_to_b64
def _pil_to_base64(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format = "PNG")
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def decode_image(base64_str):
    try:
        image = _base64_to_cv2(base64_str)
    except Exception as e:
        print(e)
    return image
###

def bound_coord(point, lower, upper):
    if point < lower: point = lower
    elif point > upper: point = upper
    
    return point

def get_face_roi(face_boxes, img):
    img_h, img_w, _ = img.shape
    dets = face_boxes(img)  # xmin, ymin, w, h
    
    max_prob = 0
    x = 0
    y = 0
    w = 0
    h = 0
    # print(f"dets = {dets}")
    
    if len(dets) != 0:
        for (x1, y1, x2, y2, prob) in dets:
            if max_prob < prob:
                max_prob = prob
                
                x1 = bound_coord(x1, 0, img_w)
                y1 = bound_coord(y1, 0, img_h)
                x2 = bound_coord(x2, 0, img_w)
                y2 = bound_coord(y2, 0, img_h)
                
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                # print(f"x, y, w, h, max_prob = {x}, {y}, {w}, {h}, {max_prob}")
        
        max_size = max(w, h)
        new_x = int(x + (w/2) - (max_size/2))
        new_y = int(y + (h/2) - (max_size/2))
        
        new_x = bound_coord(new_x, 0, img_w)
        new_y = bound_coord(new_y, 0, img_h)
        
        # print(f"new_x, new_y, max_size = {new_x}, {new_y}, {max_size}")
        img_face = img[new_y:new_y+max_size, new_x:new_x+max_size, :]
    else:
        img_face = img
        new_x = None
        new_y = None
        max_size = None
    return img_face, new_x, new_y, max_size

def vis_img(img, dets=None, text=None, size = 500):
    h, w, _ = img.shape
    scale = size / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if dets != None:
        (new_x, new_y, max_size) = dets
        new_x = int(scale*new_x)
        new_y = int(scale*new_y)
        max_size = int(scale*max_size)
        cv2.rectangle(img, (new_x, new_y), (new_x+max_size, new_y+max_size), (0, 255, 0), 1)
    
    if text != None:
        img[:50, :13*len(text)] = 0
        cv2.putText(img, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return img

def reshape_transform(tensor, height=14, width=14):
    return tensor.permute(0, 3, 1, 2)

def two_dimen_coord_to_cv_coord(x, y, img_size):
    cv_x_center = int(img_size/2)
    cv_y_center = int(img_size/2)
    
    return cv_x_center+int(x*(img_size/2)), cv_y_center+int(-y*(img_size/2))

def create_empty_canvas(img_size = 500):
    # class_list =   ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    # valence_list = [        0,    0.95, -0.85,       0.20,  -0.20,     -0.95,   -0.60,      -0.77]
    # arousal_list = [        0,    0.17, -0.38,       0.85,   0.87,      0.19,    0.75,       0.45]
    class_list =   ["Neutral", "Happy", "Sad", "Anger"]
    valence_list = [        0,   0.575, -0.85,   -0.63]
    arousal_list = [        0,   0.510, -0.38,   0.565]
    
    img = np.zeros([img_size, img_size, 3], np.uint8)
    cv2.arrowedLine(img,
                    two_dimen_coord_to_cv_coord(-1, 0, img_size),
                    two_dimen_coord_to_cv_coord(1, 0, img_size),
                    (0, 255, 0), 1, 0, 0, 0.03)
    cv2.putText(img,
                "Valence",
                two_dimen_coord_to_cv_coord(0.75, 0.05, img_size),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.arrowedLine(img,
                    two_dimen_coord_to_cv_coord(0, -1, img_size),
                    two_dimen_coord_to_cv_coord(0, 1, img_size),
                    (0, 255, 0), 1, 0, 0, 0.03)
    cv2.putText(img,
                "Arousal",
                two_dimen_coord_to_cv_coord(0.05, 0.95, img_size),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    for i in range(len(class_list)):
        cv_x, cv_y = two_dimen_coord_to_cv_coord(valence_list[i], arousal_list[i], img_size)
        cv2.circle(img, (cv_x,cv_y), 5, (0,0,225), -1)
        cv2.putText(img,
                    class_list[i],
                    two_dimen_coord_to_cv_coord(valence_list[i]-0.1, arousal_list[i]+0.05, img_size),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,225), 1, cv2.LINE_AA)
    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img

def insert_va_value(img, x, y, img_size = 500):
    cv2.circle(img,
               two_dimen_coord_to_cv_coord(x, y, img_size),
               3, (0,255,225), 4)
    return img

if __name__ == "__main__":
    import random
    img_size = 500
    img = create_empty_canvas(img_size)
    
    while True:
        img_draw = img.copy()
        img_draw = insert_va_value(img_draw, (1-(-1))*random.random()+(-1), (1-(-1))*random.random()+(-1), img_size = img_size)
        cv2.imshow("img_draw", img_draw)
        cv2.waitKey(10)
    
    # class_list =   ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    # valence_list = [        0,    0.95, -0.85,       0.20,  -0.20,     -0.95,   -0.60,      -0.77]
    # arousal_list = [        0,    0.17, -0.38,       0.85,   0.87,      0.19,    0.75,       0.45]
    
    # pred_v = [-0.1768, -0.3678, -0.6485, -0.6270, -0.5397, -0.3095, -0.1475]
    # pred_a = [-0.0776,  0.1839,  0.6581,  0.6111,  0.3571,  0.2857, -0.3793]
    # pred = [0., 0., 7., 7., 6., 6., 0.]
    
    # title = 'Valence/Arousal'
    # dpi = 80
    # width, height = 400, 400
    # legend_fontsize = 10
    # figsize = width / float(dpi), height / float(dpi)
    # fig = plt.figure(figsize=figsize)
    
    # y_x = np.array([i for i in range(-1, 1+1, 1)])
    # y_neg_x = np.array([-i for i in range(-1, 1+1, 1)])

    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # interval_y = 0.05
    # interval_x = 0.05
    # plt.xticks(np.arange(-1, 1 + interval_x, interval_x))
    # plt.yticks(np.arange(-1, 1 + interval_y, interval_y))
    # plt.grid()
    # plt.title(title, fontsize=20)
    # plt.xlabel('valence', fontsize=16)
    # plt.ylabel('arousal', fontsize=16)
    
    # plt.plot(y_x, [0, 0, 0])
    # plt.plot([0, 0, 0], y_x)
    # plt.plot(y_x, y_x)
    # plt.plot(y_x, y_neg_x)
    # plt.scatter(valence_list, arousal_list, c = "m", s = 50, alpha = .5, marker = "D")
    # plt.scatter(pred_v[5], pred_a[5])
    # plt.show()