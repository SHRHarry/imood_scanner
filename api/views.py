import cv2
import time
import webbrowser
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

from api.forms import RegisterForm, LoginForm
from api.src.controller.model_controller import ModelThreadController
from api.src.utils import create_empty_canvas, insert_va_value, post2json, vis_img, _base64_to_cv2, _cv2_to_base64

MODEL_THREAD_CONTROLLER = ModelThreadController()
MODEL_THREAD_CONTROLLER.init_model()

web_url = "http://127.0.0.1:8000/"
webbrowser.open_new_tab(web_url)

@login_required(login_url="Login")
def index(request):
    return render(request, 'index.html', locals())

#sign up page
def sign_up(request):
    form = RegisterForm()
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            redirect('/login')  #重新導向到登入畫面
    context = {
        'form': form
    }
    return render(request, 'register.html', context)

# sign in page
def sign_in(request):
    form = LoginForm()
    
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')  #重新導向到首頁
    context = {
        'form': form
    }
    
    return render(request, 'login.html', context)

# log out page
def log_out(request):
    logout(request)
    return redirect('/login') #重新導向到登入畫面

# inference page
def inference(request):
    
    return render(request, 'inference.html')

@require_http_methods(["GET"])
@csrf_exempt
def clean_va_fig(request):
    try:
        if MODEL_THREAD_CONTROLLER.is_running():
            jsonres = {
                "ReturnCode": "Service is busy.",
                "Message": "Service is busy.",
                "Result": {}
                }
            return JsonResponse(jsonres, safe=False)
        
        jsonres = MODEL_THREAD_CONTROLLER.clean_va_fig()
        
        jsonres = {
            "ReturnCode": "SUCCESS",
            "Message": "SUCCESS",
            "Result": jsonres
            }

    except Exception as err:
        print(f"[ERROR] | get_results | {err}")
    
    return JsonResponse(jsonres, safe=False)

# @require_http_methods(["GET"])
# @csrf_exempt
# def update_va_fig(request):
#     try:
#         if MODEL_THREAD_CONTROLLER.is_running():
#             jsonres = {
#                 "ReturnCode": "Service is busy.",
#                 "Message": "Service is busy.",
#                 "Result": {}
#                 }
#             return JsonResponse(jsonres, safe=False)
        
#         jsonres = MODEL_THREAD_CONTROLLER.upgate_va_fig()
        
#         jsonres = {
#             "ReturnCode": "SUCCESS",
#             "Message": "SUCCESS",
#             "Result": jsonres
#             }

#     except Exception as err:
#         print(f"[ERROR] | get_results | {err}")
    
#     return JsonResponse(jsonres, safe=False)

@require_http_methods(["POST"])
@csrf_exempt
def predict(request):
    try:
        print("[API][START] predict")
        json_data = post2json(request)
        
        if MODEL_THREAD_CONTROLLER.is_running():
            jsonres = {
                "ReturnCode": "Service is busy.",
                "Message": "Service is busy.",
                "Result": {}
                }
            return JsonResponse(jsonres, safe=False)
        
        if MODEL_THREAD_CONTROLLER.face_model == None and MODEL_THREAD_CONTROLLER.model == None:
            jsonres = {
                "ReturnCode": "Models are not loaded.",
                "Message": "Models are not loaded.",
                "Result": {}
                }
            return JsonResponse(jsonres, safe=False)
        
        img_base64 = json_data["img_data"].split("data:image/jpeg;base64,")[-1]
        img = _base64_to_cv2(img_base64)
        roi_result = MODEL_THREAD_CONTROLLER.start_get_face_roi(img)
        img_face, new_x, new_y, max_size = roi_result
        
        if new_x != None:
            dets = (new_x, new_y, max_size)
        else:
            dets = None
        
        img_resize = cv2.resize(img_face, (224,224))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        # print(f"img.shape = {img.shape}")
        
        pred_result = MODEL_THREAD_CONTROLLER.start_predict(img_resize, img, dets)
        
        valence, arousal, predicted_class, duration, pred_img, va_fig = pred_result
        
        print(f"valence = {valence}, arousal = {arousal}, predicted_class = {predicted_class}, duration = {duration:.3f} sec.")
        
        json_response = {
            "valence": valence,
            "arousal": arousal,
            "predicted_class": predicted_class,
            "pred_img": pred_img,
            "va_fig": va_fig,
            "face_roi": "data:image/jpeg;base64,"+_cv2_to_base64(img_face).decode('utf-8'),
            "duration": duration
            }
        jsonres = {
            "ReturnCode": "SUCCESS",
            "Message": "SUCCESS",
            "Result": json_response
            }
    
    except Exception as err:
        print(f"[ERROR] | predict | {err}")
    
    print("[API][DONE] predict")
    return JsonResponse(jsonres)

@require_http_methods(["POST"])
@csrf_exempt
def grad_cam(request):
    try:
        print("[API][START] grad_cam")
        json_data = post2json(request)
        
        if MODEL_THREAD_CONTROLLER.is_running():
            jsonres = {
                "ReturnCode": "Service is busy.",
                "Message": "Service is busy.",
                "Result": {}
                }
            return JsonResponse(jsonres, safe=False)
        
        if MODEL_THREAD_CONTROLLER.face_model == None and MODEL_THREAD_CONTROLLER.model == None:
            jsonres = {
                "ReturnCode": "Models are not loaded.",
                "Message": "Models are not loaded.",
                "Result": {}
                }
            return JsonResponse(jsonres, safe=False)
        
        img_base64 = json_data["img_data"].split("data:image/jpeg;base64,")[-1]
        img = _base64_to_cv2(img_base64)
        roi_result = MODEL_THREAD_CONTROLLER.start_get_face_roi(img)
        img_face, new_x, new_y, max_size = roi_result
        
        img_resize = cv2.resize(img_face, (224,224))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        
        attn_result = MODEL_THREAD_CONTROLLER.start_get_attn_map(img_resize)
        duration, attn_map = attn_result
        
        print(f"duration = {duration:.3f} sec.")
        
        json_response = {
            "attention_map": attn_map,
            "duration": duration
            }
        jsonres = {
            "ReturnCode": "SUCCESS",
            "Message": "SUCCESS",
            "Result": json_response
            }
    
    except Exception as err:
        print(f"[ERROR] | grad_cam | {err}")
    
    print("[API][DONE] grad_cam")
    return JsonResponse(jsonres, safe=False)