"""
URL configuration for imood_scanner project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from api.views import index, sign_up, sign_in, log_out, inference, predict, grad_cam, clean_va_fig

urlpatterns = [
    path('', index, name='Index'),
    path('register', sign_up, name='Register'),
    path('login', sign_in, name='Login'),
    path('logout', log_out, name='Logout'),
    path('inference', inference, name="Inference"),
    path('admin/', admin.site.urls),
    # path('api/update_va_fig', update_va_fig),
    path('api/clean_va_fig', clean_va_fig),
    path('api/predict', predict),
    path('api/grad_cam', grad_cam),
]
