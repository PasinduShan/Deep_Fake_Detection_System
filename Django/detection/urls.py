# detection/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('i_result/', views.i_result, name='i_result'),
    path('p_image/', views.p_image, name='p_image'),
    path('predict/', views.predict_image, name='predict_image'),
    path('test/', views.test, name='test'),
    path('p_video/', views.p_video, name='p_video'),
    path('p_voice/', views.p_voice, name='p_voice'),

    path('predict_voice/', views.predict_voice, name='predict_voice'),
    path('vo_result/', views.vo_result, name='vo_result'),
    path('predict_video/', views.predict_video, name='predict_video'),
    path('p_super/', name='p_super', view=views.p_super),
]
