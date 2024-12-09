
# detection/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('result.html', views.result, name='result'),  # Map root URL to homepage
    path('predict/', views.predict_image, name='predict_image'),
    path('test/', views.test, name='test'),
]
