from django.urls import path
from .views import predict_temperature

urlpatterns = [
    path('predict/', predict_temperature, name='predict_temperature')
]
