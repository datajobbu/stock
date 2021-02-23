from django.urls import path

from .views import index


app_name = 'stocks'

urlpatterns = [
    path('', index, name='overview'),
]