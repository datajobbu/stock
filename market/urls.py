from django.urls import path
from django.conf.urls import url, include

from rest_framework import routers

import market.serializers
from .views import index


app_name = 'market'

router = routers.DefaultRouter()
router.register('prices', market.serializers.PriceViewSet)

urlpatterns = [
    path('', index, name='home'),

    url(r'^api/v1/', include((router.urls, 'market'), namespace='api')),
]