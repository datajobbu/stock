from django.shortcuts import render
from django.http import HttpResponse

from .models import Price

import json

def index(request):
    prices = Price.objects.order_by('-date')[:30].values()
    
    date_list = []
    high_list = []
    low_list = []
    for i in range(len(prices)):
        high_list.append(prices[i]['high'])
        low_list.append(prices[i]['low'])
        date_list.append(prices[i]['date'].isoformat())

    high_list.reverse()
    low_list.reverse()
    date_list.reverse()

    context = {
        'prices': prices,
        'highs': high_list,
        'lows': low_list,
        'dates': json.dumps(date_list),
    }

    return render(request, 'market/home.html', context)