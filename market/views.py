from django.shortcuts import render
from django.http import HttpResponse

from .models import Price

def index(request):
    prices = Price.objects.all().values()
    
    high_list = []
    low_list = []
    idx = []
    for i in range(len(prices)):
        high_list.append(prices[i]['high'])
        low_list.append(prices[i]['low'])
        idx.append(i)

    context = {
        'prices': prices,
        'highs': high_list,
        'lows': low_list,
        'idx': idx,
    }

    return render(request, 'market/home.html', context)