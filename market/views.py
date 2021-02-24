from django.shortcuts import render
from django.http import HttpResponse

from .models import Stocks


def index(request):
    stocks = Stocks.objects.all().values()

    context = {
        'stocks': stocks,
    }

    return render(request, 'market/home.html', context)

    
