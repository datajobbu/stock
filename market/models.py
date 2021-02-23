from django.db import models


class Stocks(models.Model):
    name = models.CharField(max_length=20)
    code = models.CharField(max_length=10)