from django.db import models


class Price(models.Model):
    date = models.DateField()
    start = models.IntegerField()
    high = models.IntegerField()
    low = models.IntegerField()
    end = models.IntegerField()
    volume = models.IntegerField()