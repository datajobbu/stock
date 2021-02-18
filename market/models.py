from django.db import models

# Create your models here.
class Price(models.Model):
    date = models.DateField()
    start = models.IntegerField()
    high = models.IntegerField()
    low = models.IntegerField()
    end = models.IntegerField()
    volume = models.IntegerField()