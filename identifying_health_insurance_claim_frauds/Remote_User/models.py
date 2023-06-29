from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class insurance_claim_frauds(models.Model):

    RID= models.CharField(max_length=3000)
    Sum_Insured= models.CharField(max_length=3000)
    age= models.CharField(max_length=3000)
    sex= models.CharField(max_length=3000)
    weight= models.CharField(max_length=3000)
    bmi= models.CharField(max_length=3000)
    hereditary_diseases= models.CharField(max_length=3000)
    no_of_dependents= models.CharField(max_length=3000)
    smoker= models.CharField(max_length=3000)
    city= models.CharField(max_length=3000)
    bloodpressure= models.CharField(max_length=3000)
    diabetes= models.CharField(max_length=3000)
    regular_ex= models.CharField(max_length=3000)
    job_title= models.CharField(max_length=3000)
    claim= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



