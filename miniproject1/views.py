from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    #print(request.body)
    return HttpResponse("hello world")
# Create your views here.