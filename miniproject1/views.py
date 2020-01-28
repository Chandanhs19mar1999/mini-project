from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def index(request):
    #print(request.body)
    print(request)
    return HttpResponse("hello world")
# Create your views here.