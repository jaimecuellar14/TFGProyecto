from django.http import HttpResponse
from django.template import Template, Context
from django.shortcuts import render, redirect

def inicio(request):
    page = open("TFG/plantillas/inicio.html")
    plt = Template(page.read())
    page.close()
    context = Context()
    doc = plt.render(context)
    return HttpResponse(doc)

def predicciones(request):
    page = open("TFG/plantillas/predicciones.html")
    plt = Template(page.read())
    page.close()
    ctx = Context()
    doc = plt.render(ctx)

    return HttpResponse(doc)