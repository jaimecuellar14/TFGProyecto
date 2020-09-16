import wx
import webbrowser
import skimage.io
import os
import sys
import random
import math
import re
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MyApp(wx.App):
	def __init__(self):
		super().__init__(clearSignInt=True)

		self.InitFrame()

	def InitFrame(self):
		frame = Frame(parent=None, title="TFG - Jaime Cuéllar", pos=(100,100))
		frame.Show()


class Frame(wx.Frame):
	def __init__(self,parent,title,pos=pos):
		super().__init__(parent=parent,title=title,pos=pos)
		self.OnInit()
	def OnInit(self):
		panel = Panel(parent=self)

class Panel(wx.Panel):
	def __init__(self,parent):
		super().__init__(parent=parent)

		welcomeText = wx.StaticText(self, id=wx.ID_ANY,label="Modelo de reconocimiento de imagenes aeres", pos=(20,20))

		predictButton = wx.Button(parent=self, label = "Realizar prediccion", pos=(20,80))
		predictButton.Bind(event=wx.EVT_BUTTON, handler=self.predecir)

	def predecir(self, event):
		print("Funciono boton")
		image = skimage.imread("./images/estadio-25.JPG")
		plt.figure(figsize=(12,10))
		skimage.io.imshow(image)

if __name__ == "__main__":
	app = MyApp()
	app.MainLoop()