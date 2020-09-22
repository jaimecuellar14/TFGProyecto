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

ROOT_DIR = os.path.abspath("./Modelo/tfg_notebook/main/Mask_RCNN-master")
sys.path.append(ROOT_DIR)


from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.balloon import estadios
from samples.africa import africa
from mrcnn import config

MODEL_DIR = "./Modelo/tfg_notebook/main_1/Mask_RCNN-master/logs"
MODEL_AFRICA_DIR = "./Modelo/tfg_notebook/main/Mask_RCNN-master/logs"
DEVICE = "cpu/:0"

config = estadios.BalloonConfig()
config_africa = africa.BalloonConfig()
class InferenceConfig(config.__class__):
			GPU_COUNT=1
			IMAGES_PER_GPU=1

config = InferenceConfig()
config.display()

class InferenceConfigAfrica(config_africa.__class__):
			GPU_COUNT=1
			IMAGES_PER_GPU=1
		
config_africa = InferenceConfigAfrica()
config_africa.display()

class MyApp(wx.App):
	def __init__(self):
		super().__init__(clearSigInt=True)

		self.InitFrame()

	def InitFrame(self):
		frame = Frame(parent=None, title="TFG - Jaime Cuéllar", pos=(100,100))
		frame.Show()


class Frame(wx.Frame):
	def __init__(self,parent,title,pos=(100,100)):
		super().__init__(parent=parent,title=title,pos=pos)
		self.OnInit()
	def OnInit(self):
		panel = Panel(parent=self)

class Panel(wx.Panel):
	def __init__(self,parent):
		super().__init__(parent=parent)

		welcomeText = wx.StaticText(self, id=wx.ID_ANY,label="Modelo de reconocimiento de imagenes aereass", pos=(20,20))

		predictButton = wx.Button(parent=self, label = "Realizar prediccion (hospitales,estadios)", pos=(20,80))
		predictButton.Bind(event=wx.EVT_BUTTON, handler=self.predecir)

		predictButtonAfrica = wx.Button(parent=self, label="Prediccion zonas África",pos=(20,120))
		predictButtonAfrica.Bind(event=wx.EVT_BUTTON,handler=self.predecirAfrica)

	def predecirAfrica(self,event):
		print("predecir Africa")
		config_africa = africa.BalloonConfig()
		config_africa = InferenceConfigAfrica()
		config_africa.display()
		DATA_DIR = "./Modelo/tfg_notebook/main/dataset"
		BALLOON_DIR = DATA_DIR
		dataset_africa = africa.BalloonDataset()

		with wx.FileDialog(self, "Selecciona una imagen para predecir su clase", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST ) as fileDialog:
			if fileDialog.ShowModal() == wx.ID_CANCEL:
				return
			pathname = fileDialog.GetPath()
			try:
				with open(pathname,'r') as file:
					print(pathname)
					dataset_africa.load_balloon(BALLOON_DIR,"val")
					
					dataset_africa.prepare()
					print("Images: {}\nClasses: {}".format(len(dataset_africa.image_ids), dataset_africa.class_names))
					config_africa = InferenceConfigAfrica()
					
					with tf.device(DEVICE):
						model = modellib.MaskRCNN(mode="inference", model_dir = MODEL_AFRICA_DIR, config=config_africa)

					weights_path = model.find_last()
					
					model.load_weights(weights_path, by_name=True)
					image = skimage.io.imread(pathname)
					plt.figure(figsize=(12,10))
					skimage.io.imshow(image)
					plt.show()

					result = model.detect([image],verbose=1)
					r = result[0]
					visualize.display_instances(image, r['rois'],r['masks'],r['class_ids'],dataset_africa.class_names, r['scores'])
					
			except IOError:
				wx.LogError("ERROR")

	def predecir(self, event):
		print("Funciono boton")
		config = estadios.BalloonConfig()
		config = InferenceConfig()
		config.display()
		DATA_DIR = "./Modelo/tfg_notebook/main_1/dataset"
		BALLOON_DIR = DATA_DIR
		dataset = estadios.BalloonDataset()
		with wx.FileDialog(self,"Selecciona una imagen para predecir su clase" , style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
			if fileDialog.ShowModal() == wx.ID_CANCEL:
				return
			pathname = fileDialog.GetPath()
			try:
				with open(pathname, 'r') as file:
					print(pathname)
					dataset.load_balloon(BALLOON_DIR,"val")
					dataset.prepare()
					print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
					config = InferenceConfig()
					
					with tf.device(DEVICE):
						model = modellib.MaskRCNN(mode="inference", model_dir = MODEL_DIR, config=config)
					

					weights_path = model.find_last()
					print("Loading weights ", weights_path)
					model.load_weights(weights_path, by_name=True)

					#image = skimage.io.imread("./Modelo/tfg_notebook/main_1/dataset/val/estadio-25.JPG")

					image = skimage.io.imread(pathname)
					plt.figure(figsize=(12,10))
					skimage.io.imshow(image)
					plt.show()

					result = model.detect([image],verbose=1)
					r = result[0]
					visualize.display_instances(image, r['rois'],r['masks'],r['class_ids'],dataset.class_names, r['scores'])
			except IOError:
				wx.LogError("No se pudo abrir el archivo")
		
			
		
if __name__ == "__main__":
	app = MyApp()
	app.MainLoop()