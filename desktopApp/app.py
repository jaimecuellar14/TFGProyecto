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

from samples.balloon import balloon
from mrcnn import config

MODEL_DIR = "./Modelo/tfg_notebook/main_1/Mask_RCNN-master/logs"
DEVICE = "cpu/:0"

config = balloon.BalloonConfig()
class InferenceConfig(config.__class__):
			GPU_COUNT=1
			IMAGES_PER_GPU=1

config = InferenceConfig()
config.display()
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

		predictButton = wx.Button(parent=self, label = "Realizar prediccion", pos=(20,80))
		predictButton.Bind(event=wx.EVT_BUTTON, handler=self.predecir)

	def predecir(self, event):
		print("Funciono boton")
		#config = balloon.BalloonConfig()
		#config = InferenceConfig()
		#config.display()
		DATA_DIR = "./Modelo/tfg_notebook/main_1/dataset"
		BALLOON_DIR = DATA_DIR
		dataset = balloon.BalloonDataset()
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

		image = skimage.io.imread("./images/estadio-24.JPG")
		plt.figure(figsize=(12,10))
		skimage.io.imshow(image)
		plt.show()

		result = model.detect([image],verbose=1)
		r = result[0]
		visualize.display_instances(image, r['rois'],r['masks'],r['class_ids'],dataset.class_names, r['scores'])

if __name__ == "__main__":
	app = MyApp()
	app.MainLoop()