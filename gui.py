from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

from utils import load_model
from model import AutoEncoder
import numpy as np
from utils import denormalize, normalize
import os 
import cv2 as cv
import time


N_IMAGES = 202599


class gui():
   def __init__(self): 
      self.model = None
      self.img_path = "data/img_align_celeba_resized"

      self.tk = Tk()
      self.tk.title("Infer_image")
      self.tk.geometry("1000x600")
      #Images
      self.img1 = Canvas(self.tk, width = 256, height= 256, bg = "green")
      self.img1.place(rely = 0.5, relx = 0.2, anchor =CENTER)

      self.img2 = Canvas(self.tk, width = 256, height= 256, bg = "green")
      self.img2.place(rely = 0.5, relx = 0.8, anchor =CENTER)


      # Options
      self.options_frame = Frame(self.tk)
      self.change_path_button = Button(self.options_frame, text = "Change img path", command= self.select_img_path)
      self.label_path = Label(self.options_frame, text = self.img_path)

      self.options_frame.grid(row = 0, column = 0)
      self.change_path_button.pack(side = LEFT)
      self.label_path.pack(side = RIGHT)

      # Boutton chargement du model
      self.model_frame = Frame(self.tk)
      self.model_select = Button(self.model_frame, text = "Select a model", command = self.dir_select)
      self.model_select_label = Label(self.model_frame, text="choose an autoencoder model")

      self.model_select_label.pack(side = BOTTOM)
      self.model_select.pack()

      self.model_frame.place(relx= 0.5, rely = 0.9, anchor = CENTER )

      #Infer frame
      self.infer_frame = Frame(self.tk)
      self.infer_button = Button(self.infer_frame, text= "Infer", command = self.infer_and_show, width= 150)

      self.a_max_frame = Frame(self.infer_frame)
      self.a_min_frame = Frame(self.infer_frame)
      self.id_frame = Frame(self.infer_frame)

      self.a_min = Entry(self.a_min_frame, width = 5, justify=CENTER)
      self.a_max = Entry(self.a_max_frame, width = 5, justify=CENTER)
      self.img_id = Entry(self.id_frame, width = 10, justify=CENTER) 

      self.pb_a_max = Button(self.a_max_frame, text= "+", command = lambda: self.add_one(self.a_max,0.2))
      self.pb_a_max.grid(row = 1, column=2)
      self.pb_a_max= Button(self.a_max_frame, text= "-", command = lambda: self.add_one(self.a_max,-0.2))
      self.pb_a_max.grid(row = 1, column=0)

      self.pb_a_min = Button(self.a_min_frame, text= "+", command = lambda: self.add_one(self.a_min,0.2))
      self.pb_a_min.grid(row = 1, column=2)
      self.pb_a_min= Button(self.a_min_frame, text= "-", command = lambda: self.add_one(self.a_min,-0.2))
      self.pb_a_min.grid(row = 1, column=0)

      self.pb_img_id = Button(self.id_frame, text= "+", command = lambda: self.add_one(self.img_id,1))
      self.pb_img_id.grid(row = 1, column=2)
      self.pb_img_id= Button(self.id_frame, text= "-", command = lambda: self.add_one(self.img_id,-1))
      self.pb_img_id.grid(row = 1, column=0)

      self.reset = Button(self.infer_frame, text = "reset", command = self.reset)
      self.reset.place(relx = 0.5, rely = 0.8, anchor = CENTER)


      self.a_min.insert(0, "0")
      self.a_max.insert(0, "1")
      self.img_id.insert(0, "1")

      self.a_min_label = Label(self.a_min_frame, text = "a_min")
      self.a_max_label = Label(self.a_max_frame, text = "a_max")
      self.img_id_label = Label(self.id_frame, text = "id image") 


      self.infer_frame.place(relx=0.5, rely  = 0.5, anchor = CENTER, height = 250, width = 300)
      self.a_min_frame.place(relx = 0.2, rely = 0.5, anchor = CENTER)
      self.a_max_frame.place(relx = 0.8, rely = 0.5, anchor = CENTER)
      self.id_frame.pack(side = TOP)


      self.a_min.grid(row = 1, column = 1)
      self.a_max.grid(row = 1, column = 1)
      self.img_id.grid(row = 1, column = 1)

      self.a_min_label.grid(row = 0, column = 1)
      self.a_max_label.grid(row = 0, column = 1)
      self.img_id_label.grid(row = 0, column = 1)

      self.infer_button.pack(side = BOTTOM)

      self.tk.mainloop()


   def reset(self):
      self.a_max.delete(0,END)
      self.a_min.delete(0,END)

      self.a_max.insert(0, 1)
      self.a_min.insert(0, 0)
      self.infer_and_show()


   def add_one(self, case:Entry, n):
      mode = 'int' if abs(n) == 1  else 'float'
      id = test_id(case.get(), mode)
      if id is None:
         return
      
      case.delete(0, END)
      case.insert(0, str(id + n))
      if self.model is not None: 
         self.infer_and_show()
      

   def select_img_path(self):
      if os.path.isdir("data"):
         ini = "data"
      else:
         ini = ""
      img_path = filedialog.askdirectory(initialdir=ini, title="select image directory")
      if img_path:
         self.img_path = img_path
         self.label_path.config(text = self.img_path)




   def dir_select(self):
      """Open a gui to chose the folder of the model
      """
      if os.path.isdir("models"):
         ini = "models"
      else:
         ini = ""

      self.filename = filedialog.askdirectory(initialdir = ini, title="select Model")
      if len(self.filename) > 0:
         try:

            model = load_model(self.filename, model_type = 'ae')
            self.model = model
            attr = self.model.params.attr
            if len(attr) > 0:
               s = ' '.join(attr) 
               self.model_select_label.config(text = "attributes : " + s)

         except:
            messagebox.showinfo("No model", "No model have been loaded, please select an autoencoder folder") 
     

   def infer_and_show(self):
      if self.model == None: 
            messagebox.showinfo("No model", "No model have been loaded, please select an autoencoder folder before infering") 
            return None
      
      id = test_id(self.img_id.get(),'int')
      if id is None : 
         return None

      if id <1 or id > N_IMAGES  :
         messagebox.showinfo("Value Error", f"Veuillez choisir un nombre entre 1 (inclus) et {N_IMAGES}")
         return None
      
      a_min = test_id(self.a_min.get(), 'float')
      if a_min is None :
         return None
      
      a_max = test_id(self.a_max.get(), 'float')
      if a_max is None :
         return None
      

      image = load_image(self.img_path, id)
      image_infered =  infer(self.model, image, a_min, a_max)

      image_infered = image_infered[0]

      # Conversion en RGB
      image = cv.cvtColor(denormalize(image), cv.COLOR_BGR2RGB)
      image_infered = cv.cvtColor(image_infered, cv.COLOR_BGR2RGB)

      self.im1 =ImageTk.PhotoImage(Image.fromarray(image))
      self.im2 = ImageTk.PhotoImage(Image.fromarray(image_infered))

      self.img1.create_image(0, 0, anchor = NW, image = self.im1)
      self.img2.create_image(0, 0, anchor = NW, image = self.im2)


def test_id(n, mode = 'int'):
   try: 
      if mode =='int':
         n = int(n)
         return n
      elif mode == 'float':
         n = float(n)
         return np.round(n, 1)
   except ValueError: 
      messagebox.showinfo("TypeERror", "Entrez une valeur numérique")
      return None
      

def infer(ae: AutoEncoder, img, a_min, a_max):
   if len(img.shape) >3:
      bs = img.shape[0]
   else: 
      bs = 1
      img = np.expand_dims(img, axis = 0)


   
   y = [a_min, a_max]
   y = np.expand_dims(y, 0).astype(np.float32)
   y = np.repeat(y, bs, axis = 0)
   z = ae(img)
   decoded = denormalize(ae.decode(z, y))

   return decoded

def load_image(path, i):
   im = cv.imread(path + "/%06i.jpg" %i)

   if im.shape[0] != 256 or im.shape[1] != 256:
      im = cv.resize(im, (256,256), interpolation=cv.INTER_LANCZOS4)
   
   return normalize(im)


a = gui()
