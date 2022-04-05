import os
import shutil
import cv2
 
image_address_list = []
image_label_list = []
 
with open("artist_train.csv") as fid:
    # Вызовите readline () для чтения построчно
    for image in fid.readlines():
                 image_address_list.append (image.strip (). split (",") [0]) # адрес хранения изображения
                 image_label_list.append (image.strip (). split (",") [1]) # метка изображения



 
fls = 'dataset/Artist'
fld = '/Users/jabbson/Downloads/test/my_folders/'
 
for folder, _, files in os.walk(fls):
    for file in files:
        shutil.move(os.path.join(folder, file), fld+os.path.splitext(file)[0])
