from cProfile import label
from tempfile import template
import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from turtle import onclick
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import mahotas
from pylab import imshow, show
import skimage.feature as ski_methods
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.pipeline import Pipeline
import imutils
from scipy.spatial import distance

methods = [
            'Выберите метод',
            'SIFT',
            'Haralick',
            'Peak Local Max',
            'ColorStandDev',
            'Zernike',
            'Shi_Tomasi'
        ]

def dist(arr1, arr2):
    return np.sqrt(np.sum((arr1 - arr2)**2))

IMG_SIZE = (500, 500)
LABEL_TYPE = {
    'author': 'author',
    'style': 'style',
}
class GUI:
    def __init__(self) -> None:
        self.window = Tk()
        self.window.title("Task 3")
        self.window.geometry('1000x2000')
        self.uploaded_image_button = None
        self.uploaded_image_label = None
        self.author_label = None
        self.style_label = None

    def start(self):
        self.window.mainloop()

    def open_file(self):
        global img, file_path
        my_str = tkinter.StringVar()

        my_str.set("")
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*jpg')])
        self.file_path_src = file_path
        if file_path is not None:
            self.img_resized=Image.open(file_path)
            copy_img_resized = self.img_resized.copy()
            copy_img_resized = copy_img_resized.resize(IMG_SIZE)
            img=ImageTk.PhotoImage(copy_img_resized)
            my_str.set(file_path)

            if self.author_label:
                self.author_label.destroy()
            if self.style_label:
                self.style_label.destroy()

            if self.uploaded_image_label:
                self.uploaded_image_label.destroy()
            self.uploaded_image_label = tkinter.Label(self.window,textvariable=my_str,fg='red' )
            self.uploaded_image_label.pack()

            if self.uploaded_image_button:
                self.uploaded_image_button.destroy()
            self.uploaded_image_button = Button(self.window,image=img)
            self.uploaded_image_button.pack()
            
    def addLabel(self, text, type):
        my_font1=('times', 16, 'bold')
        if type == LABEL_TYPE['author']:
            if self.author_label:
                self.author_label.destroy()
            self.author_label = Label(self.window, text=text, anchor=CENTER, width=30,font=my_font1 )
            self.author_label.pack()
        elif type == LABEL_TYPE['style']:
            if self.style_label:
                self.style_label.destroy()
            self.style_label = Label(self.window, text=text, anchor=CENTER, width=30,font=my_font1 )
            self.style_label.pack()
    
    def uploadButton(self):
        my_font1=('times', 18, 'bold')
     
        face = Label(self.window, text='Выберите фото ', anchor=CENTER, width=30,font=my_font1 )
        face.pack()
        facebtn = Button(self.window, text ='Choose File', width=20, command = lambda: gui.open_file()) 
        facebtn.pack()

    def addRecognitionArtistButton(self):
        
        style = tkinter.ttk.Style()
        style.configure("TButton", background='#96c0eb')
        btn = tkinter.ttk.Button(self.window, text='Узнать автора', style="TButton", command=self.recognitionArtist)
        btn.pack()

    def addRecognitionStyleButton(self):
        
        style = tkinter.ttk.Style()
        style.configure("TButton", background='#96c0eb')
        btn = tkinter.ttk.Button(self.window, text='Узнать стиль', style="TButton", command=self.recognitionStyle)
        btn.pack()

    def recognitionArtist(self):
        current_method = self.method.get()
        if current_method == methods[1]:
            self.SIFT('./dataset/Artist')
        elif current_method == methods[2]:
            self.Haralick('./dataset/Artist')
        elif current_method == methods[3]:
            self.hash_classifier('./dataset/Artist')
        elif current_method == methods[4]:
            self.ColorMeanStandartDev('./dataset/Artist')
        elif current_method == methods[5]:
            # res=""
            # i=24
            # while (res!= "Rembrandt"):
            #     i=i+1
            #     res=self.Zernik('./dataset/Artist',i)
            #     print(i,res)
            self.Zernik('./dataset/Artist')
        elif current_method == methods[6]:
            self.Shi_Tomasi('./dataset/Artist')
            

    def recognitionStyle(self):
        current_method = self.method.get()
        if current_method == methods[1]:
            self.SIFT('./dataset/Style')
        elif current_method == methods[2]:
            self.Haralick('./dataset/Style')
        elif current_method == methods[3]:
            self.hash_classifier('./dataset/Style')
        elif current_method == methods[4]:
            self.ColorMeanStandartDev('./dataset/Style')
        elif current_method == methods[5]:
            # res=""
            # i=1
            # while (res!= "Cubizm"):
            #     i=i+1
            #     res=self.Zernik('./dataset/Style',i)
            #     print(i,res)
            self.Zernik('./dataset/Style')
        elif current_method == methods[6]:
            self.Shi_Tomasi('./dataset/Style')

    def addDropdown(self):
        self.method = StringVar()
        self.method.set(methods[0])

        drop = OptionMenu(self.window, self.method, *methods)
        drop.pack()

    def SIFT(self, dir_name):
        print('hello')
        open_cv_image = np.array(self.img_resized) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        gray = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        bf = cv.BFMatcher()
        best = { 'name': '', 'filepath': '', 'score': 0, 'kp': [] }
        
        for _, dirs, _ in os.walk(dir_name):
            for dir in dirs:
                print(dir)
                for _, _, files in os.walk(dir_name + '/' + dir):
                    for file in files:
                        print(file)
                        temp_im = cv.imread(dir_name + '/' + dir + '/' + file, cv.COLOR_BGR2GRAY)
                        # temp_im = cv.resize(temp_im, IMG_SIZE)
                        tmp_kp, tmp_des = sift.detectAndCompute(temp_im, None)
                        matches = bf.knnMatch(des, tmp_des, k=2)
                        good = []
                        if len(matches) < best['score']:
                            continue
                        for m, n in matches:
                            if m.distance < 0.75*n.distance:
                                good.append([m])
                        if len(good) > best['score']:
                            best['score'] = len(good)
                            best['name'] = dir
                            best['filepath'] = dir_name + '/' + dir + '/' + file
                            best['kp'] = tmp_kp
                    break
                
            break

        print(best['name'])
        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + best['name'], LABEL_TYPE['author'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + best['name'], LABEL_TYPE['style'])

        img_res = cv.imread(best['filepath'])
        # img_res = cv.resize(img_res, IMG_SIZE)
        fig, axes = plt.subplots(2, 2, figsize=(8, 3), sharex=True, sharey=True)
     
        ax = axes.ravel()
        ax[0].imshow(open_cv_image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        img_kp = cv.drawKeypoints(gray, kp, open_cv_image)
        ax[1].imshow(img_kp, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Sift')

        ax[2].imshow(img_res, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('Result')

        img_res_kp = cv.drawKeypoints(img_res, kp, img_res)
        ax[3].imshow(img_res_kp, cmap=plt.cm.gray)
        ax[3].axis('off')
        ax[3].set_title('Sift')

        fig.tight_layout()

        plt.show()

        # cv.imwrite('sift_keypoints.jpg', img)
    
    def Haralick(self, dir_name):
        print("welcome")
        image = np.array(self.img_resized) 
        image_copy = image.copy()
        image_copy = image_copy[:, :, 0]
        image_copy = mahotas.gaussian_filter(image_copy, 4)
        threshed = (image_copy > image_copy.mean())
        # making is labeled image
        labeled, n = mahotas.label(threshed)
        
        # showing image
        # print("Labelled_Image.jpg")
        # imshow(labeled)
        # show()
        
        # getting haralick features
        h_feature = mahotas.features.haralick(labeled).mean(0)

        # print("Haralick Features")
        # imshow(h_feature)
        # show()


        # train_features = []
        # train_labels = []
        # for _, dirs, _ in os.walk(dir_name):
        #     for dir in dirs:
        #         for _, _, files in os.walk(dir_name + '/' + dir):
        #             for file in files:
        #                 cv_img = cv.imread(dir_name + '/' + dir + '/' + file, cv.COLOR_BGR2GRAY)
        #                 # cv_img = cv.resize(cv_img, IMG_SIZE)
        #                 tmp_img = np.array(cv_img) 
        #                 tmp_image_copy = tmp_img.copy()
        #                 tmp_image_copy = tmp_image_copy[:, :, 0]
        #                 tmp_image_copy = mahotas.gaussian_filter(tmp_image_copy, 4)
        #                 tmp_threshed = (tmp_image_copy > tmp_image_copy.mean())
        #                 # making is labeled image
        #                 tmp_labeled, tmp_n = mahotas.label(tmp_threshed)
                        
        #                 # getting haralick features
        #                 tmp_h_feature = mahotas.features.haralick(tmp_labeled).mean(0)
        #                 train_labels.append(dir)
        #                 train_features.append(tmp_h_feature)
        #             break
        #     break

    
        # clf_svm = LinearSVC(random_state=0)

        # clf_svm.fit(train_features, train_labels)

        # pickle.dump(clf_svm, open('artist_classifier_haralick.sav', 'wb'))

        if dir_name == './dataset/Artist':
            clf_svm = pickle.load(open("artist_classifier_haralick.sav", 'rb'))
        elif dir_name == './dataset/Style':
            clf_svm = pickle.load(open("style_classifier_haralick.sav", 'rb'))

        

        prediction = clf_svm.predict(h_feature.reshape(1, -1))[0]
        cv.putText(image, prediction, (20,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + prediction, LABEL_TYPE['author'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + prediction, LABEL_TYPE['style'])

       # display the output image
        img_res = cv.resize(image, IMG_SIZE)
        cv.imshow("Test_Image", img_res)
        cv.waitKey(0)

    def peak_local_max(self,dir_name):
        print("third")
        image = np.array(self.img_resized) 
        features = ski_methods.peak_local_max(image, min_distance=1)
        image_max = ndi.maximum_filter(image, size=20, mode='constant')
        

        best = { 'name': '', 'filepath': '', 'score': 0, 'feature': []}
        
        for _, dirs, _ in os.walk(dir_name):
            for dir in dirs:
                print(dir)
                for _, _, files in os.walk(dir_name + '/' + dir):
                    for file in files:
                        print(file)
                        cv_img = cv.imread(dir_name + '/' + dir + '/' + file, cv.COLOR_BGR2GRAY)
                        cv_img = cv.resize(cv_img, IMG_SIZE)
                        tmp_img = np.array(cv_img) 
                        score = 0
                        tmp_features = ski_methods.peak_local_max(cv_img,  min_distance=1)
                        if len(features) < len(tmp_features):
                            continue
                        for i in range(len(features)):
                            for j in range(len(tmp_features)):
                                if features[i][0] == tmp_features[j][0] and features[i][1] == tmp_features[j][1]:
                                    score += 1
                        print(score)
                        if (score > best['score']):
                            best['score'] = score
                            best['name'] = dir
                            best['filepath'] = dir_name + '/' + dir + '/' + file
                            best['feature'] = tmp_features
                    break
                
            break
        print(best['name'], best['filepath'])
        

        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + best['name'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + best['name'])

        img_res = cv.imread(best['filepath'])
        print(img_res)
        img_res = cv.resize(img_res, IMG_SIZE)
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 3), sharex=True, sharey=True)

        ax = axes.ravel()
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')


        ax[1].imshow(image, cmap=plt.cm.gray)
        ax[1].autoscale(False)
        ax[1].plot(features[:, 1], features[:, 0], 'r.')
        ax[1].axis('off')
        ax[1].set_title('Peak local max')

        ax[2].imshow(img_res, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('Maximum filter')

        ax[3].imshow(img_res, cmap=plt.cm.gray)
        ax[3].autoscale(False)
        ax[3].plot(best['feature'][:, 1], best['feature'][:, 0], 'r.')
        ax[3].axis('off')
        ax[3].set_title('Peak local max')

        fig.tight_layout()

        plt.show()


    def hash_classifier(self,dir_name):
        print("third")
        image = np.array(self.img_resized) 
        resized = cv.resize(image, (8,8), interpolation = cv.INTER_AREA) #Уменьшим картинку
        gray_image = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) #Переведем в черно-белый формат
        avg=gray_image.mean() #Среднее значение пикселя
        ret, threshold_image = cv.threshold(gray_image, avg, 255, 0) #Бинаризация по порогу
        

        _hash=[]
        for x in range(8):
            for y in range(8):
                val=threshold_image[x,y]
                if val==255:
                    _hash.append(1)
                else:
                    _hash.append(0)
        

        best = { 'name': '', 'filepath': '', 'score': 9999999}

        train_features = []
        train_labels = []
        for _, dirs, _ in os.walk(dir_name):
            for dir in dirs:
                for _, _, files in os.walk(dir_name + '/' + dir):
                    for file in files:
                        print(file)
                        tmp_hash = self.CalcImageHash(dir_name + '/' + dir + '/' + file)
                        # score = self.CompareHash(_hash, tmp_hash)

                        train_labels.append(dir)
                        train_features.append(tmp_hash)
                    break
            break

    
        # clf_svm = LinearSVC(C=50,random_state=42)
        # clf_svm.fit(train_features, train_labels)

        clf_svm = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=5,random_state=1, max_iter=10000))])
        clf_svm.fit(train_features, train_labels)
        

        # pickle.dump(clf_svm, open('artist_classifier_haralick.sav', 'wb'))
        prediction = clf_svm.predict(np.array(_hash).reshape(1,-1))[0]
        cv.putText(image, prediction, (20,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + prediction, LABEL_TYPE['author'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + prediction, LABEL_TYPE['style'])

       # display the output image
        cv.imshow("Test_Image", image)
        cv.waitKey(0)


    def CalcImageHash(self,FileName):
        image = cv.imread(FileName) #Прочитаем картинку
        resized = cv.resize(image, (8,8), interpolation = cv.INTER_AREA) #Уменьшим картинку
        gray_image = cv.cvtColor(resized, cv.COLOR_BGR2GRAY) #Переведем в черно-белый формат
        avg=gray_image.mean() #Среднее значение пикселя
        ret, threshold_image = cv.threshold(gray_image, avg, 255, 0) #Бинаризация по порогу
        
        #Рассчитаем хэш
        _hash=[]
        for x in range(8):
            for y in range(8):
                val=threshold_image[x,y]
                if val==255:
                    _hash.append(1)
                else:
                    _hash.append(0)
                
        return _hash

    def CompareHash(self,hash1,hash2):
        l=len(hash1)
        i=0
        count=0
        while i<l:
            if hash1[i]!=hash2[i]:
                count=count+1
            i=i+1
        return count
    
    def ColorMeanStandartDev(self, dir_name):
        open_cv_image = np.array(self.img_resized) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        (means, stds) = cv.meanStdDev(open_cv_image)
        stats = np.concatenate([means, stds]).flatten()
        best = { 'name': '', 'filepath': '', 'dist': 9999999999 }
        
        for _, dirs, _ in os.walk(dir_name):
            for dir in dirs:
                print(dir)
                for _, _, files in os.walk(dir_name + '/' + dir):
                    for file in files:
                        print(file)
                        temp_im = cv.imread(dir_name + '/' + dir + '/' + file)
                        (tmp_means, tmp_stds) = cv.meanStdDev(temp_im)
                        tmp_stats = np.concatenate([tmp_means, tmp_stds]).flatten()
                        min_dist = dist(stats, tmp_stats)
                        if min_dist < best['dist']:
                            best['dist'] = min_dist
                            best['name'] = dir
                            best['filepath'] = dir_name + '/' + dir + '/' + file
                    break
                
            break

        print(best['name'])
        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + best['name'], LABEL_TYPE['author'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + best['name'], LABEL_TYPE['style'])
    
    def Zernik(self, dir_name):
        # image = np.array(self.img_resized)
        image = cv.imread(self.file_path_src)
        image_copy = image.copy()
        # image_copy = image_copy[:, :, 0] 
        (_, ZernikeFeatures) = self.describeShapes(image_copy)
        best = { 'name': '', 'filepath': '', 'dist': 9999999999 }
        # degree = 10
        radius = 25
        # zernik_feature = mahotas.features.zernike_moments(image_copy, radius, degree=8)

        train_features = []
        train_labels = []
        for _, dirs, _ in os.walk(dir_name):
            for dir in dirs:
                
                for _, _, files in os.walk(dir_name + '/' + dir):
                    for file in files:
                        print(file)
                        cv_img = cv.imread(dir_name + '/' + dir + '/' + file)
                        # tmp_img = np.array(cv_img) 
                        tmp_image_copy = cv_img.copy()
                        # tmp_image_copy = tmp_image_copy[:, :, 0] 
                        # tmp_zernik_feature = mahotas.features.zernike_moments(tmp_image_copy, radius, degree=8)
                        (cnts, tmpFeatures) = self.describeShapes(tmp_image_copy)
                        D = distance.cdist(ZernikeFeatures, tmpFeatures)
                        min_dist = np.min(D)
                        # min_dist = min_dist + dist(zernik_feature, tmp_zernik_feature)
                        # n=n+1
                        # train_labels.append(dir)
                        # train_features.append(tmp_zernik_feature)
                        if min_dist < best['dist']:
                            best['dist'] = min_dist
                            best['name'] = dir
                            best['filepath'] = dir_name + '/' + dir + '/' + file
                    break
                
            break

        print(best['name'])

        
        # clf_svm = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=5,random_state=2, max_iter=50000))])

        # clf_svm.fit(train_features, train_labels)

        if dir_name == './dataset/Artist':
            pass
            # clf_svm = pickle.load(open("artist_classifier_haralick.sav", 'rb'))
            # pickle.dump(clf_svm, open('artist_classifier_zernik.sav', 'wb'))
        elif dir_name == './dataset/Style':
            pass
            # clf_svm = pickle.load(open("style_classifier_haralick.sav", 'rb'))
            # pickle.dump(clf_svm, open('style_classifier_zernik.sav', 'wb'))

        

        # prediction = clf_svm.predict(zernik_feature.reshape(1, -1))[0]

        # return prediction
        # cv.putText(image, prediction, (20,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + best["name"], LABEL_TYPE['author'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + best["name"], LABEL_TYPE['style'])


    def describeShapes (self, image):
        shapeFeatures = []
        scale_percent = 60 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
  
        # convert the image to grayscale, blur it, and threshold it
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (13, 13), 0)
        thresh = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)[1]
    
        # perform a series of dilations and erosions to close holes
        # in the shapes
        thresh = cv.dilate(thresh, None, iterations=4)
        thresh = cv.erode(thresh, None, iterations=2)
    
        # detect contours in the edge map
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        # loop over the contours
        for c in cnts:
            # create an empty mask for the contour and draw it
            mask = np.zeros(resized.shape[:2], dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)
    
            # extract the bounding box ROI from the mask
            (x, y, w, h) = cv.boundingRect(c)
            roi = mask[y:y + h, x:x + w]
    
            # compute Zernike Moments for the ROI and update the list
            # of shape features
            features = mahotas.features.zernike_moments(roi, cv.minEnclosingCircle(c)[1], degree=8)
            shapeFeatures.append(features)
    
        # return a tuple of the contours and shapes
        return (cnts, shapeFeatures)

    def Shi_Tomasi(self, dir_name):
        image = np.array(self.img_resized)
        image_copy = image.copy()
        image_copy = image_copy[:, :, 0] 
        # print('1')
        # kr_feature = ski_methods.corner_kitchen_rosenfeld(image_copy, mode='constant', cval=0)
        # print('2')
        # kr_peaks = ski_methods.corner_peaks(kr_feature,min_distance=5)
        # print('3')
        # print(kr_feature)
        # imshow(kr_feature)
        # show()

        bf = cv.BFMatcher()
        best = { 'name': '', 'filepath': '', 'score': 0 }
        # kr_peaks = cv.goodFeaturesToTrack(image_copy, 100, 0.01, 10)
        im, des = self.extractFeatures(image_copy)
        train_features = []
        train_labels = []
        for _, dirs, _ in os.walk(dir_name):
            for dir in dirs:
                for _, _, files in os.walk(dir_name + '/' + dir):
                    for file in files:
                        print(file)
                        cv_img = cv.imread(dir_name + '/' + dir + '/' + file)
                        tmp_img = np.array(cv_img) 
                        tmp_image_copy = tmp_img.copy()
                        tmp_image_copy = tmp_image_copy[:, :, 0] 
                        tmp_im, tmp_des = self.extractFeatures(tmp_image_copy)
                        matches = bf.knnMatch(des, tmp_des, k=2)
                        good = []
                        if len(matches) < best['score']:
                            continue
                        for m, n in matches:
                            if m.distance < 0.75*n.distance:
                                good.append([m])
                        if len(good) > best['score']:
                            best['score'] = len(good)
                            best['name'] = dir
                            best['filepath'] = dir_name + '/' + dir + '/' + file
                            
                        # tmp_peaks = cv.goodFeaturesToTrack(image_copy, 100, 0.01, 10)
                        # train_labels.append(dir)
                        # train_features.append(tmp_peaks)
                
                    break
            break

       
        # clf_svm = Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=5,random_state=2, max_iter=50000))])

        # clf_svm.fit(train_features, train_labels)

        if dir_name == './dataset/Artist':
            pass
            # clf_svm = pickle.load(open("artist_classifier_haralick.sav", 'rb'))
            # pickle.dump(clf_svm, open('artist_classifier_zernik.sav', 'wb'))
        elif dir_name == './dataset/Style':
            pass
            # clf_svm = pickle.load(open("style_classifier_haralick.sav", 'rb'))
            # pickle.dump(clf_svm, open('style_classifier_zernik.sav', 'wb'))

        

        # prediction = clf_svm.predict(kr_peaks.reshape(1, -1))[0]

        # return prediction
        # cv.putText(image, prediction, (20,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        if dir_name == './dataset/Artist':
            self.addLabel('Автор: ' + best["name"], LABEL_TYPE['author'])
        elif dir_name == './dataset/Style':
            self.addLabel('Cтиль: ' + best["name"], LABEL_TYPE['style'])

    def extractFeatures(self,img):
        orb = cv.ORB_create()
        # detection
        pts = cv.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=7)

        # extraction
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        kps, des = orb.compute(img, kps)

        # return pts and des
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des     
    
gui = GUI()

gui.uploadButton()
gui.addDropdown()



gui.addRecognitionArtistButton()
gui.addRecognitionStyleButton()

gui.start()
    