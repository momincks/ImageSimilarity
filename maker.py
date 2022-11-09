import sys
import os
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm

class Augmenter:

    def __init__(self,probability, drop_factor):
        self.probability = probability
        self.drop_factor = drop_factor
        self.method = ["horizontal_shift",
                        "vertical_shift",
                        "horizontal_flip",
                        "vertical_flip",
                        "rotation",
                        "brightness",
                        "contrast",
                        "noise_mask",
                        "pixel_attack",
                        "pixelation",
                        "zoom",
                        "aspect_ratio",
                        "cutout",
                        ]
        self.param = [[-0.2,0.2],
                      [-0.2,0.2],
                      1,
                      1,
                      [-0.2,0.2],
                      [-0.2,0.2],
                      [-0.2,0.2],
                      [0.05,0.1],
                      [0.05,0.1],
                      [0.05,0.2],
                      [0.05,0.2],
                      [0.05,0.2],
                      [0.05,0.2],
                        ]

    def horizontal_shift(self, image, param, label):
        width = int(image.shape[1]*param)
        if width > 0:  ## shift to right
            image[:,width:,:] = image[:,:-width,:]
        elif width < 0:  ## shift to left
            image[:,:-width,:] = image[:,width:,:]   
        label = label * max(self.drop_factor,(1-abs(param))**0.5)         
        return image, label

    def vertical_shift(self, image, param, label):
        height = int(image.shape[0]*param)
        if height > 0:  ## downward shift
            image[height:,:,:] = image[:-height,:,:]
        elif height < 0:  ## upward shift
            image[:-height,:,:] = image[height:,:,:]
        label = label * max(self.drop_factor,(1-abs(param))**0.5)         
        return image, label

    def horizontal_flip(self, image, param, label):
        if param:
            image = cv2.flip(image, 1)
        return image, label

    def vertical_flip(self, image, param, label):
        if param:
            image = cv2.flip(image, 0)
        return image, label

    def rotation(self, image, param, label):
        if param:

            height, width = image.shape[:2]
            cent_x, cent_y = width // 2, height // 2

            mat = cv2.getRotationMatrix2D((cent_x, cent_y), -param, 1.0)
            cos, sin = np.abs(mat[0, 0]), np.abs(mat[0, 1])

            n_width = int((height * sin) + (width * cos))
            n_height = int((height * cos) + (width * sin))

            mat[0, 2] += (n_width / 2) - cent_x
            mat[1, 2] += (n_height / 2) - cent_y

            image = cv2.warpAffine(image, mat, (n_width, n_height))
            new_height, new_width = image.shape[:2]
            image = image[int((new_height-height)/2):int((new_height+height)/2),
                            int((new_width-width)/2):int((new_width+width)/2)]
                           
            label = label * max(self.drop_factor,(1-abs(param))**0.5)         
        return image, label

    def brightness(self, image, param, label): 
        image = image + param
        image = np.clip(image, 0., 1.)
        label = label * max(self.drop_factor,(1-abs(param))**0.5)         
        return image, label

    def contrast(self, image, param, label):
        factor = (1+param)**(1+param)
        image = np.mean(image) + factor * image - factor * np.mean(image)
        image = np.clip(image, 0., 1.)
        label = label * max(self.drop_factor,(1-abs(param))**0.5)
        return image, label

    def noise_mask(self, image, param, label):
        height, width, channel = image.shape
        noise = np.random.rand(height, width, channel)
        image = image * (1-param) + noise * param
        image = np.clip(image, 0., 1.)
        label = label * max(self.drop_factor,(1-param))    
        return image, label

    def pixel_attack(self, image, param, label):
        if param:
            height, width, channel = image.shape
            for _ in range(int(height * width * param)):
                image[random.randint(0, height-1), random.randint(0, width-1), :] = np.random.rand(channel)
            label = label * max(self.drop_factor,(1-param))
        return image, label

    def pixelation(self, image, param, label):
        factor = 1.-param
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width*factor), int(height*factor)), cv2.INTER_NEAREST)
        image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)
        label = label * max(self.drop_factor,(1-param)**0.5) 
        return image, label

    def zoom(self, image, param, label):
        factor = param/2
        height, width = image.shape[:2]
        image = image[int(height*factor):int(height*(1-factor)),int(width*factor):int(width*(1-factor))]
        label = label * max(self.drop_factor,(1-param)**0.5)
        return image, label
    
    def aspect_ratio(self, image, param, label):
        height, width = image.shape[:2]
        if random.random() > 0.5:
            image = image[:,:int(width*(1-param))]
        else:
            image = image[:int(height*(1-param)),:]
        image = cv2.resize(image,(width,height))
        label = label * max(self.drop_factor,(1-param)**0.5)
        return image, label

    def cutout(self, image, param, label):
        height, width = image.shape[:2]
        cut_size = [int(height*param),int(width*param),3]
        start_pt = [random.randint(0,cut_size[0]),random.randint(0,cut_size[1])]
        image[start_pt[0]:start_pt[0]+cut_size[0],start_pt[1]:start_pt[1]+cut_size[1]] = np.zeros(cut_size)
        label = label * max(self.drop_factor,(1-param))
        return image, label

    def run(self, image):
        label = 1.
        for step, method in enumerate(self.method):
            if random.random() <= self.probability:
                if isinstance(self.param[step],list):
                    param = random.uniform(self.param[step][0],self.param[step][1])
                else:
                    param = (random.random()<self.param[step])
                image, label = getattr(self, method)(image, param, label)
        return image, label

def filename(prefix,num,label):
    num = "0"*(6-len(str(num))) + str(num)
    name = prefix + "/" + num + "_" + str(round(label,3)) + ".jpg"
    return name

if __name__ == "__main__":
    augmenter = Augmenter(0.25,0.975)
    img_path = glob("../dataset/places365-256/*/*/*.*")
    img_path.extend(glob("../dataset/places365-256/*/*/*/*.*"))
    random.shuffle(img_path)
    for step in tqdm(range(30000)):
        img = cv2.imread(img_path[step])
        img = img.astype(np.float32)/255.
        img1 = np.copy(img)
        img2, label = augmenter.run(img)
        img3 = cv2.imread(img_path[-step-1])
        img1 = cv2.resize(img1,(224,224))
        img2 = cv2.resize(img2,(224,224))
        img3 = cv2.resize(img3,(224,224))
        # cv2.imshow("x",img1)
        # cv2.waitKey(0)
        # cv2.imshow("x",img2)
        # cv2.waitKey(0)
        label = max(label,0.9)
        cv2.imwrite(filename("../dataset/similarity/train/anchor",step,label),img1*255)
        cv2.imwrite(filename("../dataset/similarity/train/sim",step,label),img2*255)
        cv2.imwrite(filename("../dataset/similarity/train/diff",step,0.05),img3)
    random.shuffle(img_path)
    for step in tqdm(range(5000)):
        img = cv2.imread(img_path[step])
        img = img.astype(np.float32)/255.
        img1 = np.copy(img)
        img2, label = augmenter.run(img)
        img3 = cv2.imread(img_path[-step-1])
        img1 = cv2.resize(img1,(224,224))
        img2 = cv2.resize(img2,(224,224))
        img3 = cv2.resize(img3,(224,224))
        # cv2.imshow("x",img1)
        # cv2.waitKey(0)
        # cv2.imshow("x",img2)
        # cv2.waitKey(0)
        label = max(label,0.9)
        cv2.imwrite(filename("../dataset/similarity/val/anchor",step,label),img1*255)
        cv2.imwrite(filename("../dataset/similarity/val/sim",step,label),img2*255)
        cv2.imwrite(filename("../dataset/similarity/val/diff",step,0.05),img3)
        
