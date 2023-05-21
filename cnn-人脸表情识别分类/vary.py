import os
import numpy as np
import random
import scipy.ndimage as ndi
import cv2
import tensorflow as tf 
from skimage import transform,data,exposure
from PIL import Image 
#亮度变化
def Chnage_Light(image): 
    image1=exposure.adjust_gamma(image,1.5)
    return image1


def do_random_rotation(translation_factor, image_array,zoom_range):#对一张图片进行随机的翻转
    height = image_array.shape[0]
    width = image_array.shape[1]
    x_offset = np.random.uniform(0, translation_factor * width)
    y_offset = np.random.uniform(0, translation_factor * height)
    offset = np.array([x_offset, y_offset])
    scale_factor = np.random.uniform(zoom_range[0],
                                        zoom_range[1])
    crop_matrix = np.array([
        [scale_factor,0],[0,scale_factor]
    ])
    image_array = np.rollaxis(image_array,axis = -1,start = 0)
    image_channel = [ndi.interpolation.affine_transform(image_channel, crop_matrix, offset=offset, order=0, mode='nearest', cval=0.0)
                         for image_channel in image_array]
    image_array = np.stack(image_channel, axis = 0)
    image_array = np.rollaxis(image_array,0,3)
    return image_array

#添加高斯噪声
def gaussian_noise(img,mean,sigma):
    '''
    此函数将产生高斯噪声加到图片上
    :param img:原图
    :param mean:均值
    :param sigma:标准差
    :return:噪声处理后的图片
    '''

    img = img/255  #图片灰度标准化

    noise = np.random.normal(mean, sigma, img.shape) #产生高斯噪声
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值


# img_path:图片的路径
# save_path：保存的路径
# data_augmentation：放大的倍数

def generator_dataset(img_dir_path, save_path, data_augmentation,translation_factor):
    class_dirs = os.listdir(img_dir_path)#fer/train
    zoom_range = [0.75, 1.25]
    print(class_dirs)
    for class_name in class_dirs:  #class_dirs=[angry,digust,**]
        img_dirs = os.listdir(img_dir_path+"/"+class_name)#fer/train/angry
        print(img_dir_path+"/"+class_name)
        for img_name in img_dirs:
            img = cv2.imread(img_dir_path + "/" + class_name+"/"+img_name)
            for i in range(0, data_augmentation):
                rotate_img = do_random_rotation(translation_factor,img,zoom_range)
                img_noise = gaussian_noise(rotate_img, 0, 0.12) # 高斯噪声
                #print(rotate_img)
                #print(save_path + "/" + class_name+"/" + "_" + img_name)
                light_img=Chnage_Light(img_noise)
                cv2.imwrite(save_path + "/" + class_name+"/"+str(i)+"_" + img_name, img_noise)

if __name__ == '__main__':
    generator_dataset("/home/tsinghuaee85/bighw/fer/train","/home/tsinghuaee85/bighw/fer/train",10,0.1)