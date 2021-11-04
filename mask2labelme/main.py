# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/09/02 14:32:39
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   测试辅助标记的结果
'''
import cv2
import numpy as np
import glob
import os
import os.path as osp
import tqdm
from buildjson import BuildJson

def EiSeg2Labelme(DataDir,EiSegDir=None):
    if EiSegDir==None:
        EiSegDir=osp.join(DataDir,'label')
    
    

    images_path=glob.glob(osp.join(DataDir,"*[.png,bmp]"))
    # for ind,image_path in enumerate(images_path):
    for ind in tqdm.tqdm(range(len(images_path))):
        image_path=images_path[ind]
        image_name=osp.basename(image_path)
        eiseg_path=osp.join(EiSegDir,image_name.replace(osp.splitext(image_name)[1],'.png'))
        if not osp.exists(eiseg_path):
            print('[warning]not found eiseg image path in:',eiseg_path)
            continue

        eiseg_mask=cv2.imdecode(np.fromfile(eiseg_path, dtype=np.uint8), 0)
        
        save_json_dir=osp.join(DataDir,'jsons')
        if not osp.exists(save_json_dir):
            os.mkdir(save_json_dir)
        save_json_path=osp.join(save_json_dir,image_name.replace(osp.splitext(image_name)[1],'.json'))

        buildjson=BuildJson(point_precision=0.0005)
        buildjson.svae_mask_to_json(image_path,[eiseg_mask],['1'],save_json_path)

if __name__ == '__main__':
    # DataDir='E:\\业务数据\\66 AI学习图\\阴极\\偏移'
    # DataDir='E:\\业务数据\\66 AI学习图\\阴极\\角度'
    # DataDir='E:\\业务数据\\66拉SPA\\阴极\\角度'
    DataDir='E:\\业务数据\\66拉SPA\\阴极\\偏移'
    EiSeg2Labelme(DataDir)
    

