# -*- encoding: utf-8 -*-
'''
@File    :   get_images_mask.py
@Time    :   2021/11/02 15:24:57
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   多线程解析labelme生成的json文件，并生成mask
'''

import argparse
import glob
import json
import os
import os.path as osp
import threading
import numpy as np

import cv2
import loguru

from labelme_shape_utils import my_shapes_to_label, shapes_to_label

LOG=loguru.logger

class Labelme2Mask(object):
    def __init__(self,convert_mode,thread_num,issave_empty_mask) -> None:
        super(Labelme2Mask,self).__init__()
        self.convert_mode=convert_mode
        self.thread_num=thread_num
        self.issave_empty_mask=issave_empty_mask
    
    def run(self,json_dir,mask_dir,labels_name):
         # 检查数据路径及json路径
        # assert osp.exists(parsed.data_dir),LOG.error('data directory not found:{}'.format(parsed.data_dir))
        assert osp.exists(json_dir),LOG.error('json directory not found:{}'.format(json_dir))
        
        # 检查保存路径
        if not osp.exists(mask_dir):
            os.makedirs(mask_dir,exist_ok=True)

        # 获取json路径下的json文件路径
        jsons_path=[osp.join(json_dir,os.path.basename(json_name)) for json_name in glob.glob(osp.join(json_dir,'*.json'))]
        LOG.info('found {} json'.format(len(jsons_path)))
        if len(jsons_path)==0:
            LOG.error('json not found in directory:{}'.format(json_dir))
            return

        # 检查线程数
        using_thread_num=self.thread_num
        if self.thread_num>len(jsons_path):
            using_thread_num=len(jsons_path)
        
        # 多线程调用
        threads = []
        batch_size=int(len(jsons_path)/using_thread_num) #每个多线程处理的数据量
        for iter_ind in range(using_thread_num):
            # 确定该线程处理的数据量
            if (iter_ind+2)*batch_size>len(jsons_path):
                last_ind=len(jsons_path)
            else:
                last_ind=(iter_ind+1)*batch_size
            
            # 打印线程标记    
            thread_info='thread:{}'.format(iter_ind+1)

            # 多线程调用
            t=threading.Thread(target=self.export_jsons_to_masks,args=(
                jsons_path[iter_ind*batch_size:last_ind],
                labels_name,
                mask_dir,
                thread_info))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()   

        LOG.success('run done.') 

    def export_json_to_mask(self,json_path,labels_name,mask_dir,print_info):
        '''export_json_to_mask 解析1个json文件

        Args:
            json_path (str): json文件路径
            labels_name (str): json中的标签
            mask_dir (str): 保存mask的目录
            convert_mode (int, optional): 是否提取重复的mask. Defaults to 0.
            issave_empty_mask (bool, optional): 是否保存空mask. Defaults to False.
            print_info (str, optional): 不同线程的输出信息. Defaults to ''.
        '''
        # 读取json
        assert osp.exists(json_path),LOG.error('json path not found:{}'.format(json_path))
        data=json.load(open(json_path))
        image_height,image_width=data['imageHeight'],data['imageWidth']

        # 解析json上的标签字典
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            anno_label_name = shape["label"]
            # 获得指定标签
            if anno_label_name in labels_name:
                if anno_label_name in label_name_to_value:
                    label_value = label_name_to_value[anno_label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[anno_label_name] = label_value
        
        # 找不到对应标签
        if len(label_name_to_value)==1:
            label_mask=np.zeros((image_height,image_width))
            if self.issave_empty_mask:
                # 每个待解析标签都存储1个空mask
                for label_name in labels_name:
                    label_mask_dir=osp.join(mask_dir,label_name)
                    if not osp.exists(label_mask_dir):
                        os.makedirs(label_mask_dir,exist_ok=True)
                    label_mask_path=osp.join(label_mask_dir,osp.basename(json_path).replace('.json', '.png'))
                    cv2.imwrite(label_mask_path,label_mask)
                    LOG.warning('{}:save empty mask {} for label {}'.format(print_info,osp.basename(json_path),labels_name))
            LOG.warning('not found labels:{} in json:{}'.format(labels_name,json_path))
        else:
            # 获取标签字典对应的mask
            if self.convert_mode==1:
                lbl, _ = my_shapes_to_label((image_height,image_width,1), data["shapes"], label_name_to_value)
            else:
                lbl, _ = shapes_to_label((image_height,image_width,1), data["shapes"], label_name_to_value)

            # 保存mask
            for label_name in labels_name:

                #检查路径
                label_mask_dir=osp.join(mask_dir,label_name)
                if not osp.exists(label_mask_dir):
                    os.makedirs(label_mask_dir,exist_ok=True)
                label_mask_path=osp.join(label_mask_dir,osp.basename(json_path).replace('.json', '.png'))

                label_mask=np.zeros((image_height,image_width))
                if label_name  in label_name_to_value:
                    if self.convert_mode==1:
                        label_mask=lbl[label_name][:image_height,:image_width]
                        label_mask=np.where(label_mask>0,1,label_mask)
                    else:
                        label_mask=np.where(lbl[:image_height,:image_width]==label_name_to_value[label_name],1,label_mask)
                else:
                    LOG.warning('labels:{} not exist'.format(label_name))

                # 根据提取结果保存
                if np.any(label_mask):
                    cv2.imwrite(label_mask_path,label_mask)
                    LOG.success('{}:save mask {} for label {}'.format(print_info,osp.basename(json_path),label_name))
                elif not np.any(label_mask) and self.issave_empty_mask:
                    label_mask=np.zeros((image_width,image_height,1))
                    cv2.imwrite(label_mask_path,label_mask)
                    LOG.warning('{}:save empty mask {} for label {}'.format(print_info,osp.basename(json_path),label_name))
                else:
                    LOG.warning('{}:discard mask {} for label {}'.format(print_info,osp.basename(json_path),label_name))
                    continue

    def export_jsons_to_masks(self,jsons_path,labels_name,mask_dir,thread_info):
        jsons_num=len(jsons_path)
        for ind,json_path in enumerate(jsons_path):
            print_info='{} {}/{}'.format(thread_info,ind,jsons_num)
            self.export_json_to_mask(json_path,labels_name,mask_dir,print_info=print_info)


if __name__ == '__main__':
    # 初始化参数
    args=argparse.ArgumentParser("convert json to mask")
    args.add_argument('-j','--json_dir',default='',type=str,help='json directory')
    args.add_argument('-m','--mask_dir',default='',type=str,help='save mask directory')
    args.add_argument('-l','--labels_name',default=['dog'],type=list,help='mask label name')

    # 针对有重叠的目标提供2种转换模式
    # 0:重叠区域只属于一类
    # 1:重叠区域可属于多类
    args.add_argument('-c','--convert_mode',default=0,type=int,help='convert mode')
    args.add_argument('-t','--thread_num',default=1,type=int,help='number of thread')
    args.add_argument('-e','--issave_empty_mask',default=False,type=bool,help='is save empty mask?')
    parsed=args.parse_args()

    # 手动定义参数
    parsed.json_dir='data/labelme_jsons'
    parsed.mask_dir='data/masks'

    # parsed.convert_mode=1
    parsed.thread_num=3
    parsed.issave_empty_mask=False

    # 初始化处理类并运行
    labelme2mask=Labelme2Mask(parsed.convert_mode,parsed.thread_num,parsed.issave_empty_mask)
    labelme2mask.run(parsed.json_dir,parsed.mask_dir,parsed.labels_name)
