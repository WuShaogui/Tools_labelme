# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/09/02 14:32:39
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   mask图片转为labelme可再编辑的json
'''

import argparse
import glob
import json
import os
import os.path as osp
import threading

import cv2
import loguru
import numpy as np

from buildjson import BuildJson

LOG=loguru.logger

class Mask2Labelme(object):
    def __init__(self,template_json_path,point_precision,replace_mask,thread_num) -> None:
        super(Mask2Labelme,self).__init__()
        self.template_json_path=template_json_path
        self.point_precision=point_precision
        self.replace_mask=replace_mask
        self.thread_num=thread_num
    
    def run(self,image_dir,masks_dir,json_dir,label_name):
        # 检查数据目录
        assert osp.exists(image_dir),LOG.error('not found data directory:{}'.format(image_dir))
        assert osp.exists(masks_dir),LOG.error('not found mask directory:{}'.format(masks_dir))
        if not osp.exists(json_dir):
            os.makedirs(json_dir,exist_ok=True)
        
        # 获取数据路径下的数据文件路径
        extension = ['*.png', '*.bmp', '*.jpg', '*.jpeg']
        images_name = []
        for ext in extension:
            images_name+=glob.glob(osp.join(parsed.image_dir,ext))
        
        images_path=[osp.join(parsed.image_dir,image_name) for image_name in sorted(images_name)]
        if len(images_path)==0:
            LOG.error('image not found in directory:{}'.format(image_dir))
            return

        # 检查线程数
        using_thread_num=self.thread_num
        if self.thread_num>len(images_path):
            using_thread_num=len(images_path)
        
        # 多线程调用
        threads = []
        batch_size=int(len(images_path)/using_thread_num) #每个多线程处理的数据量
        for iter_ind in range(using_thread_num):
            # 确定该线程处理的数据量
            if (iter_ind+2)*batch_size>len(images_path):
                last_ind=len(images_path)
            else:
                last_ind=(iter_ind+1)*batch_size
            
            # 打印线程标记    
            thread_info='thread:{}'.format(iter_ind+1)

            # 多线程调用
            t=threading.Thread(target=self.convert_masks_to_jsons,args=(
                images_path[iter_ind*batch_size:last_ind],
                masks_dir,
                json_dir,
                label_name,
                thread_info))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()    

    def convert_mask_to_json(self,image_path,masks_dir,json_dir,label_name,print_info):
        '''convert_mask_to_json 将mask转为labelme可加载的json

        Args:
            image_path (str): 图片路径
            masks_dir (str): mask路径
            json_dir (str): json保存路径
            template_json_path (str): json初始模板
            label_name (str): mask在json中的标签
            point_precision (float): mask边缘的精细程度
            replace_mask (bool): 存在旧json是，true为更新json，否则为覆盖
            print_info (str): 线程的打印信息
        '''
        image_name=osp.basename(image_path)

        #!! 默认所有mask以png结尾
        mask_path=osp.join(masks_dir,image_name.replace(osp.splitext(image_name)[1],'.png'))
        save_json_path=osp.join(json_dir,image_name.replace(osp.splitext(image_name)[1],'.json'))
        
        # 原图对应的json不存在
        if not osp.exists(mask_path):
            LOG.warning('{} mask not found:{}'.format(print_info, mask_path))
            return
        
        # 读取mask
        mask=cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 0)

        # 读取json的初始模板
        labelme_template = json.load(open(self.template_json_path, 'r', encoding='utf-8'))
        labelme_template['shapes']=[]
        # 使用已有json文件作为初始json状态
        if osp.exists(save_json_path) and self.replace_mask:
            labelme_template = json.load(open(save_json_path, 'r', encoding='utf-8'))
            
        # 初始化转换类
        buildjson=BuildJson(
            labelme_template=labelme_template,
            point_precision=self.point_precision)

        # 解析mask并保存为json
        run_staute=buildjson.svae_mask_to_json(image_path,[mask],[label_name],save_json_path)
        if run_staute:
            LOG.success('{} json saved in {}'.format(print_info,save_json_path))

    # 多线程分发数据
    def convert_masks_to_jsons(self,images_path,masks_dir,json_dir,label_name,thread_info):
        masks_num=len(images_path)
        for ind,image_path in enumerate(images_path):
            print_info='{} {}/{}'.format(thread_info,ind,masks_num)
            self.convert_mask_to_json(image_path,masks_dir,json_dir,label_name,print_info)


if __name__ == '__main__':
    args=argparse.ArgumentParser('convert mask to json')
    args.add_argument('-i','--image_dir',default='',type=str,help='images directory')
    args.add_argument('-m','--mask_dir',default='',type=str,help='mask directory')
    args.add_argument('-s','--json_dir',default='convert_masks',type=str,help='save mask directory')
    
    # json的初始模板
    args.add_argument('-j','--template_json_path',default='mask2labelme/labelme4.5.7_template.json',type=str,help='template json path')
    # 生成json的mask标签名
    args.add_argument('-l','--label_name',default='auto_generation',type=str,help='mask label name')
    # 生成json的mask边的密集程度
    args.add_argument('-p','--point_precision',default=0.0001,type=float,help='ploy point precision')
    # 对原始json的措施：更新VS替换，怕不同类别被覆盖
    args.add_argument('-r','--replace_mask',default=False,type=bool,help='replece or update for old json')
    # 多线程数量
    args.add_argument('-t','--thread_num',default=1,type=int,help='number of thread')
    parsed=args.parse_args()

    # 自定义参数
    parsed.image_dir='/mnt/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/SPA/阴极爆点'
    parsed.mask_dir='/mnt/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/SPA/阴极爆点/masks/1/'
    parsed.json_dir='/mnt/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/SPA/阴极爆点/convert_json'
    # parsed.label_name='cat'
    parsed.point_precision=0
    # parsed.replace_mask=True
    # parsed.thread_num=2

    # 初始化处理类并运行
    mask2labelme=Mask2Labelme(parsed.template_json_path,parsed.point_precision,parsed.replace_mask,parsed.thread_num)
    mask2labelme.run(parsed.image_dir,parsed.mask_dir,parsed.json_dir,parsed.label_name)
    

