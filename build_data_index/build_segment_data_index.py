# -*- encoding: utf-8 -*-
'''
@File    :   build_segment_data_index.py
@Time    :   2021/11/04 15:26:48
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   构建分割任务的训练、测试样本的标记
'''

import argparse
import glob
import os
import os.path as osp
import random

import loguru

LOG=loguru.logger

class BuildSegementDataset(object):
    def __init__(self,test_split,isshuffle,isappend) -> None:
        super(BuildSegementDataset,self).__init__()
        self.test_split=test_split
        self.isshuffle=isshuffle
        self.isappend=isappend
    
    def run(self,images_dir,masks_dir,save_dir,separator_char='|'):
        # 检查数据路径及json路径
        assert osp.exists(images_dir),LOG.error('data directory not found:{}'.format(images_dir))
        assert osp.exists(masks_dir),LOG.error('json directory not found:{}'.format(masks_dir))

        # 检查保存路径
        if not osp.exists(save_dir):
            os.makedirs(save_dir,exist_ok=True)

        # 获取数据路径下的数据文件路径
        extension = ['*.png', '*.bmp', '*.jpg', '*.jpeg']
        images_name = []
        for ext in extension:
            images_name+=glob.glob(osp.join(images_dir,ext))
        
        images_path=[osp.join(images_dir,image_name) for image_name in sorted(images_name)]
        if len(images_path)==0:
            LOG.error('json not found in directory:{}'.format(images_dir))
            return
        
        #根据数据文件及mask文件夹获取可用的数据及标签
        availabel_images_path=[]
        availabel_masks_path=[]
        for _, image_path in enumerate(images_path):
            image_name=osp.basename(image_path)
            mask_path=osp.join(masks_dir,image_name.replace(osp.splitext(image_name)[1],'.png'))
            if osp.exists(mask_path):
                availabel_images_path.append(image_path)
                availabel_masks_path.append(mask_path)
            else:
                LOG.warning('not found mask for image:{}'.format(image_path))
        
        # 数据划分
        test_num=0
        if self.test_split<1:
            test_num=max(0,int(self.test_split*len(availabel_images_path)))
        if self.test_split>=1:
            test_num=min(self.test_split,len(availabel_images_path))
        
        samples = list(zip(availabel_images_path, availabel_masks_path))
        
        if self.isshuffle:
            random.shuffle(samples)
        
        # 写入数据
        train_index_path=osp.join(save_dir,'train_index.txt')
        test_index_path=osp.join(save_dir,'test_index.txt')
        if self.isappend:
            # 追加模式添加
            with open(train_index_path,'a') as fw:
                for availabel_image_path,availabel_mask_path in samples[:len(availabel_images_path)-test_num]:
                    sample_record='{}{}{}\n'.format(availabel_image_path,separator_char,availabel_mask_path)
                    fw.write(sample_record)

            # 追加模式添加
            with open(test_index_path,'a') as fw:
                for availabel_image_path,availabel_mask_path in samples[len(availabel_images_path)-test_num:]:
                    sample_record='{}{}{}\n'.format(availabel_image_path,separator_char,availabel_mask_path)
                    fw.write(sample_record)
            
            LOG.success('total images:{}\tappend mode,train sample:{}\ttest sample:{}'.format(len(images_path),len(availabel_images_path)-test_num,test_num))
        else:
            # 覆盖模式添加
            with open(train_index_path,'w') as fw:
                for availabel_image_path,availabel_mask_path in samples[:len(availabel_images_path)-test_num]:
                    sample_record='{}{}{}\n'.format(availabel_image_path,separator_char,availabel_mask_path)
                    fw.write(sample_record)

            # 覆盖模式添加
            with open(test_index_path,'w') as fw:
                for availabel_image_path,availabel_mask_path in samples[len(availabel_images_path)-test_num:]:
                    sample_record='{}{}{}\n'.format(availabel_image_path,separator_char,availabel_mask_path)
                    fw.write(sample_record)
            
            LOG.success('total images:{}\treplace mode,train sample:{}\ttest sample:{}'.format(len(images_path),len(availabel_images_path)-test_num,test_num))
        LOG.success('train and test index file save in:{}'.format(osp.dirname(train_index_path)))

if __name__ == '__main__':
    # 初始化参数
    args=argparse.ArgumentParser("convert json to mask")
    args.add_argument('-d','--images_dir',default='',type=str,help='数据路径')
    args.add_argument('-m','--masks_dir',default='',type=str,help='mask保存路径')
    args.add_argument('-s','--save_dir',default='./',type=str,help='生成文件的保存路径')

    args.add_argument('-p','--test_split',default=0.1,type=float,help='数据划分的比例，可固定值可比例值设置')
    args.add_argument('-f','--isshuffle',default=True,type=bool,help='是否打乱数据')
    args.add_argument('-a','--isappend',default=False,type=bool,help='是否在标记文件上执行追加操作')
    args.add_argument('-c','--separator_char',default='|',type=str,help='数据和标签之间的分隔符')
    parsed=args.parse_args()

    # 自定义参数
    parsed.images_dir='/home/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/SPA/阴极爆点/'
    parsed.masks_dir='/home/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/SPA/阴极爆点/masks/1'
    parsed.save_dir='/home/wushaogui/DataRepo/IMAGE_SCENE_SEGMENTATION/SPA/阴极爆点/dataset'

    # parsed.test_split= 0.1
    # parsed.isshuffle= False
    # parsed.isappend= True
    parsed.separator_char=' | '

    # 初始化处理类并运行
    buil_segment_dataset=BuildSegementDataset(parsed.test_split,parsed.isshuffle,parsed.isappend)
    buil_segment_dataset.run(parsed.images_dir,parsed.masks_dir,parsed.save_dir,parsed.separator_char)
