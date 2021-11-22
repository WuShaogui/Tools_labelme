# -*- encoding: utf-8 -*-
'''
@File    :   buildJson.py
@Time    :   2021/09/02 14:31:46
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   通过mask构建labelme的json
'''
import re
import cv2
import json
import numpy as np
import os.path as osp
from base64 import b64encode
import  math

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16,np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # add this line
            return obj.tolist() # add this line
        return json.JSONEncoder.default(self, obj)

# 根据三点坐标计算夹角
def __cal_ang(p1, p2, p3):
    eps = 1e-12
    a = math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1]))
    b = math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1]))
    c = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
    ang = math.degrees(
        math.acos((b ** 2 - a ** 2 - c ** 2) / (-2 * a * c + eps))
    )  # p2对应
    return ang


# 计算两点距离
def __cal_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# 边界点简化
def approx_poly_DP(contour, min_dist=10, ang_err=5):
    # print(contour.shape)  # N, 1, 2
    cs = [contour[i][0] for i in range(contour.shape[0])]
    ## 1. 先删除夹角接近180度的点
    i = 0
    while i < len(cs):
        try:
            last = (i - 1) if (i != 0) else (len(cs) - 1)
            next = (i + 1) if (i != len(cs) - 1) else 0
            ang_i = __cal_ang(cs[last], cs[i], cs[next])
            if abs(ang_i) > (180 - ang_err):
                del cs[i]
            else:
                i += 1
        except:
            i += 1
    ## 2. 再删除两个相近点与前后两个点角度接近的点
    i = 0
    while i < len(cs):
        try:
            j = (i + 1) if (i != len(cs) - 1) else 0
            if __cal_dist(cs[i], cs[j]) < min_dist:
                last = (i - 1) if (i != 0) else (len(cs) - 1)
                next = (j + 1) if (j != len(cs) - 1) else 0
                ang_i = __cal_ang(cs[last], cs[i], cs[next])
                ang_j = __cal_ang(cs[last], cs[j], cs[next])
                # print(ang_i, ang_j)  # 角度值为-180到+180
                if abs(ang_i - ang_j) < ang_err:
                    # 删除距离两点小的
                    dist_i = __cal_dist(cs[last], cs[i]) + __cal_dist(cs[i], cs[next])
                    dist_j = __cal_dist(cs[last], cs[j]) + __cal_dist(cs[j], cs[next])
                    if dist_j < dist_i:
                        del cs[j]
                    else:
                        del cs[i]
                else:
                    i += 1
            else:
                i += 1
        except:
            i += 1
    res = np.array(cs).reshape([-1, 1, 2])
    return res


class BuildJson(object):
    def __init__(self,labelme_template,point_precision=None) -> None:
        self.labelme_template = labelme_template
        if len(labelme_template['shapes'])==1 and len(labelme_template['shapes'][0]['points'])==0:
            self.labelme_template['shapes']=[]

        self.point_precision=point_precision
    
    def get_mask_shapes(self,mask,label='unnamed'):
        '''get_mask_shapes 解析mask的边缘点集

        Args:
            mask (2D array): mask数据
            min_points (int, optional): 最少的边缘点集. Defaults to 3.
            min_epsilon (int, optional): 点到边距离小于1，去掉该点. Defaults to 1.
            label (str, optional): 新生成边缘点集命名. Defaults to 'unnamed'.

        Returns:
            list(dict): 已经封装成json格式的边缘点集
        '''
        mask_shapes = []
        mask = mask.astype(np.uint8)
        mask = mask[..., -1] if len(mask.shape) == 3 else mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 原始mask
        
        for contour in contours:
            # 设置多边形拟合精度,传入<=0的参数时，按照0.001计算精度，传入(0,1)的值时，按照传入值计算精度，传入>=1的数据，直接使用该值作为精度
            min_epsilon = 1
            if self.point_precision!=None and self.point_precision<=0:
                min_epsilon=0.0005* cv2.arcLength(contour, True); #值越大，点越少，结果越粗糙
            elif self.point_precision!=None and self.point_precision>0 and self.point_precision<1:
                min_epsilon= self.point_precision * cv2.arcLength(contour, True); #值越大，点越少，结果越粗糙
            elif self.point_precision!=None and self.point_precision>=1:
                min_epsilon= self.point_precision
            
            # 在值<1时截断
            if min_epsilon<1:
                min_epsilon=1
                
            # 使用opencv求得近似多边形的点集合
            new_contour = cv2.approxPolyDP(contour, min_epsilon, True)
            
            # 自定义边界点简化
            new_contour=approx_poly_DP(new_contour)
            
            new_contour = np.reshape(new_contour, (new_contour.shape[0], new_contour.shape[2]))

            # 过滤后的边点集数量在0以上
            if len(new_contour) > 0:
                shape = {}
                shape['label'] = label
                shape['points'] = []
                for point in new_contour:
                    assert len(point.shape) == 1
                    shape['points'].append(point.tolist())
                shape['group_id'] = None

                # 不同的形状，不同的解析方法，目前只能区分point，linestrip，polygon三种类型
                if len(shape['points']) == 1:
                    shape['shape_type'] = 'point'
                elif len(shape['points']) == 2:
                    shape['shape_type'] = 'linestrip'
                else:
                    shape['shape_type'] = 'polygon'
                shape['flags'] = {}

                mask_shapes.append(shape)
            else:
                print('not found in mask')
                continue
        return mask_shapes

    def svae_mask_to_json(self,image_path,labels_mask,labels_name,save_json_path):
        # 依次追加标签信息
        for ind,label_name in enumerate(labels_name):
            self.labelme_template['shapes'].extend(self.get_mask_shapes(labels_mask[ind],label=str(label_name)))

        # 修改json信息
        assert osp.exists(image_path),'image not exist:{}'.format(image_path)
        # image=cv2.imread(image_path)
        image=cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.labelme_template['imageHeight'] = image.shape[0]
        self.labelme_template['imageWidth'] = image.shape[1]

        rel_image_path = osp.join(osp.relpath(osp.dirname(image_path),osp.dirname(save_json_path)),osp.basename(image_path))
        self.labelme_template['imagePath'] = rel_image_path
        self.labelme_template['imageData'] = b64encode(open(image_path, "rb").read()).decode('utf-8')

        # 保存json信息
        json_content = json.dumps(self.labelme_template,cls=JsonEncoder, ensure_ascii=False, indent=2, separators=(',', ': '))
        with open(save_json_path, 'w+', encoding='utf-8') as fw:
            fw.write(json_content)
        
        return 1

