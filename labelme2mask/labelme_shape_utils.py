# -*- encoding: utf-8 -*-
'''
@File    :   labelme_shape_utils.py
@Time    :   2021/11/03 16:32:20
@Author  :   Wu Shaogui
@Version :   1.0
@Contact :   wshglearn@163.com
@Desc    :   修改labelme官方提取程序
'''

import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

from labelme.logger import logger


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]

    if len(xy)==1:
        shape_type = "point"
    elif len(xy)==2:
        shape_type = "linestrip"
    else:
        shape_type = 'polygon'

    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32) #eg:(1357, 2448)
    ins = np.zeros_like(cls) #eg: (1357, 2448)

    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        if label not in label_name_to_value:
            continue
        
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)
        #[[],[]] DiaphragmBreakage 8a7818e6-a88b-11eb-be69-4fc69035cf0d polygon
        # print('here01',points,label,group_id,shape_type) 

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        # DiaphragmBreakage ('DiaphragmBreakage', UUID('8a7818e6-a88b-11eb-be69-4fc69035cf0d')) 1 2
        # print('here02',cls_name,instance,ins_id,cls_id) 
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        # print('here03',mask.shape) #(1491, 2279)

        cls[mask] = cls_id
        ins[mask] = ins_id
        # print('here04',cls.shape,ins.shape) #(1491, 2279) (1491, 2279)

    return cls, ins

# 自己附加代码，为解决粘连mask的解析问题，按label_name_to_value字典的形式返回mask
def my_shapes_to_label(img_shape, shapes, label_name_to_value):

    allclass={}
    instances = []
    for label_name in label_name_to_value:
        cls = np.zeros(img_shape[:2], dtype=np.int32)
        ins = np.zeros_like(cls)

        for shape in shapes:
            points = shape["points"]
            label = shape["label"]
            if label!=label_name:
                continue

            group_id = shape.get("group_id")
            if group_id is None:
                group_id = uuid.uuid1()
            shape_type = shape.get("shape_type", None)

            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = shape_to_mask(img_shape[:2], points, shape_type, line_width=3, point_size=1)

            cls[mask] = cls_id
            ins[mask] = ins_id
        allclass[label_name]=cls

    return allclass, instances

def labelme_shapes_to_label(img_shape, shapes):
    logger.warn(
        "labelme_shapes_to_label is deprecated, so please use "
        "shapes_to_label."
    )

    label_name_to_value = {"_background_": 0}
    for shape in shapes:
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value


def masks_to_bboxes(masks):
    if masks.ndim != 3:
        raise ValueError(
            "masks.ndim must be 3, but it is {}".format(masks.ndim)
        )
    if masks.dtype != bool:
        raise ValueError(
            "masks.dtype must be bool type, but it is {}".format(masks.dtype)
        )
    bboxes = []
    for mask in masks:
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes
