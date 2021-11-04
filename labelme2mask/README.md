本项目用于将labelme标注的图片，生成mask


## 1.脚本介绍

- labelme2mask.py 执行转换的主程序
- labelme_shape_utils.py 复制labelme安装路径下的utils/shape.py文件，并做修改

## 2.功能介绍

- 多线程提取
- 提取固定标签的mask
- 保存空mask