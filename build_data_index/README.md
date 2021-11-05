根据现有的数据目录及mask目录，生成可训练模型的标记文件

## 使用介绍
### 参数介绍
data_dir  数据路径
mask_dir  mask保存路径
save_dir  生成文件的保存路径
test_split 数据划分的比例，可固定值可比例值设置
isshuffle 是否打乱数据
isappend 是否在标记文件上执行追加操作

### 效果分析
**data_dir**
- 存放数据的目录，这个目录的文件名和mask目录的文件名一致
- 后缀名可以是[png,jpg,jpeg,bmp]等格式
- 这个目录的数据可以比mask目录的文件多，因为只转这个目录和mask目录存在对应json的文件

**mask_dir**
- 存放mask的目录，这个目录文件由labelme标记数据目录的图片生成的json
- 这个目录的文件一定是以json为后缀

**save_dir**  
- train_index.txt和test_indx.txt文件保存路径
- 该目录不存在将会被创建

test_split 
- 数据划分训练集及验证集的比例或数量
- 当设置值在(0,1)时，按比例划分，当设置值>1时，按固定值划分

isshuffle
- 数据划分前，是否打乱数据

isappend
- 是否在save_dir目录下的train_index.txt和test_indx.txt追加数据
- 注意：追加数据时，必须留意是否存在重复的数据