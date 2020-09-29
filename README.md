# table-detect
## table detect(yolo) , table line(unet) （表格检测/表格单元格定位）
links: http://59.110.234.163:9990/static/models/table-detect/   
models weights download and move to ./modes    

# 模型训练，使用labelme对图片数据标记
`
train/dataset-line/
`

###  第一步： 对表格检测

` 
python table_detect.py
`

###  第二步，使用unet对单元格进行检测  
` 
python table_ceil.py
`

## 第三步，训练表格线

` 
python train/train.py
`

# 无框表格
表格训练是训练的直线和竖线，然后直线交并组成的闭区域就是单元格。但是无边框表格比较特殊，这种要构造虚拟直线和竖线去训练。
虚拟直线不能像真正的直线那样去标注，行之间的间隔都属于虚拟线。可以把表格分为4类，横线、竖线、虚拟横线、虚拟竖线

