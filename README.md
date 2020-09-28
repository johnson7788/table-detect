# table-detect
## table detect(yolo) , table line(unet) （表格检测/表格单元格定位）
links: http://59.110.234.163:9990/static/models/table-detect/   
models weights download and move to ./modes    


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


