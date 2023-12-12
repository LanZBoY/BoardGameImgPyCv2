# BoardGameImgPyCv2 for Gogokids 程式邏輯思維桌遊

本專案為提供Gogokids桌遊的辨識結果，所寫出來的工具
共有兩個版本
1. Gogokids unity版本 - 可在IOS跟Android平台執行
2. 開發版本 - 為unity版本提供可行性的影像處理方案，開發語言為python 使用 OpenCV

而本專案為開發版本，目的是為了針對特定案例進行探索並找出可行性方案
以便未來在unity版本中作使用

# 邏輯概述

目前採用錄影中每N個畫面當作輸入，並進行棋盤偵測
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/input.jpg" title = "輸入畫面">
</div>

當偵測到棋盤格時，進行辨識路徑邏輯
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/result.jpg">
</div>

根據每一種路徑進行相似度計算，轉換為最終結果
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/final_result.jpg">
</div>

## 棋盤偵測
我們需要根據輸入的畫面找到棋盤的確切位置
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/input.jpg">
</div>
先將畫面擷取出感興趣的區域(Region of Interest, ROI)，我們使用簡單的CropImage處理
在使用者介面方面，劃出範圍並提示使用者棋盤要在所預定的框框內
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/crop.jpg">
</div>
這個做法有以下幾個優點<br>
  1. 可以減少運算量<br>
  2. 可以減少畫面中其他物件的干擾，並提高棋盤偵測的準確率<br>
<br>
在進行影像辨識前先上一層高斯模糊，可以增加偵測的準確度
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/blur.jpg">
</div>

影像轉換<br>
1. RGB轉HSV<br>
2. HSV轉灰階<br>
3. 使用MedianFilter做平滑化處理<br>
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/hsv.jpg">
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/hsv_gray.jpg">
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/hsv_gray_median.jpg">
</div>

進行Canny Edge(邊界計算)
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/canny.jpg">
</div>

從所有邊界中找出最大邊界
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/contour.jpg">
</div>

邊角偵測運算，找尋邊界中所有可能的角點
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/corner.jpg">
</div>

透過區塊MinMax找出，棋盤端點
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/select.jpg">
</div>

映射轉換
<div>
  <img src = "https://github.com/LanZBoY/BoardGameImgPyCv2/blob/master/procedure_img/result.jpg">
</div>

最後輸出結果圖，以便在棋盤辨識時使用

## 路徑辨識
