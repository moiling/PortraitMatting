# 程序说明文档

[TOC]

## 1 环境

- python ~= 3.8
- tensorboardX ~= 2.1
- numpy ~= 1.19.2
- opencv-python ~= 4.0.1
- torch ~= 1.7.0
- torchversion ~= 0.8.1
- psutil ~= 5.8.0
- pillow ~= 8.1.0


## 2 使用方法

### 2.1 训练模型

训练代码为项目根目录下的 `train.py` 文件，调用方法如下：

```bash
python train.py [-h] [--img IMG] [--trimap TRIMAP] [--matte MATTE] [--fg FG] 
				[--bg BG] [--val-out VAL_OUT] [--val-img VAL_IMG] 
				[--val-trimap VAL_TRIMAP] [--val-matte VAL_MATTE] [--ckpt CKPT] 
				[--batch BATCH] [--val-batch VAL_BATCH] [--epoch EPOCH] 
				[--sample SAMPLE] [--lr LR] [--patch-size PATCH_SIZE] 
				[--seed SEED] [-t] [-d] [-g] [-r] 
				[-m {end2end,f-net,m-net,t-net}]

optional arguments:
  -h, --help              show this help message and exit
  --img IMG               训练集中原图存放路径
  --trimap TRIMAP         训练集中三分图存放路径（若使用自动生成三分图此处可留空）
  --matte MATTE           训练集中标准透明度遮罩存放路径
  --fg FG                 训练集中标准前景图像存放路径
  --bg BG                 训练集中标准背景图像存放路径
  --val-out VAL_OUT       验证集结果临时保存路径
  --val-img VAL_IMG       验证集中原图存放路径
  --val-trimap VAL_TRIMAP 验证集中三分图存放路径（若使用自动生成三分图此处可留空）
  --val-matte VAL_MATTE   验证集中标准透明度遮罩存放路径
  --ckpt CKPT             模型参数文件保存路径
  --batch BATCH           训练时采用的 batch size
  --val-batch VAL_BATCH   测试时采用的 batch size
  --epoch EPOCH           训练 epoch 数量
  --sample SAMPLE         每 epoch 用于训练的样本数量（填 -1 表示使用所有样本进行训练）
  --lr LR                 学习率
  --patch-size PATCH_SIZE 训练时图像缩放大小
  --seed SEED             随机种子
  -t, --random-trimap     是否自动随机生成三分图
  -d, --debug             是否打印 debug 日志
  -g, --gpu               是否使用 GPU
  -r, --resume            是否加载之前训练的模型参数
  -m {end2end,f-net,m-net,t-net}, --mode {end2end,f-net,m-net,t-net}
                          工作模式
```

同目录下的 `train.sh` 文件中为使用样例。



### 2.2 测试模型

测试代码为项目根目录下的 `test.py` 文件，调用方法如下：

```bash
python test.py  [-h] [--img IMG] [--trimap TRIMAP] [--matte MATTE] [--out OUT] 
				[--ckpt CKPT] [--batch BATCH] [--patch-size PATCH_SIZE] [-d] 
				[-g] [-m {end2end,f-net,m-net,t-net}]

optional arguments:
  -h, --help              show this help message and exit
  --img IMG               测试集中原图存放路径
  --trimap TRIMAP         测试集中三分图存放路径
  --matte MATTE           测试集中标准透明度遮罩存放路径
  --out OUT               测试结果保存路径
  --ckpt CKPT             模型参数文件存放路径
  --batch BATCH           测试时采用的 batch size
  --patch-size PATCH_SIZE 测试时图像缩放大小
  -d, --debug             是否打印 debug 日志
  -g, --gpu               是否使用 GPU
  -m {end2end,f-net,m-net,t-net}, --mode {end2end,f-net,m-net,t-net}
                          工作模式
```

同目录下的 `test.sh` 文件中为使用样例。



### 2.3 使用抠图算法

抠图算法的调用代码写在项目根目录下的 `matting.py` 文件中的 `Matting` 类中，该类的接口如下：

#### 2.3.1 构造方法

1. 参数解释：

   - `checkpoint_path`： 字符串，模型文件路径；
   - `gpu`：布尔值，是否使用 GPU 运算，默认为 False。

2. 使用样例：

   ```python
   from matting import Matting
   M = Matting('/path/to/checkpoint', gpu=True)
   ```

#### 2.3.2 抠图方法

1. 方法：`matting`

2. 参数解释：

   输入：

   - `image_path`: 字符串，待抠图像的文件路径；

   - `with_img_trimp`： 布尔值，函数是否额外返回原图矩阵和预测的三分图矩阵；

   - `net_img_size`： 整型，模型处理图像的缩放尺寸（默认填 -1 表示不缩放）；

   - `max_size`： 整型，保存图像的缩放最长边尺寸（默认填 -1 表示不缩放）；

     > 如原图大小为 1000×1000，net_img_size 为 200，max_size 为 500，表示：将原图缩放为 200×200 输入模型，得到 200×200 的结果，最终缩放到 500×500 返回

   - `trimap_path`：字符串，三分图的文件路径，可用于替换模型预测的三分图（默认为空）

   输出：

   - `pred_matte`：预测的透明度遮罩矩阵

     ndarray格式矩阵，shape=(高,宽,透明度)，dtype=float32，取值范围=0-1

   - `image`：原图矩阵（仅在 `with_img_trimp` = True 时返回）

     ndarray格式矩阵，shape=(高,宽,RGB)，dtype=float32，取值范围=0-1

   - `pred_trimap`： 预测的三分图矩阵（仅在 `with_img_trimp` = True 时返回）

     ndarray格式矩阵，shape=(高,宽,三分图类别)，dtype=float32，取值范围=0-1

     其中三分图类别背景、未知区域、前景分别用 0、1、2 表示

3. 使用样例：

   ```python
   matte, img, trimap = M.matting(
       '/path/to/image', 
       return_img_trimap=True, 
       net_img_size=480, 
       max_size=-1, 
       trimap_path='/path/to/trimap'
   )
   ```

#### 2.3.3 前景预测方法

1. 方法：`cutout`

2. 参数解释：

   输入：

   - `image`：原图矩阵

     ndarray格式矩阵，shape=(高,宽,RGB)，dtype=float32，取值范围=0-1

   - `alpha`：透明度遮罩矩阵

     ndarray格式矩阵，shape=(高,宽,透明度)，dtype=float32，取值范围=0-1

   输出：

   - `cutout`：前景对象矩阵

     ndarray格式矩阵，shape=(高,宽,RGBA)，dtype=float32，取值范围=0-1

3. 使用样例：

   ```python
   cut = M.cutout(img, matte)
   ```

#### 2.3.4 图像合成方法

1. 方法：`composite`

2. 参数解释：

   输入：

   - `cutout`：前景对象矩阵

     ndarray格式矩阵，shape=(高,宽,RGBA)，dtype=float32，取值范围=0-1

   - `bg`：背景颜色

     ndarray格式矩阵，shape=(RGB)，dtype=float32，取值范围=0-1

   输出：

   - `comp`：合成后图像矩阵

     ndarray格式矩阵，shape=(高,宽,RGB)，dtype=float32，取值范围=0-1

3. 使用样例

   ```
   comp = M.composite(cutout, bg_color)
   ```

#### 2.3.5 抠图流程使用样例

```
M = Matting(checkpoint_path, gpu=True)
matte = M.matting(img_path)
cut = M.cutout(img, matte)
comp = M.composite(cut, bg_color)
```

同目录下的 `demo.py` 文件中为更完整的使用样例。



## 3 程序结构

- comp
  - estimate_fb.py：合成步骤中使用到的前景背景预测算法
- dataloader：数据集读取代码
  - dataset.py
  - prefetcher.py
  - transforms.py
- networks：模型结构
  - fnet：融合网络
    - fusionnet.py
  - mnet：抠图网络
    - dimnet.py
  - tnet：三分图预测网络
    - pspnet.py
    - resnet.py
  - matting_net.py
  - ops.py
- demo.py：抠图样例代码
- loss.py：损失函数
- matting.py：抠图方法的封装类
- test.py：测试代码
- test.sh：测试样例代码
- train.py：训练代码
- train.sh：训练样例代码
- utils.py：保存模型以及图像结果的工具代码