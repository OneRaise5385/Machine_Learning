# Datawhale夏令营-环境配置
## 安装清单：
1.	[Miniconda](https://docs.conda.io/en/latest/miniconda.html/)
2.	[Vscode](https://code.visualstudio.com/)
## Miniconda安装及配置
### 1. 下载安装
- 配置推荐：Just me、其他默认配置
### 2. 修改Power Shell执行策略
- 管理员身份打开`Windows Power Shell`
- 输入`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- 提示更改执行策略，选择全是`【A】`
### 3. 打开Miniconda
- 输入`conda init`，下次再打开cmd时系统会自动启动conda环境
## conda更换镜像源
1.	镜像源网站
- [清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/)
- [南方科技大学镜像源](https://help.mirrors.cernet.edu.cn/anaconda/)
2.	打开网址复制`.condarc`文件
3.	打开anaconda输入`notepad .condarc`
4.	将复制的文件保存后关闭
5.	输入`conda clean -i` （清楚源缓存，启用镜像源）
6.	其他镜像源在镜像网站添加即可（pytorch, Paddle）
## PYPI更换镜像源
1.	镜像源网站https://help.mirrors.cernet.edu.cn/pypi/
2.	复制换源命令在conda中运行即可
## VS Code安装
- 已经安装完成
## 创建与激活conda环境
1.	打开`conda Powershell` 输入指令`conda create -n Datawhale python=3.10`（这里是创建一个名字叫做Datawhale的conda环境）
2.	输入`conda activate Datawhale `激活Datawhale环境（在Datawhale内下载的版本运行的程序不会影响其他环境）
3.	其他操作
- 退出当前环境`conda deactivate`
- 删除整个环境`conda remove -n Datawhale –all`（删除环境时需要先推出环境）
- 查看当前环境`conda env list`
## pip的相关安装
### 1. jupyter
- 输入`pip install jupyter`
- 输入`jupyter-notebook`打开jupyter浏览器
- 关闭`jupyter ctrl + c`
### 2. 其他库的安装
- scikit-learn
- numpy
- pandas
- tqdm
- lightgbm（数据挖掘模型）
## 其他安装
- 首先查看本机支持的cuda版本，输入命令`nvidia-smi`

- 右上角显示的是本机支持的最高的cuda版本
### 1. 安装pytorch 
- https://pytorch.org/get-started/locally/
- 选择使用conda安装
### 2. 安装paddlepaddle
- https://www.paddlepaddle.org.cn/
- 选择版本进行安装
### 3. 验证pytorch
- 使用ipython打开python（如果没有ipython，使用conda安装）
- 输入`import torch`
- 输入`torch.cuda.is_available()`
- 若返回`True`则配置成功
### 4. 验证paddle
- 打开ipython
- 输入`import paddle`
- 输入`paddle.utils.run_check()`
- 若返回
- `PaddlePaddle works well on 1 GPU.`
- `PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.`
- 则配置成功
## 云端环境的使用
### 1. 云环境平台
- [百度飞桨](https://aistudio.baidu.com/aistudio/index)
- [Kaggle](https://www.kaggle.com/code)
- 阿里天池
- Google Colab
- Sagemaker Studio Lab
### 2. 百度飞桨的使用
- 打开[百度飞桨](https://aistudio.baidu.com/aistudio/index)
- 我的项目-创建项目-启动项目-选择运行环境（v100够了）
- 运行环境
