# TexturesFinder

## 环境安装

+ 推荐使用 [conda](https://docs.conda.io/en/latest/) 创建虚拟环境，确保使用 **Python 3.12**。
+ 安装所需库：
```bash
pip install -r requirements.txt
```

## 使用说明

### 打开前端界面
运行以下命令以启动前端界面，初次运行会从huggingface上下载模型，保存在 `model/` 文件夹下：
```bash
python app.py
```

### 获取随机图像的相似图像对比图
无需打开前端界面，直接运行以下命令：
```bash
python src/example.py
```

### 打包为可执行文件
使用 `pyinstaller` 打包软件，生成 `.exe` 文件：
```bash
pyinstaller app.spec
```

## 注意事项
- 确保所有依赖已正确安装。
- 打包后生成的 `.exe` 文件位于 `dist` 目录下。
- 如有问题，请检查日志或依赖版本是否匹配。