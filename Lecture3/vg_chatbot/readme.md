# 1 colab部署

本项目主要实现了google colab的部署，相关的notebook已经在项目文件中。





# 2 开发机部署

`environment.yml`在InternLM开发机上配置的基本环境，如果在类linux服务器上部署，可以创建该虚拟环境。后续操作和notebook中的程序基本一致。



## 2.1配置环境

首先配置环境，直接通过.yml文件安装：

```
conda env create -f environment.yml
```

也可以直接复制环境：

```
/root/share/install_conda_env_internlm_base.sh InternLM
```



然后激活虚拟环境：

```
conda activate InternLM
```



并在环境中安装运行 demo 所需要的依赖：

```bash
# 升级pip
python -m pip install --upgrade pip

pip install modelscope==1.9.5
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```



## 2.2 模型下载

此处我们使用的是`internlm-chat-7b`，使用如下代码下载。

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/content/drive/MyDrive/InternLM_labs/code/model', revision='v1.0.3')
```

或者：

```bash
# 可以继续未完成的下载过程，并保存到本地目录/content/model
huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir /root/model
```



## 2.3 安装LangChain相关依赖

在已完成 InternLM 的部署基础上，还需要安装以下依赖包：

```bash
pip install langchain==0.0.292
pip install gradio==4.4.0
pip install chromadb==0.4.15
pip install sentence-transformers==2.2.2
pip install unstructured==0.10.30
pip install markdown==3.3.7
```

同时，我们需要使用到开源词向量模型 [Sentence Transformer](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)。这个模块主要是进行Embedding，采用其他的模块也可以。我们使用huggingface提供的`huggingface_hub`中的`huggingface-cli`命令行工具进行下载。

```bash
pip install -U huggingface_hub
```

然后在和 `/root/data` 目录下新建python文件 `download_hf.py`。其中参数之前已经讲过：

1. `--resume-download`：断点续下；
2. `--local-dir`：指定本地保存路径。

此外，huggingface下载速度慢，我们可以定向到镜像网站下载。

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/data/model/sentence-transformer')
```



## 2.4 下载 NLTK 相关资源

我们在使用开源词向量模型构建开源词向量的时候，需要用到第三方库 `nltk` 的一些资源。我们将使用国内的镜像网站，用以下命令下载 nltk 资源并解压到服务器上。

之后使用时服务器即会自动使用已有资源，无需再次下载。

```bash
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```



## 2.5 构建Chroma数据库

由于数据库构建用到了一部分pdf文档，所以要安装langchain的pdf读取模块`UnstructuredPDFLoader`的相关依赖。

```
pip install pdf2image pdfminer opencv-python unstructured_pytesseract unstructured_inference
```

如果pdfminer缺少模块，可能是因为pdfminer在3.10下的笨笨问题，需要重装pdfminer.six。



数据在`/vg_data`子目录下。



接着构建数据库：

```
python /root/mydb/demo/create_chroma_db.py
```



## 2.6部署web_demo

```
python /root/mydb/demo/web_demo.py
```

