# 云游智灵 · 知识图谱 + 大语言模型旅游助手

一个融合知识图谱推荐与大语言模型（ChatGLM-6B）的旅游智能助手，提供检索增强问答（RAG）、偏好推荐、城市路线查询与旅游攻略生成等功能。前端为静态站点，后端基于 Flask，内置 KGCN 推荐模型与 Chroma 向量库。

## 功能概览
- 智能对话（RAG + ChatGLM-6B）
  - 检索 CSV 文档构建的向量库，结合 ChatGLM 进行上下文增强回答
  - 接口：`/process_input`
- 偏好推荐（KGCN）
  - 采集点击偏好并用 KGCN 进行 Top-K 景点推荐
- 城市路线查询
  - 前端基于百度地图 JS
- 旅游攻略推荐（文本主题相关度）
  - 基于 `文本主题关注度.csv` 的余弦相似度推荐
- 基础页面
  - `index.html` 主页、`chat.html` 对话页、`tag.html` 偏好页、`map.html` 地图页、`contact.html` 团队页、`login.html`/`register.html` 登录注册

## 目录结构
- `skk-0811-16/`
  - `app.py` Flask 后端主程序
  - `templates/` 前端页面与静态资源
  - `data/` 知识图谱与点击数据（`encoded_entities.tsv`、`relation.tsv`、`shuffled_encoded_KG.tsv`、`new_click.tsv`）
  - `model/model.pth` KGCN 预训练模型
  - `LangChain.py`、`chat.py` 向量库与命令行交互脚本
  - `ChatGLM-6B-main/` ChatGLM 相关代码与依赖说明
  - `VectorStore/` 持久化向量库目录（首次运行自动构建）
- 根目录
  - `YunnanTravelKG.csv`、`KG_.csv` 文档数据源
  - 其他辅助脚本与素材

## 环境要求
- Python 3.9+
- Windows/NVIDIA GPU（建议 ≥ 13GB 显存），CUDA 驱动可用
- 首次运行需自动下载 `THUDM/chatglm-6b` 模型

## 安装与运行
- 安装依赖
  - `pip install flask transformers==4.27.1 cpm_kernels torch>=1.10 langchain chromadb sentencepiece accelerate scikit-learn pandas requests pydantic`
- 首次构建向量库
  - 后端会检测 `VectorStore/` 是否存在，缺失则从 `KG_.csv` 或 `YunnanTravelKG.csv` 加载并持久化
- 启动后端
  - `cd skk-0811-16 && python app.py`
  - 访问：`http://127.0.0.1:5000/index.html`
- GPU 注意
  - 当前代码默认 `.half().cuda()`。如无 GPU，可修改为 CPU 推理（将 `.half().cuda()` 改为 `.float()` 并移除 `.cuda()`），但性能与可行性取决于机器配置。

## 关键模块
- 检索增强问答（RAG）
  - 向量库：`Chroma`（首次运行持久化到 `VectorStore/`）
  - 数据源：`KG_.csv` 或 `YunnanTravelKG.csv`
  - 接口：`POST /process_input`
    - 请求：`{"input": "你的问题"}`
    - 响应：`{"output": "回答文本"}`
- 偏好推荐（KGCN）
  - 采集接口：`POST /get_random_words` 随机给出候选
  - 推荐接口：`POST /commadation` 根据选择写入 `new_click.tsv` 并输出 Top-K
  - 模型加载与推荐逻辑：`KG` 类
- 攻略推荐
  - 余弦相似度计算：`TextRecommendation`
  - 接口：`POST /tuijian_gonglue`
- 路由与页面
  - `index.html`、`chat.html`）、`map.html`、`contact.html`、`tag.html`

## 数据与模型
- 知识图谱数据
  - `encoded_entities.tsv` 名称-实体 ID 映射
  - `relation.tsv`、`shuffled_encoded_KG.tsv` 等用于构图与训练
  - `new_click.tsv` 用户点击日志（运行时追加）
- 推荐模型
  - `model/model.pth` 为预训练的 KGCN 权重
- 文档数据源
  - `KG_.csv`、`YunnanTravelKG.csv` 用于构建向量库

## 常见问题
- 模型下载
  - HuggingFace 拉取 `THUDM/chatglm-6b` 可能受网络限制，可配置镜像或手动下载
- 显存不足
  - 尝试降低精度或使用量化/CPU 推理（需自行评估性能与延迟）
- 向量库重建
  - 删除 `VectorStore/` 后重启会自动重建

## 许可与致谢
- ChatGLM-6B 相关许可参见子目录中的 `LICENSE` 与 `MODEL_LICENSE`
- 致谢：THUDM ChatGLM、LangChain、Chroma、Baidu Map、PyTorch、Transformers 等开源项目
