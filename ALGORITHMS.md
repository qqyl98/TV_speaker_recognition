# 核心算法概览

## 数据预处理与分段
- **SRT 时间解析与对话分组**：`parse_srt` 通过正则拆分字幕块，并将时间字符串转换为秒；`group_dialogues_by_interval` 按相邻对白的时间差（默认 10 秒）进行分组，用于后续音频切分与聚类。实现位于 `preprocessor.py`。
- **音频提取与切分**：`extract_wav_from_mp4` 调用 MoviePy 从 MP4 中导出 16 kHz WAV，`extract_segments_from_video` 依据字幕时间段裁剪音频为查询片段，并记录起止时间、文本与组号。

## 视觉先验
- **人脸对齐与特征提取（神经网络）**：`actorfaces.py` 通过 `load_pretrained_model` 实例化 AdaFace 的 IR-50 网络（`build_model('ir_50')` 构建 ResNet-IR50 主干并加载 `pretrained/adaface_ir50_webface4m.ckpt` 权重），并置为 `eval` 模式；`align.get_aligned_face` 先对帧中人脸做几何对齐，`to_input` 按 AdaFace 预处理规范（BGR 归一化、通道转置）生成张量以馈入该神经网络。
- **参考库构建与相似度匹配**：`load_reference_embeddings` 对参考 JPG 逐张运行 AdaFace 模型、对同一演员的特征取均值得到模板向量；`recognize_faces_in_segments` 在每个对话组均匀采样帧批量前向传递，使用 `torch.nn.functional.cosine_similarity` 将帧特征与参考模板比对（阈值 0.4）筛出候选演员列表 `seen_actors`，作为后续语音分类的视觉先验。

## 音频聚类
- **说话人嵌入提取（声纹识别神经网络）**：`compute_clusters` 初始化 ReDimNet 说话人模型（卷积 + Transformer 编码器的端到端声纹/说话人表征网络）并对每个片段调用 `extract_embedding`，保证 1 秒窗口、单声道、16 kHz 输入。
- **时间分组与增量聚类算法（基于余弦相似度的在线单遍聚类）**：
  - `group_by_gap` 按对白间隔（≤10 秒）将片段切成连续对话组，避免跨场景误聚。
  - `cluster_within_group` 逐段扫描同组片段：
    1. 计算当前片段与已有每个簇所有成员的余弦相似度。
    2. 若该片段与某簇所有成员相似度都大于 0.4，且该簇内部任意两成员的相似度都不低于 0.1（`cluster_valid` 约束簇内一致性），则将片段加入该簇。
    3. 若没有簇满足条件，则创建新簇。
  - 该策略等价于单遍、阈值驱动的增量聚类（非层次/非 KMeans），依赖余弦相似度阈值控制簇的凝聚程度，直接在 `clusterer.py` 内完成，无需额外库。

## 语音分类
- **参考库均值嵌入（声纹识别神经网络）**：`VoiceClassifier._build_reference_library` 加载 `dataset/ref` 下角色 WAV，调用同一 ReDimNet 声纹/说话人神经网络提取嵌入后对每个角色求均值作为参考向量。
- **滑窗投票识别**：`classify` 对聚类结果逐簇处理，短片段直接取单窗嵌入，长片段用 `_extract_embeddings_sliding` 的 1 秒滑窗；与参考库做余弦相似度，若包含视觉先验角色则相似度乘以 1.5，加权多数票与平均相似度确定簇的最终角色与置信度。

## 输出融合
- **角色映射与字幕写出**：`save_labeled_srt` 将分类结果展开、按字幕顺序排序，并用 `seconds_to_srt_time` 生成 SRT 时间轴，输出“角色名: 文本”字幕文件，完成多模态识别闭环。
