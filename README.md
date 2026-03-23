# Skill Library for LIBERO-VLA

独立的技能库构建、聚类、检索和可视化代码，与 LIBERO 框架核心代码分离。

## 目录结构

```
Skill_Lib/
├── __init__.py                    # 包说明
├── README.md                      # 本文件
│
│  ── 核心模块 ──
├── QwenPlanner.py                 # Qwen VLM 规划器接口
├── test_skill_lib.py              # 技能库主构建器（编码器、聚类、main入口）
├── verbnet_utils.py               # 共享 VerbNet 子任务解析
├── build_contrastive_skill_emb.py # 对比学习投影训练
├── build_task_artifacts.py        # 渲染和缓存任务初始观测
├── skill_retriever.py             # 三层层级技能检索
├── film_encoder.py                # FiLM 条件化 CLIP+ResNet 编码器
│
│  ── 脚本工具 ──
├── get_planner_output.py          # 规划器 demo 脚本
├── recluster_verbnet.py           # 使用 VerbNet 层级重新聚类
├── recluster_2layer.py            # 使用 2 层分割重新聚类
├── visualize_segmented.py         # 分段嵌入的 t-SNE 可视化
├── patch_2layer.py                # 给 test_skill_lib 打 2 层聚类补丁
├── patch_verbnet_1stlayer.py      # 给 test_skill_lib 打 VerbNet L1 补丁
├── run_skill_test.sh              # 运行技能库构建的 shell 脚本
│
│  ── 数据（符号链接） ──
├── skill_lib_results*/            # -> ../skill_lib_results*/ 结果数据
└── tmp/                           # -> ../tmp/ 子任务缓存
```

## 使用方式

### 从仓库根目录运行
```bash
# 构建技能库
python Skill_Lib/test_skill_lib.py --dist_threshold 0.6 --output_dir skill_lib_results

# 使用 VerbNet 重新聚类
python Skill_Lib/recluster_verbnet.py --source_dir skill_lib_results_oat --output_dir skill_lib_results_verbnet

# 获取规划器输出
python Skill_Lib/get_planner_output.py --model_path Qwen/Qwen3-VL-4B-Instruct

# 可视化
python Skill_Lib/visualize_segmented.py

# 对比学习训练
python Skill_Lib/build_contrastive_skill_emb.py --source_dir skill_lib_results_full --output_dir skill_lib_results_contrastive
```

### Python 导入
```python
# 直接导入（推荐）
from Skill_Lib.verbnet_utils import _parse_subtask_verbnet
from Skill_Lib.skill_retriever import HierarchicalSkillRetriever
from Skill_Lib.film_encoder import CLIPFiLMSkillEncoder

# 旧路径仍可用（通过 shim 重定向）
from libero.lifelong.test_skill_lib import _parse_subtask_verbnet
```

## 训练集成

以下文件仍保留在 `libero/lifelong/` 中（因为它们与 LIBERO 训练框架深度耦合）：
- `libero/lifelong/algos/skill_library.py` — SkillLibraryBuilder 算法
- `libero/lifelong/datasets.py` — SkillLabeledVLDataset
- `libero/lifelong/main_skill.py` — Hydra 训练入口
- `libero/configs/config_skill.yaml` — 训练配置

## 备份文件

原始文件已备份为 `.bak` 后缀，确认无误后可删除：
```bash
rm -f *.bak libero/lifelong/*.bak libero/lifelong/models/modules/*.bak
```
