# Minibatch Mode 使用说明

## 概述

Minibatch mode 允许你从外部 JSON 文件指定要运行的特定 episode 列表，而不是运行整个数据集。这对于测试特定场景或断点续跑非常有用。

## 启动方式

### 方法 1: 在配置文件中指定

在配置文件中（如 `config/habitat_eval_hm3dv1.yaml`）添加 `minibatch` 字段：

```yaml
# 其他配置...
minibatch: "path/to/your/minibatch.json"
minibatch_output_dir: ""  # 可选：自定义输出目录，用于断点续跑
minibatch_resume: false   # 可选：是否从 continue.txt 继续（默认 false）
```

### 方法 2: 通过命令行参数传递（Hydra 覆盖）

**注意**：由于配置文件中已经包含了 `minibatch` 字段（默认值为 `null`），你可以直接使用：

```bash
python habitat_evaluation.py --dataset hm3dv1 minibatch="path/to/your/minibatch.json"
```

或者同时指定输出目录和是否续跑：

```bash
python habitat_evaluation.py --dataset hm3dv1 \
    minibatch="path/to/your/minibatch.json" \
    minibatch_output_dir="/path/to/output" \
    minibatch_resume=true
```

**如果配置文件没有 `minibatch` 字段**，需要使用 `+` 前缀来添加新配置项：

```bash
python habitat_evaluation.py --dataset hm3dv1 +minibatch="path/to/your/minibatch.json"
```

**Hydra 语法说明**：
- `key=value`：覆盖已存在的配置项
- `+key=value`：添加新的配置项（仅在配置文件中不存在该字段时使用）

## JSON 文件格式

创建一个 JSON 文件，格式如下：

```json
{
  "episodes": [
    {
      "scene": "scene_id_1",
      "episode_id": 0
    },
    {
      "scene": "scene_id_1",
      "episode_id": 1
    },
    {
      "scene": "scene_id_2",
      "episode_id": 5
    }
  ]
}
```

### 字段说明

- `episodes`: 必需，包含要运行的 episode 列表
- `scene`: episode 所属的场景 ID（字符串）
- `episode_id`: episode 的 ID（整数）

### 示例 JSON 文件

创建一个文件 `minibatch_example.json`：

```json
{
  "episodes": [
    {
      "scene": "00003",
      "episode_id": 0
    },
    {
      "scene": "00003",
      "episode_id": 1
    },
    {
      "scene": "00005",
      "episode_id": 0
    }
  ]
}
```

## 输出目录

### 自动生成（默认）

如果不指定 `minibatch_output_dir`，系统会自动生成输出目录：

```
/home/hdd2/chaiqi/Apexnav/videos/test_{dataset}_{split}_minibatch_{timestamp}
```

例如：
```
/home/hdd2/chaiqi/Apexnav/videos/test_hm3dv1_val_minibatch_20240101_120000
```

### 自定义输出目录

在配置中设置 `minibatch_output_dir` 可以指定固定输出目录，便于断点续跑：

```yaml
minibatch_output_dir: "/path/to/my/minibatch_output"
```

## 断点续跑

默认情况下，minibatch mode 不会继承历史进度（从 0 开始）。如果需要从上次中断的地方继续：

```yaml
minibatch_resume: true
```

或者通过命令行：

```bash
python habitat_evaluation.py --dataset hm3dv1 \
    minibatch="minibatch.json" \
    minibatch_resume=true
```

## 完整示例

### 1. 创建 minibatch JSON 文件

`my_minibatch.json`:
```json
{
  "episodes": [
    {"scene": "00003", "episode_id": 0},
    {"scene": "00003", "episode_id": 1},
    {"scene": "00005", "episode_id": 0}
  ]
}
```

### 2. 运行 minibatch mode

```bash
python habitat_evaluation.py --dataset hm3dv1 minibatch="my_minibatch.json"
```

### 3. 使用自定义输出目录和断点续跑

```bash
python habitat_evaluation.py --dataset hm3dv1 \
    minibatch="my_minibatch.json" \
    minibatch_output_dir="./outputs/my_minibatch" \
    minibatch_resume=true
```

## 注意事项

1. **Scene ID 匹配**: 代码会自动处理不同的 scene ID 格式（如 `hm3d_v0.2/00003` 或 `00003`），只要后缀匹配即可。

2. **找不到 Episode**: 如果指定的 episode 不存在，会打印警告并跳过，继续处理下一个。

3. **进度跟踪**: 在 minibatch mode 下，进度条显示的是 minibatch 列表的大小，而不是整个数据集的 episode 数量。

4. **输出文件**: 所有输出（视频、记录文件等）都会保存到 minibatch 指定的输出目录中。

## 故障排除

如果遇到 "episode not found" 警告：
- 检查 scene ID 是否正确
- 检查 episode_id 是否存在于该 scene 中
- 查看日志中的候选 scene ID 匹配信息

如果 minibatch 文件加载失败：
- 检查 JSON 文件格式是否正确
- 检查文件路径是否正确（相对路径或绝对路径）
- 确保文件编码为 UTF-8

