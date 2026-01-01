#!/bin/bash
CONDA_ENV="py39"
# --- 配置区域 ---
GPU_SAM=0       # Pane 2: 左下
GPU_BLIP=1      # Pane 3: 右下
GPU_DINO=1      # Pane 0: 左上
GPU_YOLO=0     # Pane 1: 右上

SESSION="apexnav_vlm"

tmux kill-session -t $SESSION 2>/dev/null

tmux new-session -d -s $SESSION -n "VLM_Servers"

tmux split-window -h -t $SESSION:0

tmux split-window -v -t $SESSION:0.0

tmux split-window -v -t $SESSION:0.2

tmux select-layout -t $SESSION tiled

# --- 此时窗格编号分布如下 ---
#  0 (左上)  |  2 (右上)
# -----------+-----------
#  1 (左下)  |  3 (右下)
# -------------------------

# --- 2. 发送命令 (对号入座) ---

# Pane 0 (左上): Grounding Dino (GPU 2)
tmux send-keys -t $SESSION:0.0 ' ' C-m
tmux send-keys -t $SESSION:0.0 "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION:0.0 "export CUDA_VISIBLE_DEVICES=$GPU_DINO" C-m
tmux send-keys -t $SESSION:0.0 "python -m vlm.detector.grounding_dino --port 12181" C-m

# Pane 1 (左下): SAM (GPU 0) - 注意：根据上面的切分逻辑，左下是1
tmux send-keys -t $SESSION:0.1 ' ' C-m
tmux send-keys -t $SESSION:0.1 "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION:0.1 "export CUDA_VISIBLE_DEVICES=$GPU_SAM" C-m
tmux send-keys -t $SESSION:0.1 "python -m vlm.segmentor.sam --port 12183" C-m

# Pane 2 (右上): YOLOv7 (GPU 3)
tmux send-keys -t $SESSION:0.2 ' ' C-m
tmux send-keys -t $SESSION:0.2 "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION:0.2 "export CUDA_VISIBLE_DEVICES=$GPU_YOLO" C-m
tmux send-keys -t $SESSION:0.2 "python -m vlm.detector.yolov7 --port 12184" C-m

# Pane 3 (右下): BLIP2 (GPU 1)
tmux send-keys -t $SESSION:0.3 ' ' C-m
tmux send-keys -t $SESSION:0.3 "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION:0.3 "export CUDA_VISIBLE_DEVICES=$GPU_BLIP" C-m
tmux send-keys -t $SESSION:0.3 "python -m vlm.itm.blip2itm --port 12182" C-m

# 接入会话
tmux attach-session -t $SESSION