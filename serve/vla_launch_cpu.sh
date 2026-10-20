#!/bin/bash
# vla_launch_cpu.sh <plan> <out.jsonl> <parallel> [lp_threads=2] [mesa_dev=9]: pure-VLA eval lanes with Mesa llvmpipe SOFTWARE rendering (no GPU graphics)
PLAN=$1; OUT=$2; P=${3:-48}; TH=${4:-2}; DEV=${5:-9}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LD_LIBRARY_PATH=/data/jinghui.wang/miniconda3/envs/mesa/lib __EGL_VENDOR_LIBRARY_FILENAMES=/data/jinghui.wang/miniconda3/envs/mesa/share/glvnd/egl_vendor.d/50_mesa.json EGL_PLATFORM=surfaceless GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=$TH MUJOCO_EGL_DEVICE_ID=$DEV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VIA_CAM_SCHED=1
export VIA_ROBOCASA_SPLIT=${VIA_ROBOCASA_SPLIT:-pretrain} VIA_NATIVE_RESET=${VIA_NATIVE_RESET:-1}; unset CUDA_VISIBLE_DEVICES
[ -n "$VIA_VLA_BRIDGE_URLS" ] || { echo "VIA_VLA_BRIDGE_URLS not set"; exit 2; }
nl -ba "$PLAN" | xargs -P "$P" -L 1 bash -c 'cd ~/ailab-via-demo-r9s && timeout 3000 /data/jinghui.wang/a100/home/weizhijie/ailab-via/.venv/bin/python /tmp/vla_eval.py $1 $2 "'"$OUT"'" >> "'"$OUT"'.log" 2>&1'
echo VLA_DONE $(date +%H:%M)
