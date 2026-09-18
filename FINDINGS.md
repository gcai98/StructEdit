# StructEdit 前期发现 (2026-09-18, AutoDL 无卡模式)

## 1. OrionEdit 的 information-flow mask 不覆盖 syn→ref
证据: models/transformer_orion.py L69-73 注释 + L706 _apply_orion_information_flow_mask
实际 block 调用仅 5 条: ref→txt, ref→syn, ref→src, src→txt, src→syn
即只保护条件表征不被噪声污染 (reverse-causal)。
**syn query → ref key 完全开放, 无任何空间约束。**
STAGE_A_LAYERS=20 (硬 -inf), STAGE_B_LAYERS=20 (软 soft_beta 减法)。
=> 软硬分层调度有现成先例可引用, 但用在条件保护方向, 与本方案不冲突。

## 2. 多 reference 预合成为单一 canvas
证据: models/pipeline_orion_edit.py L203-232 _build_composited_reference_canvas
横向拼 strip -> 等比缩放 -> 居中贴到 source 尺寸的 canvas。
编辑任务限 1-3 张 (L671), fusion 恰好 2 张 (L678)。
=> ref 分支无 per-reference token 段, 仅靠 2D RoPE 隐式区分。
    (论文里必须回应: RoPE 提供位置差异, 但无 "位置 k <-> 目标区域 k" 的绑定)

## 3. per-ref token 区间可解析计算 (未实测)
_pack_latents (L493-496): 2x2 patchify, 行优先 flatten, t = row*w_p + col
坐标链:
  ref_k 原图 -> strip (x_k = Σ_{i<k} w_i, strip_w=Σw_i, strip_h=max h_i)
  -> scale = min(canvas_w/strip_w, canvas_h/strip_h)
  -> canvas (paste_x=(cw-new_w)//2, paste_y=(ch-new_h)//2)
  -> VAE resize: calculate_dimensions(VAE_IMAGE_SIZE, ar)
  -> // (vae_scale_factor*2) 得 patch 网格
  -> 行优先展开 + seg["ref"].start 偏移
实现: ref_slices.py
预计改动量: 50-80 行, 全在已有 hook 点, 不碰主干不碰训练。

## 4. 容量不均衡 (第二缺陷) —— 修正
官方样例 example3/example4 均用等尺寸 ref (576x1024 x2), 占比 0.422/0.422, 均衡。
=> 不能用官方样例作证据, 必须自建尺寸失配输入。
=> 另注: example3 中 strip(1152x1024) 贴进 canvas(1024x768) 后左右各留 80px 黑边,
   纯黑填充占 canvas 15.6%, 同样进入 ref 分支成为无效 token。
=> output 尺寸 (1184,896) != source (1024,768), 证实 pipeline 内部有 calculate_dimensions
   重对齐, 坐标映射必须以运行时 vae_image_sizes 为准。

## 5. 复现计划变更 (重要)
repo 无 eval 脚本 / 无 metric 实现 / 无 test split (ls -R 确认, 只有
inference.py train.py config/base.py models/*)。
OrionEditBench 无官方划分, README 明示 "partition based on your experimental needs"。
论文用 Qwen-Image-Edit-2509, repo 用 2511。
=> 放弃对齐论文数字。对照 = 自己跑的 OrionEdit-2511 + 自建固定 split (种子写死, 存 json)。
=> 评测链需自建: CLIP-I / DINO / DreamSim / 背景 PSNR。

## 6. 训练配置 (config/base.py, 供后续参考)
rank=256, lr=7e-5, bs=1, grad_accum=6, max_train_steps=3000,
resolution=1024, bf16, 8bit adam, gradient_checkpointing, offload=True

## 7. GPU 模式待实测
- [ ] pipeline.vae_scale_factor 实际值; vae_image_sizes 真实网格
- [ ] build_image_stream_slices 的 seg["ref"] 运行时 start/stop
- [ ] L686-688 `width // multiple_of * multiple_of` 截断的偏移误差
- [ ] 第 4 条 padding 实验 (优先, 一天内可出结论)

## 8. 环境备忘 (AutoDL)
- 无卡模式: 0.5 核 / 2GB 内存 -> hf 默认 xet 会 OOM, 必须 HF_HUB_DISABLE_XET=1 + --max-workers 1
- turbo 代理和 pip/apt 镜像互斥, 装包前要 unset proxy
- 数据盘 100G; 底模 57.7G + OrionEditBench 50G 装不下, bench 要边下边解包边删 tar
- 底模无冗余文件可排除 (transformer 5 片 40G + text_encoder 4 片 16.6G + vae 0.25G)

## 9. 环境实际版本 (与 yml 的偏差)
Python 3.12.3 (yml 要 3.10) —— venv 复用镜像自带 python
torch 2.9.0+cu128 / torchvision 0.24.0+cu128  ✓
transformers 4.57.1 ✓ / peft 0.17.1 ✓ / accelerate 1.11.0 ✓
diffusers: yml 要 0.36.0.dev0 (PyPI 无), 实装 <填实际版本>
huggingface_hub 必须锁 0.36.0 —— diffusers>=0.40 会强升到 1.x, 与 transformers 4.57.1 冲突
numpy 2.5.2 (yml 2.2.6), pillow 12.3.0 (yml 12.0.0) —— 由 torchvision 拉取, 未降级
锁文件: structedit/requirements.lock.txt