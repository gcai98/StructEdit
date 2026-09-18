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
diffusers: yml 要 0.36.0.dev0 (PyPI 无), 实装 0.36.0 (正式版)
huggingface_hub 必须锁 0.36.0 —— diffusers>=0.40 会强升到 1.x, 与 transformers 4.57.1 冲突
numpy 2.5.2 (yml 2.2.6), pillow 12.3.0 (yml 12.0.0) —— 由 torchvision 拉取, 未降级
锁文件: structedit/requirements.lock.txt

## 11. 官方样例数据点 (2026-09-18, 无卡模式算出)
example3: ref1/ref2 均 576x1024, source 1024x768, output 1184x896
  -> canvas boxes [(80,0,512,768), (512,0,944,768)], 面积占比 0.422 / 0.422 (均衡)
example4: ref1/ref2 均 576x1024, 无 source (fusion 任务)
=> 官方双 ref 样例均用等尺寸 ref, 看不到容量不均衡; 第 4 条必须自建失配输入验证
=> example3 中 strip(1152x1024) 贴进 canvas(1024x768) 后左右各 80px 黑边,
   纯黑占 canvas 约 15.6%, 同样进入 ref 分支成为无效 token
=> output(1184,896) != source(1024,768), 证实 pipeline 内部 calculate_dimensions 重对齐,
   坐标映射必须以运行时 vae_image_sizes 为准, 不能用原图尺寸

## 12. 代码组织约定
- OrionEdit clone 保持原样, 一个字符不改 (baseline 可验证性)
- 所有改动在 StructEdit repo 内, 通过 monkey patch 替换 T._apply_orion_information_flow_mask
- paths.py 负责 sys.path 注入, 本地/服务器自动适配
- 本地是唯一编辑源, 服务器只 git pull 后运行

## 14. padding 实验的理论预测 (2026-09-18, 尚未上机验证)
控制变量: ref2 像素内容不变, 仅在上下加黑边改变其宽高比。
| pad | ref2尺寸    | ref1占比 | ref2占比 | ref2有效 |
|-----|------------|---------|---------|---------|
| 0.0 | 576x1024   | 0.4219  | 0.4219  | 0.4219  |
| 0.25| 576x1536   | 0.1875  | 0.2812  | 0.1875  |
| 0.5 | 576x2048   | 0.1055  | 0.2109  | 0.1055  |
| 1.0 | 576x3072   | 0.0469  | 0.1406  | 0.0469  |

**核心发现: 容量是耦合的, 不是各自独立的。**
strip_h = max(h for w,h in refs) -> 任意一张 ref 变高会把整条 strip 拉高,
等比缩放后所有 ref 一起缩小。ref1 内容未动, 容量却从 0.4219 降到 0.0469 (1/9)。

=> 待验证的两个现象:
   (a) ref2 保真度随 pad 下降 (预期)
   (b) **ref1 保真度也随 pad 下降** (反直觉, 无法用语义解释, 只能归因于架构)
=> 实验要跑两组 prompt:
   A: "replace the character in Figure 3 with the character in Figure 1."  (ref2 是干扰项)
   B: 双角色替换, 两张 ref 都参与
   若 prompt_A 下 padding ref2 仍影响输出, 证据最强。

## 15. smoke test 结果 (2026-09-18, RTX 4090 49GB)
vae_scale_factor = 8, latent_channels = 16, patch grid divisor = 16
  -> ref_slices.py 的 //16 假设**已验证**
out_size (1184,896) != src (1024,768), 内部 calculate_dimensions 重对齐确认
peak GPU mem 44.95 GB / 48 GB -> 4090 是推理下限, 无余量; 训练必须换 A800-80GB
VAE decode 需 enable_tiling()+enable_slicing(), 否则最后一步 OOM (需 5.12GB 连续分配)

### 与官方 output 的定性差异 (example3, 我们的 prompt 未必与官方一致)
source: 单个背对镜头的皮卡丘, 路左侧
ref1: 猴子道士(正面站立持杖), ref2: 白袍女孩(正面站立)
官方 output: 两角色背影并排走, 姿态适配 source, 有地面阴影, ref 脸部简化
我们的 output: 两角色正面站立, **照搬 ref 原始姿态**, 无地面阴影,
              右下角残留皮卡丘黄色块
=> 疑似 structure-appearance conflict 的实例: 输出应为 source_pose + ref_identity,
   实际是 ref_pose + ref_identity, source 的结构信息未被吸收
=> 待确认: 差异是否来自 prompt 措辞 (官方 example3 prompt 未公开)

## 17. padding 实验结论: 假设被证伪 (2026-09-18, 5 seeds x 4 pad)
官方 prompt, 每格 5 个 seed, DINO 全图相似度:
| pad  | ref1 面积 | ref1 DINO 均值 | 标准差 | ref2 DINO 均值 |
|------|----------|---------------|--------|---------------|
| 0.0  | 0.4219   | 0.2444        | 0.0445 | 0.2541        |
| 0.25 | 0.1875   | 0.2059        | 0.0649 | 0.2873        |
| 0.5  | 0.1055   | 0.2108        | 0.0783 | 0.2933        |
| 1.0  | 0.0469   | 0.2246        | 0.0811 | 0.3259        |

**均值变动幅度 0.039 < 标准差 0.044-0.081, 且不单调。**
结论: padding 造成的 token 容量缩水 (0.42 -> 0.047, 9倍) 对 ref 保真度
**没有可检出的影响**。第 4 条作为失败模式不成立 (作为架构事实仍成立)。

注: n=1 那轮 A 组呈现 0.66->0.56 的完美单调, 是偶然。
    事先定死判读标准 (>0.05 才算信号 + 需多 seed 验证) 避免了误判。

## 18. 评测指标失效 (必须先解决)
同一对图, n=1 轮 ref1 DINO=0.66, 5-seed 轮只有 0.24, 唯一区别是 prompt:
- n=1: "replace the character in Figure 3 with the character in Figure 1" (无姿态约束)
- 5-seed: 官方 prompt, 含 "with their backs facing the camera"
ref1 原图是正面站姿, 输出要求背影 -> **DINO 全图相似度主要在测姿态, 不是 identity**。

=> DINO 全图余弦相似度不适用于本任务。候选替代:
   (a) 先检测/裁剪输出中角色所在区域再比对
   (b) 换用对姿态鲁棒的 identity 特征
   (c) 以官方 output 为参照系, 测与其的差距
   (d) 对 identity 错配这类离散失败, 直接用人工标注/GPT-4V 判定, 不用连续指标
指标未定之前, 不再跑连续指标类实验。

## 19. 当前结论可靠性排序
确凿 (代码/文档级证据, 无需实验):
  - 指令层面无 per-reference 可寻址性 (第16条)
  - information-flow mask 不覆盖 syn->ref 方向 (第1条)
  - 坐标链可解析, vae_scale_factor=8 / grid divisor=16 (第15条实测)
成立但模型不敏感:
  - 容量不均衡与耦合 (第14条的机制成立, 第17条证明无行为影响)
已证伪:
  - padding 导致保真度下降 (第17条)
已降级:
  - structure-appearance conflict (第15条, 系 prompt 缺姿态约束所致)

**尚无已验证的失败模式。下一步应直接做 binding 实验:
  两个相似角色 + 两张 ref, 看 identity 是否搞反。
  该实验不依赖连续指标, 肉眼可判, 且是核心假设。**

## 20. seg 运行时实测 (2026-09-18)
build_image_stream_slices 在 transformer_orion.py:677 (不在 pipeline_tools)
example3 输入 (ref 576x1024 x2, src 1024x768):
  L_syn = L_ref = L_src = 4144
  slices: syn (0,4144) ref (4144,8288) src (8288,12432)
**4144 = 74 x 56** -> latent 网格 74x56, 对应 1184x896 (= smoke 输出尺寸)

=> 修正第 3/15 条: 网格不能用 原图尺寸//16 推算。
   calculate_dimensions(VAE_IMAGE_SIZE, ar) 把所有图重采样到统一像素预算,
   保持宽高比。ref_slices.py 的 grid_w/grid_h 必须从运行时取, 不能内部算。
=> per-ref token 区间: ref 段 [4144, 8288), 网格 74x56, 行优先。
   猴 ≈ 列 0-36, 女孩 ≈ 列 37-73, 再按 paste_x 和黑边裁边缘。
   实验 2 的 mask 坐标基础已具备。
