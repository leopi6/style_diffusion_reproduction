import yaml
import os
import copy
import itertools
import torch
from main import run_experiment  # 导入我们重构的函数

# ----------------------------------------------------
# 1. 定义超参数搜索空间 (根据 ar 的聊天)
# ----------------------------------------------------
# ar 提到 (lambda_style, lambda_l1, lambda_dir)
# 但你的旧代码只有 lambda_l1 和 lambda_dir (lambda_style 在 `ar` 的新代码里)
# 我将使用 `ar` 提到过的参数
param_grid = {
    'lambda_l1': [5, 10, 20],
    'lambda_dir': [0.1, 1, 10],
    'k': [5, 10]  # 训练轮数 (epochs)
}

# `ar` 提到的其他参数 (你可以稍后添加到 param_grid 中)
# 'style_removal_s_for': [20, 40],
# 'style_removal_s_rev': [20, 40],
# 'style_transfer_s_for': [40],
# 'style_transfer_s_rev': [6, 10],

# ----------------------------------------------------
# 2. 加载基础配置文件
# ----------------------------------------------------
base_config_path = "configs/default.yaml"
print(f"加载基础配置: {base_config_path}")
with open(base_config_path, "r") as f:
    base_cfg = yaml.safe_load(f)

# ----------------------------------------------------
# 3. 准备工作 (预计算 Latents)
# ----------------------------------------------------
print("--- 步骤 1: 运行 Style Removal 和 Precompute Latents ---")
# 我们只需要运行一次 'style_removal' 来准备数据
# 我们复用 'default.yaml' 里的 'run_id' (例如 'test_run')
setup_cfg = copy.deepcopy(base_cfg)
setup_cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
setup_cfg['use_fp16'] = True
setup_cfg['run_id'] = "sweep_setup"  # 专门为此设置一个run_id
setup_cfg['run_mode'] = "style_removal"

# 自动设置路径
setup_cfg['content_processed_path'] = os.path.join(setup_cfg['output_path'], setup_cfg['run_id'], "content_processed")
setup_cfg['style_processed_path'] = os.path.join(setup_cfg['output_path'], setup_cfg['run_id'], "style_processed",
                                                 "style.jpg")
setup_cfg['style_latent_path'] = os.path.join(setup_cfg['output_path'], setup_cfg['run_id'], "style_latents",
                                              "style.pt")
setup_cfg['content_latent_path'] = os.path.join(setup_cfg['output_path'], setup_cfg['run_id'], "content_latents")

# 运行 Style Removal
run_experiment(setup_cfg)

# 运行 Precompute Latents
setup_cfg['run_mode'] = "style_transfer"
setup_cfg['precompute_latents'] = True
run_experiment(setup_cfg)

print("--- 步骤 1 完成: Latents 已在 'output/sweep_setup' 中准备就绪 ---")

# ----------------------------------------------------
# 4. 生成所有参数组合
# ----------------------------------------------------
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"--- 步骤 2: 开始超参数搜索，共 {len(combinations)} 次运行 ---")

# ----------------------------------------------------
# 5. 循环运行所有实验
# ----------------------------------------------------
for i, params in enumerate(combinations):

    # 复制基础配置，防止互相污染
    run_cfg = copy.deepcopy(base_cfg)

    # A. 创建一个唯一的 run_id
    run_id = f"sweep_run_{i + 1}"
    for key, val in params.items():
        run_id += f"_{key}_{val}"

    run_cfg['run_id'] = run_id

    # B. 应用新的超参数
    run_cfg.update(params)

    # C. 设置为 "style_transfer" 模式
    run_cfg['run_mode'] = "style_transfer"

    # D. [关键] 跳过预计算，使用我们第一步的结果
    run_cfg['precompute_latents'] = False

    # E. [关键] 将路径指向我们第一步生成的 latents
    run_cfg['content_processed_path'] = setup_cfg['content_processed_path']
    run_cfg['style_processed_path'] = setup_cfg['style_processed_path']
    run_cfg['style_latent_path'] = setup_cfg['style_latent_path']
    run_cfg['content_latent_path'] = setup_cfg['content_latent_path']

    # F. 确保使用GPU和FP16
    run_cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_cfg['use_fp16'] = True

    # G. 运行实验
    print(f"\n--- 运行 {i + 1}/{len(combinations)}: {run_id} ---")
    print(f"参数: {params}")

    try:
        run_experiment(run_cfg)
        print(f"--- 运行 {i + 1} 完成. 结果保存在: output/{run_id} ---")
    except Exception as e:
        print(f"--- 运行 {i + 1} (ID: {run_id}) 失败! ---")
        print(f"错误: {e}")
        print(f"参数: {params}")
        print("--- 继续下一次运行 ---")

print("--- 超参数搜索全部完成! ---")