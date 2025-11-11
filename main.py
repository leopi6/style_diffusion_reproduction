# --- 这是 main.py (V9重构版) ---
# --- (已重构为函数，修复OOM、FP16、TypeError，添加进度条) ---

import sys
import os
import traceback
import argparse
import yaml
import logging
from PIL import Image
from datasets import load_dataset
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import clip
import numpy as np
from tqdm.auto import tqdm, trange

# ----------------------------------------------------
# 关键：从 src 和 guided_diffusion 文件夹导入依赖
# ----------------------------------------------------
# (确保 guided-diffusion 文件夹在)
current_working_dir = os.getcwd()
guided_diffusion_path = os.path.join(current_working_dir, 'guided-diffusion')
if guided_diffusion_path not in sys.path:
    sys.path.insert(0, guided_diffusion_path)
    print(f"--- 路径修复 ---")
    print(f"已将 {guided_diffusion_path} 添加到 Python 路径")

try:
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    from src.helper import *
    from src.style_removal import *
    from src.style_transfer import *

    print("--- 所有模块导入成功! ---")
except ImportError as e:
    print(f"--- 导入错误: {e} ---")
    print("请确认 guided-diffusion 和 src 目录存在")


# ----------------------------------------------------
# ‼️ [重构] 核心函数（原main） ‼️
# ----------------------------------------------------
def run_experiment(cfg):
    """
    运行一次完整的实验 (style_removal 或 style_transfer)
    cfg: 一个配置字典 (dict)
    """
    try:
        print(f"--- 启动实验: {cfg.get('run_id')} ---")
        print(f"--- 模式: {cfg.get('run_mode')} ---")

        os.makedirs(cfg['output_path'], exist_ok=True)
        run_output_path = os.path.join(cfg['output_path'], cfg['run_id'])
        os.makedirs(run_output_path, exist_ok=True)

        log_file_name = f"{cfg['run_id']}.log"
        log_file_path = os.path.join(run_output_path, log_file_name)

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(cfg['run_id'])  # 使用唯一logger
        logger.info(f"--- 启动实验 ---")
        logger.info(f"Run parameters: {cfg}")

        # ----------------------------------
        # 模式一: Style Removal
        # ----------------------------------
        if cfg['run_mode'] == 'style_removal':
            logger.info("Starting style removal...")

            options = model_and_diffusion_defaults()
            options.update({
                'attention_resolutions': '32,16,8',
                'class_cond': False,
                'diffusion_steps': cfg['style_removal_s_for'],  # 使用 config
                'image_size': cfg['image_size'],
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_fp16': cfg.get('use_fp16', True),  # (V7 OOM 修复)
                'use_scale_shift_norm': True,
            })
            model, diffusion = create_model_and_diffusion(**options)
            state_dict = torch.load(cfg['pretrained_model_path'], map_location=cfg['device'], weights_only=True)
            model.load_state_dict(state_dict)
            model.eval().to(cfg['device'])

            # (V9 FP16 修复)
            if options['use_fp16']:
                model.convert_to_fp16()
                print("模型已转换为 FP16 (Style Removal)")

            print("预训练模型加载完毕")

            # (Kaggle Debug 修复) 只处理风格图片
            content_output_path = os.path.join(run_output_path, "content_processed")
            os.makedirs(content_output_path, exist_ok=True)
            print(f"处理内容图片 (灰度化)... 保存到 {content_output_path}")
            content_files = [f for f in os.listdir(cfg['content_path']) if
                             f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for image_file in tqdm(content_files, desc="处理内容图片"):
                image = Image.open(os.path.join(cfg['content_path'], image_file)).convert('RGB')
                image_luma = rgb_to_luma_601(image)
                Image.fromarray(image_luma).save(os.path.join(content_output_path, image_file))

            print(f"处理风格图片 (DDIM反转): {cfg['style_path']}")
            style_output_path = os.path.join(run_output_path, "style_processed")
            os.makedirs(style_output_path, exist_ok=True)
            style_image = Image.open(cfg['style_path']).convert('RGB')
            style_image_luma = rgb_to_luma_601(style_image)
            x0 = prepare_image_as_tensor(Image.fromarray(style_image_luma), image_size=cfg['image_size'],
                                         device=cfg['device'])

            print("开始风格图片的前向扩散...")
            t = torch.tensor([diffusion.num_timesteps - 1]).to(cfg['device'])
            ddim_timesteps_forward = np.linspace(0, diffusion.num_timesteps - 1, cfg['style_removal_s_for'], dtype=int)
            x_t = ddim_deterministic(x0, model, diffusion, ddim_timesteps_forward, cfg['device'], logger=logger)

            print("开始风格图片的反向扩散 (风格移除)...")
            ddim_timesteps_backward = np.linspace(0, cfg['style_removal_s_for'] - 1, cfg['style_removal_s_rev'],
                                                  dtype=int)
            ddim_timesteps_backward = ddim_timesteps_backward[::-1]
            assert ddim_timesteps_backward[-1] == 0
            x0_est = ddim_deterministic(x_t, model, diffusion, ddim_timesteps_backward, device=cfg['device'],
                                        logger=logger)
            torch.save(x0_est, os.path.join(style_output_path, 'style.pt'))

            image_recon = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_recon = ((image_recon + 1) / 2).clip(0, 1)
            gray_image = image_recon[..., 0]
            gray_image = (gray_image * 255).astype(np.uint8)
            Image.fromarray(gray_image, mode='L').save(os.path.join(style_output_path, "style.jpg"))

            print("--- 风格移除完成 ---")

        # ----------------------------------
        # 模式二: Style Transfer
        # ----------------------------------
        elif cfg['run_mode'] == 'style_transfer':
            logger.info("Starting style transfer...")

            options = model_and_diffusion_defaults()
            options.update({
                'attention_resolutions': '32,16,8',
                'class_cond': False,
                'diffusion_steps': cfg['style_transfer_s_for'],
                'image_size': cfg['image_size'],
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_fp16': cfg.get('use_fp16', True),  # (V7 OOM 修复)
                'use_scale_shift_norm': True,
            })

            if cfg['precompute_latents']:
                logger.info("Precomputing latents...")
                print("开始预计算 latents (precompute_latents: True)")

                model, diffusion = create_model_and_diffusion(**options)
                state_dict = torch.load(cfg['pretrained_model_path'], map_location=cfg['device'], weights_only=True)
                model.load_state_dict(state_dict)
                model.eval().to(cfg['device'])

                # (V9 FP16 修复)
                if options['use_fp16']:
                    model.convert_to_fp16()
                    print("模型已转换为 FP16 (Latent Precomputation)")

                print("预训练模型加载完毕")
                ddim_timesteps_forward = np.linspace(0, diffusion.num_timesteps - 1, cfg['style_transfer_s_for'],
                                                     dtype=int)

                content_latent_output_path = os.path.join(run_output_path, "content_latents")
                os.makedirs(content_latent_output_path, exist_ok=True)
                print(f"计算内容 latents... 保存到 {content_latent_output_path}")
                content_processed_files = [f for f in os.listdir(cfg['content_processed_path']) if
                                           f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                for image_file in tqdm(content_processed_files, desc="计算内容Latents"):
                    image = Image.open(os.path.join(cfg['content_processed_path'], image_file))
                    image_tensor = prepare_image_as_tensor(image, image_size=cfg['image_size'], device=cfg['device'])
                    image_latent = ddim_deterministic(image_tensor, model, diffusion, ddim_timesteps_forward,
                                                      cfg['device'], logger=logger)
                    torch.save(image_latent,
                               os.path.join(content_latent_output_path, f"{image_file.lower().split('.')[0]}.pt"))

                style_latent_output_path = os.path.join(run_output_path, "style_latents")
                os.makedirs(style_latent_output_path, exist_ok=True)
                print(f"计算风格 latent... 保存到 {style_latent_output_path}")
                style = Image.open(cfg['style_processed_path'])
                style_tensor = prepare_image_as_tensor(style, image_size=cfg['image_size'], device=cfg['device'])
                style_latent = ddim_deterministic(style_tensor, model, diffusion, ddim_timesteps_forward, cfg['device'],
                                                  logger=logger)
                torch.save(style_latent, os.path.join(style_latent_output_path, "style.pt"))
                print("--- Latents 预计算完成 ---")

                del model, diffusion
                torch.cuda.empty_cache()

            logger.info("Starting style transfer fine-tuning...")
            print("--- 开始风格迁移微调 (Style Transfer Fine-tuning) ---")
            print(">>> [OOM修复] 已启动，强制 Batch Size = 1")

            model, diffusion = create_model_and_diffusion(**options)
            state_dict = torch.load(cfg['pretrained_model_path'], map_location=cfg['device'], weights_only=True)
            model.load_state_dict(state_dict)
            model.to(cfg['device'])

            # (V9 FP16 修复)
            if options['use_fp16']:
                model.convert_to_fp16()
                print("模型已转换为 FP16 (Fine-tuning)")

            print("微调：加载预训练模型")

            original_style = Image.open(cfg['style_path'])
            original_style_tensor = prepare_image_as_tensor(original_style, image_size=cfg['image_size'],
                                                            device=cfg['device'])
            style_latent = torch.load(os.path.join(cfg['style_latent_path']), map_location=cfg['device'],
                                      weights_only=True)
            print("微调：加载原始风格图片 和 风格latent")

            content_latents_path = os.path.join(cfg['content_latent_path'])
            content_latents_files = [f for f in os.listdir(content_latents_path) if f.lower().endswith(('.pt'))]
            print(f"微调：找到 {len(content_latents_files)} 个内容 latents 文件")

            clip_model, clip_preprocess = clip.load("ViT-B/32", device=cfg['device'])
            print("微调：加载 CLIP 模型")

            num_epochs = cfg['k']

            for epoch in trange(num_epochs, desc="微调 Epochs"):
                for latent_file in tqdm(content_latents_files, desc=f"Epoch {epoch + 1} Batches", leave=False):
                    current_latent = torch.load(
                        os.path.join(content_latents_path, latent_file),
                        map_location=cfg['device'],
                        weights_only=True
                    )
                    content_latents_batch = [current_latent]

                    current_lr = cfg['lr'] * (cfg['lr_multiplier'] ** epoch)

                    # (V6 TypeError 修复)
                    model = style_diffusion_fine_tuning(
                        original_style_tensor,
                        style_latent,
                        content_latents_batch,
                        model,
                        diffusion,
                        clip_model,
                        clip_preprocess,
                        cfg['style_transfer_s_rev'],
                        1,
                        cfg['k_s'],
                        current_lr,
                        1.0,  # 传入中立的 lr_multiplier
                        cfg['lambda_l1'],
                        cfg['lambda_dir'],
                        cfg['device'],
                        logger=logger,
                    )

                    del current_latent, content_latents_batch
                    torch.cuda.empty_cache()

            print(">>> `style_diffusion_fine_tuning` 循环执行完毕 <<<")
            torch.save(model.state_dict(), os.path.join(run_output_path, f"finetuned_style_model.pt"))
            print(f"微调后的模型已保存到: {run_output_path}")

            logger.info("Generating stylized images using fine-tuned model...")
            print("--- 开始使用微调后的模型生成风格化图片 ---")

            stylized_output_path = os.path.join(run_output_path, "content_stylized")
            os.makedirs(stylized_output_path, exist_ok=True)

            for latent_file in tqdm(content_latents_files, desc="生成风格化图片"):
                x_t = torch.load(
                    os.path.join(content_latents_path, latent_file),
                    map_location=cfg['device'],
                    weights_only=True
                ).clone().to(cfg['device'])

                ddim_timesteps_backward = np.linspace(0, cfg['style_transfer_s_for'] - 1, cfg['style_transfer_s_rev'],
                                                      dtype=int)
                ddim_timesteps_backward = ddim_timesteps_backward[::-1]

                x0_est = ddim_deterministic(x_t, model, diffusion, ddim_timesteps_backward, device=cfg['device'])

                stylized_image = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
                stylized_image = ((stylized_image + 1) / 2).clip(0, 1)
                stylized_image = (stylized_image * 255).astype(np.uint8)
                Image.fromarray(stylized_image).save(
                    os.path.join(stylized_output_path, f"{latent_file.split('.')[0]}.jpg"))

                del x_t
                torch.cuda.empty_cache()

            logger.info("Stylized images generated.")
            print("--- 风格化图片生成完毕 ---")

    except Exception as e:
        print(f"--- 实验 {cfg['run_id']} 失败 ---")
        print("详细错误信息:")
        traceback.print_exc()
        if logger:
            logger.error(f"--- 实验 {cfg['run_id']} 失败 ---")
            logger.error(traceback.format_exc())


# ----------------------------------------------------
# ‼️ [重构] 启动器 ‼️
# ----------------------------------------------------
if __name__ == "__main__":
    # 这个块现在只负责加载配置并调用上面的函数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"--- 独立运行: 加载 {args.config} ---")

    # 自动设置路径 (Kaggle Debug中学到的)
    # 这使得config文件更整洁
    cfg['content_processed_path'] = os.path.join(cfg['output_path'], cfg['run_id'], "content_processed")
    cfg['style_processed_path'] = os.path.join(cfg['output_path'], cfg['run_id'], "style_processed", "style.jpg")
    cfg['style_latent_path'] = os.path.join(cfg['output_path'], cfg['run_id'], "style_latents", "style.pt")
    cfg['content_latent_path'] = os.path.join(cfg['output_path'], cfg['run_id'], "content_latents")
    cfg['use_fp16'] = True  # 强制开启
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_experiment(cfg)