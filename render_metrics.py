#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scene import Scene, SceneDataset
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import (
    # preprocess3dgs_and_all2all,
    # render
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
)
import torchvision
from utils.general_utils import (
    safe_state,
    set_args,
    init_distributed,
    set_log_file,
    set_cur_iter,
)
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu_for_eval
from gaussian_renderer.workload_division import (
    start_strategy_final,
    DivisionStrategyHistoryFinal,
)
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import utils.general_utils as utils
from utils.loss_utils import l2_loss, cosine_loss
from xy_utils.memory import index_to_raw
from xy_utils.visual import vpca_embeddings


def render_set(model_path, name, iteration, views, gaussians, pipeline, scene, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    emb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "embedding")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(emb_path, exist_ok=True)

    dataset = SceneDataset(views)

    set_cur_iter(iteration)
    generated_cnt = 0

    num_cameras = len(views)
    strategy_history = DivisionStrategyHistoryFinal(
        dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )
    progress_bar = tqdm(
        range(1, num_cameras + 1),
        desc="Rendering progress",
        disable=(utils.LOCAL_RANK != 0),
    )

    l2_dist = {}
    cosine_dist = {}
    for idx in range(1, num_cameras + 1, args.bsz):
        progress_bar.update(args.bsz)

        num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
        batched_cameras = dataset.get_batched_cameras(num_camera_to_load)
        batched_strategies, gpuid2tasks = start_strategy_final(
            batched_cameras, strategy_history
        )
        load_camera_from_cpu_to_all_gpu_for_eval(
            batched_cameras, batched_strategies, gpuid2tasks
        )

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras,
            gaussians,
            pipeline,
            background,
            batched_strategies=batched_strategies,
            mode="test",
        )
        batched_image, batched_embeddings, _ = render_final(batched_screenspace_pkg, batched_strategies, use_embed=args.use_embed)

        for camera_id, (image, gt_camera, embedding) in enumerate(
            zip(batched_image, batched_cameras, batched_embeddings)
        ):
            actual_idx = idx + camera_id
            if args.sample_freq != -1 and actual_idx % args.sample_freq != 0:
                continue
            if generated_cnt == args.generate_num:
                break
            # Uncomment for not regenerating images.
            # if os.path.exists(
            #     os.path.join(render_path, "{0:05d}".format(actual_idx) + ".png")
            # ):
            #     continue
            if args.l != -1 and args.r != -1:
                if actual_idx < args.l or actual_idx >= args.r:
                    continue

            generated_cnt += 1

            if (
                image is None or len(image.shape) == 0
            ):  # The image is not rendered locally.
                image = torch.zeros(
                    gt_camera.original_image.shape, device="cuda", dtype=torch.float32
                )

            # process image
            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

            # process embeddings
            model_names = ["clip", "siglip", "dinov2", "seem", "llama3", "llamav"]
            for model in model_names:
                if not getattr(args, "use_{}".format(model)):
                    continue
                emb_proj = scene.emb_proj_ops[model]
                emb_mem = scene.emb_mem_ops[model]
                gt_feature = gt_camera.original_embeddings_backup[model].to('cuda') # [h,w,c]
                pred = embedding[getattr(args, f'{model}_bit')[0]:getattr(args, f'{model}_bit')[1],:,:]
                
                model_embed_path = os.path.join(emb_path, model)
                os.makedirs(model_embed_path, exist_ok=True)
                cv2.imwrite(
                    os.path.join(model_embed_path, "{0:05d}".format(actual_idx) + ".png"),
                    vpca_embeddings(pred.cpu()),
                )

                embedding_resized = F.interpolate(
                    pred[None,], (gt_feature.shape[0], gt_feature.shape[1]), mode='bilinear', align_corners=True
                )[0]
                raw_feature = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=args.softmax_temp).float()
                
                if model not in l2_dist and model not in cosine_dist:
                    l2_dist[model] = []
                    cosine_dist[model] = []

                if torch.isnan(l2_loss(raw_feature, gt_feature, fdim=-1)) or torch.isnan(cosine_loss(raw_feature, gt_feature, dim=-1)):
                    print("NaN detected in distance calculation. Skipping this image.")
                    continue

                l2_dist[model] += [l2_loss(raw_feature, gt_feature, fdim=-1)]
                cosine_dist[model] += [cosine_loss(raw_feature, gt_feature, dim=-1)]
            
            torchvision.utils.save_image(
                image,
                os.path.join(render_path, "{0:05d}".format(actual_idx) + ".png"),
            )
            torchvision.utils.save_image(
                gt_image,
                os.path.join(gts_path, "{0:05d}".format(actual_idx) + ".png"),
            )

            model_embed_path = os.path.join(emb_path, "all")
            os.makedirs(model_embed_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(model_embed_path, "{0:05d}".format(actual_idx) + ".png"),
                vpca_embeddings(embedding),
            )

            # release memory usage
            gt_camera.original_image = None
            gt_camera.original_embeddings_backup = None

        if generated_cnt == args.generate_num:
            break
    
    # Format and print results
    print("Cosine Distances:")
    for model, tensor_list in cosine_dist.items():
        print(f"{model:<7}: {torch.tensor(tensor_list).mean().item():.4f}")
    
    print("L2 Distances:")
    for model, tensor_list in l2_dist.items():
        print(f"{model:<7}: {torch.tensor(tensor_list).mean().item():.4f}")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree, dataset.emb_degree, args.use_embed)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False, _eval=True)
        scene.load_weights(args.load_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                args.load_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                scene,
                background,
            )

        if not skip_test:
            render_set(
                args.load_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                scene,
                background,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--distributed_load", action="store_true")  # TODO: delete this.
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    init_distributed(args)
    # This script only supports single-gpu rendering.
    # I need to put the flags here because the render() function need it.
    # However, disable them during render.py because they are only needed during training.

    log_file = open(
        args.model_path
        + f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
        "w",
    )
    set_log_file(log_file)

    ## Prepare arguments.
    # Check arguments
    init_args(args)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    # Set up global args
    set_args(args)

    print_all_args(args, log_file)

    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(
        lp.extract(args),
        args.iteration,
        pp.extract(args),
        args.skip_train,
        args.skip_test,
    )
