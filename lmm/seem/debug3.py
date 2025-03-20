import sys
import os
import numpy as np

import cv2
import torch
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as d2_utils

from vlcore.trainer.utils.misc import move_batch_to_device, cast_batch_to_half
from vlcore.xy_utils.image2html.visualizer import VL
from vlcore.utils.arguments import load_opt_from_config_file
from vlcore.utils.distributed import init_distributed 
from vlcore.utils.constants import COCO_PANOPTIC_CLASSES
from vlcore.modeling.BaseModel import BaseModel
from vlcore.modeling import build_model

from .grid_sample import create_circular_grid_masks
from .matrix_nms import matrix_nms, resolve_mask_conflicts

metadata = MetadataCatalog.get('coco_2017_train_panoptic')


def build_transform_gen(min_scale, max_scale=None):
    augmentation = []
    augmentation.extend([
        T.ResizeShortestEdge(
            min_scale, max_size=max_scale
        ),
    ])
    return augmentation

def main(args=None):
    '''
    build args
    '''
    seem_cfg = "/data/xueyanz/code/vlcore_v2.0/vlcore/configs/seem/davitd5_unicl_lang_v1.yaml"
    seem_ckpt = "/data/xueyanz/checkpoints/seem/seem_davit_d5.pt"
    image_folder = "/data/xueyanz/data/tandt/train/images"
    image_pths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith(".jpg")]

    opt_seem = load_opt_from_config_file(seem_cfg)
    opt_seem = init_distributed(opt_seem)
    opt_seem['MODEL']['ENCODER']['NAME'] = 'transformer_encoder_deform'

    model = BaseModel(opt_seem, build_model(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()
    model.model.metadata = metadata

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

            for image_pth in image_pths:
                dataset_dict = {}
                image = d2_utils.read_image(image_pth, format="RGB")
                d2_utils.check_image_size(dataset_dict, image)
                tfm_gens = build_transform_gen(min_scale=640, max_scale=1333)
                image_ori, _ = T.apply_transform_gens(tfm_gens, image)
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
                dataset_dict.update({"image": images})
                batched_inputs = [dataset_dict]


                batched_inputs = move_batch_to_device(batched_inputs, 'cuda')
                batched_inputs = cast_batch_to_half(batched_inputs)
                
                images = [x["image"].to(model.model.device) for x in batched_inputs] # x["image"] is a tensor with shape [3, h, w] on cuda.
                images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
                images = ImageList.from_tensors(images, model.model.size_divisibility)

                queries_grounding = None
                features = model.model.backbone(images.tensor)
                mask_features, _, multi_scale_features = model.model.sem_seg_head.pixel_decoder.forward_features(features)

                _, _, h, w = images.tensor.shape
                all_spatial_samples = create_circular_grid_masks(h, w, dot_spacing=60, dot_radius=12)
                interval = 10

                acc_masks = []
                acc_scores = []
                acc_labels = []
                for i in range(0, len(all_spatial_samples), interval):
                    spatial_samples = all_spatial_samples[i:i+interval]
                
                    extra = {}
                    pos_masks = [spatial_samples.to(model.model.device)]
                    pos_masks = ImageList.from_tensors(pos_masks, model.model.size_divisibility).tensor.unbind(0)

                    neg_masks = [(spatial_samples.to(model.model.device) & False)]
                    neg_masks = ImageList.from_tensors(neg_masks, model.model.size_divisibility).tensor.unbind(0)
                    extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})
                    
                    queries_grounding = None
                    results = model.model.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='seg')

                    v_emb = results['pred_smaskembs']
                    pred_smasks = results['pred_smasks']

                    s_emb = results['pred_pspatials']
                    diag_mask = ~(torch.eye(model.model.sem_seg_head.predictor.attention_data.extra['spatial_query_number'], device=s_emb.device).repeat_interleave(model.model.sem_seg_head.predictor.attention_data.extra['sample_size'],dim=0)).bool()
                    offset = torch.zeros_like(diag_mask, device=s_emb.device).float()
                    offset.masked_fill_(diag_mask, float("-inf"))

                    pred_logits = v_emb @ s_emb.transpose(1,2) + offset[None,]
                    bs,_,ns = pred_logits.shape
                    _,_,h,w = pred_smasks.shape

                    logits_idx_y = pred_logits.max(dim=1)[1]
                    logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)[:,None].repeat(1, logits_idx_y.shape[1])
                    logits_idx = torch.stack([logits_idx_x, logits_idx_y]).view(2,-1).tolist()

                    pred_stexts = results['pred_stexts']
                    pred_object_embs = pred_stexts[logits_idx].reshape(bs,ns,-1)
                    pred_object_probs = results['pred_slogits'][logits_idx].softmax(dim=-1)
                    pred_object_probs[:,-1] *= 0.001
                    pred_object_probs, pred_object_ids = pred_object_probs.max(dim=-1)
                    outputs = {"embeddings" : pred_object_embs[0]}
                    pred_masks = F.interpolate(results['prev_mask'], size=images.tensor.shape[-2:], mode='bilinear', align_corners=False)[:,:,:batched_inputs[0]['image'].shape[-2], :batched_inputs[0]['image'].shape[-1]]
                    
                    acc_masks += [pred_masks]
                    acc_scores += [pred_object_probs]
                    acc_labels += [pred_object_ids]

                acc_masks = torch.cat(acc_masks, dim=1)[0]
                acc_scores = torch.cat(acc_scores, dim=0)
                acc_ids = torch.cat(acc_labels, dim=0)
                
                sorted_indices = torch.argsort(acc_scores, descending=True)
                
                _h, _w = h//4, w//4
                _acc_masks = F.interpolate(acc_masks[None], size=(_h, _w), mode='bilinear', align_corners=False)[0]
                _acc_masks = _acc_masks[sorted_indices] > 0
                acc_masks = acc_masks[sorted_indices]
                acc_scores = acc_scores[sorted_indices]
                acc_ids = acc_ids[sorted_indices]

                final_score = matrix_nms(_acc_masks.cuda(), acc_ids.cuda(), acc_scores.cuda())
                keep = final_score > 0.60
                pred_masks = acc_masks[keep] > 0.0
                pred_scores = final_score[keep]
                pred_ids = acc_ids[keep]
                pred_masks = resolve_mask_conflicts(pred_masks, pred_scores)

                # visualization
                # Need to uncomment: outputs.update(self.update_spatial_results(outputs))
                image = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()[:,:,::-1]
                visual_masks = pred_masks.float()
                visual_masks = visual_masks.cpu().numpy()
                visual_mask = VL.overlay_all_masks_to_image(image, visual_masks)
                cv2.imwrite("mask.png", visual_mask)
                import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
    sys.exit(0)