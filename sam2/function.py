import os
import pandas as pd
import monai
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sam2.util import *
from sam2.modeling.sam2_utils import select_memory_by_similarity_and_iou, update_memory_bank

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def validation_step(args, pack, net, memory_bank_list, device):
    """
    Perform validation on a single batch.
    
    Args:
        args: Arguments containing model configuration
        pack: Batch data dictionary containing 'image', 'mask', 'boxes', etc.
        net: The network model
        memory_bank_list: List of memory bank elements
        device: CUDA device
        
    Returns:
        loss: Loss value for this batch
        eiou: IoU score for this batch
        edice: Dice score for this batch
        memory_bank_list: Updated memory bank list
    """

    name = pack['case']
    imgs = pack['image'].to(dtype = torch.float32, device = device)
    masks = pack['mask'].to(dtype = torch.float32, device = device)
    boxes_torch = pack['boxes'].to(device=device)

    '''test'''
    with torch.no_grad():
        """Compute the image features on given frames."""         
        backbone_out = net.forward_image(imgs)
        # expand the features to have the same dimension as the number of objects
        expanded_image = imgs.expand(args.b, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                args.b, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(args.b, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
        _, vision_feats, vision_pos_embeds, feat_sizes = net._prepare_backbone_features(expanded_backbone_out)

        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        B = args.b
        C = net.hidden_dim
        H, W = feat_sizes[-1] 

        
        memory, memory_pos, sampled_indices = select_memory_by_similarity_and_iou(
            memory_bank_list=memory_bank_list,
            current_vision_feats=vision_feats[-1],
            batch_size=B,
            device=device
        )

        pix_feat_with_mem = net.memory_attention(
            curr=vision_feats[-1:],
            curr_pos=vision_pos_embeds[-1:],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=0
            )
        backbone_features = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        sparse_embeddings, dense_embeddings = net.sam_prompt_encoder(
            points=None, 
            boxes=boxes_torch,
            masks=None,
            batch_size=B,
        )

        
        low_res_multimasks, ious, sam_output_tokens, _ = net.sam_mask_decoder(
            image_embeddings=backbone_features, # (B, 256, 64, 64)
            image_pe=net.sam_prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                            mode="bilinear", align_corners=False)

        # '''memory encoder'''  
        if args.update_memory_bank_during_val:
            if net.non_overlap_masks_for_mem_enc and not net.training:
                # optionally, apply non-overlapping constraints to the masks (it's applied
                # in the batch dimension and should only be used during eval, where all
                # the objects come from the same video under batch size 1).
                high_res_multimasks = net._apply_non_overlapping_constraints(
                    high_res_multimasks
                )
            # scale the raw mask logits with a temperature before applying sigmoid
            is_mask_from_pts = False  # Since we're using bbox prompt
            binarize = net.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
            if binarize and not net.training:
                mask_for_mem = (high_res_multimasks > 0).float()
            else:
                # apply sigmoid on the raw mask logits to turn them into range (0, 1)
                mask_for_mem = torch.sigmoid(high_res_multimasks)
            # apply scale and bias terms to the sigmoid probabilities
            if net.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * net.sigmoid_scale_for_mem_enc
            if net.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + net.sigmoid_bias_for_mem_enc
            maskmem_out = net.memory_encoder(
                backbone_features, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
            )
            maskmem_features = maskmem_out["vision_features"]
            maskmem_pos_enc = maskmem_out["vision_pos_enc"] 
                
            # Update memory bank with new features only if enabled
            memory_bank_list = update_memory_bank(
                memory_bank_list=memory_bank_list,
                maskmem_features=maskmem_features,
                maskmem_pos_enc=maskmem_pos_enc[0] if isinstance(maskmem_pos_enc, list) else maskmem_pos_enc,
                ious=ious,
                backbone_features=backbone_features,
                device=device
            )

        # binary mask and calculate loss, iou, dice
        loss = combined_seg_loss(
            logits=high_res_multimasks,
            masks=masks,
            iou_preds=ious,
            dice_weight=1.0,    
            focal_weight=0.5,   
            iou_weight=0.5,    
            alpha_focal=0.25,
            gamma_focal=2.0,
            use_l1_iou=True
        )
        high_res_multimasks = (high_res_multimasks> 0.5).float()

        eiou, edice = eval_seg(high_res_multimasks, masks, [0.5])

    return loss.item(), eiou, edice, memory_bank_list

 
def validation_step_sam2(args, pack, net, memory_bank_list, device):
    """
    Perform validation on a single batch using standard SAM2 inference.
    
    Args:
        args: Arguments containing model configuration
        pack: Batch data dictionary containing 'image', 'mask', 'boxes', etc.
        net: The network model
        memory_bank_list: List of memory bank elements (not used in this function)
        device: CUDA device
        
    Returns:
        loss: Loss value for this batch
        eiou: IoU score for this batch
        edice: Dice score for this batch
        memory_bank_list: Unchanged memory bank list
    """

    name = pack['case']
    imgs = pack['image'].to(dtype = torch.float32, device = device)
    masks = pack['mask'].to(dtype = torch.float32, device = device)
    boxes_torch = pack['boxes'].to(device=device)

    '''test'''
    with torch.no_grad():
        """Compute the image features on given frames."""         
        backbone_out = net.forward_image(imgs)
        _, vision_feats, _, _ = net._prepare_backbone_features(backbone_out)
        
        # Add no_mem_embed if needed
        if net.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + net.no_mem_embed
        
        # Prepare features in the format expected by SAM decoder
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(imgs.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        
        img_embed = feats[-1]  # (B, 256, 64, 64)
        high_res_features = feats[:-1]  # List of high resolution features

        B = args.b
        
        # Prepare box prompts in the format expected by SAM
        # Convert boxes from (B, 4) to (B, 2, 2) format
        # box_coords = boxes_torch.reshape(-1, 2, 2)
        # box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
        # box_labels = box_labels.repeat(B, 1)
        # concat_points = (box_coords, box_labels)

        # Encode prompts
        # sparse_embeddings, dense_embeddings = net.sam_prompt_encoder(
        #     points=concat_points,
        #     boxes=None,
        #     masks=None,
        # )
        sparse_embeddings, dense_embeddings = net.sam_prompt_encoder(
            points=None, 
            boxes=boxes_torch,
            masks=None,
            batch_size=B,
        )

        # Decode masks
        low_res_multimasks, ious, sam_output_tokens, _ = net.sam_mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=net.sam_prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # Convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks, 
            size=(args.image_size, args.image_size),
            mode="bilinear", 
            align_corners=False
        )

        # Calculate loss
        loss = combined_seg_loss(
            logits=high_res_multimasks,
            masks=masks,
            iou_preds=ious,
            dice_weight=1.0,    
            focal_weight=0.5,   
            iou_weight=0.5,    
            alpha_focal=0.25,
            gamma_focal=2.0,
            use_l1_iou=True
        )
        
        # Binarize masks for evaluation
        high_res_multimasks = (high_res_multimasks > 0.5).float()

        # Calculate metrics
        eiou, edice = eval_seg(high_res_multimasks, masks, [0.5])

    return loss.item(), eiou, edice, memory_bank_list

def validation_step_medsam2(args, step, pack, net, device):
    # eval mode
    # net.eval()
    GPUdevice = device

    # init
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0

    name = pack['case']
    imgs = pack['image'].to(dtype = torch.float32, device = device)
    masks = pack['mask'].to(dtype = torch.float32, device = device)
    boxes_torch = pack['boxes'].to(device=device)

    with torch.no_grad():
        """Compute the image features on given frames."""         

        to_cat_memory = []
        to_cat_memory_pos = []
        to_cat_image_embed = []
        
        if 'pt' in pack:
            pt_temp = pack['pt'].to(device = GPUdevice)
            pt = pt_temp.unsqueeze(1)
            point_labels_temp = pack['p_label'].to(device = GPUdevice)
            point_labels = point_labels_temp.unsqueeze(1)
            coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
        else:
            coords_torch = None
            labels_torch = None

                  
        backbone_out = net.forward_image(imgs)
        _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
        B = vision_feats[-1].size(1)

        """ memory condition """
        if len(memory_bank_list) == 0:
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

        else:
            for element in memory_bank_list:
                maskmem_features = element[0]
                maskmem_pos_enc = element[1]
                to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed
                
            memory_stack_ori = torch.stack(to_cat_memory, dim=0)
            memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
            image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

            vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64) 
            vision_feats_temp = vision_feats_temp.reshape(B, -1)

            image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
            vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
            similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

            similarity_scores = F.softmax(similarity_scores, dim=1) 
            sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]

            memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
            memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

            memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
            memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

            vision_feats[-1] = net.memory_attention(
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0
                )

        feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        
        image_embed = feats[-1]
        high_res_feats = feats[:-1]


        '''prompt encoder'''         

            
        points=(coords_torch, labels_torch)
        flag = True

        se, de = net.sam_prompt_encoder(
            points=None, #(coords_torch, labels_torch)
            boxes=boxes_torch,
            masks=None,
            batch_size=B,
        )

        
        low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=net.sam_prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False, 
            repeat_image=False,  
            high_res_features = high_res_feats
        )

        # prediction
        pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                        mode="bilinear", align_corners=False)
                                        
        """ memory encoder """
        maskmem_features, maskmem_pos_enc = net._encode_new_memory( 
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_multimasks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=flag)  
            
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
        maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
        maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)

        """ memory bank """
        if len(memory_bank_list) < 16:
            for batch in range(maskmem_features.size(0)):
                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                            (maskmem_pos_enc[batch].unsqueeze(0)),
                                            iou_predictions[batch, 0],
                                            image_embed[batch].reshape(-1).detach()])
        
        else:
            for batch in range(maskmem_features.size(0)):
                
                memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                        memory_bank_maskmem_features_norm.t())

                current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                min_similarity_index = torch.argmin(similarity_scores) 
                max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                    if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                        memory_bank_list.pop(max_similarity_index) 
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                    (maskmem_pos_enc[batch].unsqueeze(0)),
                                                    iou_predictions[batch, 0],
                                                    image_embed[batch].reshape(-1).detach()])

        # binary mask and calculate loss, iou, dice
        loss = combined_seg_loss(
            logits=pred,
            masks=masks,
            iou_preds=iou_predictions,
            dice_weight=1.0,    
            focal_weight=0.5,   
            iou_weight=0.5,    
            alpha_focal=0.25,
            gamma_focal=2.0,
            use_l1_iou=True
        )
        pred = (pred> 0.5).float()

        eiou, edice = eval_seg(pred, masks, [0.5])

    return loss.item(), eiou, edice, memory_bank_list

def validation_step_sam(args, step, pack, net, device):
    image = pack['image'].to(dtype = torch.float32, device = device)
    gt2D = pack['mask'].to(dtype = torch.float32, device = device)
    boxes_np = pack['boxes'].detach().cpu().numpy()
    name = pack['case']

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    
    with torch.no_grad():
        if args.use_amp:
            ## AMP
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                medsam_pred = net(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )
        else:
            medsam_pred = net(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())

        pred = (medsam_pred> 0.5).float()

        eiou, edice = eval_seg(pred, gt2D, [0.5])

    return loss.item(), eiou, edice
