import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from sam2.build_sam import build_sam2
from segment_anything import sam_model_registry
import sam2.function as function
from sam2.util import vis_image
import numpy as np
import pickle
import os


class MedicalSegmenter:
    """Simple API for medical image segmentation using SAMed2 and other models"""
    
    def __init__(self, model_type='samed2', checkpoint_path=None, device='cuda:0', memory_bank_size=640):
        """
        Initialize the segmenter
        
        Args:
            model_type: 'samed2', 'medsam', 'sam2', 'medsam2', or 'sam'
            checkpoint_path: Path to model checkpoint
            device: Device to run on (default: 'cuda:0')
            memory_bank_size: Size of memory bank for SAMed2 (default: 640)
        """
        self.model_type = model_type
        self.device = torch.device(device)
        self.image_size = 1024
        self.memory_bank_size = memory_bank_size
        
        # Load model
        print(f"Loading {model_type} model...")
        if model_type in ['samed2', 'medsam2', 'sam2']:
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            config_map = {
                'samed2': 'sam2_hiera_s',
                'medsam2': 'sam2_hiera_t_original',
                'sam2': 'sam2_hiera_s_original'
            }
            self.model, _ = build_sam2(config_map[model_type], checkpoint_path, device=device)
        else:  # medsam or sam
            sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            self.model = function.MedSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
            ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
        
        # Load memory bank for SAMed2 if available
        self.memory_bank_list = []
        if model_type == 'samed2':
            memory_bank_path = f"memory_bank_list_{memory_bank_size}.pkl"
            if os.path.exists(memory_bank_path):
                with open(memory_bank_path, "rb") as f:
                    self.memory_bank_list = pickle.load(f)
                print(f"Loaded memory bank with {len(self.memory_bank_list)} entries")
            else:
                print("No memory bank found, using empty memory")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def predict(self, image, box=None):
        """
        Segment a medical image
        
        Args:
            image: PIL Image, numpy array, or path to image
            box: [x1, y1, x2, y2] bounding box coordinates (0-1024)
                 If None, uses center 80% of image
        
        Returns:
            dict with keys:
                - 'mask': Binary segmentation mask (numpy array)
                - 'iou': Predicted IoU score
                - 'coverage': Percentage of image covered by mask
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # CRITICAL: Duplicate to batch size 2 (model requires minimum batch size of 2)
        image_batch = torch.cat([image_tensor, image_tensor], dim=0)
        
        # Default box if not provided
        if box is None:
            h, w = self.image_size, self.image_size
            box = [w*0.1, h*0.1, w*0.9, h*0.9]
        
        # Also duplicate box for batch size 2
        box_tensor = torch.tensor([box, box], device=self.device)
        
        # Run inference
        with torch.no_grad():
            if self.model_type == 'samed2':
                # SAMed2 inference with memory attention
                backbone_out = self.model.forward_image(image_batch)
                _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)
                
                # Prepare for memory attention
                B = 2  # Batch size is 2
                C = self.model.hidden_dim
                H, W = feat_sizes[-1]
                
                # Use memory bank if available, otherwise use empty memory
                if self.memory_bank_list:
                    # Import the memory selection function
                    from sam2.function import select_memory_by_similarity_and_iou
                    
                    memory, memory_pos, sampled_indices = select_memory_by_similarity_and_iou(
                        memory_bank_list=self.memory_bank_list,
                        current_vision_feats=vision_feats[-1],
                        batch_size=B,
                        device=self.device
                    )
                else:
                    # Use empty memory for single image without memory bank
                    memory = torch.zeros(1, 1, C, device=self.device)
                    memory_pos = torch.zeros(1, 1, C, device=self.device)
                
                # Apply memory attention (critical for SAMed2)
                pix_feat_with_mem = self.model.memory_attention(
                    curr=vision_feats[-1:],
                    curr_pos=vision_pos_embeds[-1:],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                )
                backbone_features = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                
                # Prepare high resolution features if available
                if len(vision_feats) > 1:
                    high_res_features = [
                        x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                        for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
                    ]
                else:
                    high_res_features = None
                
                # Encode prompts
                sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                    points=None, boxes=box_tensor, masks=None, batch_size=B
                )
                
                # Decode masks
                low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
                    image_embeddings=backbone_features,
                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features
                )
                
                # Upscale masks
                masks = F.interpolate(low_res_masks.float(), 
                                    size=(self.image_size, self.image_size),
                                    mode="bilinear", align_corners=False)
                mask = (masks[0, 0] > 0.5).cpu().numpy()
                iou = ious[0].item()
            else:
                # Implement other models as needed
                raise NotImplementedError(f"Model {self.model_type} not implemented yet")
        
        # Calculate coverage
        coverage = mask.sum() / mask.size * 100
        
        return {
            'mask': mask,
            'iou': iou,
            'coverage': coverage
        }
    
    def visualize(self, image, mask, save_path='result.jpg', gt_mask=None):
        """
        Create and save visualization
        
        Args:
            image: Original image (PIL Image or path)
            mask: Segmentation mask (numpy array)
            save_path: Where to save the visualization
            gt_mask: Optional ground truth mask (numpy array)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert to tensor format for vis_image
        image_tensor = self.transform(image).unsqueeze(0)
        image_batch = torch.cat([image_tensor, image_tensor], dim=0)
        
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_batch = torch.cat([mask_tensor, mask_tensor], dim=0)
        
        # Handle optional GT mask
        if gt_mask is not None:
            gt_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0).unsqueeze(0)
            gt_batch = torch.cat([gt_tensor, gt_tensor], dim=0)
            vis_image(image_batch, mask_batch, gt_batch, save_path)
        else:
            # No GT mask - just visualize predictions
            vis_image(image_batch, mask_batch, save_path=save_path)
        
        print(f"Visualization saved to: {save_path}")


# Convenience function for quick usage
def segment(image_path, box=None, model='samed2', checkpoint='checkpoints/latest_epoch_0217.pth'):
    """
    Quick segmentation function
    
    Example:
        result = segment('medical_image.png', box=[100, 100, 900, 900])
        mask = result['mask']
        print(f"IoU: {result['iou']:.4f}")
    """
    segmenter = MedicalSegmenter(model, checkpoint)
    return segmenter.predict(image_path, box) 