import os
import torch.nn.functional as F
import torch
import sys
import cv2
import numpy as np
from PIL import Image
import json


# New imports for the diffuser pipeline
from src.pipeline_flux_kontext_control import FluxKontextControlPipeline
from src.transformer_flux import FluxTransformer2DModel

import tempfile
from safetensors.torch import load_file, save_file

_original_load_lora_weights = FluxKontextControlPipeline.load_lora_weights

def _patched_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs):
    """自动转换混合格式的 LoRA 并添加 transformer 前缀"""
    weight_name = kwargs.get("weight_name", "pytorch_lora_weights.safetensors")
    
    if isinstance(pretrained_model_name_or_path_or_dict, str):
        if os.path.isdir(pretrained_model_name_or_path_or_dict):
            lora_file = os.path.join(pretrained_model_name_or_path_or_dict, weight_name)
        else:
            lora_file = pretrained_model_name_or_path_or_dict
        
        if os.path.exists(lora_file):
            state_dict = load_file(lora_file)
            
            # 检查是否需要转换格式或添加前缀
            needs_format_conversion = any('lora_A.weight' in k or 'lora_B.weight' in k for k in state_dict.keys())
            needs_prefix = not any(k.startswith('transformer.') for k in state_dict.keys())
            
            if needs_format_conversion or needs_prefix:
                print(f"🔄 Processing LoRA: {lora_file}")
                if needs_format_conversion:
                    print(f"   - Converting PEFT format to diffusers format")
                if needs_prefix:
                    print(f"   - Adding 'transformer.' prefix to keys")
                
                converted_state = {}
                converted_count = 0
                
                for key, value in state_dict.items():
                    new_key = key
                    
                    # 步骤 1: 转换 PEFT 格式到 diffusers 格式
                    if 'lora_A.weight' in new_key:
                        new_key = new_key.replace('lora_A.weight', 'lora.down.weight')
                        converted_count += 1
                    elif 'lora_B.weight' in new_key:
                        new_key = new_key.replace('lora_B.weight', 'lora.up.weight')
                        converted_count += 1
                    
                    # 步骤 2: 添加 transformer 前缀（如果还没有的话）
                    if not new_key.startswith('transformer.'):
                        new_key = f'transformer.{new_key}'
                    
                    converted_state[new_key] = value
                
                if needs_format_conversion:
                    print(f"   ✅ Converted {converted_count} PEFT keys")
                print(f"   ✅ Total keys: {len(converted_state)}")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = os.path.join(temp_dir, weight_name)
                    save_file(converted_state, temp_file)
                    return _original_load_lora_weights(self, temp_dir, **kwargs)
            else:
                print(f"✅ LoRA already in correct format: {lora_file}")
    
    # 不需要转换，使用原始方法
    return _original_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs)

# 应用 monkey patch
FluxKontextControlPipeline.load_lora_weights = _patched_load_lora_weights
print("✅ Monkey patch applied to FluxKontextPipeline.load_lora_weights")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..', 'comfy_extras')))

from train.src.condition.edge_extraction import InformativeDetector, HEDDetector
from utils_node import BlendInpaint, JoinImageWithAlpha, GrowMask, InvertMask, ColorDetector
from segment_anything import sam_model_registry, SamPredictor

TEST_MODE = False

class KontextEditModel():
    def __init__(self, base_model_path="black-forest-labs/FLUX.1-Kontext-dev", device="cuda",
                 aux_lora_dir="models/v2_ckpt", easycontrol_base_dir="models/v2_ckpt",
                 aux_lora_weight_name="puzzle_lora.safetensors",
                 aux_lora_weight=1.0):
        # Keep necessary preprocessors
        self.mask_processor = GrowMask()
        self.scribble_processor = HEDDetector.from_pretrained()
        self.lineart_processor = InformativeDetector.from_pretrained()
        self.color_processor = ColorDetector()
        self.blender = BlendInpaint()

        # Initialize the new pipeline (Kontext version)
        self.device = device
        self.pipe = FluxKontextControlPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_pretrained(
            base_model_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            device=self.device
        )
        self.pipe.transformer = transformer
        self.pipe.to(self.device, dtype=torch.bfloat16)

        control_lora_config = {
            "local": {
                "path": os.path.join(easycontrol_base_dir, "local_lora.safetensors"),
                "lora_weights": [1.0],
                "cond_size": 512,
            },
            "removal": {
                "path": os.path.join(easycontrol_base_dir, "removal_lora.safetensors"),
                "lora_weights": [1.0],
                "cond_size": 512,
            },
            "edge": {
                "path": os.path.join(easycontrol_base_dir, "edge_lora.safetensors"),
                "lora_weights": [1.0],
                "cond_size": 512,
            },
            "color": {
                "path": os.path.join(easycontrol_base_dir, "color_lora.safetensors"),
                "lora_weights": [1.0],
                "cond_size": 512,
            },
        }
        self.pipe.load_control_loras(control_lora_config)

        # Aux LoRA for foreground mode
        self.aux_lora_weight_name = aux_lora_weight_name
        self.aux_lora_dir = aux_lora_dir
        self.aux_lora_weight = aux_lora_weight
        self.aux_adapter_name = "aux"
        
        from safetensors.torch import load_file as _sft_load
        aux_path = os.path.join(self.aux_lora_dir, self.aux_lora_weight_name)
        if os.path.isfile(aux_path):
            self.pipe.load_lora_weights(aux_path, adapter_name=self.aux_adapter_name)
            print(f"Loaded aux LoRA: {aux_path}")
            # Ensure aux LoRA is disabled by default; it will be enabled only in foreground_edit
            self._disable_aux_lora()
        else:
            print(f"Aux LoRA not found at {aux_path}, foreground mode will run without it.")


    # gamma is now applied inside the pipeline based on control_dict

    def _tensor_to_pil(self, tensor_image):
        # Converts a ComfyUI-style tensor [1, H, W, 3] to a PIL Image
        return Image.fromarray(np.clip(255. * tensor_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def _pil_to_tensor(self, pil_image):
        # Converts a PIL image to a ComfyUI-style tensor [1, H, W, 3]
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def clear_cache(self):
        for name, attn_processor in self.pipe.transformer.attn_processors.items():
            if hasattr(attn_processor, 'bank_kv'):
                attn_processor.bank_kv.clear()
            if hasattr(attn_processor, 'bank_attn'):
                attn_processor.bank_attn = None

    def _enable_aux_lora(self):
        self.pipe.enable_lora()
        self.pipe.set_adapters([self.aux_adapter_name], adapter_weights=[self.aux_lora_weight])
        print(f"Enabled aux LoRA '{self.aux_adapter_name}' with weight {self.aux_lora_weight}")

    def _disable_aux_lora(self):
        self.pipe.disable_lora()
        print("Disabled aux LoRA")

    def _expand_mask(self, mask_tensor: torch.Tensor, expand: int = 0) -> torch.Tensor:
        if expand <= 0:
            return mask_tensor
        expanded = self.mask_processor.expand_mask(mask_tensor, expand=expand, tapered_corners=True)[0]
        return expanded

    def _tensor_mask_to_pil3(self, mask_tensor: torch.Tensor) -> Image.Image:
        mask_01 = torch.clamp(mask_tensor, 0.0, 1.0)
        if mask_01.ndim == 3 and mask_01.shape[-1] == 3:
            mask_01 = mask_01[..., 0]
        if mask_01.ndim == 3 and mask_01.shape[0] == 1:
            mask_01 = mask_01[0]
        pil = self._tensor_to_pil(mask_01.unsqueeze(-1).repeat(1, 1, 3))
        return pil

    def _apply_black_mask(self, image_tensor: torch.Tensor, binary_mask: torch.Tensor) -> Image.Image:
        # image_tensor: [1, H, W, 3] in [0,1]
        # binary_mask: [H, W] or [1, H, W], 1=mask area (white)
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[0]
        mask_bool = (binary_mask > 0.5)
        img = image_tensor.clone()
        img[0][mask_bool] = 0.0
        return self._tensor_to_pil(img)

    def edge_edit(self,
                image, colored_image, positive_prompt, 
                base_mask, add_mask, remove_mask, 
                fine_edge, 
                edge_strength, color_strength,
                seed, steps, cfg):

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare mask and original image
        original_image_tensor = image.clone()
        original_mask = base_mask
        original_mask = self._expand_mask(original_mask, expand=10)
        
        image_pil = self._tensor_to_pil(image)
        control_dict = {}
        lineart_output = None

        # Determine control type: color or edge
        if not torch.equal(image, colored_image):
            print("Apply color control")
            colored_image_pil = self._tensor_to_pil(colored_image)
            # Create color block condition
            color_image_np = np.array(colored_image_pil)
            downsampled = cv2.resize(color_image_np, (32, 32), interpolation=cv2.INTER_AREA)
            upsampled = cv2.resize(downsampled, (256, 256), interpolation=cv2.INTER_NEAREST)
            color_block = Image.fromarray(upsampled)
            # Create grayscale condition
            
            control_dict = {
                "type": "color",
                "spatial_images": [color_block],
                "gammas": [color_strength]
            }
        else:
            print("Apply edge control")
            if fine_edge == "enable":
                lineart_image = self.lineart_processor(np.array(self._tensor_to_pil(image.cpu().squeeze())), detect_resolution=1024, style="contour", output_type="pil")
                lineart_output = self._pil_to_tensor(lineart_image)
            else:
                scribble_image = self.scribble_processor(np.array(self._tensor_to_pil(image.cpu().squeeze())), safe=True, resolution=512, output_type="pil")
                lineart_output = self._pil_to_tensor(scribble_image)
            
            if lineart_output is None:
                raise ValueError("Preprocessor failed to generate lineart.")

            # Apply user sketches to the lineart
            add_mask_resized = F.interpolate(add_mask.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
            remove_mask_resized = F.interpolate(remove_mask.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)

            bool_add_mask_resized = (add_mask_resized > 0.5)
            bool_remove_mask_resized = (remove_mask_resized > 0.5)

            lineart_output[bool_remove_mask_resized] = 0.0
            lineart_output[bool_add_mask_resized] = 1.0

            control_dict = {
                "type": "edge",
                "spatial_images": [self._tensor_to_pil(lineart_output)],
                "gammas": [edge_strength]
            }

        # Prepare debug/output images
        colored_image_np = np.array(self._tensor_to_pil(colored_image))
        debug_image = lineart_output if lineart_output is not None else self.color_processor(colored_image_np, detect_resolution=1024, output_type="pil")

        # Run inference
        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=128,
            control_dict=control_dict,
        ).images[0]
        self.clear_cache()

        result_tensor = self._pil_to_tensor(result_pil)
        # final_image = self.blender.blend_inpaint(result_tensor, original_image_tensor, original_mask, kernel=10, sigma=10)[0]
        final_image = result_tensor
        return (final_image, debug_image, original_mask)

    def object_removal(self,
                       image, positive_prompt, 
                       remove_mask, 
                       local_strength,
                       seed, steps, cfg):
        
        generator = torch.Generator(device=self.device).manual_seed(seed)

        original_image_tensor = image.clone()
        original_mask = remove_mask
        original_mask = self._expand_mask(remove_mask, expand=10)
        
        image_pil = self._tensor_to_pil(image)
        # Prepare spatial image: original masked to black in the remove area
        spatial_pil = self._apply_black_mask(image, original_mask)
        # Note: mask is not passed to pipeline; we use it only for blending
        control_dict = {
            "type": "removal",
            "spatial_images": [spatial_pil],
            "gammas": [local_strength]
        }

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            control_dict=control_dict,
        ).images[0]
        self.clear_cache()

        result_tensor = self._pil_to_tensor(result_pil)
        # final_image = self.blender.blend_inpaint(result_tensor, original_image_tensor, original_mask, kernel=10, sigma=10)[0]
        final_image = result_tensor
        return (final_image, self._pil_to_tensor(spatial_pil), original_mask)

    def local_edit(self,
                   image, positive_prompt, fill_mask, local_strength,
                   seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        original_image_tensor = image.clone()
        original_mask = self._expand_mask(fill_mask, expand=10)
        image_pil = self._tensor_to_pil(image)

        spatial_pil = self._apply_black_mask(image, original_mask)
        control_dict = {
            "type": "local",
            "spatial_images": [spatial_pil],
            "gammas": [local_strength]
        }

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=128,
            control_dict=control_dict,
        ).images[0]
        self.clear_cache()

        result_tensor = self._pil_to_tensor(result_pil)
        # final_image = self.blender.blend_inpaint(result_tensor, original_image_tensor, original_mask, kernel=10, sigma=10)[0]
        final_image = result_tensor
        return (final_image, self._pil_to_tensor(spatial_pil), original_mask)

    def foreground_edit(self,
                        merged_image, positive_prompt,
                        add_prop_mask, fill_mask, fix_perspective, grow_size,
                        seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        edit_mask = torch.clamp(self._expand_mask(add_prop_mask, expand=grow_size) + fill_mask, 0.0, 1.0)
        final_mask = self._expand_mask(edit_mask, expand=25)
        if fix_perspective == "enable":
            positive_prompt = positive_prompt + " Fix the perspective if necessary."
        # Prepare edited input image: inside edit_mask but outside add_prop_mask set to white
        img = merged_image.clone()
        base_mask = (edit_mask > 0.5)
        add_only = (add_prop_mask <= 0.5) & base_mask  # [1, H, W] bool
        add_only_3 = add_only.squeeze(0).unsqueeze(-1).expand(-1, -1, img.shape[-1])  # [H, W, 3]
        img[0] = torch.where(add_only_3, torch.ones_like(img[0]), img[0])

        image_pil = self._tensor_to_pil(img)

        # Enable aux LoRA only for foreground
        self._enable_aux_lora()

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=128,
            control_dict=None,
        ).images[0]

        # Disable aux LoRA afterwards
        self._disable_aux_lora()

        final_image = self._pil_to_tensor(result_pil)
        # final_image = self.blender.blend_inpaint(final_image, img, final_mask, kernel=10, sigma=10)[0]
        return (final_image, self._pil_to_tensor(image_pil), edit_mask)

    def kontext_edit(self,
                     image, positive_prompt,
                     seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image_pil = self._tensor_to_pil(image)

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=128,
            control_dict=None,
        ).images[0]

        final_image = self._pil_to_tensor(result_pil)
        mask = torch.zeros((1, final_image.shape[1], final_image.shape[2]), dtype=torch.float32, device=final_image.device)
        return (final_image, image, mask)

    def process(self, image, colored_image, 
                 merged_image, positive_prompt,
                total_mask, add_mask, remove_mask, add_prop_mask, fill_mask, 
                fine_edge, fix_perspective, edge_strength, color_strength, local_strength, grow_size,
                seed, steps, cfg, flag="precise_edit"):
        if flag == "foreground":
            return self.foreground_edit(merged_image, positive_prompt, add_prop_mask, fill_mask, fix_perspective, grow_size, seed, steps, cfg)
        elif flag == "local":
            return self.local_edit(image, positive_prompt, fill_mask, local_strength, seed, steps, cfg)
        elif flag == "removal":
            return self.object_removal(image, positive_prompt, remove_mask, local_strength, seed, steps, cfg)
        elif flag == "precise_edit":
            return self.edge_edit(
                image, colored_image, positive_prompt,
                total_mask, add_mask, remove_mask,
                fine_edge,
                edge_strength, color_strength,
                seed, steps, cfg
            )
        elif flag == "kontext":
            return self.kontext_edit(image, positive_prompt, seed, steps, cfg)
        else:
            raise ValueError("Invalid Editing Type: {}".format(flag))


class SAM():
    def __init__(self):
        self.join_alpha = JoinImageWithAlpha()
        self.invert_mask = InvertMask()
        self.predictor = None
        # Initialize immediately with default or ask user to call load_model
        self.load_model()

    def load_model(self, model_type='vit_b', checkpoint_path='models/sam/sam_vit_b_01ec64.pth', device='cuda'):
        # You need to download the checkpoint manually: 
        # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        if not os.path.exists(checkpoint_path):
            print(f"Warning: SAM Checkpoint not found at {checkpoint_path}. Please download it.")
            return
            
        print(f"Loading SAM model: {model_type} from {checkpoint_path}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def morphological_operations(self, mask, kernel_size=11, iterations=1):
        mask_for_open_close = mask.clone()
        mask_for_close_open = mask.clone()
        
        for i in range(iterations):
            eroded = -F.max_pool2d(-mask_for_open_close, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            dilated = F.max_pool2d(opened, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            open_then_close = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            mask_for_open_close = open_then_close
        
        for i in range(iterations):
            dilated = F.max_pool2d(mask_for_close_open, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            eroded = -F.max_pool2d(-closed, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            close_then_open = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            mask_for_close_open = close_then_open
        
        final_mask = torch.min(open_then_close, close_then_open)
        return final_mask

    def process(self, image, keep_model_loaded=True, coordinates_positive=None, coordinates_negative=None, individual_objects=False, bboxes=None, mask=None):
        if self.predictor is None:
            self.load_model()
            if self.predictor is None:
                raise RuntimeError("SAM model not loaded.")

        # Prepare image for SAM (numpy uint8)
        # image tensor is [1, H, W, 3] float 0-1
        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        self.predictor.set_image(image_np)

        input_point = []
        input_label = []
        
        # Process points
        if coordinates_positive:
            coords = json.loads(coordinates_positive) if isinstance(coordinates_positive, str) else coordinates_positive
            for p in coords:
                input_point.append([p['x'], p['y']])
                input_label.append(1) # 1 = foreground
                
        if coordinates_negative:
            coords = json.loads(coordinates_negative) if isinstance(coordinates_negative, str) else coordinates_negative
            for p in coords:
                input_point.append([p['x'], p['y']])
                input_label.append(0) # 0 = background

        # Process bbox
        input_box = None
        if bboxes:
            
            box_list = []
            for box in bboxes:
                box_list.append(list(box))
            
            if len(box_list) > 0:
                input_box = np.array(box_list)

        if len(input_point) > 0:
            input_point = np.array(input_point)
            input_label = np.array(input_label)
        else:
            input_point = None
            input_label = None

        # Predict
        # We use multimask_output=False to get single best mask
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        
        # masks: [1, H, W]
        mask_np = masks[0]
        
        # Convert back to tensor [1, H, W]
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        
        invert_mask = self.invert_mask.invert(mask)[0]
        image_with_alpha = self.join_alpha.join_image_with_alpha(image, invert_mask)[0]

        return (image_with_alpha, mask)
