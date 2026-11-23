"""Single-image WeVisOne pixel sidecar for callers with captured backbone features.

Requires Tencent/WeVisOne-VL (tested revision 9a6bcd85404505df0d05cfb5d97512ccacf35dee)
on PYTHONPATH. No tokenizer, backbone weights or network access needed here.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from wevisone_vl.config import Qwen3VLSegConfig
from wevisone_vl.mask_decoder import WeVisOneMaskDecoder
from wevisone_vl.feature_bridge import premerge_tokens_to_2d


class PixelDecoder:
    def __init__(self, model_dir, device='cpu', dtype=torch.float32):
        root = Path(model_dir)
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        data = torch.load(root / 'mask_decoder.pt', weights_only=True, map_location='cpu')
        cfg = Qwen3VLSegConfig.from_dict(data['cfg'])
        if cfg.architecture_version != 'wevisone_vl_v1':
            raise ValueError(f'Unsupported decoder version {cfg.architecture_version}')
        self.layers = tuple(cfg.vision_feature_layers)
        self.seg_token_id = data['seg_token_id']
        self.merge_size = data['spatial_merge_size']
        self.canvas = cfg.cnn_image_size
        self.llm_dim = config['text_config']['hidden_size']
        self.net = WeVisOneMaskDecoder(visual_in_dims=[config['vision_config']['hidden_size']] * len(self.layers),
                                       llm_dim=self.llm_dim, cfg=cfg)
        self.net.load_state_dict(data['state_dict'], strict=True)
        self.device, self.dtype = device, dtype
        self.net.to(device=device, dtype=dtype).eval()

    @torch.inference_mode()
    def decode(self, image, grid_hw, premerge, obj_hidden, boxes01, task):
        """Return [N, original H, W], bool for segmentation, float32 for matting.

        premerge[layer] is [grid_h*grid_w,C] in Qwen merger-input order.
        obj_hidden is [N,D] from last LM block output at actual <OBJ> positions.
        boxes01 is [N,4], normalized xyxy in the same response order.
        """
        if task not in ('matting', 'segmentation'):
            raise ValueError('task must be matting or segmentation')
        boxes = torch.as_tensor(boxes01, device=self.device, dtype=self.dtype)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.shape != (len(boxes), 4) or obj_hidden.shape != (len(boxes), self.llm_dim):
            raise ValueError('expected one [4] box and one <OBJ> hidden vector per instance')
        if not ((boxes >= 0) & (boxes <= 1)).all() or not (boxes[:, 2:] > boxes[:, :2]).all():
            raise ValueError('invalid normalized xyxy boxes')
        if len(boxes) == 0:
            return np.empty((0, image.height, image.width), bool if task == 'segmentation' else np.float32)
        features = [premerge_tokens_to_2d(premerge[i], grid_hw, self.merge_size).to(self.device, self.dtype)
                    for i in self.layers]
        rgb = image.convert('RGB').resize((self.canvas, self.canvas), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255).permute(2, 0, 1).contiguous()
        pixels = pixels.unsqueeze(0).to(self.device, self.dtype)
        hidden = obj_hidden.to(self.device, self.dtype)
        maps = []
        for box, token in zip(boxes.split(1), hidden.split(1)):
            logits = self.net(image=pixels, visual_features_2d=features, t_seg=token,
                              boxes01=box, grid_hw=grid_hw)['mask_logits']
            prob = F.interpolate(logits.reshape(1, 1, *logits.shape[-2:]).float().sigmoid(),
                                 size=(image.height, image.width), mode='bilinear', align_corners=False)[0, 0]
            maps.append((prob >= 0.5 if task == 'segmentation' else prob).cpu().numpy())
        return np.stack(maps)
