"""Compare a real single-instance prediction with the official decoder, on CUDA.

With the official checkout on PYTHONPATH, run:
python wevisone_pixel_gate.py MODEL_DIR OFFICIAL_CHECKOUT/demo/examples/images/seabreeze.jpg
"""
import argparse

import numpy as np
from PIL import Image
import torch
from wevisone_vl import WeVisOneVL
from wevisone_vl.feature_bridge import VisualFeatureCapture
from wevisone_pixel import PixelDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir')
    parser.add_argument('image')
    parser.add_argument('--query', default='person')
    args = parser.parse_args()
    ref = WeVisOneVL(args.model_dir, device='cuda', decoder_batch_size=1)
    adapter = PixelDecoder(args.model_dir, ref.device, ref.dtype)
    image = Image.open(args.image).convert('RGB')
    result = ref.predict(image, task='matting', query=args.query, max_new_tokens=128, decode_matting=False)
    captured = []

    def capture(_module, _args, kwargs):
        captured.append((kwargs['t_seg'], kwargs['grid_hw']))

    hook = ref.mask_decoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with VisualFeatureCapture(ref.model, adapter.layers) as vision:
            boxes, masks = ref.decode_masks(image, result.prompt, result.response)
    finally:
        hook.remove()
    if not masks:
        raise RuntimeError(f'No reference instance for query: {result.response}')
    inputs = (image, captured[0][1], vision.premerge_features, torch.cat([h for h, _ in captured]), boxes)
    expected = np.stack(masks)
    alpha = adapter.decode(*inputs, 'matting')
    mask = adapter.decode(*inputs, 'segmentation')
    assert np.isfinite(alpha).all() and ((alpha > 0) & (alpha < 1)).any()
    np.testing.assert_array_equal(alpha, expected)
    np.testing.assert_array_equal(mask, expected >= 0.5)
    print(f'{result.response}\nshape={alpha.shape} alpha_max_error={np.max(abs(alpha-expected))} '
          f'mask_mismatches={np.count_nonzero(mask != (expected >= 0.5))}')


if __name__ == '__main__':
    main()
