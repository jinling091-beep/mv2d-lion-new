import argparse
from pathlib import Path

import torch

'''
python LION/weights/convert_lion_spconv_checkpoint.py \
  --src /media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/weights/checkpoint_epoch_36_nus_mamba.pth \
  --dst /media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/weights/checkpoint_epoch_36_nus_mamba_spconv2.pth

python - <<'PY'
import torch

path = "/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/weights/checkpoint_epoch_36_nus_mamba_spconv2.pth"

checkpoint = torch.load(
    path,
    map_location="cpu",
)

state_dict = checkpoint.get(
    "model_state",
    checkpoint.get("state_dict", checkpoint),
)

for key, value in state_dict.items():
    if (
        key.endswith("sub_conv.0.weight")
        and "backbone_3d" in key
        and value.ndim == 5
    ):
        print(key, tuple(value.shape))
PY

'''


TARGET_KEYS = {
    'backbone_3d.linear_1.downsample_list.0.sub_conv.0.weight',
    'backbone_3d.linear_1.downsample_list.1.sub_conv.0.weight',
    'backbone_3d.dow1.sub_conv.0.weight',

    'backbone_3d.linear_2.downsample_list.0.sub_conv.0.weight',
    'backbone_3d.linear_2.downsample_list.1.sub_conv.0.weight',
    'backbone_3d.dow2.sub_conv.0.weight',

    'backbone_3d.linear_3.downsample_list.0.sub_conv.0.weight',
    'backbone_3d.linear_3.downsample_list.1.sub_conv.0.weight',
    'backbone_3d.dow3.sub_conv.0.weight',

    'backbone_3d.linear_4.downsample_list.0.sub_conv.0.weight',
    'backbone_3d.linear_4.downsample_list.1.sub_conv.0.weight',
    'backbone_3d.dow4.sub_conv.0.weight',
}


def normalized_key(key):
    """Remove wrappers only for matching; saved key remains unchanged."""
    prefixes = (
        'module.',
        'model.',
        'pts_backbone.model.',
    )

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f'Checkpoint must be a dict, got {type(checkpoint)}'
        )

    if 'model_state' in checkpoint:
        return checkpoint['model_state'], 'model_state'

    if 'state_dict' in checkpoint:
        return checkpoint['state_dict'], 'state_dict'

    # Bare state_dict
    if checkpoint and all(
        isinstance(key, str)
        for key in checkpoint.keys()
    ):
        return checkpoint, '<bare_state_dict>'

    raise KeyError(
        'Cannot find model_state or state_dict in checkpoint'
    )


def convert_checkpoint(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if src_path.resolve() == dst_path.resolve():
        raise ValueError(
            'Source and destination must be different. '
            'Do not overwrite the original checkpoint.'
        )

    print(f'Loading: {src_path}')
    checkpoint = torch.load(
        str(src_path),
        map_location='cpu',
    )

    state_dict, state_field = get_state_dict(checkpoint)

    print(f'State field: {state_field}')
    print(f'Parameter count: {len(state_dict)}')

    converted = []
    already_converted = []
    missing_targets = set(TARGET_KEYS)

    for raw_key, value in list(state_dict.items()):
        key = normalized_key(raw_key)

        if key not in TARGET_KEYS:
            continue

        missing_targets.discard(key)

        if not torch.is_tensor(value):
            raise TypeError(
                f'{raw_key} is not a Tensor: {type(value)}'
            )

        if value.ndim != 5:
            raise RuntimeError(
                f'{raw_key}: expected a 5D tensor, '
                f'got shape={tuple(value.shape)}'
            )

        old_shape = tuple(value.shape)

        # Already in current spconv layout:
        # [out, kz, ky, kx, in]
        if old_shape == (128, 3, 3, 3, 128):
            already_converted.append(raw_key)
            print(
                f'Already converted: {raw_key}: '
                f'{old_shape}'
            )
            continue

        # Official checkpoint layout:
        # [out, in, kz, ky, kx]
        if old_shape != (128, 128, 3, 3, 3):
            raise RuntimeError(
                f'Unexpected shape for {raw_key}: '
                f'{old_shape}. Refusing automatic conversion.'
            )

        new_value = value.permute(
            0, 2, 3, 4, 1
        ).contiguous()

        new_shape = tuple(new_value.shape)

        if new_shape != (128, 3, 3, 3, 128):
            raise RuntimeError(
                f'Conversion failed for {raw_key}: '
                f'{old_shape} -> {new_shape}'
            )

        # Keep a plain Tensor; state_dict values do not need
        # to be wrapped in torch.nn.Parameter.
        state_dict[raw_key] = new_value

        converted.append(raw_key)

        print(
            f'Converted: {raw_key}: '
            f'{old_shape} -> {new_shape}'
        )

    if missing_targets:
        print('\nWarning: target keys not found:')
        for key in sorted(missing_targets):
            print(f'  {key}')

    if not converted and not already_converted:
        raise RuntimeError(
            'No target LION spconv weights were found. '
            'Check checkpoint format and key prefixes.'
        )

    dst_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        str(dst_path),
    )

    print('\nConversion completed')
    print(f'Converted: {len(converted)}')
    print(f'Already converted: {len(already_converted)}')
    print(f'Saved to: {dst_path}')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--src',
        required=True,
        help='Original LION checkpoint',
    )
    parser.add_argument(
        '--dst',
        required=True,
        help='Converted checkpoint',
    )

    args = parser.parse_args()

    convert_checkpoint(
        args.src,
        args.dst,
    )


if __name__ == '__main__':
    main()