import contextlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES


class _PointFeatureEncoderStub:
    def __init__(self, num_point_features):
        self.num_point_features = num_point_features


class _LIONDatasetStub:
    def __init__(self, class_names, point_cloud_range, voxel_size, num_point_features):
        self.class_names = class_names
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        ).astype(np.int64)
        self.depth_downsample_factor = None
        self.point_feature_encoder = _PointFeatureEncoderStub(num_point_features)


@BACKBONES.register_module()
# @DETECTORS.register_module()
class LION3DDetectorAdapter(BaseModule):
    """Wrap a LION/OpenPCDet detector with MV2DFusion's point-branch API."""

    use_fsd_anno = False

    def __init__(
        self,
        cfg_file,
        lion_root='../LION-main',
        ckpt_path=None,
        class_names=None,
        freeze=False,
        norm_eval=True,
        output_channels=128,
        point_cloud_range=None,
        voxel_size=None,
        num_point_features=None,
        strict_ckpt=False,
        map_location='cpu',
        restore_intensity=False, #处理点云强度问题
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.restore_intensity = restore_intensity
        self.lion_root = self._resolve_lion_root(lion_root)
        self.cfg_file = self._resolve_path(cfg_file, [Path.cwd(), self.lion_root, self.lion_root / 'tools'])
        self.ckpt_path = ckpt_path
        self.strict_ckpt = strict_ckpt
        self.map_location = map_location
        self.freeze = freeze
        self.norm_eval = norm_eval
        self.output_channels = output_channels
        self._ckpt_loaded = False

        self.lion_cfg = self._load_lion_cfg()
        self.class_names = list(class_names or self.lion_cfg.CLASS_NAMES)
        data_cfg = self.lion_cfg.DATA_CONFIG
        self.point_cloud_range = list(point_cloud_range or data_cfg.POINT_CLOUD_RANGE)
        self.virtual_voxel_size = list(voxel_size or self._find_voxel_size(data_cfg))
        self.num_point_features = int(
            num_point_features or len(data_cfg.POINT_FEATURE_ENCODING.used_feature_list)
        )

        dataset = _LIONDatasetStub(
            self.class_names,
            self.point_cloud_range,
            self.virtual_voxel_size,
            self.num_point_features,
        )
        from pcdet.models import build_network

        self.model = build_network(
            model_cfg=self.lion_cfg.MODEL,
            num_class=len(self.class_names),
            dataset=dataset,
        )

        voxel_channels = getattr(getattr(self.model, 'backbone_3d', None), 'num_point_features', output_channels)
        query_channels = int(self.lion_cfg.MODEL.DENSE_HEAD.HIDDEN_CHANNEL)
        self.voxel_feat_proj = (
            torch.nn.Linear(voxel_channels, output_channels) if output_channels and voxel_channels != output_channels else None
        )
        self.query_feat_proj = (
            torch.nn.Linear(query_channels, output_channels) if output_channels and query_channels != output_channels else None
        )

        if self.freeze:
            self._freeze_parameters()

    @staticmethod
    def _resolve_path(path, roots):
        path = Path(path)
        if path.is_absolute():
            return path
        for root in roots:
            candidate = (Path(root) / path).resolve()
            if candidate.exists():
                return candidate
        return (Path.cwd() / path).resolve()

    @classmethod
    def _resolve_lion_root(cls, lion_root):
        repo_root = Path(__file__).resolve().parents[5]
        roots = [Path.cwd(), repo_root, repo_root / 'MV2DFusion']
        path = cls._resolve_path(lion_root, roots)
        if not (path / 'pcdet').exists():
            raise FileNotFoundError(f'Cannot find LION pcdet package under {path}')
        return path

    def _load_lion_cfg(self):
        lion_root = str(self.lion_root)
        if lion_root not in sys.path:
            sys.path.insert(0, lion_root)

        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file

        cfg = EasyDict()
        cfg.ROOT_DIR = self.lion_root
        cfg.LOCAL_RANK = 0

        old_cwd = os.getcwd()
        try:
            os.chdir(self.lion_root / 'tools')
            cfg_from_yaml_file(str(self.cfg_file), cfg)
        finally:
            os.chdir(old_cwd)
        return cfg

    @staticmethod
    def _find_voxel_size(data_cfg):
        for processor in data_cfg.DATA_PROCESSOR:
            if 'VOXEL_SIZE' in processor:
                return processor.VOXEL_SIZE
        raise KeyError('LION DATA_CONFIG.DATA_PROCESSOR must define VOXEL_SIZE')

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self):
        super().init_weights()
        if self.ckpt_path is None or self._ckpt_loaded:
            return
        ckpt_path = self._resolve_path(self.ckpt_path, [Path.cwd(), self.lion_root, self.lion_root / 'tools'])
        # #check real pth root
        # print('=' * 80)
        # print('[LION adapter] configured checkpoint:', self.ckpt_path)
        # print('[LION adapter] resolved checkpoint:', ckpt_path)
        # print('[LION adapter] real checkpoint:', ckpt_path.resolve())
        # print('[LION adapter] exists:', ckpt_path.is_file())

        # if ckpt_path.is_file():
        #     print(
        #         '[LION adapter] size:',
        #         f'{ckpt_path.stat().st_size / 1024**2:.2f} MB',
        #     )

        # print('=' * 80)
        if not ckpt_path.exists():
            msg = f'LION checkpoint not found: {ckpt_path}'
            if self.strict_ckpt:
                raise FileNotFoundError(msg)
            warnings.warn(msg)
            return

        checkpoint = torch.load(str(ckpt_path), map_location=self.map_location)
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))

        #check pth shape
        probe_key = (
            'backbone_3d.linear_1.'
            'downsample_list.0.sub_conv.0.weight'
        )

        print(
            '[LION adapter] checkpoint fields:',
            list(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else type(checkpoint),
        )

        if probe_key in state_dict:
            print(
                '[LION adapter] raw checkpoint shape:',
                probe_key,
                tuple(state_dict[probe_key].shape),
            )
        else:
            matched_probe_keys = [
                key for key in state_dict
                if key.endswith(probe_key)
            ]

            print(
                '[LION adapter] matched raw probe keys:',
                matched_probe_keys,
            )

            for key in matched_probe_keys:
                print(
                    '[LION adapter] raw shape:',
                    key,
                    tuple(state_dict[key].shape),
                )

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            key = key[7:] if key.startswith('module.') else key
            key = key[len('model.'):] if key.startswith('model.') else key
            key = key[len('pts_backbone.model.'):] if key.startswith('pts_backbone.model.') else key
            cleaned_state_dict[key] = value
        # print(
        #     '[LION adapter] final cleaned shape:',
        #     probe_key,
        #     tuple(cleaned_state_dict[probe_key].shape),
        # )

        # print(
        #     '[LION adapter] model expected shape:',
        #     probe_key,
        #     tuple(self.model.state_dict()[probe_key].shape),
        # )

        # assert (
        #     tuple(cleaned_state_dict[probe_key].shape)
        #     == tuple(self.model.state_dict()[probe_key].shape)
        # ), (
        #     f'Checkpoint was changed before load_state_dict: '
        #     f'checkpoint={tuple(cleaned_state_dict[probe_key].shape)}, '
        #     f'model={tuple(self.model.state_dict()[probe_key].shape)}'
        # )
        # missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        # if self.strict_ckpt and (missing or unexpected):
        #     raise RuntimeError(
        #         f'LION checkpoint mismatch: '
        #         f'{len(missing)} missing, '
        #         f'{len(unexpected)} unexpected keys.\n'
        #         f'Missing: {missing}\n'
        #         f'Unexpected: {unexpected}'
        #     )

        # keep_vars=True：取得模型真实 Parameter/Buffer 的引用。
        model_state_dict = self.model.state_dict(
            keep_vars=True
        )

        checkpoint_keys = set(
            cleaned_state_dict.keys()
        )
        model_keys = set(
            model_state_dict.keys()
        )

        missing = sorted(
            model_keys - checkpoint_keys
        )
        unexpected = sorted(
            checkpoint_keys - model_keys
        )


        # 检查所有同名参数的shape。
        shape_mismatches = {}

        for key in sorted(
            model_keys & checkpoint_keys
        ):
            source = cleaned_state_dict[key]
            target = model_state_dict[key]

            if not torch.is_tensor(source):
                shape_mismatches[key] = (
                    f'checkpoint value is '
                    f'{type(source)}',
                    tuple(target.shape),
                )
                continue

            if source.shape != target.shape:
                shape_mismatches[key] = (
                    tuple(source.shape),
                    tuple(target.shape),
                )


        if shape_mismatches:
            mismatch_lines = []

            for key, (
                checkpoint_shape,
                model_shape,
            ) in shape_mismatches.items():
                mismatch_lines.append(
                    f'{key}: '
                    f'checkpoint={checkpoint_shape}, '
                    f'model={model_shape}'
                )

            raise RuntimeError(
                'LION checkpoint shape mismatch '
                'before direct copy:\n'
                + '\n'.join(mismatch_lines)
            )


        print(
            '[LION adapter] direct checkpoint load:',
            f'missing={len(missing)},',
            f'unexpected={len(unexpected)}',
        )

        if missing:
            print('[LION adapter] missing keys:')

            for key in missing:
                print(f'  {key}')

        if unexpected:
            print(
                '[LION adapter] unexpected keys:'
            )

            for key in unexpected:
                print(f'  {key}')


        if self.strict_ckpt and (
            missing or unexpected
        ):
            raise RuntimeError(
                f'LION checkpoint key mismatch: '
                f'{len(missing)} missing, '
                f'{len(unexpected)} unexpected'
            )


        # 所有key和shape验证完成后直接复制。
        # 这一步不调用Module.load_state_dict，
        # 因此不会触发spconv的权重布局加载钩子。
        with torch.no_grad():
            for key in sorted(
                model_keys & checkpoint_keys
            ):
                source = cleaned_state_dict[key]
                target = model_state_dict[key]

                if isinstance(
                    source,
                    torch.nn.Parameter,
                ):
                    source = source.detach()

                source = source.to(
                    device=target.device,
                    dtype=target.dtype,
                )

                target.copy_(source)


        probe_key = (
            'backbone_3d.linear_1.'
            'downsample_list.0.sub_conv.0.weight'
        )

        loaded_probe = (
            self.model
            .state_dict()[probe_key]
            .detach()
            .cpu()
        )

        checkpoint_probe = (
            cleaned_state_dict[probe_key]
            .detach()
            .cpu()
        )

        if not torch.equal(
            loaded_probe,
            checkpoint_probe,
        ):
            max_difference = (
                loaded_probe
                - checkpoint_probe
            ).abs().max().item()

            raise RuntimeError(
                f'Direct checkpoint copy verification '
                f'failed for {probe_key}; '
                f'max_difference={max_difference}'
            )

        print(
            '[LION adapter] direct copy verified:',
            probe_key,
            tuple(loaded_probe.shape),
        )


        self._ckpt_loaded = True

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if self.freeze:
            self._freeze_parameters()
        return self

    def _points_to_batch(self, points):
        batched_points = []
        for batch_idx, point in enumerate(points):
            if hasattr(point, 'tensor'):
                point = point.tensor
            # point = point.float()
            # clone 很重要，避免原地修改 MV2DFusion 数据
            point = point.float().clone()

            # MV2DFusion 的 NormalizePoints 将 intensity 除以 255，
            # 恢复成 LION/OpenPCDet checkpoint 使用的原始范围
            if self.restore_intensity:
                if point.size(1) <= 3:
                    raise ValueError(
                        f'Point tensor has only {point.size(1)} dimensions; '
                        'intensity column 3 is missing.'
                    )
                point[:, 3] = point[:, 3] * 255.0


            if point.size(1) < self.num_point_features:
                pad = point.new_zeros((point.size(0), self.num_point_features - point.size(1)))
                point = torch.cat([point, pad], dim=1)
            elif point.size(1) > self.num_point_features:
                point = point[:, :self.num_point_features]

            batch_col = point.new_full((point.size(0), 1), batch_idx)
            batched_points.append(torch.cat([batch_col, point], dim=1))
        return torch.cat(batched_points, dim=0)

    @staticmethod
    def _boxes_to_tensor(boxes):
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        boxes = boxes.float()
        if boxes.size(-1) < 9:
            pad = boxes.new_zeros((boxes.size(0), 9 - boxes.size(-1)))
            boxes = torch.cat([boxes, pad], dim=-1)
        elif boxes.size(-1) > 9:
            boxes = boxes[:, :9]
        return boxes

    def _gt_to_batch(self, gt_bboxes_3d, gt_labels_3d, device):
        if gt_bboxes_3d is None or gt_labels_3d is None:
            return None
        box_list = []
        max_gt = max((len(labels) for labels in gt_labels_3d), default=0)
        if max_gt == 0:
            return torch.zeros((len(gt_labels_3d), 0, 10), dtype=torch.float32, device=device)
        for boxes, labels in zip(gt_bboxes_3d, gt_labels_3d):
            boxes = self._boxes_to_tensor(boxes).to(device)
            labels = labels.to(device=device, dtype=torch.float32) + 1
            gt = torch.cat([boxes, labels[:, None]], dim=-1)
            if gt.size(0) < max_gt:
                pad = gt.new_zeros((max_gt - gt.size(0), gt.size(1)))
                gt = torch.cat([gt, pad], dim=0)
            box_list.append(gt)
        return torch.stack(box_list, dim=0)

    def _build_batch_dict(self, points, gt_bboxes_3d=None, gt_labels_3d=None):
        batch_points = self._points_to_batch(points)
        batch_dict = {
            'points': batch_points,
            'batch_size': len(points),
        }
        gt_boxes = self._gt_to_batch(gt_bboxes_3d, gt_labels_3d, batch_points.device)
        if gt_boxes is not None:
            batch_dict['gt_boxes'] = gt_boxes
        return batch_dict

    def _encoded_voxel_centers(self, sparse_tensor):
        coors = sparse_tensor.indices.long()
        device = coors.device
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device, dtype=torch.float32)
        pc_range = torch.tensor(self.point_cloud_range, device=device, dtype=torch.float32)

        spatial_shape = torch.tensor(sparse_tensor.spatial_shape, device=device, dtype=torch.float32)
        grid_size = torch.tensor(
            (np.asarray(self.point_cloud_range[3:6]) - np.asarray(self.point_cloud_range[:3]))
            / np.asarray(self.virtual_voxel_size),
            device=device,
            dtype=torch.float32,
        )
        z_stride = torch.clamp(grid_size[2] / spatial_shape[0], min=1.0)
        effective_voxel_size = voxel_size.clone()
        effective_voxel_size[2] = effective_voxel_size[2] * z_stride
        return (coors[:, [3, 2, 1]].float() + 0.5) * effective_voxel_size[None, :] + pc_range[None, :3]

    def _format_queries(self, pred_dicts, device):
        query_feats, query_xyz, query_pred, query_cat = [], [], [], []
        for pred_dict in pred_dicts:
            boxes = pred_dict['pred_boxes'].to(device).detach()
            scores = pred_dict['pred_scores'].to(device).detach()
            labels = pred_dict['pred_labels'].to(device=device, dtype=torch.long).detach()
            if labels.numel() > 0 and labels.min() >= 1:
                labels = labels - 1
            labels = labels.clamp(min=0, max=len(self.class_names) - 1)

            feats = pred_dict.get('pred_query_feats', None)
            if feats is None:
                channels = self.output_channels or int(self.lion_cfg.MODEL.DENSE_HEAD.HIDDEN_CHANNEL)
                feats = boxes.new_zeros((boxes.size(0), channels))
            else:
                feats = feats.to(device).detach()
                if self.query_feat_proj is not None:
                    feats = self.query_feat_proj(feats).detach()

            pred_tail = boxes[:, 3:].detach()
            if pred_tail.size(-1) < 6:
                pred_tail = torch.cat([pred_tail, pred_tail.new_zeros((pred_tail.size(0), 6 - pred_tail.size(-1)))], dim=-1)
            elif pred_tail.size(-1) > 6:
                pred_tail = pred_tail[:, :6]

            query_feats.append(feats)
            query_xyz.append(boxes[:, :3].detach())
            query_pred.append(torch.cat([pred_tail, scores[:, None]], dim=-1).detach())
            query_cat.append(labels.detach())
        return query_feats, query_xyz, query_pred, query_cat

    def _format_output(self, batch_dict, losses=None):
        sparse_tensor = batch_dict['encoded_spconv_tensor']
        voxel_feats = sparse_tensor.features
        if self.voxel_feat_proj is not None:
            voxel_feats = self.voxel_feat_proj(voxel_feats)
        voxel_coors = sparse_tensor.indices.long()
        voxel_xyz = self._encoded_voxel_centers(sparse_tensor).detach()

        pred_dicts = batch_dict.get('final_box_dicts', None)
        if pred_dicts is None:
            pred_dicts, _ = self.model.post_processing(batch_dict)
        query_feats, query_xyz, query_pred, query_cat = self._format_queries(pred_dicts, voxel_feats.device)

        return {
            'losses': losses or {},
            'voxel_feats': voxel_feats,
            'voxel_coors': voxel_coors,
            'voxel_xyz': voxel_xyz,
            'query_feats': query_feats,
            'query_cat': query_cat,
            'query_xyz': query_xyz,
            'query_pred': query_pred,
        }

    def _run_lion(self, batch_dict):
        for module in self.model.module_list:
            batch_dict = module(batch_dict)
        return batch_dict

    @staticmethod
    def _clone_pred_dict(pred_dict):
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in pred_dict.items()
        }

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore=None):
        batch_dict = self._build_batch_dict(points, gt_bboxes_3d, gt_labels_3d)
        context = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with context:
            batch_dict = self._run_lion(batch_dict)
            loss, tb_dict = batch_dict['loss'], batch_dict.get('tb_dict', {})
            if 'final_box_dicts' not in batch_dict:
                with torch.no_grad():
                    batch_dict['final_box_dicts'] = self.model.dense_head.get_bboxes(
                        self._clone_pred_dict(batch_dict['lion_preds'])
                    )

        losses = {'loss_trans': loss}
        for key, value in tb_dict.items():
            if key == 'loss_trans' or key.startswith('loss_'):
                continue
            if isinstance(value, torch.Tensor):
                losses[f'stat_{key}'] = value.detach()
            else:
                losses[f'stat_{key}'] = loss.new_tensor(value)
        return self._format_output(batch_dict, losses)

    def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None, return_res=False):
        batch_dict = self._build_batch_dict(points)
        with torch.no_grad():
            batch_dict = self._run_lion(batch_dict)
            if 'final_box_dicts' not in batch_dict:
                batch_dict['final_box_dicts'], _ = self.model.post_processing(batch_dict)
        return self._format_output(batch_dict)
