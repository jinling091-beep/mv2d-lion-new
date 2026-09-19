# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class TwoStageDecoderFinetuneHook(Hook):
    """两阶段解码层微调 hook。

    阶段 1（前 ``switch_ratio`` 比例的 iters）：只训练新增的最后一层解码器
    （``decoder.layers[last]`` 及其 ``cls_branches`` / ``reg_branches`` /
    ``dyn_q_prob_branch`` 分支），其余参数全部冻结，让新层在稳定的预训练
    输出上先站稳。学习率从 ``warmup_ratio * stage1_lr`` 线性升温到
    ``stage1_lr``。

    阶段 2：解冻全部解码层，所有可训练参数统一用 ``stage2_lr`` 微调，并按
    余弦衰减到 ``min_lr_ratio * stage2_lr``。

    注意：本 hook 接管学习率调度，配置里应把 ``lr_config`` 设为 ``None``。
    """

    def __init__(self,
                 switch_ratio=0.15,
                 stage1_lr=1e-4,
                 stage2_lr=1e-5,
                 min_lr_ratio=0.1,
                 warmup_iters=300,
                 warmup_ratio=1.0 / 3,
                 head_attr='fusion_bbox_head',
                 freeze_non_last=True):
        self.switch_ratio = switch_ratio
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.min_lr_ratio = min_lr_ratio
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.head_attr = head_attr
        self.freeze_non_last = freeze_non_last

    @staticmethod
    def _unwrap(model):
        return model.module if is_module_wrapper(model) else model

    def before_run(self, runner):
        model = self._unwrap(runner.model)
        head = getattr(model, self.head_attr)
        num_layers = head.num_pred
        last = num_layers - 1
        assert last >= 1, ('TwoStageDecoderFinetuneHook 需要 num_layers >= 2，'
                           f'当前 num_layers={num_layers}')

        self.switch_iter = int(runner.max_iters * self.switch_ratio)
        self._switched = False
        self._warmup_iters = min(self.warmup_iters, self.switch_iter)

        # 所有解码层分支（含最后一层）的前缀
        branch_markers = (
            f'{self.head_attr}.transformer.decoder.layers.',
            f'{self.head_attr}.cls_branches.',
            f'{self.head_attr}.reg_branches.',
            f'{self.head_attr}.dyn_q_prob_branch.',
        )
        # 最后一层（index = num_layers - 1）的前缀
        last_markers = (
            f'{self.head_attr}.transformer.decoder.layers.{last}.',
            f'{self.head_attr}.cls_branches.{last}.',
            f'{self.head_attr}.reg_branches.{last}.',
            f'{self.head_attr}.dyn_q_prob_branch.{last}.',
        )

        # 记录每个参数原始 requires_grad，供阶段 2 恢复
        self._orig_requires_grad = {}
        for name, p in model.named_parameters():
            self._orig_requires_grad[name] = p.requires_grad

        # 阶段 1：只保留最后一层可训练
        for name, p in model.named_parameters():
            if any(m in name for m in last_markers):
                p.requires_grad_(True)
            elif self.freeze_non_last:
                # 严格"只训第 7 层"：冻结除最后一层外的所有参数
                p.requires_grad_(False)
            elif any(m in name for m in branch_markers):
                # 只冻结前 num_layers-1 层解码器，其他模块照常训练
                p.requires_grad_(False)

        runner.logger.info(
            f'[TwoStageDecoderFinetuneHook] 阶段1：只训练第 {last} 层'
            f'（共 {num_layers} 层），冻结其余参数，'
            f'switch_iter={self.switch_iter}')

    def before_train_iter(self, runner):
        if runner.iter >= self.switch_iter and not self._switched:
            self._switched = True
            model = self._unwrap(runner.model)
            for name, p in model.named_parameters():
                p.requires_grad_(self._orig_requires_grad.get(name, True))
            runner.logger.info(
                '[TwoStageDecoderFinetuneHook] 阶段2：解冻全部解码层，'
                f'整体以 {self.stage2_lr} 低学习率微调')

        self._set_lr(runner)

    def _set_lr(self, runner):
        if runner.iter < self.switch_iter:
            # 阶段 1：warmup 升温到 stage1_lr，之后保持
            if runner.iter < self._warmup_iters:
                progress = runner.iter / max(self._warmup_iters, 1)
                lr = self.stage1_lr * (self.warmup_ratio +
                                       (1.0 - self.warmup_ratio) * progress)
            else:
                lr = self.stage1_lr
        else:
            # 阶段 2：cosine 从 stage2_lr 衰减到 min_lr_ratio * stage2_lr
            total = max(runner.max_iters - self.switch_iter, 1)
            progress = min((runner.iter - self.switch_iter) / total, 1.0)
            lr = self.min_lr_ratio * self.stage2_lr + 0.5 * (
                self.stage2_lr - self.min_lr_ratio * self.stage2_lr) * (
                1.0 + math.cos(math.pi * progress))

        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = lr

    def after_train_iter(self, runner):
        """临时诊断：第一个 iter 后打印第 7 层各参数梯度状态与 loss 信息。

        排查 grad_norm 一直为 0.0 的问题，定位后可删除本方法。
        """
        if getattr(self, '_dbg_done', False):
            return
        self._dbg_done = True
        model = self._unwrap(runner.model)
        head = getattr(model, self.head_attr)
        last = head.num_pred - 1
        keys = (f'.layers.{last}.', f'.cls_branches.{last}.',
                f'.reg_branches.{last}.', f'.dyn_q_prob_branch.{last}.')
        n_frozen = n_none = n_zero = n_nonzero = 0
        frozen_names, none_names, zero_names, nonzero_names = [], [], [], []
        for name, p in model.named_parameters():
            if any(k in name for k in keys):
                if not p.requires_grad:
                    n_frozen += 1
                    if len(frozen_names) < 3:
                        frozen_names.append(name)
                elif p.grad is None:
                    n_none += 1
                    if len(none_names) < 3:
                        none_names.append(name)
                elif p.grad.abs().max().item() == 0.0:
                    n_zero += 1
                    if len(zero_names) < 3:
                        zero_names.append(name)
                else:
                    n_nonzero += 1
                    if len(nonzero_names) < 3:
                        nonzero_names.append(
                            (name, round(p.grad.abs().max().item(), 6)))

        loss = runner.outputs.get('loss', None)
        loss_info = None
        if loss is not None:
            loss_info = (round(float(loss.detach().cpu()), 6),
                         bool(loss.requires_grad))

        runner.logger.info(
            f'[debug] 第{last}层: 非零={n_nonzero}, 零={n_zero}, '
            f'None={n_none}, 冻结={n_frozen}; loss={loss_info}')
        runner.logger.info(f'[debug] 非零示例={nonzero_names}')
        runner.logger.info(f'[debug] 零梯度示例={zero_names}')
        runner.logger.info(f'[debug] None示例={none_names}')

