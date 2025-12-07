from typing import Literal, Optional, Tuple, Dict, Any
import torch

class StatsCollector:
    """统一收集诊断指标的类"""
    def __init__(self, prefix: str = "infer_correction"):
        self.prefix = prefix
        self.stats: Dict[str, Any] = {}
        self.tensor_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def add(self, name: str, value: Any):
        """添加标量指标"""
        self.stats[f"{self.prefix}/{name}"] = value.item() if torch.is_tensor(value) else value
    
    def add_tensor_stat(self, name: str, tensor: torch.Tensor, mask: torch.Tensor):
        """添加张量统计指标（延迟计算）"""
        self.tensor_stats[name] = (tensor, mask)
    
    def compute_tensor_stats(self):
        """严格遵循原始代码的数据移动策略"""
        for name, (tensor, mask) in self.tensor_stats.items():
            # 1. 确保在同一设备上
            if tensor.device != mask.device:
                mask = mask.to(tensor.device)
            
            # 2. 直接在原始代码风格中计算：先筛选，再移动到CPU
            mask=mask.bool()
            valid = tensor[mask]
            
            # 3. 严格按照原始代码逻辑处理
            if valid.numel() > 0:
                # 关键：先detach()再item()，确保在CPU上计算
                valid_cpu = valid.detach().cpu()
                self.add(f"{name}_mean", valid_cpu.mean().item())
                self.add(f"{name}_std", valid_cpu.std(unbiased=False).item() if valid_cpu.numel() > 1 else 0.0)
                self.add(f"{name}_min", valid_cpu.min().item())
                self.add(f"{name}_max", valid_cpu.max().item())
            else:
                self.add(f"{name}_mean", 0.0)
                self.add(f"{name}_std", 0.0)
                self.add(f"{name}_min", 0.0)
                self.add(f"{name}_max", 0.0)
                
        self.tensor_stats.clear()
    
    def get_metrics(self) -> Dict[str, float]:
        """获取所有指标"""
        return self.stats.copy()

class InferCorrectionHandler:
    """处理重要性采样校正和样本拒绝的核心类"""
    def __init__(self, pipeline_config: "PPOConfig"):
        self.pipeline_config = pipeline_config
        self.stats = StatsCollector()
    
    def __call__(
        self,
        old_log_probs: torch.Tensor,
        infer_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        pg_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        主入口：执行重要性采样校正和样本拒绝
        
        Args:
            old_log_probs: 历史策略的log概率 [B, T]
            infer_log_probs: 生成时策略的log概率 [B, T]
            response_mask: 有效token掩码 [B, T]
            pg_loss: 原始策略梯度损失 [B, T]
        
        Returns:
            weighted_loss: 重加权后的损失
            final_mask: 最终保留的token掩码
            metrics: 诊断指标字典
        """
        # 1. 对齐形状
        infer_log_probs = self._align_shapes(old_log_probs, infer_log_probs, response_mask)
        
        # 2. 计算IS权重
        ratio, raw_is_weight, is_weight = self._compute_is_weights(old_log_probs, infer_log_probs, response_mask)
        
        # 3. 收集基础统计
        self._collect_base_stats(ratio, response_mask)
        
        # 4. 应用拒绝策略
        keep_mask = response_mask.clone()
        keep_mask = self._apply_token_rejection(ratio, keep_mask)
        keep_mask = self._apply_catastrophic_rejection(ratio, keep_mask, response_mask)
        keep_mask = self._apply_sequence_rejection(ratio, keep_mask, response_mask)
        
        # 5. 计算拒绝统计
        self._collect_rejection_stats(ratio, raw_is_weight, keep_mask, response_mask)
        
        # 6. 重加权损失
        weighted_loss = pg_loss * is_weight
        
        # 7. 计算KL指标
        self._compute_kl_metrics(old_log_probs, infer_log_probs, keep_mask, response_mask)
        
        # 8. 批量计算张量统计
        self.stats.compute_tensor_stats()
        
        return weighted_loss, keep_mask, self.stats.get_metrics()
    
    def _align_shapes(
        self,
        old_log_probs: torch.Tensor,
        infer_log_probs: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """对齐log概率张量形状"""
        if infer_log_probs.shape[1] == old_log_probs.shape[1] + 1:
            infer_log_probs = infer_log_probs[:, 1:]
        
        assert old_log_probs.shape == infer_log_probs.shape == response_mask.shape, (
            f"Shape mismatch: old_log_probs {old_log_probs.shape}, "
            f"infer_log_probs {infer_log_probs.shape}, "
            f"response_mask {response_mask.shape}"
        )
        return infer_log_probs
    
    def _compute_is_weights(
        self,
        old_log_probs: torch.Tensor,
        infer_log_probs: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算重要性采样权重
        
        Returns:
            ratio: 原始重要性比率 [B, T]
            raw_is_weight: 未裁剪的IS权重 [B, T]
            is_weight: 裁剪后的IS权重 [B, T]
        """
        log_ratio = old_log_probs - infer_log_probs
        ratio = torch.exp(log_ratio)
        
        if self.pipeline_config.infer_is_mode == "token":
            raw_is_weight = ratio
        elif self.pipeline_config.infer_is_mode == "sequence":
            # 序列级IS：使用序列总log-ratio
            log_ratio_sum = self._masked_sum(log_ratio, response_mask, dim=-1).unsqueeze(-1)
            seq_ratio = torch.exp(log_ratio_sum)
            raw_is_weight = seq_ratio.expand_as(ratio)
            # 收集序列级统计
            self.stats.add_tensor_stat("seq_ratio", seq_ratio.squeeze(-1), torch.ones_like(seq_ratio.squeeze(-1), dtype=torch.bool))
        else:  # "None" or any other value
            raw_is_weight = torch.ones_like(ratio)
        
        # 裁剪IS权重
        is_weight = raw_is_weight.clamp(
            min=self.pipeline_config.infer_is_threshold_min,
            max=self.pipeline_config.infer_is_threshold_max
        ).detach()
        
        return ratio, raw_is_weight, is_weight
    
    def _collect_base_stats(self, ratio: torch.Tensor, response_mask: torch.Tensor):
        """收集基础统计指标"""
        self.stats.add_tensor_stat("token_ratio", ratio, response_mask)
        
        if self.pipeline_config.infer_is_mode in ("token", "sequence"):
            # 1. 裁剪比例统计（现有代码）
            clipped_low = ratio <= self.pipeline_config.infer_is_threshold_min
            clipped_high = ratio >= self.pipeline_config.infer_is_threshold_max
            clipped = clipped_low | clipped_high
            self.stats.add("token_clip_low_frac", self._agg_loss(clipped_low.float(), response_mask))
            self.stats.add("token_clip_high_frac", self._agg_loss(clipped_high.float(), response_mask))
            self.stats.add("token_clip_frac", self._agg_loss(clipped.float(), response_mask))
            
            # 2. 添加缺失的：裁剪后权重的分布统计
            if self.pipeline_config.infer_is_mode == "token":
                # 重新计算裁剪后的权重
                is_weight = ratio.clamp(
                    min=self.pipeline_config.infer_is_threshold_min,
                    max=self.pipeline_config.infer_is_threshold_max
                )
                # 添加缺失的统计
                self.stats.add_tensor_stat("token_is_weight", is_weight, response_mask)
            
            elif self.pipeline_config.infer_is_mode == "sequence":
                # 序列级IS权重已在_compute_is_weights中添加
                pass
    
    def _apply_token_rejection(
        self,
        ratio: torch.Tensor,
        keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """应用token级拒绝策略"""
        if not self.pipeline_config.enable_token_reject:
            return keep_mask
        
        ratio_too_high = ratio > self.pipeline_config.infer_token_mask_threshold_max
        ratio_too_low = ratio < self.pipeline_config.infer_token_mask_threshold_min
        token_reject = ratio_too_high | ratio_too_low
        
        # 更新掩码：丢弃被拒绝的token
        new_keep_mask = keep_mask & (~token_reject)
        
        # 收集统计
        self.stats.add("token_reject_low_frac", self._agg_loss(ratio_too_low.float(), keep_mask))
        self.stats.add("token_reject_high_frac", self._agg_loss(ratio_too_high.float(), keep_mask))
        
        return new_keep_mask
    
    def _apply_catastrophic_rejection(
        self,
        ratio: torch.Tensor,
        keep_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """应用灾难性拒绝策略"""
        if not self.pipeline_config.enable_catastrophic_reject:
            return keep_mask
        
        # 识别灾难性token
        catastrophic = (ratio < self.pipeline_config.infer_catastrophic_threshold) & response_mask
        
        # 检查哪些序列包含灾难性token
        seq_has_catastrophic = catastrophic.any(dim=-1, keepdim=True)
        
        # 更新掩码：丢弃包含灾难性token的整个序列
        new_keep_mask = keep_mask & (~seq_has_catastrophic)
        
        # 收集统计
        catastrophic_token_frac = self._agg_loss(catastrophic.float(), response_mask)
        self.stats.add("catastrophic_token_frac", catastrophic_token_frac)
        
        # 计算包含灾难性token的序列比例
        seq_has_valid = response_mask.any(dim=-1)
        seq_has_catastrophic_flat = catastrophic.any(dim=-1) & seq_has_valid
        catastrophic_seq_frac = (
            seq_has_catastrophic_flat.sum().float() / seq_has_valid.sum().float()
            if seq_has_valid.sum() > 0 else 0.0
        )
        self.stats.add("catastrophic_seq_frac", catastrophic_seq_frac)
        
        return new_keep_mask
    
    def _apply_sequence_rejection(
        self,
        ratio: torch.Tensor,
        keep_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """应用序列级拒绝策略"""
        if self.pipeline_config.enable_seq_reject in (None, "None", "none"):
            return keep_mask
        
        # 计算序列级比率
        if self.pipeline_config.enable_seq_reject == "sequence":
            log_ratio_agg = self._masked_sum(torch.log(ratio), response_mask, dim=-1)
        elif self.pipeline_config.enable_seq_reject == "geometric":
            log_ratio_agg = self._masked_mean(torch.log(ratio), response_mask, dim=-1)
        else:
            return keep_mask
        
        seq_ratio = torch.exp(log_ratio_agg)
        
        # 识别要拒绝的序列
        seq_too_high = seq_ratio > self.pipeline_config.infer_seq_mask_threshold_max
        seq_too_low = seq_ratio < self.pipeline_config.infer_seq_mask_threshold_min
        seq_reject = (seq_too_high | seq_too_low).unsqueeze(-1)
        
        # 更新掩码
        new_keep_mask = keep_mask & (~seq_reject)
        
        # 收集统计
        seq_has_valid = response_mask.any(dim=-1)
        total_valid_seqs = seq_has_valid.sum().item()
        
        seq_reject_low = seq_too_low & seq_has_valid
        seq_reject_high = seq_too_high & seq_has_valid
        
        seq_reject_low_frac = seq_reject_low.sum().item() / total_valid_seqs if total_valid_seqs > 0 else 0.0
        seq_reject_high_frac = seq_reject_high.sum().item() / total_valid_seqs if total_valid_seqs > 0 else 0.0
        
        self.stats.add("seq_reject_low_frac", seq_reject_low_frac)
        self.stats.add("seq_reject_high_frac", seq_reject_high_frac)
        
        return new_keep_mask
    
    def _collect_rejection_stats(
        self,
        ratio: torch.Tensor,
        raw_is_weight: torch.Tensor,
        keep_mask: torch.Tensor,
        response_mask: torch.Tensor
    ):
        """收集拒绝相关的统计指标"""
        
        # 计算被拒绝的token
        rejected_mask = response_mask & (~keep_mask)
        self.stats.add("reject_frac", self._agg_loss(rejected_mask.float(), response_mask))
        

        # 仅在序列拒绝启用时计算序列级拒绝率
        if self.pipeline_config.enable_seq_reject not in (None, "None", "none"):
            seq_has_valid = response_mask.any(dim=-1)
            seq_completely_rejected = (~keep_mask).all(dim=-1) & seq_has_valid
            total_valid_seqs = seq_has_valid.sum().item()
            rejected_seqs = seq_completely_rejected.sum().item()
            seq_reject_frac = rejected_seqs / total_valid_seqs if total_valid_seqs > 0 else 0.0
            self.stats.add("seq_reject_frac", seq_reject_frac)
        else:
            # 未启用时显式设为0.0
            self.stats.add("seq_reject_frac", 0.0)
        

        if self.pipeline_config.infer_is_mode in ("token", "sequence"):
            # 使用已计算的rejected_mask
            clipped_mask = ((raw_is_weight <= self.pipeline_config.infer_is_threshold_min) | 
                        (raw_is_weight >= self.pipeline_config.infer_is_threshold_max)) & response_mask
            
            clip_and_reject_frac = self._agg_loss((clipped_mask & rejected_mask).float(), response_mask)
            clip_or_reject_frac = self._agg_loss((clipped_mask | rejected_mask).float(), response_mask)
            
            self.stats.add("token_clip_and_reject_frac", clip_and_reject_frac)
            self.stats.add("token_clip_or_reject_frac", clip_or_reject_frac)
        else:
            # 关键：为未启用IS的情况提供默认值
            self.stats.add("token_clip_and_reject_frac", 0.0)
            self.stats.add("token_clip_or_reject_frac", 0.0)
    
    def _compute_kl_metrics(
        self,
        old_log_probs: torch.Tensor,
        infer_log_probs: torch.Tensor,
        keep_mask: torch.Tensor,
        response_mask: torch.Tensor
    ):
        """计算KL散度指标"""
        # 原始KL（所有有效token）
        inferkl_orig = self._compute_approx_kl(infer_log_probs, old_log_probs, response_mask, kl_penalty="kl")
        inferkl_orig_agg = self._agg_loss(inferkl_orig, response_mask)
        self.stats.add("inferkl", inferkl_orig_agg)
        
        # 拒绝后KL（仅保留的token）
        inferkl_final = self._compute_approx_kl(infer_log_probs, old_log_probs, keep_mask, kl_penalty="kl")
        inferkl_final_agg = self._agg_loss(inferkl_final, keep_mask)
        self.stats.add("inferkl_reject", inferkl_final_agg)
    
    # --- 辅助方法（使用已有工具函数）---
    def _compute_approx_kl(
        self,
        log_probs: torch.Tensor,
        log_probs_base: torch.Tensor,
        action_mask: torch.Tensor,
        kl_penalty: str = "kl"
    ) -> torch.Tensor:
        """使用已有的compute_approx_kl函数计算近似KL散度"""
        from roll.utils.functionals import compute_approx_kl
        return compute_approx_kl(
            log_probs=log_probs,
            log_probs_base=log_probs_base,
            action_mask=action_mask,
            kl_penalty=kl_penalty
        )
    
    def _agg_loss(self, loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """使用已有的agg_loss函数聚合损失"""
        from roll.utils.functionals import agg_loss
        return agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode=self.pipeline_config.loss_agg_mode
        )
    
    def _masked_sum(self, tensor: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """使用已有的masked_sum函数在掩码区域求和"""
        from roll.utils.functionals import masked_sum
        return masked_sum(tensor, mask, dim=dim)
    
    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """使用已有的masked_mean函数在掩码区域计算均值"""
        from roll.utils.functionals import masked_mean
        return masked_mean(tensor, mask, dim=dim)