# 训推差异修复


## 简介

训推差异是由于RL训练过程中，训练器和生成器之间由于后端不同（vLLM vs SGLang vs FSDP vs Megatron），精度不同（FP8 vs FP16 vs BF16 vs FP32），形成了一种类似off-policy gap，会导致训练不稳定和策略崩溃。


## 实现原理


修复训推差异大致可分为两种方法（1）对训练器和生成器进行策略修正（2）使用infer_log_probs直接代替old_log_probs(trainer)进行PPO ratio计算。第二种方案比较直接，我们着重说明第一种方法。

### 对训练器和生成器进行策略修正

### IS权重矫正
通过对训练器（old_log_probs）和生成器（infer_log_prob）之间进行重要性采样矫正，弥合训推差异。与off-policy算法类似，IS权重矫正可以区分token级别和sequence级别，只能选择一个。

### MASK过滤掉不符合条件的样本
与IS权重修正不同的是，此方法对于超过阈值的样本直接进行mask遮掩，过滤掉不符合的样本。涉及的方法有（1）token级别：过滤掉不符合条件的token（2）灾难性token：过滤掉出现灾难性严重偏差的token的句子样本（3）sequence级别：对sequence进行IS计算，过滤掉不符合的句子样本（4）sequence级别，使用几何平均来计算IS权重，但指标也更为敏感


## 关键参数配置

生成器是否返回infer_log_probs

GeneratingArguments：

```yaml
actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: fp16
  generating_args:
    max_new_tokens: ${response_length}
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: ${num_return_sequences_in_group}
    logprobs: 1
```
当logprobs:大于0时，返回infer_log_probs


-------

参数的配置在PPOConfig和中，关键配置参数如下：

```yaml
infer_correction: true 

infer_is_mode: token       # 可选 token sequence 
infer_is_threshold_min: 0.0
infer_is_threshold_max: 2.0     # 1.5~5.0

enable_token_reject: true
infer_token_mask_threshold_min: 0.0
infer_token_mask_threshold_max: 2.0 # 2~10

enable_catastrophic_reject: true
infer_catastrophic_threshold: 1e-4

enable_seq_reject: sequence  可选None sequence geometric
infer_seq_mask_threshold_min: 0.1
infer_seq_mask_threshold_max: 10
```


### infer_correction
- **含义**：控制是否启用训推差异修复机制。若启用，系统将使用 `infer_log_probs` 对策略梯度进行矫正。
- **默认值**：`false`

### infer_is_mode
- **含义**：指定重要性采样（IS）权重的计算粒度。
- **可选值**：
  - `"token"`：每个 token 独立计算 IS 权重
  - `"sequence"`：基于整个 response 序列的 log-ratio 求和后广播至所有 token
  - `"none"`（或未设置）：不应用 IS 加权，权重恒为 1
- **默认值**：若未设置，默认为 `"token"`
- **注意**：不可同时使用多种模式，仅能选择其一。

### infer_is_threshold_min
- **含义**：IS 权重的下限阈值，用于裁剪过小的权重以控制方差。
- **默认值**：`0.0`
- **建议**：通常保留为 `0.0`，以保持无偏性下界

### infer_is_threshold_max
- **含义**：IS 权重的上限阈值，防止极端大的权重主导梯度。
- **默认值**：`2.0`
- **建议**：`"token"`级别推荐为 `1.5 ~ 5.0` `"sequence"`级别推荐为2.0 - 10.0

### enable_token_reject
- **含义**：是否启用 token 级别的样本拒绝机制。
- **默认值**：`false`
- **作用**：结合 `infer_token_mask_threshold_min/max`，mask 掉 IS ratio 超出合法区间的 token。

### infer_token_mask_threshold_min
- **含义**：token 级 IS ratio（`old_log_probs / infer_log_probs` 的指数）的下限。
- **默认值**：`0.0`
- **典型值**：`0.0`通常可设为1/max

### infer_token_mask_threshold_max
- **含义**：遮掩token 级 IS ratio 的上限。
- **默认值**：`2.0`
- **典型范围**：`1.5 ~ 5.0`

### enable_catastrophic_reject
- **含义**：是否启用“灾难性偏差”检测并拒绝整句样本。
- **默认值**：`false`
- **触发条件**：只要序列中存在任一 token 满足 `ratio < infer_catastrophic_threshold`，则整句被 mask。

### infer_catastrophic_threshold
- **含义**：灾难性拒绝的 ratio 下限阈值。
- **默认值**：`1e-4`
- **解释**：当 `infer_log_probs` 远大于 `old_log_probs`（即生成器过于“自信”），导致 `ratio = exp(old - infer)` 极小

### enable_seq_reject
- **含义**：是否启用序列级别的拒绝机制，以及使用何种聚合方式。
- **可选值**：
  - `null` / `false`：禁用
  - `"sequence"`：使用 log-ratio **求和** 计算序列 IS ratio
  - `"geometric"`：使用 log-ratio **平均**（等价于几何平均概率）计算序列 IS ratio
- **默认值**：`null`

### infer_seq_mask_threshold_min
- **含义**：遮掩序列级 IS ratio 的下限。
- **默认值**：`0.1`
- **典型值**：通常可设为1/max，当使用`"geometric"`时，最好强制设为1/max


### infer_seq_mask_threshold_max
- **含义**：遮掩序列级 IS ratio 的上限。
- **默认值**：`10.0`
- **典型范围**：当使用`"sequence"`时，推荐`2.0 ~ 10.0`，但随着长度增加可适当放宽。当使用`"geometric"`时，推荐设置为1.0001 - 1.001



## 使用建议

1. 通常情况下，old_log_prob << infer_log_porb, 阈值的下限就比较重要了。并不建议使用sequence级别的IS或MASK

