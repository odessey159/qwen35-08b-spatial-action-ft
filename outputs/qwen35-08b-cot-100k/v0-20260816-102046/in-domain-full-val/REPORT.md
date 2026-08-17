# CoT-100K in-domain 自由生成评测

95% CI 使用按 `scene_id` 重采样的 1000 次 percentile cluster bootstrap。

## 主指标

| 指标 | A（正确图） | A′（换图） |
|:---|---:|---:|
| action sequence exact | 91.59% | 37.55% |
| CF pair exact | 79.14% | 0.22% |
| step position match recall | 96.65% | 71.72% |

A − A′ action exact = **+54.03%** （95% CI +52.43% 到 +55.53%）。

## 感知 / 规划归因与格式健康度（A）

| 指标 | 数值 |
|:---|---:|
| state fact P / R / F1 | 87.60% / 88.21% / 87.90% |
| P(action 对 \| state exact) | 100.00% |
| P(action 对 \| state 错) | 83.75% |
| strict structure valid | 100.00% |
| action vocab 越界率 | 0.00% |
| object vocab 越界率 | 0.00% |
| placeholder copy | 0.00% |
| 同切片 train-fitted text oracle | 62.54%（coverage 99.96%） |

## 必报切片（A）

| 切片 | n | action exact | 95% CI | state F1 | text oracle |
|:---|---:|---:|:---:|---:|---:|
| cf:no | 6231 | 94.32% | 93.55%–95.13% | 89.07% | 74.42% |
| cf:yes | 3720 | 87.02% | 85.38%–88.46% | 85.77% | 42.63% |
| container_type:Cabinet | 52 | 88.46% | 78.26%–96.43% | 85.25% | 46.15% |
| container_type:Drawer | 2772 | 86.76% | 84.84%–88.61% | 84.49% | 40.87% |
| container_type:Fridge | 752 | 88.83% | 86.45%–90.92% | 92.59% | 47.74% |
| container_type:Microwave | 68 | 73.53% | 64.44%–83.33% | 88.46% | 48.53% |
| container_type:Safe | 76 | 89.47% | 81.39%–95.24% | 91.35% | 48.68% |
| plan_length:1 | 1245 | 87.47% | 85.10%–89.72% | 92.75% | 100.00% |
| plan_length:2 | 7476 | 94.54% | 93.82%–95.24% | 89.59% | 62.03% |
| plan_length:4 | 615 | 74.15% | 70.41%–77.87% | 81.07% | 55.45% |
| plan_length:6 | 615 | 81.46% | 77.91%–84.75% | 81.05% | 0.00% |
| receptacle_state:closed | 1860 | 90.97% | 89.52%–92.41% | 86.13% | 0.00% |
| receptacle_state:open | 1860 | 83.06% | 80.66%–85.04% | 85.42% | 85.27% |
| task_group:clean | 1447 | 95.23% | 94.10%–96.40% | 90.73% | 83.28% |
| task_group:counterfactual_put | 3720 | 87.02% | 85.38%–88.46% | 85.77% | 42.63% |
| task_group:open_close | 1062 | 92.47% | 90.47%–94.21% | 88.75% | 67.89% |
| task_group:pickup | 1481 | 99.86% | 99.66%–100.00% | 92.30% | 99.73% |
| task_group:slice | 1057 | 96.50% | 94.76%–97.85% | 88.76% | 67.17% |
| task_group:toggle | 1184 | 85.98% | 83.24%–88.65% | 84.82% | 44.26% |
