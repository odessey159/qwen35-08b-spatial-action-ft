# CoT-100K in-domain 自由生成评测

95% CI 使用按 `scene_id` 重采样的 1000 次 percentile cluster bootstrap。

## 主指标

| 指标 | A（正确图） | A′（换图） |
|:---|---:|---:|
| action sequence exact | 91.85% | 37.16% |
| CF pair exact | 79.62% | 0.11% |
| step position match recall | 96.71% | 71.59% |

A − A′ action exact = **+54.69%** （95% CI +53.09% 到 +56.30%）。

## 感知 / 规划归因与格式健康度（A）

| 指标 | 数值 |
|:---|---:|
| state fact P / R / F1 | 87.55% / 88.25% / 87.90% |
| P(action 对 \| state exact) | 100.00% |
| P(action 对 \| state 错) | 84.33% |
| strict structure valid | 100.00% |
| action vocab 越界率 | 0.00% |
| object vocab 越界率 | 0.00% |
| placeholder copy | 0.00% |
| 同切片 train-fitted text oracle | 62.54%（coverage 99.96%） |

## 必报切片（A）

| 切片 | n | action exact | 95% CI | state F1 | text oracle |
|:---|---:|---:|:---:|---:|---:|
| cf:no | 6231 | 94.61% | 93.80%–95.42% | 89.12% | 74.42% |
| cf:yes | 3720 | 87.23% | 85.66%–88.64% | 85.66% | 42.63% |
| container_type:Cabinet | 52 | 84.62% | 72.22%–94.23% | 83.78% | 46.15% |
| container_type:Drawer | 2772 | 86.90% | 85.00%–88.70% | 84.42% | 40.87% |
| container_type:Fridge | 752 | 90.29% | 87.79%–92.44% | 92.84% | 47.74% |
| container_type:Microwave | 68 | 69.12% | 60.93%–77.59% | 86.85% | 48.53% |
| container_type:Safe | 76 | 86.84% | 78.37%–93.42% | 88.08% | 48.68% |
| plan_length:1 | 1245 | 88.27% | 85.85%–90.74% | 92.76% | 100.00% |
| plan_length:2 | 7476 | 94.84% | 94.13%–95.48% | 89.64% | 62.03% |
| plan_length:4 | 615 | 73.82% | 69.82%–77.36% | 80.87% | 55.45% |
| plan_length:6 | 615 | 80.81% | 77.20%–84.07% | 80.87% | 0.00% |
| receptacle_state:closed | 1860 | 90.97% | 89.54%–92.32% | 86.03% | 0.00% |
| receptacle_state:open | 1860 | 83.49% | 81.22%–85.39% | 85.30% | 85.27% |
| task_group:clean | 1447 | 95.37% | 94.14%–96.56% | 90.72% | 83.28% |
| task_group:counterfactual_put | 3720 | 87.23% | 85.66%–88.64% | 85.66% | 42.63% |
| task_group:open_close | 1062 | 93.50% | 91.73%–95.13% | 89.39% | 67.89% |
| task_group:pickup | 1481 | 99.86% | 99.66%–100.00% | 92.34% | 99.73% |
| task_group:slice | 1057 | 96.31% | 94.78%–97.50% | 88.41% | 67.17% |
| task_group:toggle | 1184 | 86.57% | 83.73%–89.33% | 84.93% | 44.26% |
