# CoT-100K in-domain 自由生成评测

95% CI 使用按 `scene_id` 重采样的 1000 次 percentile cluster bootstrap。

## 主指标

| 指标 | A（正确图） | A′（换图） |
|:---|---:|---:|
| action sequence exact | 88.06% | 38.06% |
| CF pair exact | 73.22% | 0.11% |
| step position match recall | 95.09% | 70.55% |

A − A′ action exact = **+50.00%** （95% CI +47.76% 到 +52.24%）。

## 感知 / 规划归因与格式健康度（A）

| 指标 | 数值 |
|:---|---:|
| state fact P / R / F1 | 84.40% / 85.31% / 84.85% |
| P(action 对 \| state exact) | 100.00% |
| P(action 对 \| state 错) | 79.20% |
| strict structure valid | 100.00% |
| action vocab 越界率 | 0.00% |
| object vocab 越界率 | 0.00% |
| placeholder copy | 0.00% |
| 同切片 train-fitted text oracle | 61.98%（coverage 99.78%） |

## 必报切片（A）

| 切片 | n | action exact | 95% CI | state F1 | text oracle |
|:---|---:|---:|:---:|---:|---:|
| cf:no | 3148 | 91.74% | 90.51%–92.84% | 86.82% | 74.27% |
| cf:yes | 1852 | 81.80% | 79.00%–84.35% | 81.49% | 41.09% |
| container_type:Drawer | 1428 | 80.04% | 76.44%–83.28% | 79.93% | 39.15% |
| container_type:Fridge | 362 | 89.78% | 86.54%–92.98% | 91.24% | 47.79% |
| container_type:Microwave | 40 | 65.00% | 55.26%–76.50% | 84.86% | 47.50% |
| container_type:Safe | 22 | 95.45% | 86.36%–100.00% | 91.98% | 45.45% |
| plan_length:1 | 554 | 85.92% | 83.01%–88.87% | 91.85% | 100.00% |
| plan_length:2 | 3702 | 92.36% | 91.18%–93.51% | 87.43% | 63.16% |
| plan_length:4 | 372 | 66.40% | 60.29%–72.89% | 76.24% | 55.65% |
| plan_length:6 | 372 | 70.16% | 64.54%–76.12% | 76.56% | 0.00% |
| receptacle_state:closed | 926 | 85.53% | 82.56%–88.37% | 81.96% | 0.00% |
| receptacle_state:open | 926 | 78.08% | 74.84%–81.09% | 81.01% | 82.18% |
| task_group:clean | 741 | 92.85% | 90.38%–95.13% | 88.00% | 85.83% |
| task_group:counterfactual_put | 1852 | 81.80% | 79.00%–84.35% | 81.49% | 41.09% |
| task_group:open_close | 555 | 92.43% | 89.96%–94.63% | 87.80% | 70.45% |
| task_group:pickup | 741 | 99.73% | 99.31%–100.00% | 91.39% | 98.52% |
| task_group:slice | 555 | 93.69% | 91.12%–95.95% | 87.61% | 59.28% |
| task_group:toggle | 556 | 76.98% | 72.28%–81.64% | 78.65% | 45.32% |
