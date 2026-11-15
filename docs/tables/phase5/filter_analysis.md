## Fallback rule F (uncertain boxes only where no box >= 0.5)

### kvasir_test

| Run | Dice | IoU | Pixel prec. | Pixel recall | Missed | Box P / R / F1 |
|---|---|---|---|---|---|---|
| B | 0.831 [0.792, 0.871] | 0.780 | 0.873 | 0.810 | 18 | 0.877 / 0.865 / 0.871 |
| T | 0.893 [0.871, 0.913] | 0.832 | 0.929 | 0.890 | 1 | 0.807 / 0.947 / 0.871 |
| C | 0.891 [0.868, 0.912] | 0.831 | 0.926 | 0.892 | 1 | 0.802 / 0.942 / 0.867 |
| O | 0.952 [0.943, 0.958] | 0.912 | 0.979 | 0.931 | 0 | 1.000 / 1.000 / 1.000 |
| F_T | 0.895 [0.872, 0.915] | 0.834 | 0.936 | 0.886 | 1 | 0.852 / 0.942 / 0.894 |
| F_C | 0.893 [0.870, 0.914] | 0.833 | 0.933 | 0.888 | 1 | 0.847 / 0.937 / 0.890 |

| Pair | Dice difference [95% CI] | better / worse / tie |
|---|---|---|
| F_T − B | +0.064 [+0.035, +0.093] | 17 / 0 / 183 |
| F_T − T | +0.002 [-0.003, +0.006] | 4 / 5 / 191 |
| F_T − C | +0.003 [-0.003, +0.010] | 16 / 15 / 169 |
| F_C − B | +0.062 [+0.035, +0.091] | 17 / 0 / 183 |
| F_C − T | -0.000 [-0.006, +0.005] | 9 / 13 / 178 |
| F_C − C | +0.001 [-0.003, +0.007] | 8 / 10 / 182 |

| Run | good | extra_box | under_segmentation | over_segmentation | false_positive | missed |
|---|---|---|---|---|---|---|
| B | 159 | 15 | 6 | 2 | 0 | 18 |
| T | 160 | 29 | 7 | 3 | 0 | 1 |
| C | 160 | 28 | 7 | 4 | 0 | 1 |
| O | 199 | 0 | 1 | 0 | 0 | 0 |
| F_T | 168 | 21 | 7 | 3 | 0 | 1 |
| F_C | 168 | 20 | 7 | 4 | 0 | 1 |

### clinicdb

| Run | Dice | IoU | Pixel prec. | Pixel recall | Missed | Box P / R / F1 |
|---|---|---|---|---|---|---|
| B | 0.677 [0.648, 0.710] | 0.628 | 0.712 | 0.671 | 144 | 0.890 / 0.686 / 0.774 |
| T | 0.760 [0.736, 0.785] | 0.698 | 0.800 | 0.764 | 75 | 0.820 / 0.780 / 0.799 |
| C | 0.757 [0.734, 0.783] | 0.695 | 0.795 | 0.765 | 75 | 0.815 / 0.776 / 0.795 |
| O | 0.945 [0.942, 0.948] | 0.899 | 0.966 | 0.928 | 0 | 1.000 / 1.000 / 1.000 |
| F_T | 0.756 [0.731, 0.782] | 0.695 | 0.805 | 0.753 | 75 | 0.859 / 0.763 / 0.808 |
| F_C | 0.756 [0.731, 0.782] | 0.695 | 0.802 | 0.754 | 75 | 0.857 / 0.762 / 0.807 |

| Pair | Dice difference [95% CI] | better / worse / tie |
|---|---|---|
| F_T − B | +0.079 [+0.060, +0.098] | 67 / 0 / 545 |
| F_T − T | -0.003 [-0.010, +0.002] | 10 / 26 / 576 |
| F_T − C | -0.001 [-0.007, +0.005] | 53 / 56 / 503 |
| F_C − B | +0.078 [+0.060, +0.097] | 67 / 0 / 545 |
| F_C − T | -0.004 [-0.010, +0.002] | 33 / 51 / 528 |
| F_C − C | -0.002 [-0.008, +0.004] | 28 / 33 / 551 |

| Run | good | extra_box | under_segmentation | over_segmentation | false_positive | missed |
|---|---|---|---|---|---|---|
| B | 409 | 22 | 26 | 9 | 2 | 144 |
| T | 426 | 59 | 34 | 16 | 2 | 75 |
| C | 424 | 58 | 34 | 19 | 2 | 75 |
| O | 610 | 0 | 0 | 2 | 0 | 0 |
| F_T | 445 | 29 | 42 | 17 | 4 | 75 |
| F_C | 444 | 29 | 42 | 18 | 4 | 75 |

### Corrupted Kvasir-SEG test (F_C)

| Setting | F_C − B | F_C − C |
|---|---|---|
| blur_s1 | +0.036 [+0.015, +0.060] | +0.008 [+0.001, +0.018] |
| blur_s2 | +0.044 [+0.021, +0.071] | +0.009 [+0.002, +0.016] |
| blur_s3 | +0.084 [+0.051, +0.121] | +0.002 [-0.011, +0.014] |
| brightness_contrast_s1 | +0.053 [+0.026, +0.083] | +0.007 [-0.004, +0.021] |
| brightness_contrast_s2 | +0.055 [+0.027, +0.087] | +0.005 [-0.008, +0.019] |
| brightness_contrast_s3 | +0.044 [+0.022, +0.070] | +0.004 [-0.007, +0.017] |
| noise_s1 | +0.067 [+0.039, +0.099] | -0.007 [-0.021, +0.004] |
| noise_s2 | +0.054 [+0.029, +0.084] | -0.019 [-0.036, -0.003] |
| noise_s3 | +0.019 [+0.005, +0.039] | +0.000 [+0.000, +0.000] |

## Box-level signals (T run, kept boxes; AUROC for separating true from false boxes)

| Dataset | Boxes | Signal | n (true) | AUROC [95% CI] |
|---|---|---|---|---|
| kvasir_test | all kept boxes | YOLO confidence | 243 (196) | 0.861 [0.794, 0.916] |
| kvasir_test | all kept boxes | SAM 2 score | 243 (196) | 0.685 [0.594, 0.768] |
| kvasir_test | all kept boxes | confidence x SAM score | 243 (196) | 0.861 [0.794, 0.917] |
| kvasir_test | all kept boxes | no box >= 0.5 in image | 243 (196) | 0.804 [0.734, 0.872] |
| kvasir_test | uncertain band 0.25-0.5 | YOLO confidence | 39 (17) | 0.448 [0.254, 0.641] |
| kvasir_test | uncertain band 0.25-0.5 | SAM 2 score | 39 (17) | 0.527 [0.340, 0.708] |
| kvasir_test | uncertain band 0.25-0.5 | confidence x SAM score | 39 (17) | 0.471 [0.278, 0.666] |
| kvasir_test | uncertain band 0.25-0.5 | no box >= 0.5 in image | 39 (17) | 0.766 [0.647, 0.875] |
| clinicdb | all kept boxes | YOLO confidence | 615 (504) | 0.820 [0.782, 0.855] |
| clinicdb | all kept boxes | SAM 2 score | 615 (504) | 0.656 [0.598, 0.709] |
| clinicdb | all kept boxes | confidence x SAM score | 615 (504) | 0.822 [0.785, 0.857] |
| clinicdb | all kept boxes | no box >= 0.5 in image | 615 (504) | 0.666 [0.620, 0.711] |
| clinicdb | uncertain band 0.25-0.5 | YOLO confidence | 117 (61) | 0.519 [0.418, 0.626] |
| clinicdb | uncertain band 0.25-0.5 | SAM 2 score | 117 (61) | 0.469 [0.366, 0.580] |
| clinicdb | uncertain band 0.25-0.5 | confidence x SAM score | 117 (61) | 0.512 [0.406, 0.615] |
| clinicdb | uncertain band 0.25-0.5 | no box >= 0.5 in image | 117 (61) | 0.678 [0.594, 0.762] |

## Uncertain boxes by context

| Dataset | Image has a box ≥ 0.5 | Boxes | True polyp | Share | SAM score true / false |
|---|---|---|---|---|---|
| clinicdb | no | 76 | 50 | 0.66 | 0.863 / 0.881 |
| clinicdb | yes | 41 | 11 | 0.27 | 0.879 / 0.903 |
| kvasir_test | no | 25 | 16 | 0.64 | 0.914 / 0.926 |
| kvasir_test | yes | 14 | 1 | 0.07 | 0.896 / 0.848 |
