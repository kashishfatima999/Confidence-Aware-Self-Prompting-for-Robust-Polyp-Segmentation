# Phase 4 Report



## 2. Evaluation Metrics



### 2.1 Segmentation Metrics



Segmentation performance is evaluated using Dice coefficient, Intersection over Union (IoU), pixel precision, pixel recall, and pixel accuracy. These metrics are calculated for each image, with dataset-level mean values also reported.



- **Dice coefficient (F1-score):** Measures the overlap between the predicted polyp mask and the ground-truth mask. It is calculated as \(2TP/(2TP+FP+FN)\). Higher values indicate better segmentation.

- **Intersection over Union (IoU):** Measures the ratio of the intersection to the union of the predicted and ground-truth masks: \(TP/(TP+FP+FN)\).

- **Pixel Precision:** Measures the proportion of predicted polyp pixels that are correctly classified: \(TP/(TP+FP)\).

- **Pixel Recall:** Measures the proportion of ground-truth polyp pixels correctly detected: \(TP/(TP+FN)\).

- **Pixel Accuracy:** Measures the proportion of all pixels classified correctly: \((TP+TN)/(TP+TN+FP+FN)\).



Dice and IoU are the primary segmentation metrics because they focus on overlap between predicted and ground-truth polyp regions. Pixel accuracy is reported but should not be interpreted as a reliable indicator of segmentation quality. Endoscopic images contain large background regions, so a model may achieve high pixel accuracy while failing to detect a polyp.



An empty prediction against a non-empty ground-truth mask receives a Dice and IoU score of zero. Images where both masks are empty receive a score of one, since neither contains a polyp to detect.



### 2.2 Detection Metrics



Detection performance is evaluated separately using precision, recall, and F1-score. Predicted bounding boxes are matched to ground-truth boxes using a greedy one-to-one matching procedure. A match is accepted when box IoU is at least 0.5. Unmatched predictions count as false positives, while unmatched ground-truth boxes count as false negatives.



### 2.3 Mask-Area Error



Instead of using RMSE as a measure of segmentation quality, the evaluation reports absolute mask-area error in percentage points. This is the absolute difference between the percentage of image area occupied by the predicted polyp mask and the percentage occupied by the ground-truth mask.



This provides an interpretable measure of how much the predicted polyp area differs from the annotated area. However, it does not measure spatial overlap, so it is interpreted alongside Dice and IoU.



### 2.4 Per-Image and Pooled Evaluation



The evaluation reports both the mean of per-image Dice and IoU scores and pooled Dice and IoU. Per-image averages give each image equal weight, while pooled metrics calculate overlap after accumulating pixel counts across the dataset. Reporting both provides a view of typical image-level performance and overall pixel-level performance.



The evaluation also calculates 95% percentile-bootstrap confidence intervals for mean per-image Dice and IoU using 1,000 resamples and a fixed random seed of 42.



## 4. Bias and Subgroup Analysis



### 4.1 Analysis Method



The subgroup analysis evaluates whether segmentation performance varies across datasets and image characteristics. The factors examined are dataset, polyp size, brightness, contrast, image width, image height, polyp count, and corruption type and severity. Continuous characteristics are divided into low, medium, and high tertiles using the baseline dataset distributions. The same baseline-derived group boundaries are applied to the corresponding baseline, threshold-only, and CASP results.



For each subgroup, the analysis reports the number of images (n), mean per-image Dice for baseline (B), threshold-only (T), and CASP (C), miss rate, false-positive boxes per image, and the paired difference in Dice between CASP and baseline (C − B). The 95% confidence intervals are calculated using 1,000 paired bootstrap resamples. A confidence interval that includes zero does not provide clear evidence of an improvement in that subgroup. Corrupted-image evaluations are reported separately because threshold-only results were not generated for those runs.



### 4.2 Dataset



CASP improves mean Dice on both evaluation datasets. The improvement is larger on CVC-ClinicDB (+0.0802) than on Kvasir-SEG (+0.0607). Baseline performance is also lower on ClinicDB, with a higher miss rate of 23.5%, compared with 9.0% on Kvasir. These differences indicate that performance depends partly on the evaluation dataset and its image characteristics. The results should therefore not be interpreted as evidence that performance will transfer equally to every dataset.



| Dataset | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---:|---|---|---|---|

| CVC-ClinicDB | 612 | 0.677 / 0.760 / 0.757 | 23.5 / 12.3 / 12.3 | 0.090 / 0.181 / 0.186 | +0.0802 \[0.0604, 0.1001] |

| Kvasir-SEG | 200 | 0.831 / 0.893 / 0.891 | 9.0 / 0.5 / 0.5 | 0.125 / 0.235 / 0.240 | +0.0607 \[0.0318, 0.0902] |



### 4.3 Polyp Size



CASP improves Dice across all polyp-size groups in both clean datasets. On ClinicDB, the largest gains occur in the low- and high-size groups, while the medium-size group has a smaller improvement. Kvasir shows a similar pattern: the high-size group has the largest gain, while the medium-size group's confidence interval includes zero. The results suggest that performance differs across polyp-size groups, although the subgroup analysis alone cannot establish that size is the cause of these differences.



| Dataset | Size group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---|---:|---|---|---|---|

| ClinicDB | Low | 204 | 0.614 / 0.718 / 0.714 | 29.9 / 15.7 / 15.7 | 0.059 / 0.118 / 0.123 | +0.1000 \[0.0636, 0.1382] |

| ClinicDB | Medium | 204 | 0.782 / 0.825 / 0.823 | 14.2 / 8.8 / 8.8 | 0.074 / 0.127 / 0.132 | +0.0409 \[0.0170, 0.0685] |

| ClinicDB | High | 204 | 0.635 / 0.735 / 0.735 | 26.5 / 12.3 / 12.3 | 0.137 / 0.299 / 0.304 | +0.0996 \[0.0665, 0.1375] |

| Kvasir | Low | 67 | 0.811 / 0.872 / 0.872 | 10.4 / 1.5 / 1.5 | 0.045 / 0.090 / 0.090 | +0.0611 \[0.0192, 0.1122] |

| Kvasir | Medium | 66 | 0.863 / 0.901 / 0.899 | 6.1 / 0.0 / 0.0 | 0.167 / 0.273 / 0.273 | +0.0366 \[-0.0063, 0.0906] |

| Kvasir | High | 67 | 0.819 / 0.906 / 0.903 | 10.4 / 0.0 / 0.0 | 0.164 / 0.343 / 0.358 | +0.0842 \[0.0279, 0.1505] |



### 4.4 Brightness



CASP improves Dice across all brightness groups in both clean datasets. On ClinicDB, the low- and medium-brightness groups show larger gains than the high-brightness group. On Kvasir, the largest gain occurs in the low-brightness group, while the high-brightness group's confidence interval includes zero. The different patterns across datasets suggest that brightness-related performance should be evaluated separately for each dataset rather than assuming one brightness level is consistently more difficult.



| Dataset | Brightness group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---|---:|---|---|---|---|

| ClinicDB | Low | 204 | 0.609 / 0.711 / 0.709 | 28.9 / 14.7 / 14.7 | 0.127 / 0.260 / 0.265 | +0.0996 \[0.0622, 0.1391] |

| ClinicDB | Medium | 204 | 0.635 / 0.741 / 0.738 | 27.5 / 14.2 / 14.2 | 0.078 / 0.142 / 0.137 | +0.1028 \[0.0687, 0.1381] |

| ClinicDB | High | 204 | 0.787 / 0.826 / 0.825 | 14.2 / 7.8 / 7.8 | 0.064 / 0.142 / 0.157 | +0.0381 \[0.0151, 0.0643] |

| Kvasir | Low | 67 | 0.820 / 0.908 / 0.909 | 11.9 / 1.5 / 1.5 | 0.030 / 0.060 / 0.060 | +0.0897 \[0.0283, 0.1574] |

| Kvasir | Medium | 66 | 0.835 / 0.895 / 0.894 | 9.1 / 0.0 / 0.0 | 0.091 / 0.167 / 0.167 | +0.0586 \[0.0180, 0.1123] |

| Kvasir | High | 67 | 0.838 / 0.876 / 0.871 | 6.0 / 0.0 / 0.0 | 0.254 / 0.478 / 0.493 | +0.0338 \[-0.0059, 0.0810] |



### 4.5 Contrast



CASP improves all three contrast groups on ClinicDB. The medium-contrast group has the largest observed gain. On Kvasir, the high-contrast group has the largest gain, while the medium-contrast group has the smallest gain and a confidence interval that includes zero. This difference between datasets indicates that contrast-related performance is not uniform. The relatively high false-positive box rates in some groups also show that reducing missed detections can come with additional false-positive predictions.



| Dataset | Contrast group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---|---:|---|---|---|---|

| ClinicDB | Low | 204 | 0.614 / 0.696 / 0.693 | 28.9 / 18.1 / 18.1 | 0.064 / 0.152 / 0.157 | +0.0786 \[0.0459, 0.1165] |

| ClinicDB | Medium | 204 | 0.714 / 0.804 / 0.801 | 20.6 / 8.8 / 8.8 | 0.093 / 0.176 / 0.176 | +0.0871 \[0.0540, 0.1227] |

| ClinicDB | High | 204 | 0.703 / 0.779 / 0.778 | 21.1 / 9.8 / 9.8 | 0.113 / 0.216 / 0.225 | +0.0747 \[0.0426, 0.1103] |

| Kvasir | Low | 67 | 0.871 / 0.920 / 0.921 | 6.0 / 0.0 / 0.0 | 0.090 / 0.179 / 0.179 | +0.0507 \[0.0103, 0.1034] |

| Kvasir | Medium | 66 | 0.858 / 0.884 / 0.882 | 4.5 / 0.0 / 0.0 | 0.212 / 0.364 / 0.364 | +0.0232 \[-0.0143, 0.0723] |

| Kvasir | High | 67 | 0.764 / 0.875 / 0.871 | 16.4 / 1.5 / 1.5 | 0.075 / 0.164 / 0.179 | +0.1078 \[0.0508, 0.1758] |



### 4.6 Image Resolution: Width



ClinicDB images have a fixed width, so they form a single group and cannot support a comparison across width levels. Kvasir images are divided into low-, medium-, and high-width groups. CASP improves Dice in all three groups, with the largest observed gain in the high-width group. These results describe the tested images and do not establish that image width itself causes the performance differences.



| Dataset | Width group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---|---:|---|---|---|---|

| ClinicDB | All | 612 | 0.677 / 0.760 / 0.757 | 23.5 / 12.3 / 12.3 | 0.090 / 0.181 / 0.186 | +0.0802 \[0.0604, 0.1001] |

| Kvasir | Low | 67 | 0.830 / 0.902 / 0.897 | 9.0 / 0.0 / 0.0 | 0.090 / 0.134 / 0.149 | +0.0672 \[0.0214, 0.1218] |

| Kvasir | Medium | 83 | 0.854 / 0.896 / 0.896 | 6.0 / 0.0 / 0.0 | 0.169 / 0.289 / 0.289 | +0.0417 \[0.0029, 0.0890] |

| Kvasir | High | 50 | 0.793 / 0.876 / 0.877 | 14.0 / 2.0 / 2.0 | 0.100 / 0.280 / 0.280 | +0.0837 \[0.0227, 0.1606] |



### 4.7 Image Resolution: Height



As with width, ClinicDB has a fixed image height and therefore only has an overall group. Kvasir results show positive CASP improvements across all three height groups. The largest gain is in the high-height group, followed by the medium-height group. The low-height group has a smaller improvement. These findings should be interpreted as subgroup associations rather than evidence that image height independently determines segmentation quality.



| Dataset | Height group | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---|---:|---|---|---|---|

| ClinicDB | All | 612 | 0.677 / 0.760 / 0.757 | 23.5 / 12.3 / 12.3 | 0.090 / 0.181 / 0.186 | +0.0802 \[0.0604, 0.1001] |

| Kvasir | Low | 72 | 0.813 / 0.854 / 0.851 | 8.3 / 1.4 / 1.4 | 0.208 / 0.319 / 0.333 | +0.0378 \[0.0028, 0.0816] |

| Kvasir | Medium | 75 | 0.854 / 0.926 / 0.924 | 8.0 / 0.0 / 0.0 | 0.067 / 0.133 / 0.133 | +0.0699 \[0.0169, 0.1256] |

| Kvasir | High | 53 | 0.821 / 0.900 / 0.900 | 11.3 / 0.0 / 0.0 | 0.094 / 0.264 / 0.264 | +0.0790 \[0.0193, 0.1523] |



### 4.8 Polyp Count



CASP improves Dice for both single- and multiple-polyp images. On ClinicDB, multiple-polyp images have substantially lower Dice than single-polyp images for all three methods. This indicates that images containing multiple polyps are a more challenging subgroup in this dataset. However, the multiple-polyp group contains only 30 images on ClinicDB and six on Kvasir. The Kvasir confidence interval is particularly wide and includes zero, so its multiple-polyp result is inconclusive. Larger and more balanced samples are needed before drawing strong conclusions about this subgroup.



| Dataset | Polyp count | n | Dice B/T/C | Miss rate B/T/C (%) | FP boxes/image B/T/C | C − B \[95% CI] |

|---|---|---:|---|---|---|---|

| ClinicDB | Single | 582 | 0.689 / 0.771 / 0.770 | 23.2 / 12.0 / 12.0 | 0.093 / 0.187 / 0.192 | +0.0811 \[0.0615, 0.1016] |

| ClinicDB | Multiple | 30 | 0.452 / 0.527 / 0.513 | 30.0 / 16.7 / 16.7 | 0.033 / 0.067 / 0.067 | +0.0612 \[0.0117, 0.1188] |

| Kvasir | Single | 194 | 0.831 / 0.894 / 0.893 | 9.3 / 0.5 / 0.5 | 0.129 / 0.242 / 0.247 | +0.0615 \[0.0331, 0.0923] |

| Kvasir | Multiple | 6 | 0.816 / 0.854 / 0.854 | 0.0 / 0.0 / 0.0 | 0.000 / 0.000 / 0.000 | +0.0377 \[0.0000, 0.1130] |



### 4.9 Corruption Type and Severity



CASP produces higher mean Dice than baseline across all nine tested corruption conditions. The size of the improvement varies by corruption type and severity. However, improvement over baseline does not mean that the model is robust under every condition. Severe noise (noise\_s3) is a major failure case: baseline Dice is 0.0155 and CASP Dice is only 0.0346, while miss rates remain 98.0% and 96.0%, respectively. This demonstrates that CASP does not recover reliable segmentation under the most severe noise condition. Threshold-only results are unavailable for these corrupted runs because that experiment was not generated for the corrupted datasets.



| Corruption | n | Dice B/C | Miss rate B/C (%) | FP boxes/image B/C | C − B \[95% CI] |

|---|---:|---|---|---|---|

| Blur, severity 1 | 200 | 0.842 / 0.870 | 7.5 / 2.5 | 0.115 / 0.225 | +0.0280 \[0.0059, 0.0539] |

| Blur, severity 2 | 200 | 0.811 / 0.846 | 10.5 / 4.0 | 0.090 / 0.235 | +0.0353 \[0.0103, 0.0631] |

| Blur, severity 3 | 200 | 0.724 / 0.806 | 18.5 / 5.5 | 0.105 / 0.270 | +0.0821 \[0.0452, 0.1204] |

| Brightness/contrast, severity 1 | 200 | 0.837 / 0.883 | 8.5 / 1.5 | 0.100 / 0.230 | +0.0461 \[0.0149, 0.0778] |

| Brightness/contrast, severity 2 | 200 | 0.834 / 0.885 | 8.0 / 1.5 | 0.110 / 0.220 | +0.0503 \[0.0187, 0.0847] |

| Brightness/contrast, severity 3 | 200 | 0.799 / 0.838 | 12.5 / 6.0 | 0.085 / 0.205 | +0.0397 \[0.0136, 0.0690] |

| Noise, severity 1 | 200 | 0.800 / 0.875 | 10.0 / 1.0 | 0.125 / 0.270 | +0.0749 \[0.0436, 0.1087] |

| Noise, severity 2 | 200 | 0.687 / 0.761 | 20.0 / 11.5 | 0.155 / 0.300 | +0.0738 \[0.0441, 0.1066] |

| Noise, severity 3 | 200 | 0.015 / 0.035 | 98.0 / 96.0 | 0.005 / 0.005 | +0.0192 \[0.0047, 0.0388] |



### 4.10 Pipeline Biases and Limitations



The evaluation reveals several limitations that may affect how fairly the pipeline performs across different image groups. First, the two datasets have different baseline performance, which suggests that dataset-specific characteristics influence the results. Second, some subgroups are much smaller than others. In particular, multiple-polyp images are underrepresented, making conclusions about that group less reliable. Third, the fixed image dimensions in ClinicDB prevent meaningful within-dataset comparisons by width and height. Fourth, the corruption experiment shows that severe noise can cause near-total segmentation failure even when CASP improves the average Dice score. Finally, CASP often increases false-positive boxes per image compared with baseline. This is an important trade-off: lower miss rates and higher Dice do not necessarily mean fewer false detections.



The subgroup analysis is descriptive and does not isolate the independent effect of each image characteristic. Brightness, contrast, resolution, polyp size, and dataset may be related to one another. The tertile boundaries are also dataset-specific, so a low-brightness group in one dataset does not necessarily represent the same absolute brightness range in another. The results should therefore be interpreted as evidence of performance variation across the evaluated groups, not as proof that a particular characteristic causes the variation.



### 4.11 Recommendations for a Fairer and More Robust Pipeline



A fairer pipeline should be trained and evaluated using images from multiple datasets rather than relying heavily on one dataset's characteristics. Training data should include a more balanced representation of small, medium, and large polyps, single- and multiple-polyp images, and different brightness, contrast, and resolution conditions. Severe image corruptions should also be represented during training and testing so that robustness limitations can be identified before deployment.



The pipeline should be evaluated separately on each dataset and subgroup, with confidence intervals and sample counts reported alongside average performance. Thresholds should be selected using validation data that represents the intended deployment domain, rather than assuming that one threshold will work equally well across all datasets. Any dataset-specific threshold adjustment should be documented and evaluated on held-out data to avoid overfitting. Finally, false-positive detections and missed polyps should both be considered when selecting operating thresholds. These changes would help prevent improvements in average Dice from hiding poor performance on smaller or more difficult subgroups.

