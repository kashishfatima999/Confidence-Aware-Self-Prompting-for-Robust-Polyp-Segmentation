# Bias analysis: Single vs multiple polyps

```text
                                dataset      factor subgroup   n  dice_B  miss_rate_B  fp_boxes_per_image_B  dice_T  miss_rate_T  fp_boxes_per_image_T  dice_C  miss_rate_C  fp_boxes_per_image_C  c_minus_b  ci_low  ci_high  paired_n
                               clinicdb polyp_count multiple  30  0.4516       0.3000                0.0333  0.5271       0.1667                0.0667  0.5128       0.1667                0.0667     0.0612  0.0117   0.1188        30
                               clinicdb polyp_count   single 582  0.6889       0.2320                0.0928  0.7715       0.1203                0.1873  0.7700       0.1203                0.1924     0.0811  0.0615   0.1016       582
               kvasir_corrupted_blur_s1 polyp_count multiple   6  0.8904       0.0000                0.0000     NaN          NaN                   NaN  0.9290       0.0000                0.0000     0.0385  0.0000   0.1156         6
               kvasir_corrupted_blur_s1 polyp_count   single 194  0.8405       0.0773                0.1186     NaN          NaN                   NaN  0.8682       0.0258                0.2320     0.0277  0.0044   0.0549       194
               kvasir_corrupted_blur_s2 polyp_count multiple   6  0.8592       0.0000                0.0000     NaN          NaN                   NaN  0.8677       0.0000                0.0000     0.0084 -0.0107   0.0361         6
               kvasir_corrupted_blur_s2 polyp_count   single 194  0.8092       0.1082                0.0928     NaN          NaN                   NaN  0.8454       0.0412                0.2423     0.0361  0.0108   0.0646       194
               kvasir_corrupted_blur_s3 polyp_count multiple   6  0.4818       0.3333                0.0000     NaN          NaN                   NaN  0.7135       0.1667                0.0000     0.2316  0.0000   0.5068         6
               kvasir_corrupted_blur_s3 polyp_count   single 194  0.7311       0.1804                0.1082     NaN          NaN                   NaN  0.8085       0.0515                0.2784     0.0775  0.0420   0.1161       194
kvasir_corrupted_brightness_contrast_s1 polyp_count multiple   6  0.8159       0.0000                0.0000     NaN          NaN                   NaN  0.8926       0.0000                0.0000     0.0767  0.0000   0.2300         6
kvasir_corrupted_brightness_contrast_s1 polyp_count   single 194  0.8378       0.0876                0.1031     NaN          NaN                   NaN  0.8829       0.0155                0.2371     0.0451  0.0151   0.0778       194
kvasir_corrupted_brightness_contrast_s2 polyp_count multiple   6  0.7686       0.0000                0.0000     NaN          NaN                   NaN  0.9191       0.0000                0.0000     0.1505  0.0000   0.3742         6
kvasir_corrupted_brightness_contrast_s2 polyp_count   single 194  0.8364       0.0825                0.1134     NaN          NaN                   NaN  0.8835       0.0155                0.2268     0.0472  0.0157   0.0822       194
kvasir_corrupted_brightness_contrast_s3 polyp_count multiple   6  0.6149       0.1667                0.0000     NaN          NaN                   NaN  0.7412       0.1667                0.0000     0.1263  0.0000   0.3498         6
kvasir_corrupted_brightness_contrast_s3 polyp_count   single 194  0.8044       0.1237                0.0876     NaN          NaN                   NaN  0.8414       0.0567                0.2113     0.0370  0.0102   0.0652       194
              kvasir_corrupted_noise_s1 polyp_count multiple   6  0.6856       0.0000                0.0000     NaN          NaN                   NaN  0.7977       0.0000                0.0000     0.1121  0.0000   0.3362         6
              kvasir_corrupted_noise_s1 polyp_count   single 194  0.8038       0.1031                0.1289     NaN          NaN                   NaN  0.8776       0.0103                0.2784     0.0738  0.0420   0.1055       194
              kvasir_corrupted_noise_s2 polyp_count multiple   6  0.4123       0.3333                0.0000     NaN          NaN                   NaN  0.7355       0.1667                0.0000     0.3233  0.0741   0.5981         6
              kvasir_corrupted_noise_s2 polyp_count   single 194  0.6953       0.1959                0.1598     NaN          NaN                   NaN  0.7614       0.1134                0.3093     0.0661  0.0360   0.0998       194
              kvasir_corrupted_noise_s3 polyp_count multiple   6  0.0000       1.0000                0.0000     NaN          NaN                   NaN  0.0000       1.0000                0.0000     0.0000  0.0000   0.0000         6
              kvasir_corrupted_noise_s3 polyp_count   single 194  0.0159       0.9794                0.0052     NaN          NaN                   NaN  0.0357       0.9588                0.0052     0.0198  0.0049   0.0441       194
                            kvasir_test polyp_count multiple   6  0.8160       0.0000                0.0000  0.8536       0.0000                0.0000  0.8536       0.0000                0.0000     0.0377  0.0000   0.1130         6
                            kvasir_test polyp_count   single 194  0.8312       0.0928                0.1289  0.8944       0.0052                0.2423  0.8926       0.0052                0.2474     0.0614  0.0331   0.0923       194
```
