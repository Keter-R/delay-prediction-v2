| Model                  | Accuracy        | AUC             | F1_score        | GMean           | Sensitivity     | Specificity     |
|------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| gcn_temporal           | 0.798±0.011     | **0.816**±0.017 | 0.366±0.020     | **0.748**±0.022 | 0.694±0.039     | 0.808±0.011     |
| gcn_knn                | 0.731±0.021     | 0.788±0.016     | 0.301±0.009     | 0.711±0.022     | 0.691±0.060     | 0.735±0.027     | 
| mlp                    | 0.795±0.012     | 0.807±0.016     | 0.356±0.018     | 0.738±0.019     | 0.677±0.033     | 0.806±0.013     | 
| random_forest          | 0.903±0.004     | 0.780±0.013     | 0.209±0.030     | 0.384±0.029     | 0.153±0.024     | 0.972±0.003     | 
| random_forest_balanced | 0.770±0.011     | 0.808±0.013     | 0.338±0.013     | 0.737±0.012     | **0.700**±0.021 | 0.776±0.013     | 
| svm                    | **0.916**±0.000 | 0.500±0.000     | 0.000±0.000     | 0.000±0.000     | 0.000±0.000     | **1.000**±0.000 | 
| svm_balanced           | 0.802±0.006     | 0.749±0.016     | **0.367**±0.014 | 0.746±0.017     | 0.686±0.032     | 0.813±0.006     | 
| regression             | 0.915±0.001     | 0.808±0.016     | 0.017±0.006     | 0.093±0.016     | 0.009±0.003     | 0.998±0.001     | 
| regression_balanced    | 0.791±0.007     | 0.811±0.016     | 0.356±0.013     | 0.743±0.016     | 0.689±0.029     | 0.801±0.008     | 