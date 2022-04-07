# Implementation for KNN serach by kd-tree and ball-tree
The datasets refer to [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

Because using complete test dataset will cost long time to get results, here just take some records as the test dataset.

## Table 1: Result from implementation by kd-tree (k=3)
| Dataset | Building Time (s) | Searching Time (s) | Number of Calculations|
| - | :-: |:-: |:-: |
| ijcnn1 | 0.356 | 0.777 | 12694.25 |
| mnist| 7.35 | 5.415 | 60000 |
| shuttle| 0.247 | 0.439 | 5900.8 |

## Table 2: Result from implementation by ball-tree (k=3)
| Dataset | Building Time (s) | Searching Time (s) | Number of Calculations|
| - | :-: |:-: |:-: |
| ijcnn1 | 21.571 | 0.301 | 4643.7 |
| mnist| 77.306 | 3.533 | 40698.6 |
| shuttle| 17.805 | 0.168 | 2730.1 |