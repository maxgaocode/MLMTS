## MLTVD: Multi-level Contrastive Representation Learning with Temporal–Variable Decoupling for Multivariate Time Series


## Framework

![Framework](Framework.png)



## Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies:

* python 3.8
* torch 1.13.0



## MLTVD
The data folder contains ten benchmark datasets (), the data description is shown in the fllows:


python train.py



| Datasets | ID | Train | Test | Dim | Length | Class |
|----------|---------|------|-----|--------|----------|--------|
| AWR      | 1      |275|   300|9     | 144|25
| EC       | 2      |261|   263|3     |1751 |4
| FD       | 3      |5890| 3524|144   |62 |2
| FM       | 4      |316|  100 |28    |50 |2
| HMD      | 5      |160|  74  |10    |400 |4
| HB       | 6      |204|  205 |61    |405 |2
| PEMS     | 7      |267|  173 |963   |144 |7
| PS       | 8      |151|  152 |6     |30  |4
| SRS1     | 9      |268|  293 |6     |896 |2
| SRS2     | 10     |200|  180 |7     |1152|2
