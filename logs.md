## PHASE 1 . Optimizer testing

|   Optimizer   |            LR Planning             |       Results       |
|:-------------:|:----------------------------------:|:-------------------:|
|   Adam Decay  |   Constant LR (expertise) 0.001    |       0.739932      |
|   Adam Decay  |   Constant LR (Finder-1exp) 0.01   |        ToRun        |
|  Adam Default |        Step LR (Finder) 0.1        |        ToRun        |
| Adam Nesterov |        Step LR (Finder) 0.1        |        ToRun        |
|  SGD Momentum |   Constant LR (Expertise) 0.01     |       0.747039      |
|  SGD Momentum |      Step LR (Expertise) 0.01      |    Running gpu21    |
|  SGD Momentum |      Constant LR (Finder) 1        |      Discarded      |
|  SGD Momentum |   Constant LR (Finder-1exp) 0.1    |      Discarded      |
|  SGD Default  |      Constant LR (Finder) 1        |    Running gpu11    |
|  SGD Default  |        Step LR (Finder) 1          |    Running gpu18    |