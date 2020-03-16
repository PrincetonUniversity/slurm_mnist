```
module load anaconda3 rh/devtoolset/8
R
> install.packages("tensorflow", method="conda", conda="/usr/licensed/anaconda3/2019.10/bin/conda")
> install.packages(c("keras", "reshape2", "ggplot2"))  # keras 2.2.5.0
> use_condaenv("~/.conda/envs/r-reticulate/")
> library(tensorflow)
> tf$constant("Hellow Tensorflow")
```
