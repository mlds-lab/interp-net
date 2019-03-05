# Interpolation-Prediction Networks
In this work, we present a new deep learning architecture for addressing the problem of supervised learning with sparse and irregularly sampled multivariate time
series. The architecture is based on the use of a semi-parametric interpolation
network followed by the application of a prediction network. The interpolation
network allows for information to be shared across multiple dimensions of a multivariate time series during the interpolation stage, while any standard deep learning model can be used for the prediction network.

We use a two layer interpolation network. The first interpolation layer performs a semi-parametric univariate interpolation for each of the D time series separately while the second layer merges information from across all of the D time series at each reference time point by taking into account the correlations among the time series. 

## Requirements
The code requires Python 2.7. The file [requirements.txt](requirements.txt) contains the full list of
required Python modules.

## Usage
For running our model on univariate time series (UWave dataset):
```bash
python src/univariate_example.py --epochs 1000 --hidden_units 128 --ref_points 128 --batch_size 1024
```

## References
Satya Narayan Shukla and Benjamin Marlin. Interpolation-prediction networks for irregularly sampled time series. In International Conference on Learning Representations, 2019. \[[pdf](https://openreview.net/pdf?id=r1efr3C9Ym)\]

## Contact
For more  details, please contact <snshukla@cs.umass.edu>. 
