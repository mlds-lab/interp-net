# Interpolation-Prediction Networks
In this work, we present a new deep learning architecture for addressing the problem of supervised learning with sparse and irregularly sampled multivariate time
series. The architecture is based on the use of a semi-parametric interpolation
network followed by the application of a prediction network. The interpolation
network allows for information to be shared across multiple dimensions of a multivariate time series during the interpolation stage, while any standard deep learning model can be used for the prediction network.

We use a two layer interpolation network. The first interpolation layer performs a semi-parametric univariate interpolation for each of the D time series separately while the second layer merges information from across all of the D time series at each reference time point by taking into account the correlations among the time series. 


## Reference
> Satya Narayan Shukla and Benjamin Marlin. Interpolation-prediction networks for irregularly sampled time series. In International Conference on Learning Representations, 2019. \[[pdf](https://openreview.net/pdf?id=r1efr3C9Ym)\]


## Requirements
The code requires Python 3.7. The file [requirements.txt](requirements.txt) contains the full list of
required Python modules.

## Usage
For running our model on univariate time series (UWave dataset):
```bash
python src/univariate_example.py --epochs 1000 --hidden_units 2048 --ref_points 128 --batch_size 2048
```
To reproduce the results on MIMIC-III Dataset, first you need to have an access to the dataset which can be requested [here](https://mimic.physionet.org/gettingstarted/access/). Once your application to access MIMIC has been approved, you can download the [data](https://physionet.org/works/MIMICIIIClinicalDatabase/). MIMIC is provided as a collection of comma-separated (CSV) files. You can use these [scripts](https://physionet.org/works/MIMICIIIClinicalDatabase/) to import the csv files into a database. Assuming you installed postgres while creating the database, you need to install psycopg2 using
```bash
pip3 install psycopg2
```
Once the database has been created, run these scripts in order.
```bash
python src/mimic_data_extraction.py
python src/multivariate_example.py --epochs 1000 --reference_points 192 --hours_from_adm 48 --batch_size 256 --gpus 4
```

## Contact
For more  details, please contact <snshukla@cs.umass.edu>. 
