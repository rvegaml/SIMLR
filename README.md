# SIMLR

This repository cointains the code to reproduce the main results of the manuscript [SIMLR: Machine Learning inside the SIR model for COVID-19 Forecasting](https://arxiv.org/abs/2106.01590). 
There are six jupyter notebooks on that folder. 
All the experiments were run using an e2-standard-4 (4 vCPUs, 16 GB memory) computer in the Google Cloud Platform.


- **CDC_models.ipynb**: It contains the code used to compile the predictions of the models submitted to the CDC. The dataset required to run this script was not included due to the size, but it is publicly available.
- **Comparison_CDC.ipynb**: It contains the code to create the graphs that compare SIMLR with the models submitted to the CDC. It uses the files created by the previous notebook.
- **Model_Canada_Provinces.ipynb**: It contains the data to predict the number of cases 1 to 4 weeks in advance in the 6 biggest provinces in Canada.
- **Model_US_Country.ipynb**: Similar to the previous one, but for the predictions on US at the country level.
- **Model_US_States.ipynb**: Similar to the previous one, but for the predictions on US at the state level.
- **SIR_Simulations.ipynb**: Code to create the simulated SIR, and to show how a simple SIR model with time-varying parameters can describe the complexities of the COVID-19 dynamics.


This repository folder in addition contains the in-house developed python library *MLib*, which contains custom code for inference in probabilistic graphical models.

## About the datasets
All the datasets used in this repository are publicly available. The relevant spredsheets are included in here for reproducibility purposes.
For information about the new number of reported cases and deaths, we used the publicly available [COVID-19 Data Repository by the Center for Systems Science and Engineering at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) 
For policy tracking we used the [OxCGRT](https://github.com/OxCGRT/covid-policy-tracker).
For comparing our approach with other models we used the publicly available predictions at the [COVID-19 Forecast Hub](https://github.com/reichlab/covid19-forecast-hub)