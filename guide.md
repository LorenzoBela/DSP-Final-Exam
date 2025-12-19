To Decompose, Denoise, and Interpret a real-world time series signal by data
preprocessing, advancing to spectral analysis via FFT, and concluding with a correlation-
based assessment of its drivers.
Process
Data Preprocessing
Import Data - Load the dataset into a programming environment using the pandas
library
Clean Data - Handle any missing or inconsistent entries, Resample the data only
if required for uniform sampling or frequency alignment (e.g., hourly or daily
averages), and clearly document the decision and method. Apply Discrete Fast
Fourier Transform (FFT)
Transform a primary (could be more than one) time-series signal Parameter -
Use FFT to analyze each time-series signal data separately.
o Identify the dominant frequencies (daily, weekly, etc. cycles).
Plot Results
o Create a frequency spectrum plot for each parameter to visualize periodic
components.
Analyze Results
Interpret Frequencies
o Identify and interpret the dominant peaks in the FFT spectrum for each
parameter.
o Discuss any periodic trends observed
o Apply Spectral filtering (Low-Pass filter, High-pass filter, Band-pass filter,
Butterworth filter, etc.) and write a complete analysis of the findings using
Inverse FFT.
Correlate Parameters
o Analyze how changes in each data component correlate with one
another.
Final Documentation
Introduction - Describe the dataset used (source size, variables, engineering
relevance)
Methodology – Describe how the analysis was conducted (Data preprocessing steps),
Python tools and libraries used, and their purpose and the analytical methods applied
(signal analysis)
Results - Present frequency spectral plot and highlight significant findings. This includes
the spectral-filtered plot and its time-domain signal
Discussion - Discuss the implications of identified trends and periodic behaviors.
Conclusion - Summarize the findings and their relevance to real-world applications.
Objectives
DIGITAL SIGNAL PROCESSING LAB 3 | P a g e
Submission Requirements
• Raw data, Cleaned Data
• Jupyter Notebook with complete explanation of codes and plot/ visualizations
• Documentation
Notable code blocks to use:
Data loading using Pandas library import pandas as pd
df = pd.read_csv(‘sample_data.csv’)
print(df)
Calculating and Visualizing the
Correlation Matrix
import seaborn as sns import
matplotlib.pyplot as plt
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
correlation_matrix,
annot=True,
fmt=".2f",
cmap='coolwarm',
linewidths=.5,
cbar_kws={'label': 'Correlation
Coefficient'}
)
plt.title('Correlation Matrix of
Time Series Parameters')
plt.show()
Introduction
Methodology
Results and Discussion
Conclusion
References