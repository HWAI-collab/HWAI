# GP Risk Stratification

This project aims to analyze GP clinics data, perform risk stratification, and evaluate various models for risk prediction.

## Scripts

### 1. riskstratageodata.py

#### Description

This script retrieves geodata for GP clinics, calculates distances to the nearest hospitals, and adds geostatistical features to the dataset.

#### Functions

- `get_distance`: Calculates the driving distance between two locations using the Google Maps API.
- `find_nearest_hospital`: Finds the nearest hospital to a given clinic location.
- `main`: Main function to read GP clinics, hospitals, socioeconomic index, and rurality index data from CSV files, calculate the distance from each clinic to the nearest hospital, merge the socioeconomic and rurality indexes with the clinics data, and save the resulting data to a new CSV file.

#### Input Files

- `gp_clinics.csv`: Contains GP clinics data with columns 'Postcode' and 'Address'.
- `hospitals.csv`: Contains hospital data with an 'Address' column.
- `socioeconomic_index.csv`: Contains socioeconomic index data with a 'Postcode' column.
- `rurality_index.csv`: Contains rurality index data with a 'Postcode' column.
- `googlemapsapikey.txt`: Contains the Google Maps API key.

#### Output File

- `gp_clinics_with_geostats.csv`: Contains GP clinics data with added socioeconomic index, rurality index, and distance to the nearest hospital.

### 2. riskstratamodelling.py

#### Description

This script performs risk stratification modeling using various classification models.

#### Functions

- `main`: Loads GP clinics data, performs preprocessing and feature engineering, applies K-means clustering to assign risk levels, and evaluates various classification models on the data.

#### Input File

- `gp_data.csv`: Contains GP clinics data with various features.

#### Output

Prints the accuracy and classification report for each model.

## Requirements

- `pandas`
- `scikit-learn`
- `googlemaps`

Install the required packages using:

```bash
pip install pandas scikit-learn googlemaps
```
### Usage

Ensure all input files are present in the same directory as the scripts.
Run riskstratageodata.py to retrieve geodata and add geostatistical features.
Run riskstratamodelling.py to perform risk stratification modeling.

### Contact
For any queries or issues, please contact jarred@orthovision.com.au