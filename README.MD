## Weather Forecast Example Model
#### The project scope is a weather forecasting model based on behavioral analysis of the last 33 hours (hour-by-hour forecast) with `Random Forest Classifier`. The program automatically saves and loads the last trained model for prediction.

### Requirements
	pip install -r requirements.txt

### Usage
   `--train_0` Default script for training with the provided weatherHistory.csv

   `--predict_0` Standard Script for forecasting based on the last trained model (provided weatherHistory.csv)

### Training steps `--train0`

#### 1. Example of the treated table.

| Summary | Precip Type | Temperature (C) | Apparent Temperature (C) | Humidity | Wind Speed (km/h) | Wind Bearing (degrees) | Visibility (km) | Pressure (millibars) |
|---------|-------------|-----------------|--------------------------|----------|-------------------|------------------------|-----------------|----------------------|
| 19      | 0           | 9.472222        | 7.388889                 | 0.89     | 14.1197           | 251.0                  | 15.8263         | 1015.13              |
| 19      | 0           | 9.355556        | 7.227778                 | 0.86     | 14.2646           | 259.0                  | 15.8263         | 1015.63              |
| 17      | 0           | 9.377778        | 9.377778                 | 0.89     | 3.9284            | 204.0                  | 14.9569         | 1015.94              |
| 19      | 0           | 8.288889        | 5.944444                 | 0.83     | 14.1036           | 269.0                  | 15.8263         | 1016.41              |
| 17      | 0           | 8.755556        | 6.977778                 | 0.83     | 11.0446           | 259.0                  | 15.8263         | 1016.51              |

#### 2. Convert the Summary column from strings to single integer labels.
	New labels for summary for prediction:

	Partly Cloudy: 1
	Mostly Cloudy: 2
	Overcast: 3
	Foggy: 4
	Breezy and Mostly Cloudy: 5
	Clear: 6
	Breezy and Partly Cloudy: 7
	Breezy and Overcast: 8
	Humid and Mostly Cloudy: 9
	Humid and Partly Cloudy: 10
	Windy and Foggy: 11
	Windy and Overcast: 12
	Breezy and Foggy: 13
	Windy and Partly Cloudy: 14
	Breezy: 15
	Dry and Partly Cloudy: 16
	Windy and Mostly Cloudy: 17
	Dangerously Windy and Partly Cloudy: 18
	Dry: 19
	Windy: 20
	Humid and Overcast: 21
	Light Rain: 22
	Drizzle: 23
	Windy and Dry: 24
	Dry and Mostly Cloudy: 25
	Breezy and Dry: 26
	Rain: 27

#### 3. Only the last 33 rows (hours) of the dataframe are used for training.

	Close the plot window to continue...

	Specific number of unique classes for y: 3
	Prediction labels: [1 1 1 1 1 1 6 1 1 1 6]

	Model trained successfully. Score: 100.0 %

	Precision: 100.0 %
	Mean Squared Error: 0.0 %
	Recall Score: 100.0 %
	F1 Score: 100.0 %

### Expected forecast
- This dataframe used for example prediction was taken a few hours (rows) before the last 33 hours (rows) of the dataframe used for training...


	    df = pd.DataFrame( #The correct answer will not be included in the prediction
            {
                'Precip Type': ["rain"],
                'Temperature (C)': [13.844444444444445],
                'Apparent Temperature (C)': [13.844444444444445],
                'Humidity': [0.84],
                'Wind Speed (km/h)': [0.0],
                'Wind Bearing (degrees)': [0.0],
                'Visibility (km)': [15.826300000000002],
                'Pressure (millibars)': [1017.82]
            }
        )

`Simulating a Prediction weather conditions...`

| Precip Type | Temperature (C) | Apparent Temperature (C) | Humidity | Wind Speed (km/h) | Wind Bearing (degrees) | Visibility (km) | Pressure (millibars) |
|-------------|-----------------|--------------------------|----------|-------------------|------------------------|-----------------|----------------------|
| 0           | 13.844444       | 13.844444                | 0.84     | 0.0               | 0.0                    | 15.8263         | 1017.82              |


`Prediction label: Clear`
   
### Observations
- `labels.json` It is generated after training containing a unique number assigned to each string label of the Dataframe column chosen as the response for training and prediction.

- The larger the size of the dataset used for training, the less accurate the model is; the redundancy of labels for very different behavioral patterns at different times generates convergence in prediction. Consider including less data like just Morning, Afternoon and Evening of each day in the dataset for training on larger time intervals like weeks, months...

