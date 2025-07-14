## Weather Forecast Example Model
#### The project scope is a weather forecasting model based on behavioral analysis of the last 33 hours (hour-by-hour forecast) with `Random Forest Classifier`. The program automatically saves and loads the last trained model for prediction.

### Usage
   `--train_0` Default script for training with the provided weatherHistory.csv

   `--predict_0` Standard Script for forecasting based on the last trained model (provided weatherHistory.csv)

### Training steps `--train0`

#### 1. Conversion of string labels to unique integers and floats multiplied by 1000

| Summary | Precip Type | Temperature (C) | Apparent Temperature (C) | Humidity | Wind Speed (km/h) | Wind Bearing (degrees) | Visibility (km) | Pressure (millibars) |
|---------|-------------|------------------|---------------------------|----------|--------------------|-------------------------|------------------|------------------------|
| 1       | 1           | 9472             | 7388                      | 890      | 14119              | 251000                  | 15826            | 1015130                |
| 1       | 1           | 9355             | 7227                      | 860      | 14264              | 259000                  | 15826            | 1015630                |
| 2       | 1           | 9377             | 9377                      | 890      | 3928               | 204000                  | 14956            | 1015940                |
| 1       | 1           | 8288             | 5944                      | 830      | 14103              | 269000                  | 15826            | 1016410                |
| 2       | 1           | 8755             | 6977                      | 830      | 11044              | 259000                  | 15826            | 1016510                |

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


	    df  =  pd.DataFrame(
		    {
			    'Summary': ["Clear"], #The correct answer will not be included in the prediction
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
	    
	    df = df.drop('Summary', Axis=1)

`Simulating a Prediction weather conditions...`

| Precip Type | Temperature (C) | Apparent Temperature (C) | Humidity | Wind Speed (km/h) | Wind Bearing (degrees) | Visibility (km) | Pressure (millibars) |
|-------------|------------------|---------------------------|----------|--------------------|-------------------------|------------------|------------------------|
| 1           | 13844            | 13844                    | 840      | 0                  | 0                       | 15826            | 1017820                |


`Prediction label: Clear`
   
### Observations
- `labels.json` It is generated after training containing a unique number assigned to each string label of the Dataframe column chosen as the response for training and prediction.

- The larger the size of the dataset used for training, the less accurate the model is; the redundancy of labels for very different behavioral patterns at different times generates convergence in prediction. Consider including less data like just Morning, Afternoon and Evening of each day in the dataset for training on larger time intervals like weeks, months...

