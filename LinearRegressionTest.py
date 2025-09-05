# 线性归回课程实例，源地址：
# https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_taxi.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_m

# 运行之前确保这些模块已经安装，在console运行，可能需要先科学上网。其中tensorFlow是大头约三百多兆
# pip install google-ml-edu==0.1.3 keras~=3.8.0 matplotlib~=3.10.0 numpy~=2.0.0 pandas~=2.2.0 tensorflow~=2.18.0
# print('\n\nAll requirements successfully installed.')

# 防止TensorFlow oneDNN警告，可加可不加
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# region Code - Load dependencies -----------------

# data
import numpy as np
import pandas as pd

# machin learning
import keras
import ml_edu.experiment
import ml_edu.results

# data visualization
import plotly.express as px

# endregion -----------------

# region - Read dataset -----------------

# chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
chicago_taxi_dataset = pd.read_csv("chicago_taxi_train.csv")  # 网不好的时候读本地文件

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
# print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

# endregion -----------------

# region - View dataset statistics -----------------

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')

answer = '''
What is the maximum fare? 				              Answer: $159.25
What is the mean distance across all trips? 		Answer: 8.2895 miles
How many cab companies are in the dataset? 		  Answer: 31
What is the most frequent payment type? 		    Answer: Credit Card
Are any features missing data? 				          Answer: No
'''

# You should be able to find the answers to the questions about the dataset
# by inspecting the table output after running the DataFrame describe method.
#
# Run this code cell to verify your answers.

# What is the maximum fare?
max_fare = training_df['FARE'].max()
print("What is the maximum fare? 				Answer: ${fare:.2f}".format(fare=max_fare))

# What is the mean distance across all trips?
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? 		Answer: {mean:.4f} miles".format(mean=mean_distance))

# How many cab companies are in the dataset?
num_unique_companies = training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? 		Answer: {number}".format(number=num_unique_companies))

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("What is the most frequent payment type? 		Answer: {type}".format(type=most_freq_payment_type))

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? 				Answer:", "No" if missing_values == 0 else "Yes")

# endregion ----------------


# region - View correlation matrix -------------------
training_df.corr(numeric_only=True)

# Which feature correlates most strongly to the label FARE?
answer = '''
The feature with the strongest correlation to the FARE is TRIP_MILES.
As you might expect, TRIP_MILES looks like a good feature to start with to train
the model. Also, notice that the feature TRIP_SECONDS has a strong correlation
with fare too.
'''
print(answer)

# Which feature correlates least strongly to the label FARE?
answer = '''The feature with the weakest correlation to the FARE is TIP_RATE.'''
print(answer)

# View pairplot
px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"]).show()  # 加上.show才会在本地IDE里显示画图


# endregion -------------------

# region - Define ML functions -------------------

def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple linear regression model."""
    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model topography into code that Keras can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                  loss="mean_squared_error",
                  metrics=metrics)

    return model


def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        label_name: str,
        settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(x=features,
                        y=label,
                        batch_size=settings.batch_size,
                        epochs=settings.number_epochs)

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


print("SUCCESS: defining linear regression functions complete.")

# endregion --------------------

# region - Experiment 1 ---------------------

# The following variables are the hyperparameters.
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=20,
    batch_size=50,
    input_features=['TRIP_MILES']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_1 = create_model(settings_1, metrics)

experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')

# How many epochs did it take to converge on the final model?
answer = """
Use the loss curve to see where the loss begins to level off during training.

With this set of hyperparameters:

  learning_rate = 0.001
  epochs = 20
  batch_size = 50

it takes about 5 epochs for the training run to converge to the final model.
"""
print(answer)

# How well does the model fit the sample data?
answer = '''
It appears from the model plot that the model fits the sample data fairly well.
'''
print(answer)

# endregion --------------------

print("the end")
