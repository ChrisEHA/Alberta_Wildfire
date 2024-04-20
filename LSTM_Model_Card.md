# Fire Prediction Model

* Model Name: Alberta Wildfire Prediction
* Model Version: 1.0
* Model Type: Binary Classification
* Developers:
    * Christopher Aitken, E.I.T.  
    * Raymond Xu, E.I.T.  
    * Mohammed Shah, E.I.T
    * Dana Shatara
* Release Date: 2024-04-19

## Intended Use

### Background

Wildfires occur due to various factors such as climate change, human activity, and landscape characteristics. Fires pose a risk to the environment and endanger lives, properties, and critical infrastructure.

Predicting fire occurrence with accuracy and efficiency is crucial for timely evacuation planning, resource allocation, and effective firefighting strategies.

Artificial intelligence has been used in wildfire management since the 1990s. Since then, rapid progression has been made in applying machine learning (ML) methods in the field.

### Problem Statement

*There is a need for reliable fire occurrence prediction from wildfire management government agencies that will help with firefighting resource strategies.*

* Primary Use: Predicting wildfire occurrences based on input features ranging from climate data and fire related metrics.
* Intended Users: Wildfire researchers, environmental agencies, emergency responders of government agency.
* Out-of-Scope Use Cases: Prediction of other natural disasters, non-environmental applications.

## Model/Data Description

* See Model_Diagrams.md for detailed data flow diagrams.
* Data Used: Wildfire dataset sourced from:
  * Alberta Wildfire Dataset
  * ERA5 dataset
  * FWI dataset.
* Preprocessing involved feature scaling and encoding categorical variables. Biases may exist due to uneven geographical distribution of data.
* Features: Input features include weather conditions, geographical data, vegetation type, and historical wildfire occurrences.
* Model Architecture: LSTM neural network with multiple layers. Adam optimizer used with a learning rate of 0.001. Binary cross-entropy loss function. Batch size of 32. Time step not utilized.

## Training and Evaluation

* Training Procedure: Model trained on GPU (AMD Ryzen 7) using TensorFlow 2.0 and Python 3.7. Trained with Hyperband Parameters.
* Evaluation Metrics:
  * Precision:  0.309
  * Recall:     0.169
  * F1:         0.218
  * ROC-AUC:    0.85

## Ethical Considerations

* Fairness and Bias: Considerations of fairness and bias arise when examining the historical wildfire dataset, including the possibility of neglected wildfires in particular areas or discrepancies in recording the causes of wildfires.
* Privacy: Privacy is of paramount significance, especially when it comes to the welfare of individuals impacted by wildfires as well as those represented in the training data.
* Security: Addressing security concerns is essential in order to ensure the confidentiality of the model and datasets from potential threats.

## Limitations and Recommendations

* Known Limitations: The model might encounter certain limitations, such as incomplete or biased historical wildfire data, potential deficiencies in particular geographical regions due to inadequate information availability, or difficulties in generalizing predictions from training to actual scenarios.
* Recommendations for Use: It is important to provide users with comprehensive documentation that clarifies the model's assumptions, limitations, and uncertainties in order to ensure effective and ethical utilization. Continuous monitoring and evaluation of the model's performance, in addition to periodic revisions and modifications are required to maintain its ongoing effectiveness and ethical implementation. In addition, establishing user training and awareness programs may promote responsible and informed utilization of the model by all stakeholders.

## Additional Information

* Contact Information: caitken@ualberta.ca
