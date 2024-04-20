# Fire Prediction Model

-   Model Name: Alberta Wildfire Occurence Prediction
-   Model Version: 1.0
-   Model Type: LSTM - Binary Classification
-   Developers:
    -   Christopher Aitken, E.I.T.\
    -   Raymond Xu, E.I.T.\
    -   Mohammed Shah, E.I.T.
-   Release Date: 2024-04-19

# Intended Use

-   Background:

> Wildfires occur due to various factors such as climate change, human
> activity and landscape characteristics. Fires pose a risk to the
> environment and endanger lives, properties and critical
> infrastructure.

> Predicting fire occurrence with accuracy and efficiency, is crucial
> for timely evacuation planning, resource allocation and effective
> firefighting strategies.

> Artificial intelligence has been used in wildfire management since
> 1990s. Since then, rapid progression has been made on applying machine
> learning (ML) methods in the field.

**Problem Statement**

*There is a need for reliable fire occurrence prediction from wildfire
management government agencies that will help with firefighting resource
strategies.*

-   Primary Use: Predicting wildfire occurrences based on input features
    ranging from climate data and fire related metrics

-   Intended Users: Wildfire researchers, environmental agencies,
    emergency responders of government agency

-   Out-of-Scope Use Cases: Prediction of other natural disasters,
    non-environmental applications.

# Model/Data Description

-   Data Used: Wildfire dataset sourced from;

    -   Alberta Wildfire Dataset (2000-2021)
    -   ERA5 reanalysed dataset
    -   Fire Weather Index (FWI) dataset

-   Preprocessing involved feature scaling and encoding categorical
    variables. Biases may exist due to uneven geographical distribution
    of data.

-   Features: Input features include weather conditions, geographical
    data, vegetation type, and historical wildfire occurrences.

-   Model Architecture: LSTM neural network with multiple layers. Adam
    optimizer used with an adopted learning rate between 0.0001 to 0.02.
    Binary cross-entropy loss function. Batch size of 64. Time step was
    not utilized.

# Training and Evaluation

-   Model trained on GPU (AMD Ryzen 7) using TensorFlow 2.0 and Python
    3.7. Hyperparameter searching was completed using Hyperband.
-   Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
-   Baseline Comparison: Compared against Logistic Regression and Random
    Forest models.

# Ethical Considerations

-   Fairness and Bias: Considerations of fairness and bias arise when
    examining the historical wildfire dataset, including the possibility
    of neglected wildfires in particular areas or discrepancies in
    recording the causes of wildfires. Strategies for tackling biases
    involve performing audits on the dataset to identify and eliminate
    any biases, while employing algorithms that prioritize fairness and
    eliminate errors during model training and ensuring a diverse
    representation in the training data to mitigate demographic biases.

-   Privacy: Privacy is of paramount significance, especially when it
    comes to the welfare of individuals impacted by wildfires as well as
    those represented in the training data. It is essential that we
    protect the privacy rights of individuals by concealing or
    aggregating personal information, such as exact location data or
    identifiable defining features. It is important to ensure compliance
    with relevant privacy regulations, such as GDPR or HIPAA, throughout
    the data handling process.

-   Security: Addressing security concerns is essential in order to
    ensure the confidentiality of the model and datasets from potential
    threats. It is important to implement strong security measures, such
    as encryption of confidential information, establishing access
    controls and conducting regular security audits, in order to
    maintain the security and confidentiality regarding both the data
    and the model.

# Limitations and Recommendations

-   Known Limitations: The model might encounter certain limitations,
    such as incomplete or biased historical wildfire data, potential
    deficiencies in particular geographical regions due to inadequate
    information availability, or difficulties in generalizing
    predictions from training to actual scenarios. In addition, the
    model\'s predictive accuracy may be impacted by limitations in the
    spatial and temporal resolution of climate data. It is imperative to
    openly acknowledge these limitations and include uncertainty
    estimates in model outputs to effectively manage expectations.

-   Recommendations for Use: It is important to provide users with
    comprehensive documentation that clarifies the model\'s assumptions,
    limitations and uncertainties in order to ensure effective and
    ethical utilization. It is essential to communicate recommendations
    for interpreting model predictions and understanding confidence
    levels. Additionally, providing guidelines for incorporating
    additional contextual information when making decisions based on
    model outputs is required. Continuous monitoring and evaluation of
    the model\'s performance, in addition to periodic revisions and
    modifications are required to maintain its ongoing effectiveness and
    ethical implementation. In addition, establishing user training and
    awareness programs may promote responsible and informed utilization
    of the model by all stakeholders.

# Additional Information

-   Contact Information: <rxu6@ualberta.ca>, <caitken@ualberta.ca>