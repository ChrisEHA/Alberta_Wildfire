## Data sources and handling diagrams

**Overview of data flow for model training.**

![Training Diagram](/Model_Diagrams/Wildfire_ML_dataFlow.png)

=================================================
**Detailed diagram of point selection**

![Point Selection Diagram](/Model_Diagrams/Point_Selection.png)

=================================================
**Model prediction diagram**

![Prediction Diagram](/Model_Diagrams/Prediction_diagram.png)

=================================================
**Detailed feature preprocessing for LSTM**

![Prediction Diagram](/Model_Diagrams/LSTM_preprocessing.png)

=================================================
**Next Steps**
- Discuss the results with fire experts from Forest Fire Services of NRCan for their inputs.
    - Better understanding of importance of precision and recall
- Incorporate some features that link to the civil structures such as distance to the nearby towns and road.
- Try incorporate macro climate variability variables such as Southern Oscillation Index, & El Niño-Southern Oscillation.
- Expand dataset to include 2020 to 2022 wildfires
- Incorporate LSTM into the ensemble model
- Explore preprocessing filter steps to handle outliers (locations that models consistently have troubles predicting)
