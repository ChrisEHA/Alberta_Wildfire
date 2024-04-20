from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
set_config(transform_output = "pandas")


## Begin functions
def get_predictions_pipeline(ann_predictor,rf_predictor,svm_predictor):
    ann = FunctionTransformer(ann_predictor.predict, validate=False)
    rf = FunctionTransformer(rf_predictor.predict, validate=False)
    svm = FunctionTransformer(svm_predictor.predict, validate=False)

    Participant_preditcions = FeatureUnion([
        ('ann', ann),
        ('rf', rf),
        ('svm',svm),
    ])

    # Create prediction feature space
    predictions_pipeline = Pipeline([
        ('predictions', Participant_preditcions),
    ])

    return predictions_pipeline
