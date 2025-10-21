class DecodedPipelineWrapper:
    def __init__(self, pipeline, label_encoder):
        self.pipeline = pipeline
        self.le = label_encoder

    def predict(self, order):
        y_pred_encoded = self.pipeline.predict([order])
        y_pred_decoded = self.le.inverse_transform(y_pred_encoded)
        return str(y_pred_decoded[0])
    
    def probability(self, order):
        probabilities = self.pipeline.predict_proba([order])
        probability_max = round(probabilities.max()*100, 2)
        return probability_max

