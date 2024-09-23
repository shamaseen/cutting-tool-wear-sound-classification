import numpy as np
import onnxruntime as rt

class soundModel:
    def __init__(self, model_path) -> None:
        self.interpreter = rt.InferenceSession(model_path)
        self.input_details = self.interpreter.get_inputs()[0].name
        self.labels={0:"BASE520",1:"BASE635",2:"BROKEN520",3:"BROKEN635",4:"FRESH520",5:"FRESH635",6:"MODERATE520",7:"MODERATE635"}

    def predict(self, input_sample):
        if type(input_sample)!=np.ndarray:
            input_sample=input_sample.numpy()
        predict =self.interpreter.run(None, {self.input_details: input_sample.astype(np.float32)})[0]
        predict=predict.argmax(-1)
        return [self.labels[i] for i in predict]
