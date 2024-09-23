import torch

class model_inferance:
    def __init__(self,model_path):
        self.model=torch.load(model_path)
        self.labels={0:"BASE520",1:"BASE635",2:"BROKEN520",3:"BROKEN635",4:"FRESH520",5:"FRESH635",6:"MODERATE520",7:"MODERATE635"}
        self.model.eval()
    def predict(self,input):
        with torch.inference_mode():
            predict=self.model(input)
        predict=predict.argmax(-1)
        return [self.labels[i] for i in predict.numpy()]
    