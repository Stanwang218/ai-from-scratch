class basecluster:
    def __init__(self) -> None:
        pass
    
    def fit(self):
        raise NotImplementedError
    
    def pred(self):
        raise NotImplementedError
    
    def score(self):
        raise NotImplementedError