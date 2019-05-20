import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(input_data, train_data, method='mean', normalize = False):
  
    if method not in ['mean', 'median', 'zeros']:
        raise NameError("Invalid method. Try 'mean', 'median' or 'zeros.'")
        
    output_data = input_data.copy()
        
    if method == 'mean':
        output_data = input_data.fillna(train_data.mean())
    elif method == 'median':
        output_data = input_data.fillna(train_data.median())
    elif method == 'zeros':
        output_data = input_data.fillna(0)
    
    if normalize:
        scalers = {}
        
        for label, content in output_data.iteritems():
            scaler = StandardScaler()
            output_data[label] = scaler.fit_transform(np.expand_dims(content, axis = 1))
            
            scalers[label] = scaler
        
        return output_data, scalers
    
    return output_data, None
    
        