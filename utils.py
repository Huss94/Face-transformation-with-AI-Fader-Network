import numpy as np

def prepare_attributes(attributes,param_attr):
    attr = None
    for p in param_attr:
        for i in range(2):
            formated_attr = np.reshape(attributes[p] == i, (len(attributes[p]), 1))
            if attr is None:
                attr = np.array(formated_attr.astype(np.float32))
            else:
                attr = np.append(attr, formated_attr,axis = 1)
    return attr


