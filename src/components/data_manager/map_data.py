import numpy as np

cholecT45_array=np.load('cholecT45_array.npy') 
cholecSeg8k_array=np.load('cholecSeg8k_array.npy') 

#print(cholecT45_array)
#print(cholecSeg8k_array)

def get_data_mapping():

    mapping=(cholecT45_array[None,:] == cholecSeg8k_array[:,None]).all(-1).any(0)
    
    return mapping


def get_matching_data(mapping):
    
    matching_data = cholecT45_array[mapping]
    
    return matching_data

    
if __name__=='__main__':
    mapping=get_data_mapping()
    matching_data=get_matching_data(mapping=mapping)
    
    print(matching_data.shape)
    print(matching_data)
    
    np.save('matching_data_array.npy', matching_data) 