from sklearn.preprocessing import StandardScaler
import random

def get_framedata(data,channels, frameSize, overlap):
    wlen = data.shape[1]
    step = frameSize - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = np.zeros((frameNum-math.ceil(frameSize/step),frameSize*19))
    for i in range(frameNum-math.ceil(frameSize/step)):
        singleFrame = data[:,np.arange(i * step, i * step+frameSize)].reshape(1,-1)
        frameData[i,0:singleFrame.shape[1]] = singleFrame
    return frameData.T

def gen_data_cross(xin,bigrate,lambd=0.75):
    
    bigsize = int(xin.shape[1]*bigrate) #(channels*framesize,trails)
    new_data = np.zeros((xin.shape[0],bigsize))
   
    for i in range(bigsize):
        data_sel1 = xin[:,random.randint(0,xin.shape[1]-1)]
        data_sel2 = xin[:,random.randint(0,xin.shape[1]-1)]
        new_data[:,i] = lambd*data_sel1+(1-lambd)*data_sel2
    return new_data

def get_data_label(matdata,channels,f_size,o_lap,sub_list,genrate):
    perSframe = []
    for i in range(len(matdata)):
        for j in sub_list:
            if((i+1)==j):
                if(perSframe ==[]):
                    if(genrate == 0.0):
                        perSframe = get_framedata(matdata[i],channels,f_size,o_lap)
                    else:
                        perSframe = get_framedata(matdata[i],channels,f_size,o_lap)
                        gen_data = gen_data_cross(perSframe,bigrate=genrate,lambd=0.45)
                        perSframe = np.concatenate((perSframe,gen_data),axis=1)
                else:
                    if(genrate==0.0):
                        perSframe = np.hstack((perSframe,get_framedata(matdata[i],channels,f_size,o_lap)))
                    else:
                        tempframe = get_framedata(matdata[i],channels,f_size,o_lap)
                        gen_data = gen_data_cross(tempframe,bigrate=genrate,lambd=0.45)
                        perSframe = np.concatenate((perSframe,tempframe,gen_data),axis=1)
    return perSframe

# voting experiment on sub
def get_sub_split():
    all_sub = list(range(1,15))

    test_sub = random.sample(all_sub, 4)
    remaining_sub = [i for i in all_sub if i not in test_sub]
    valid_sub = random.sample(remaining_sub, 2)
    train_sub = [i for i in remaining_sub if i not in valid_sub] 
    return test_sub,valid_sub,train_sub

def cross_data(channels,f_size,o_lap,cEE_list,pEE_list):

    test_sub,valid_sub,train_sub = get_sub_split()
    train_datas = get_data_label(cEE_list,channels,f_size,o_lap,train_sub,1.2)
    valid_datas = get_data_label(cEE_list,channels,f_size,o_lap,valid_sub,1.2)
    test_datas = get_data_label(cEE_list,channels,f_size,o_lap,test_sub,0.0)

    test_sub,valid_sub,train_sub = get_sub_split()
    train_datah = get_data_label(pEE_list,channels,f_size,o_lap,train_sub,1.2)
    valid_datah = get_data_label(pEE_list,channels,f_size,o_lap,valid_sub,1.2)
    test_datah = get_data_label(pEE_list,channels,f_size,o_lap,test_sub,0.0)
    
   

    train_data = np.hstack((train_datas,train_datah)).T
    train_label = to_categorical(np.hstack((np.zeros(train_datas.shape[1]),np.ones(train_datah.shape[1]))))
    
    valid_data = np.hstack((valid_datas,valid_datah)).T
    valid_label = to_categorical(np.hstack((np.zeros(valid_datas.shape[1]),np.ones(valid_datah.shape[1]))))

    test_data = np.hstack((test_datas,test_datah)).T
    test_label = to_categorical(np.hstack((np.zeros(test_datas.shape[1]),np.ones(test_datah.shape[1]))))
    
    train_data,valid_data,test_data = standardize_data(train_data,valid_data,test_data,channels,f_size)

    return train_data,train_label,valid_data,valid_label,test_data,test_label


def voting_accuracy(model, test_data, test_labels, segment_length=5):
    predictions = model.predict({'ee_in': test_data}, verbose=0)
    predicted_labels = np.argmax(predictions[1], axis=1)

    true_labels = np.argmax(test_labels, axis=1)
    correct_votes = 0
    total_votes = 0

    for i in range(0, len(predicted_labels) - segment_length + 1):
        segment_preds = predicted_labels[i:i + segment_length]
        segment_true = true_labels[i + segment_length - 1]  
        voted_label = np.bincount(segment_preds).argmax()

        if voted_label == segment_true:
            correct_votes += 1
        total_votes += 1
    accuracy = correct_votes / total_votes
    return accuracy