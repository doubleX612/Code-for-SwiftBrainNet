import dill
from scipy import signal
from tensorflow.python.keras.optimizers import adam_v2
from SwiftBrainNet import SwiftBrainNet
from data_preprocessing import get_matdata,cross_data,voting_accuracy


def categorical_focal_loss(alpha, gamma=2.):
    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        sd_weight = 1e-10
        return K.mean(K.sum(loss, axis=-1))-sd_weight*K.mean(K.sqrt(K.var(y_pred,axis=-1)))
    return categorical_focal_loss_fixed

f_size = 25 # 25 equals 100ms with 250Hz sample frequency and 50 equals 100ms with 500Hz sample frequency
o_lap = 0
acc_res = []
sub_res = []
i_list = []
start = time.time()
np.random.seed(0)
tf.random.set_seed(0)
channels = 19
 
sEE_list,hEE_list = get_matdata()    
window_ee_channel = signal.hamming(f_size)
window_ee_channels = np.expand_dims(window_ee_channel,0).repeat(channels,axis=0).flatten()
train_data,train_label,valid_data,valid_label,test_data,test_label = cross_data(19,f_size,o_lap,sEE_list,hEE_list)
train_data = np.concatenate((train_data,valid_data),axis=0)
train_label = np.concatenate((train_label,valid_label),axis=0)

# Xtrain, Ytrain,Xvalid, Yvalid = loocv_data_new(60,f_size,o_lap,sEE_list,hEE_list,0.0,sel_num)
shuffle_num = np.random.permutation(len(train_data))
train_data=   train_data[shuffle_num, :]
train_label = train_label[shuffle_num,:]
train_data  = np.multiply(window_ee_channels,train_data)
valid_data  = np.multiply(window_ee_channels,valid_data)
test_data  = np.multiply(window_ee_channels,test_data)

del sEE_list,hEE_list


# checkp = ModelCheckpoint(filepath='./val_select/train_by_person.h5',verbose=0,period=30,
#                 save_weights_only=True,monitor='val_loss',mode='min',save_best_only='True')

eemodel2 = multi_eemodel(60,5,int(f_size*channels))
eemodel2 = SwiftBrainNet(0.45,4,4,int(f_size*channels))

adm = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,amsgrad=False,epsilon=1e-08, decay=0.0, clipnorm=1.0)
eemodel2.compile(optimizer=adm,
                loss={'ee_rebuildout': 'mean_squared_error','ee_classout':dill.loads(dill.dumps(categorical_focal_loss(gamma=2, alpha=[[0.25,0.25]])))},
                # loss={'ee_rebuildout': 'mean_squared_error','ee_classout': 'categorical_crossentropy'},
                loss_weights={'ee_rebuildout': 0.001, 'ee_classout': 0.999},
                # loss={'ee_classout': 'categorical_crossentropy'},
                metrics= {'ee_classout':metrics.categorical_accuracy})#[metrics.categorical_accuracy, sensitivity, specificity]

eemodel2.fit({'ee_in':train_data},{'ee_rebuildout':train_data,'ee_classout':train_label},
              batch_size=2048,epochs=300,
              validation_data=({'ee_in': test_data}, {'ee_rebuildout': test_data, 'ee_classout': test_label}),
              # callbacks = [checkp],
              verbose=1)
# eemodel2.load_weights('./val_select/train_by_person.h5')
scores = eemodel2.evaluate({'ee_in':test_data},
                 {'ee_rebuildout':test_data,'ee_classout':test_label}, verbose=1)
print("%s: %.2f%%" % (eemodel2.metrics_names[3], scores[3]*100))

voting_acc = voting_accuracy(eemodel2, test_data, test_label)
print(f'Voting Accuracy: {voting_acc * 100:.2f}%')

voting_acc = voting_accuracy(eemodel2, test_data, test_label,9)
print(f'Voting Accuracy: {voting_acc * 100:.2f}%')

voting_acc = voting_accuracy(eemodel2, test_data, test_label,15)
print(f'Voting Accuracy: {voting_acc * 100:.2f}%')