from __future__ import print_function
import argparse
import torch
import numpy as np
from utils import calculate_log as callog
import models
import os
from utils import lib_generation
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.autograd import Variable
from utils import lib_regression
import argparse
from sklearn.linear_model import LogisticRegressionCV


def Mahalanobis_Generate(pre_trained_net, train_dataset, in_dataset, out_dataset):
    
    model_path = './trained_models/'+pre_trained_net+'.pth'
    outf = './output/'+pre_trained_net+'/'
    
    if os.path.isdir(outf) == False:
        os.mkdir(outf)
    torch.cuda.manual_seed(0)

    model = torch.load(model_path)
    try:
        model = model.embedding_net
        #print("model is siamese or triplet")    
    except:
        pass
        #print("model is just resnet")    
    model.cuda()
    print('load model: ' + pre_trained_net)
    batch_size=256
     
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    out_test_loader = torch.utils.data.DataLoader(out_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    temp=in_dataset.root    
    in_dataset_name = temp.split('/')[-1]
    temp=out_dataset.root    
    out_dataset_name = temp.split('/')[-1]
    
    model.eval()
    if in_dataset_name== 'MNIST' or in_dataset_name=="FashionMNIST":
        temp_x = torch.rand(256,1,28,28).cuda()
    else:
        temp_x = torch.rand(256,3,32,32).cuda()
    temp_x = Variable(temp_x)
    if len(model.feature_list(temp_x)) == 2:
        temp_list = model.feature_list(temp_x)[1]
    else :
        temp_list = model.feature_list(temp_x)
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    if in_dataset_name == 'CIFAR100':
        num_classes =100
    else:
        num_classes=10
    #print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader)
    
    #print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, outf, \
                                                        True, sample_mean, precision, i, magnitude, in_dataset_name)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
              
        print('Out-distribution: ' + out_dataset_name) 
        for i in range(num_output):
            M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader,num_classes, outf, \
                                                         False, sample_mean, precision, i, magnitude, in_dataset_name)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
        file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), in_dataset_name , out_dataset_name))
        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)
        
def Mahalanobis_Regression(pre_trained_net, in_dataset, out_dataset):
    temp=in_dataset.root    
    in_dataset_name = temp.split('/')[-1]
    temp=out_dataset.root    
    out_dataset_name = temp.split('/')[-1]
    
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    list_best_results, list_best_results_index = [], []
    print('In-distribution: ', in_dataset_name)
    outf = './output/'+pre_trained_net+'/'    
    print('Out-of-distribution: ', out_dataset_name)
    
    best_tnr, best_result, best_index = 0, 0, 0
    list_best_results_out, list_best_results_index_out = [], []
    for score in score_list:
        total_X, total_Y = lib_regression.load_characteristics(score, in_dataset_name, out_dataset_name, outf)
        X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out_dataset_name)
        X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
        Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
        X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
        Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        y_pred = lr.predict_proba(X_train)[:, 1]
        
        results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
        if best_tnr < results['TMP']['TNR']:
            best_tnr = results['TMP']['TNR']
            best_index = score
            best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
    list_best_results_out.append(best_result)
    list_best_results_index_out.append(best_index)
    list_best_results.append(list_best_results_out)
    list_best_results_index.append(list_best_results_index_out)
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    
    for in_list in list_best_results:
        print('in_distribution: ' + in_dataset_name + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_dataset_name)
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1