import os
import torch
from utils import lib_generation
from utils import calculate_log as callog

def Baseline_and_ODIN(pre_trained_net, in_dataset, out_dataset):
    # set the path to pre-trained model and output
    model_path = './trained_models/'+pre_trained_net+'.pth'
    outf = './output/'+pre_trained_net+'/'
    if os.path.isdir(outf) == False:
        os.mkdir(outf)
    torch.cuda.manual_seed(0)
    
    
    # load networks
    model = torch.load(model_path)
    model.cuda()
    print('load model: ' + pre_trained_net)
    
    # load dataset
    batch_size=32 
    kwargs = {'num_workers': 2, 'pin_memory': True}    
    test_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    out_test_loader = torch.utils.data.DataLoader(out_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    temp=in_dataset.root    
    in_dataset_name = temp.split('/')[-1]
    print('load target data: ', in_dataset_name) 
    temp=out_dataset.root
    out_dataset_name = temp.split('/')[-1]
    print('Out-distribution: ' + out_dataset_name)
    
    # measure the performance
    M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    T_list = [1, 10, 100, 1000]
    base_line_list = []
    ODIN_best_tnr = [0]
    ODIN_best_results = [0]
    ODIN_best_temperature = [-1]
    ODIN_best_magnitude = [-1]
    
    for T in T_list:
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(model, test_loader, magnitude, temperature, outf, True, in_dataset_name)
            out_count = 0
            print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude)) 
            lib_generation.get_posterior(model, out_test_loader, magnitude, temperature, outf, False, in_dataset_name)
            if temperature == 1 and magnitude == 0:
                test_results = callog.metric(outf, ['PoT'])
                base_line_list.append(test_results)
            else:
                val_results = callog.metric(outf, ['PoV'])
                if ODIN_best_tnr[out_count] < val_results['PoV']['TNR']:
                    ODIN_best_tnr[out_count] = val_results['PoV']['TNR']
                    ODIN_best_results[out_count] = callog.metric(outf, ['PoT'])
                    ODIN_best_temperature[out_count] = temperature
                    ODIN_best_magnitude[out_count] = magnitude
            out_count += 1
    
    # print the results
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print('Baseline method: in_distribution: ' + in_dataset_name + '==========')
    count_out = 0
    for results in base_line_list:
        print('out_distribution: '+ out_dataset_name)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('')
        count_out += 1
        
    print('ODIN method: in_distribution: ' + in_dataset_name + '==========')
    count_out = 0
    for results in ODIN_best_results:
        print('out_distribution: '+ out_dataset_name)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('temperature: ' + str(ODIN_best_temperature[count_out]))
        print('magnitude: '+ str(ODIN_best_magnitude[count_out]))
        print('')
        count_out += 1