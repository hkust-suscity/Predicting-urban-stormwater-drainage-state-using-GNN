import os

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--expid',type=str,default='link_cap_len6_6_f_r',help='experiment id')
parser.add_argument('--data',type=str,default='data/link_cap_len6_6_f_r',help='data path')
parser.add_argument('--seq_length',type=int,default=6,help='')#input sequence length
parser.add_argument('--out_length',type=int,default=6,help='')#output sequence length
parser.add_argument('--blocks',type=int,default=7,help='')#block is the number of TCN.Btw,TCN can have 1 or 2 layers,this project use 1 layer;if use 2 layers,'new_dilation'and'additional_scope' should be modified in model.py as well.
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')#two dimensions, network state and rainfall
parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--batch_size',type=int,default=2,help='batch size')
parser.add_argument('--num_nodes',type=int,default=667,help='number of nodes')#667 for link states, 664 for node states
parser.add_argument('--adjdata',type=str,default='kwdata/sensor_graph/adj_mat_link.pkl',help='adj data path')#two pkl files for link or node, 'adj_mat_link.pkl' or 'adj_mat_node.pkl'
parser.add_argument('--adjtype',type=str,default='transition',help='adj type')
parser.add_argument('--gcn_bool',default=True,action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',default=False,action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',default=True,action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',default=True,action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--print_every',type=int,default=1,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./modelpath/',help='save path')

args = parser.parse_args()

def main():
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    engine = trainer(scaler, args.in_dim, args.out_length, args.blocks,args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    print("start training...",flush=True)
    his_loss = []
    val_time = []
    train_time = []
    histrain_loss = []
    his_mape = []
    histrain_mape = []
    his_rmse = []
    histrain_rmse = []
    his_r2 = []
    histrain_r2 = []
    his_nrmse = []
    histrain_nrmse = []

    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_r2 = []
        train_nrmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_r2.append(metrics[3])
            train_nrmse.append(metrics[4])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train R2: {:.4f}, Train NRMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1],train_r2[-1], train_nrmse[-1]),flush=True)

        t2 = time.time()
        train_time.append(t2-t1)

        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_r2 = []
        valid_nrmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_r2.append(metrics[3])
            valid_nrmse.append(metrics[4])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_r2 = np.mean(train_r2)
        mtrain_nrmse = np.mean(train_nrmse)

        histrain_loss.append(mtrain_loss)
        histrain_mape.append(mtrain_mape)
        histrain_rmse.append(mtrain_rmse)
        histrain_r2.append(mtrain_r2)
        histrain_nrmse.append(mtrain_nrmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_r2 = np.mean(valid_r2)
        mvalid_nrmse = np.mean(valid_nrmse)

        his_loss.append(mvalid_loss)
        his_mape.append(mvalid_mape)
        his_rmse.append(mvalid_rmse)
        his_r2.append(mvalid_r2)
        his_nrmse.append(mvalid_nrmse)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid R2: {:.4f},Valid NRMSE: {:.4f},Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,mvalid_r2, mvalid_nrmse,  (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
        torch.save(engine.model.state_dict(), args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + "_" + str(round(mvalid_mape, 2)) + "_"+str(round(mvalid_rmse,2))+"_"+str(round(mvalid_r2,2))+"_"+str(round(mvalid_nrmse,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    #testing
    bestid = np.argmin(his_loss) #the model with the minimum mean valid loss
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    amae = []
    amape = []
    armse = []
    ar2 = []
    anrmse = []

    for i in range(args.out_length):
        if args.out_length == 1:
            pred = scaler.inverse_transform(yhat)
            real = realy[:, :, 0]
        else:
            pred = scaler.inverse_transform(yhat[:,:,i])
            real = realy[:,:,i]
        metrics = util.metric(pred, real)

        log = 'Evaluate best model on test data for Horizon{:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, TEST R2:{:.4f} Test NRMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1],metrics[2],metrics[3], metrics[4]))

        fields = [args.expid, args.seq_length, engine.model.blocks, args.epochs, round(his_loss[bestid], 4),
                  round(metrics[0], 4), round(metrics[1], 4), round(metrics[2], 4), round(metrics[3], 4),
                  round(metrics[4], 4), i + 1, args.out_length]
        with open(r'results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        ar2.append(metrics[3])
        anrmse.append(metrics[4])

    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, TEST R2:{:.4f} Test NRMSE: {:.4f}'
    print(log.format(args.out_length,np.mean(amae),np.mean(amape),np.mean(armse),np.mean(ar2),np.mean(anrmse)))

    bestname = args.save+"_"+args.expid+"_"+str(args.epochs)+"_best_"+str(round(his_loss[bestid],2))+"_"+str(round(metrics[0],2))+"_"+str(round(metrics[1],2))+"_"+str(round(metrics[2],2))+"_"+str(round(metrics[3],2))+"_"+str(round(metrics[4],2))+".pth"
    torch.save(engine.model.state_dict(),bestname)

    fields = [args.expid, args.seq_length, engine.model.blocks, args.epochs, round(his_loss[bestid], 4),
              round(np.mean(amae), 4), round(np.mean(amape), 4), round(np.mean(armse), 4),
              round(np.mean(ar2), 4), round(np.mean(anrmse), 4), "mean", args.out_length,bestname]
    with open(r'results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))


