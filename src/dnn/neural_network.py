import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from dnn.convolutional_layer import ConvolutionalLayer
from dnn.fully_connected_layer import FullyConnected
from dnn.input_waveform_layer import InputWaveformLayer

def encode_onehot(classes, y_train):
    targets = y_train.reshape(-1).astype(int)
    return np.eye(classes)[targets]

def decode_onehot(h):
    ypred = np.argmax(h, 1)
    return ypred.reshape(-1,1)

class NeuralNetwork:
    def __init__(self, name, model_layers, objective, train_type='sgd', optimizer='adam'):
        self.name = name
        self.layers = model_layers
        self.objective = objective
        self.train_type = train_type
        self.optimizer = optimizer
        self.x_train = None

    def __forward(self, x, useInputLayer=False):
        h = x
        for layer in self.layers:
            if isinstance(layer, InputWaveformLayer) and not useInputLayer:
                continue
            h = layer.forward(h)
        return h

    def __backward(self, y, h, epoch, eta):
        grad = self.objective.gradient(y, h)
        for layer in reversed(self.layers):
            if isinstance(layer, InputWaveformLayer):
                continue
            newgrad = layer.backward(grad, eta)
            if isinstance(layer, FullyConnected):
                if self.optimizer == 'adam':
                    layer.updateWeights_adam(grad, epoch, eta)
                else:
                    layer.updateWeights(grad, epoch, eta)
            grad = newgrad

    def __moving_avg(self, values, window):
        return np.convolve(values, np.repeat(1.0, window)/window, 'valid')

    def train(self, x_train, y_train, epochs, batch_size, eta, criteria, random_seed=0, x_valid=None, y_valid=None):

        if x_valid is None:
            print('### ERROR ###')
            print('Training must have validadion data')
            print()
            return

        print("Training...")
        np.random.seed(random_seed)
        train_loss = []
        # generate training spectrograms
        xin = self.layers[0].forward(x_train)
        self.x_train = xin.reshape(xin.shape[0], 1, xin.shape[1], xin.shape[2])

        # generate validation spectrograms
        xvin = self.layers[0].forward(x_valid)
        x_valid = xvin.reshape(xvin.shape[0], 1, xvin.shape[1], xvin.shape[2])

        # setup overall metrics
        training_totals = {'correct':[], 'false_pos':[], 'false_neg':[], 'objective':[]}
        validation_totals = {'correct':[], 'false_pos':[], 'false_neg':[], 'objective':[]}
        batchavg_training_totals = {'correct':[], 'false_pos':[], 'false_neg':[], 'objective':[]}

        # log file data
        report = '###### Acoustic Crack Detection #######\n\n'
        report += 'Datasets:'
        report += f'\n\tTraining Size: {self.x_train.shape}'
        report += f'\n\tValidation Size: {x_valid.shape}'
        report += '\n\n'
        report += 'Layer Model:\n'
        for n,l in enumerate(self.layers):
            if (isinstance(l,FullyConnected)):
                report += f'\tLayer {n}: {type(l).__name__} {l.getWeights().shape}\n'
            elif (isinstance(l,ConvolutionalLayer)):
                k = l.kernels_shape
                report += f'\tLayer {n}: {type(l).__name__} (chan_in:{k[0]}  chan_out:{k[1]}  K={k[2]}x{k[3]})\n'
            else:
                report += f'\tLayer {n}: {type(l).__name__}\n'
        report += f'\tObjective: {type(self.objective).__name__}\n'

        # init log files
        with open('../plots/acd_report.txt', 'w') as f:
            f.write(report)
        with open('../plots/training_totals.json', 'w') as f:
            f.write(report)
        with open('../plots/validation_totals.json', 'w') as f:
            f.write(report)

        print(report)
        

        if self.train_type == 'sgd': # stochastic mini batch training
            train_loss_itr = []
            avg_win = 5
            loss = None
            loss_change = 1
            old_loss = 1

            # overall timers
            batch_times = []
            epoch_times = []
            begin_time = time.perf_counter()

            for n in range(1, epochs):
                # epoch timer
                e_tic = time.perf_counter()
                
                print(f'\nEpoch {n}:', end='', flush=True)
                train_precision_epoch = {'correct':[], 'false_pos':[], 'false_neg':[]}

                for b in range(batch_size): # per batch
                    # batch timer
                    b_tic = time.perf_counter()

                    print(f'\n|__Batch {b}:', end='', flush=True)
                    #random samples
                    indecies= np.random.randint(0, len(y_train), int(len(y_train)/batch_size))
                    xt = np.take(self.x_train, indecies, 0)
                    yt = np.take(y_train, indecies, 0)
                    ht = []
                    for x, y in zip(xt, yt): # per train sample
                        print('.', end='', flush=True)
                        
                        #forwards
                        h = self.__forward(x)
                        ht.append(h)
                        
                        #backwards
                        self.__backward(y, h, n, eta)
                    
                    # checking precision per mini-batch 
                    ht = np.array(ht)
                    print('\nprecision:')
                    prec = self.precision(yt,ht)
                    for p in prec.keys():
                        train_precision_epoch[p].append(prec[p])
                        print(f'{p} : {prec[p]}%')

                    # evaluate training
                    loss = self.objective.eval(yt, ht)
                    train_loss_itr.append(loss)

                    if b == batch_size -1:
                        train_loss.append(loss) # train loss per epoch

                    # batch timer
                    b_toc = time.perf_counter()
                    batch_times.append(b_toc - b_tic)
                    print(f'\nBatch Time: {batch_times[-1]:0.4f} seconds')

                    if len(train_loss) > 0 and b == batch_size-1:
                        avg_win_index = len(train_loss_itr) - avg_win
                        new_loss = self.__moving_avg(train_loss_itr[avg_win_index:], avg_win).mean()
                        loss_change = abs((new_loss-old_loss)/old_loss)

                        print(f'\nEnd of Epoch: {n}')
                        print(f'\tloss: {loss}')
                        print(f'\tabs change in loss: {loss_change}')

                        #### END OF EPOCH METRICS ####
                        # track avg batch precision over epochs
                        for p in train_precision_epoch.keys():
                            val = 0
                            count = len(train_precision_epoch[p])
                            for n in train_precision_epoch[p]:
                                val += n/float(count)
                            batchavg_training_totals[p].append(val)
                            print(f'\tAvg {p} in Epoch: {val}')
                        

                        ### TESTING ON TRAIN AND VALIDATION
                        # avg epoch training metrics report
                        print('\nTesting Train Data ( . x10):')
                        test_results_train = []
                        for i,x in enumerate(self.x_train):
                            if i%10 == 0:
                                print('.', end='', flush=True)
                            test_results_train.append(self.__forward(x))
                        print('complete')
                        
                        # store training precision
                        test_prec = self.precision(y_train, np.array(test_results_train))
                        for k in test_prec.keys():
                            training_totals[k].append(test_prec[k])
                            print(f'train {k}: {test_prec[k]}')
                        # store training objective
                        training_totals['objective'].append(self.objective.eval(y_train, np.array(test_results_train)))
                        print('test J results:', training_totals['objective'][-1])

                        # epoch validation metrics report
                        print('\nTesting Validation Data ( . x10):')
                        test_results_valid = []
                        for i,x in enumerate(x_valid):
                            if i%10 == 0:
                                print('.', end='', flush=True)
                            test_results_valid.append(self.__forward(x))
                        print('complete')

                        # store validation precision
                        valid_prec = self.precision(y_valid, np.array(test_results_valid))
                        for k in valid_prec.keys():
                            validation_totals[k].append(valid_prec[k])
                            print(f'valid {k}: {valid_prec[k]}')
                        # store validation objective
                        validation_totals['objective'].append(self.objective.eval(y_valid, np.array(test_results_valid)))
                        print('test J results:', validation_totals['objective'][-1])

                        ### PLOTS
                        # plot objective
                        self.plot_j(train_loss, "Loss vs Epoch", len(train_loss), xLabel="Epochs", yLabel="Loss (J)")
                        # plot precision per epoch
                        self.plot_precision(training_totals, 'training')
                        self.plot_precision(validation_totals, 'validation')
                        self.plot_epoch_all(training_totals, validation_totals)

                        # timer stuff
                        end_time = time.perf_counter()
                        e_toc = time.perf_counter()
                        epoch_times.append(e_toc - e_tic)
                        print(f'\nEpoch Time: {epoch_times[-1]:0.4f} seconds')

                        #### REPORTING
                        end_report = f'\nTraining Time: {end_time - begin_time:0.4f} seconds\n'
                        end_report += f'Avg Batch Time: {np.mean(batch_times):0.4f} seconds\n'
                        end_report += f'Avg Epoch Time: {np.mean(epoch_times):0.4f} seconds\n\n'
                        end_report += f'Max Epochs: {epochs}\n'
                        end_report += f'Total Epochs: {n}\n'
                        end_report += f'eta: {eta}\n'
                        end_report += f'Batch Size: {batch_size}\n'
                        end_report += f'Objective Threshold: {criteria}\n\n'
                        end_report += 'Accuracy:\n'
                        end_report += f"Train Start: {training_totals['correct'][0]:0.4f}%\n"
                        end_report += f"Train End: {training_totals['correct'][-1]:0.4f}%\n"
                        end_report += f"Train FP: {training_totals['false_pos'][-1]:0.4f}%\n"
                        end_report += f"Train FN: {training_totals['false_neg'][-1]:0.4f}%\n\n"
                        end_report += f"Valid Start: {validation_totals['correct'][0]:0.4f}%\n"
                        end_report += f"Valid End: {validation_totals['correct'][-1]:0.4f}%\n"
                        end_report += f"Valid FP: {validation_totals['false_pos'][-1]:0.4f}%\n"
                        end_report += f"Valid FN: {validation_totals['false_neg'][-1]:0.4f}%\n\n"
                        end_report += f"Final Train J: {training_totals['objective'][-1]:0.8f}\n"
                        end_report += f"Final Valid J: {validation_totals['objective'][-1]:0.8f}\n"

                        # overwrite data log
                        with open('../plots/acd_report.txt', 'r+') as f:
                            foo = f.read()
                            f.seek(0)
                            f.write(report)
                            f.write(end_report)
                            f.truncate
                        
                        # write all data to json files for post-graphing if needed
                        with open('../plots/training_totals.json', 'r+', encoding='utf-8') as f:
                            foo = f.read()
                            f.seek(0)
                            json.dump(training_totals, f, ensure_ascii=False, indent=4)
                            f.truncate

                        with open('../plots/validation_totals.json', 'r+', encoding='utf-8') as f:
                            foo = f.read()
                            f.seek(0)
                            json.dump(validation_totals, f, ensure_ascii=False, indent=4)
                            f.truncate

                        ### Break out if early termination
                        if loss_change <= criteria:
                            break

                        old_loss = new_loss

                if loss_change <= criteria:
                    break
           

            print(f'\nResults End of Epoch: {n}')
            print(f'\tloss: {loss}')
            print(f'\tabs change in loss: {loss_change}')
            print(f'\taccuracy: {self.straight_accuracy(y_train,test_results_train)}')
            print('\tprecision:')
            for p in prec.keys():
                print(f'\t  {p} : {prec[p]}%')

            self.plot_j(train_loss, "Loss vs Epoch", len(train_loss), xLabel="Epochs", yLabel="Loss (J)")
        return train_loss

    def predict(self, X):
        return self.__forward(X, True)

    def straight_accuracy(self, Y_hot, h):
        return np.mean(np.argmax(Y_hot, axis=1)==np.argmax(h, axis=1))[...,None]*100

    def accuracy(self, y_true, y_pred):
        match_count = 0
        N = 0
        if type(y_pred) is list:
            N = len(y_pred)
        else:
            N = y_pred.shape[0]

        for i in range(N):
            ypred = y_pred[i]
            ytrue = y_true[i]
            if(ypred == ytrue):
                match_count += 1.0
        accuracy = match_count / N
        return accuracy

    def precision(self, y_true, y_pred):
        match_count = 0
        false_pos = 0
        false_neg = 0

        for i in range(0,y_pred.shape[0]):
            h_pred = np.zeros(y_pred[i].size)
            h_pred[np.argmax(y_pred[i])] = 1
            h = h_pred
            if np.array_equal(np.atleast_2d(h), y_true[i]):
                match_count += 1
            else:
                if h[1]==1:
                    false_pos += 1
                else:
                    false_neg += 1
        vals = {}
        vals['correct'] = match_count/y_true.shape[0] * 100
        vals['false_pos'] = false_pos/y_true.shape[0] * 100
        vals['false_neg'] = false_neg/y_true.shape[0] * 100
        return vals


    def plot_j(self, train_loss, title, xLim, xLabel, yLabel):
        plt.plot(train_loss, color="blue", label="training")
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.xlim(0,xLim)
        #plt.tight_layout()
        plt.legend()
        path = "../plots/"
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = path+self.name+" "+title+".png"
        print(" saving plot ["+filename+"]")
        plt.savefig(filename)
        plt.close()

    def plot_precision(self, p_dict, title):
        plt.plot(p_dict['correct'], color="blue", label="Correct")
        plt.plot(p_dict['false_pos'], label="False Positive")
        plt.plot(p_dict['false_neg'], label="False Negative")
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Percent')
        plt.xlim(0,len(p_dict['correct']))
        #plt.tight_layout()
        plt.legend()
        path = "../plots/"
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = path+self.name+'_'+title+'.png'
        print(" saving plot ["+filename+"]")
        plt.savefig(filename)
        plt.close()

    # def plot_test_obj(self, train_j, valid_j):
    #     plt.plot(train_j, label="Training")
    #     plt.plot(valid_j, label="Validation")
    #     plt.title('Objective')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('J')
    #     plt.xlim(0,len(train_j))
    #     plt.tight_layout()
    #     plt.legend()
    #     path = "../plots/"
    #     if not os.path.isdir(path):
    #         os.mkdir(path)
    #     filename = path+self.name+"_TandVobjective.png"
    #     print(" saving plot ["+filename+"]")
    #     plt.savefig(filename)
    #     plt.close()

    def plot_epoch_all(self, train_dict, valid_dict):
        plt.plot(train_dict['correct'], label="Train Correct")
        plt.plot(train_dict['false_pos'], label="Train False Positive")
        plt.plot(train_dict['false_neg'], label="Train False Negative")
        plt.plot(valid_dict['correct'], label="Valid Correct")
        plt.plot(valid_dict['false_pos'], label="Valid False Positive")
        plt.plot(valid_dict['false_neg'], label="Valid False Negative")
        plt.title('Precision vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Percent')
        plt.xlim(0,len(train_dict['correct']))
        plt.tight_layout()
        plt.legend()
        path = "../plots/"
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = path+self.name+"_all_precision.png"
        print(" saving plot ["+filename+"]")
        plt.savefig(filename)
        plt.close()

        plt.plot(train_dict['objective'], label="Training Objective")
        plt.plot(valid_dict['objective'], label="Validation Objective")
        plt.title('Objective vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Objective')
        plt.xlim(0,len(train_dict['correct']))
        plt.tight_layout()
        plt.legend()
        path = "../plots/"
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = path+self.name+"_all_objective.png"
        print(" saving plot ["+filename+"]")
        plt.savefig(filename)
        plt.close()


