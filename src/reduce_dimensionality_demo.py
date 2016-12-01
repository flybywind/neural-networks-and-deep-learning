# -*- encoding: utf8 -*-
import os
import sys
import timeit
import numpy as np 
import pylab
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStream
from utils import load_data
import six.moves.cPickle as pickle    
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Imag
# configure
seed = np.random.randint(2**30)
# seed = 1019117481  # nan occur
rng_np = np.random.RandomState(seed)
rng_theano = RandomStream(seed)

class LOG(object):
    def __init__(self, buffering = 1):
        self.buffering = buffering
        self.log_file = None
    def set_path(self, path): 
        if not (self.log_file is None):
            self.log_file.close()
        self.log_file = open(path, mode="w", buffering=self.buffering)

    def print(self, *all_args):
        assert(not self.log_file is None)    
        print(*all_args, file=self.log_file)

log = LOG()
# theano.config.exception_verbosity = "high"
# theano.config.optimizer = "fast_compile"
# theano.config.warn_float64 = "raise"

class NNEncoder(object):
    """ 基于神经网络的encoder，feature detector是自学习的
    """
    def __init__(self, n_in, n_out, 
        inpt = None,
        corruption = 0):
        """ 初始化NNEncoder
        :param n_in:   输入维度
        :param n_out:  输出维度
        :param lr:     学习因子
        :param corruption: 破坏因子
        """

        rand_bound = 4. * np.sqrt(6./(n_in+n_out))
        self.W = theano.shared(rng_np.uniform(
            low  = -rand_bound,
            high =  rand_bound,
            size =  (n_in, n_out)
        ).astype(theano.config.floatX), 
            name = "W", borrow = True)
        self.b = theano.shared(rng_np.uniform(
            low  = -rand_bound,
            high =  rand_bound,
            size =  (n_out, )
        ).astype(theano.config.floatX), 
            name = "b", borrow = True)
        self.c = theano.shared(rng_np.uniform(
            low  = -rand_bound,
            high =  rand_bound,
            size =  (n_in, )
        ).astype(theano.config.floatX), 
            name = "c", borrow = True)

        if inpt is None:
            self.input = T.matrix("visible")
        else:
            self.input = inpt

        # 以下4个变量和input共同构成 CD-1 链
        self.input.name = "v^0"
        self.encode_out = self._encode_out(self.input, corruption)  
        self.encode_out.name = "h^0"
        self.reconstruct_out = self.reconstruct_var(self.encode_out)
        self.reconstruct_out.name = "v^1"
        self.encode_output1 = self._encode_out(self.reconstruct_out)
        self.encode_output1.name = "h^1"

    def _encode_out(self, inpt, corruption = 0.):
        if corruption > 0. and corruption < 1:
            inpt = rng_theano.binomial(size = inpt.shape, 
                p = 1 - corruption,
                dtype=theano.config.floatX) * inpt
        return self.output_var(inpt)
        # 这段代码除了增加运算时间，没啥用处
        # probs_mean = T.flatten(expect)
        # binominal_sample, _ = theano.scan(
        #     lambda x: rng_theano.binomial(p=x, size=(1,)), sequences=probs_mean)
        # return T.reshape(binominal_sample, expect.shape)

    def output_var(self, inpt):
        return T.nnet.sigmoid(T.dot(inpt, self.W) + self.b)
   
    def reconstruct_var(self, inpt):
        return T.nnet.sigmoid(T.dot(inpt, self.W.T) + self.c)
    
    def output_func(self):
        return theano.function([self.input], self.output_var(self.input))

    def reconstruct_func(self):
        return theano.function([self.input], 
                self.reconstruct_var(
                    self.output_var(self.input)
                ))

    def reconstruct_error(self):
        return -T.sum(T.sum(
            self.input * T.log(self.reconstruct_out)
            + (1 - self.input) * T.log(self.reconstruct_out), 0))

    def update(self, lr, mini_batch):
        update = [(self.W, self.W + lr * 
            (T.dot(self.input.T, self.encode_out) - 
             T.dot(self.reconstruct_out.T, self.encode_output1))/mini_batch
        )]
        update.append((self.b, self.b + lr * 
            (T.mean(self.encode_out - self.encode_output1, 0)) 
        ))
        update.append((self.c, self.c + lr * 
            (T.mean(self.input - self.reconstruct_out, 0))
        ))
        return update

    def enegy(self):
        diag = T.diagonal(
            T.dot(T.dot(self.input, self.W), self.encode_out.T))

        return T.mean(- diag
                - T.dot(self.encode_out, self.b)
                - T.dot(self.input, self.c))
    
    def train(self, tr, lr = 0.01, mini_batch = 100, epoch = 1000, debug=False, out_freq = 10):
        indx = T.lscalar("index")
        tr_sample_total = tr.get_value().shape[0]
        tr_batch_num = tr_sample_total // mini_batch 

        if debug == True:
            def debug_monitor(i, node, fn):
                for (indx, var) in enumerate(node.outputs):
                    name = var.name
                    if name == "v^0" or name == "v^1" or name == "h^0" or name == "h^1":
                        log.print("debug_monitor, current node:", type(var), name)
                        log.print(fn.outputs[indx][0].shape, fn.outputs[indx][0])
                        log.print("======================\n")
            mode = theano.compile.MonitorMode(
                post_func=debug_monitor).excluding('local_elemwise_fusion', 'inplace')
            train_func = theano.function(
                inputs = [indx],
                outputs = [self.enegy(), self.reconstruct_error()],
                updates = self.update(lr, mini_batch),
                givens = {
                    self.input: tr[indx*mini_batch : (indx+1)*mini_batch]
                },
                mode = mode
            )
        else:
            train_func = theano.function(
                inputs = [indx],
                outputs = [self.enegy(), self.reconstruct_error()],
                updates = self.update(lr, mini_batch),
                givens = {
                    self.input: tr[indx*mini_batch : (indx+1)*mini_batch]
                }
            )
        enegy_out = []
        recons_err_out = []
        out_freq = tr_batch_num//out_freq
        if out_freq == 0:
            out_freq = tr_batch_num - 1
        for e in range(epoch):
            for mb in range(tr_batch_num):
                tr_value = tr.get_value()[mb*mini_batch : (mb+1)*mini_batch]
                if debug == True:
                    log.print("mini-batch input:", tr_value)
                enegy, rec_err = train_func(mb)
                if mb % out_freq == 0:
                    log.print("Epoch [%d] Iter [%s], enegy: %f, recons_err_out: %s" % (e, mb, enegy, rec_err))
                    enegy_out.append(enegy)
                    recons_err_out.append(rec_err)
            log.print("End epoch", e, "\n========================\n")

        return enegy_out, recons_err_out

class StackEncoder(object):
    """ 级联Encoder 
    """
    def __init__(self, inpt, n_in, layers_node, layers_corruption):
        """ 初始化StackEncoder
            :param inpt: 符合变量，输入
            :param n_in: 输入向量维度
            :param laysers_node: 每一层encoder的节点数，将根据这个变量生成NNEncoder对象列表
            :param laysers_corruption: 每一层的corruption值
        """    
        assert(len(layers_node) > 0)
        assert(len(layers_node) == len(layers_corruption))
        self.last_dw = []
        self.last_db = []
        self.W = []
        self.b = []
        self.input = inpt

        last_input_num = n_in
        last_input_var = inpt
        self.encoder_layers = []
        cur_layer = None
        for (nn, corup) in zip(layers_node, layers_corruption):
            self.last_dw.append(
                theano.shared(np.zeros((last_input_num, nn), 
                    dtype=theano.config.floatX)))
            self.last_db.append(
                theano.shared(np.zeros((nn, ), 
                    dtype=theano.config.floatX)))
            cur_layer = NNEncoder(
                last_input_num, nn, 
                last_input_var, corup)
            self.encoder_layers.append(cur_layer)
            self.W.append(cur_layer.W)
            self.b.append(cur_layer.b)
            last_input_var = cur_layer.output_var(last_input_var)
            last_input_num = nn

        self.y_encode = last_input_var
        last_encoder = self.encoder_layers[-1]
        self.x_recon = last_encoder.reconstruct_var(self.y_encode)
        self.x_recon.name = "x_recon"
        for el in self.encoder_layers[-2::-1]:
            self.x_recon = el.reconstruct_var(self.x_recon)

        
    def encoder(self):
        return theano.function([self.input], -T.log(1./self.y_encode - 1))

    def reconstruct(self):
        return theano.function([self.input], self.x_recon)
            
    def cross_entropy_error(self):
        return -T.sum(
                T.mean(self.input * T.log(self.x_recon)+
                (1. - self.input) * T.log(1 - self.x_recon), 0))

    def pre_train(self, train_data, mini_batch, epoch, learn_rate):
        """ pre_train 获得W的初始值
            :param learn_rate: array
        """
        assert(len(self.encoder_layers) == len(learn_rate))
        param_name = "mb%d_ep%d_lr%s.log" % (
                mini_batch, epoch,
                "_".join([str(x) for x in learn_rate]))
        log.set_path("logs/pretrain_%s.log" % param_name)
        tr = train_data
        s = timeit.default_timer()
        for lr, el in zip(learn_rate, self.encoder_layers):
            el.train(tr, 
                lr = lr, mini_batch = mini_batch, 
                epoch = epoch,
                out_freq = 5)
            encode_func = el.output_func()
            tr = theano.shared(encode_func(tr.get_value()),
                borrow = True)
        pretrain_model = "models/pretrain_%s.pkl" % param_name
        with open(pretrain_model, "wb") as f:
            pickle.dump(self, f)

        e = timeit.default_timer()
        log.print("pre_train time: %.2fs" % (e - s))
    def train(self, dataset, mini_batch, epoch, learn_rate, momentum_ratio):
        tr = dataset[0][0]
        va = dataset[1][0]
        te = dataset[2][0]
        train_batch_num = tr.get_value().shape[0] // mini_batch
        index = T.lscalar("batch-index")
        updates = []
        error_func = self.cross_entropy_error()
        for last_dw, last_db, w, b in zip(self.last_dw, self.last_db, self.W, self.b):
            dw = T.grad(error_func, w)
            db = T.grad(error_func, b)
            if momentum_ratio > 0:
                cur_dw = - learn_rate * dw + momentum_ratio * last_dw
                cur_db = - learn_rate * db + momentum_ratio * last_db
                updates.extend([
                    (w, w + cur_dw),
                    (b, b + cur_db),
                    (last_dw, cur_dw),
                    (last_db, cur_db)])
            else:
                updates.extend([
                    (w, w - learn_rate * dw),
                    (b, b - learn_rate * db)])
        
        params_name = "mb%d_ep%d_lr%s_me%s" % (mini_batch, epoch, learn_rate, momentum_ratio)
        log.set_path("logs/stack_encoder_%s.log" % params_name)
        def debug_monitor(i, node, fn):
            for (indx, output) in enumerate(fn.outputs):
                if (not isinstance(output[0], np.random.RandomState) and 
                    np.isnan(output[0]).any()):
                    log.print('*** NaN detected ***')
                    theano.printing.debugprint(node)
                    log.print('Inputs : %s' % [input[0] for input in fn.inputs])
                    log.print('Outputs: %s' % [output[0] for output in fn.outputs])
                    log.print("======================\n")
                    raise(Exception("MeetNaN"))

        mode = theano.compile.MonitorMode(
            post_func=debug_monitor).excluding('local_elemwise_fusion', 'inplace')
        
        train_model = theano.function(
            inputs = [index],
            outputs = error_func,
            updates = updates,
            givens = {
                self.input: tr[index*mini_batch : (index+1)*mini_batch]
            },
            mode=mode
        )
        valid_model = theano.function(
            inputs = [],
            outputs = error_func,
            givens = {
                self.input: va
            },
            mode=mode
        )
        

        left_epoch = epoch
        check_frequence = train_batch_num//3
        minimum_error = np.inf
        improve_ratio = 0.99
        if check_frequence == 0:
            check_frequence = train_batch_num - 1
        tr_error_ary = []
        va_error_ary = []
        e = 0
        s = timeit.default_timer()

        while left_epoch > 0:
            left_epoch -= 1
            e += 1
            for mb in range(train_batch_num):
                tr_error = train_model(mb)
                if mb % check_frequence == 0:
                    va_error = valid_model()
                    tr_error_ary.append(tr_error)
                    va_error_ary.append(va_error)

                    log.print("epoch %d, iter %d, left-epoch %d: train_error[%.2f], validation_error[%.2f]" % 
                            (e, mb, left_epoch, tr_error, va_error))
                    if va_error < minimum_error * improve_ratio:
                        minimum_error = va_error
                        min_ep = e 
                        if left_epoch < epoch / 2:
                            left_epoch += epoch / 2
                        log.print("    Find a better model! Left epoch:", left_epoch)
                        with open("models/stack_encoder_%s.pkl" % params_name, 'wb') as f:
                            pickle.dump(self, f)

        e = timeit.default_timer()
        log.print("fine tuneing train time: %.2fs" % (e - s))       
        return tr_error_ary, va_error_ary

def encoder_demo(tr):
    # 在这套参数时，可以明显看到corruption的作用
    # 当lr = 1时，corruption能明显加速训练过程，从enegy的值上看，大概加速10倍
    # 如果没有corruption，enegy还没有lr = 0.1时小
    # 所以lr必须足够大才能发现问题，之前lr都有点小。
    # 但是，lr变大的结果就是，重构误差也变得非常大。有明显的corruption痕迹
    # 所以，lr还是维持在0.1吧，增加epoch试试
    # mini_batch = 30
    # epoch = 2
    # learning_rate = 1
    # corruption = 0.
    # train_out_freq = 100
    mini_batch = 100
    epoch = 10
    learning_rate = 0.1
    corruption = 0.2
    train_out_freq = 50
    n_out = 100
    debug_v = False
    v = T.matrix("visible")
    encoder = NNEncoder(n_in = 784, 
        n_out = n_out, 
        inpt = v,
        corruption = corruption)
    
    params_name = "n%d-lr%s-ep%d-mb%d-crp%s" % (n_out, 
        learning_rate,
        epoch,
        mini_batch,
        corruption)
    
    print(params_name)
    log.set_path("logs/reduce_dim_%s.log" % params_name)
    
    start_time = timeit.default_timer()
    enegy_out, recons_err_out = encoder.train(tr, 
        lr = learning_rate, mini_batch = mini_batch, epoch = epoch,
        debug = debug_v, out_freq = train_out_freq)
    end_time = timeit.default_timer()
    log.print("eclapse time: %f s" % (end_time - start_time))

    
    with open("models/encoder_%s.pkl"%params_name, 'wb') as f:
        pickle.dump(encoder, f)
 
    va = dataset[1][0].get_value()[0:100, :]
    image = Image.fromarray(tile_raster_images(
        X=va,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('img/origin.png')
    recon_func = encoder.reconstruct_func()
    recon_out = recon_func(va)
    image = Image.fromarray(tile_raster_images(
        X=recon_out,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save("img/recon_%s.png" % params_name)

    image = Image.fromarray(tile_raster_images(
        X=encoder.W.get_value().T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save("img/encoder-W_%s.png"%params_name)

    plt.subplot(211)
    plt.plot(enegy_out, ".-")
    plt.title("enegy")
 
    plt.subplot(212)
    plt.plot(recons_err_out, ".-")
    plt.title("reconstruction error")
    plt.show()

def compare_model(test_data, model_path):
    X = test_data[0].get_value()
    Y = test_data[1].eval()
    model = pickle.load(open(model_path, 'rb'))
    encode_func = model.encoder()
    encode_out = encode_func(X)
    recon_func = model.reconstruct()
    recon_out = recon_func(X[0:100])
    
    pca0 = RandomizedPCA(n_components=1000)
    pca1 = RandomizedPCA(n_components=500)
    pca2 = RandomizedPCA(n_components=250)
    pca3 = RandomizedPCA(n_components=2)

    pca_m0 = pca0.fit(X)
    pca_out0 = pca_m0.transform(X)
    pca_m1 = pca1.fit(pca_out0)
    pca_out1 = pca_m1.transform(pca_out0)
    pca_m2 = pca2.fit(pca_out1)
    pca_out2 = pca_m2.transform(pca_out1)
    pca_m3 = pca3.fit(pca_out2)
    pca_final = pca_m3.transform(pca_out2)
    pca_recon = pca_m3.inverse_transform(pca_final[0:100])
    pca_recon = pca_m2.inverse_transform(pca_recon)
    pca_recon = pca_m1.inverse_transform(pca_recon)
    pca_recon = pca_m0.inverse_transform(pca_recon)

    image = Image.fromarray(tile_raster_images(
        X=X[0:100],
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('img/origin.png')
    image = Image.fromarray(tile_raster_images(
        X=recon_out,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save("img/recon_stk.png")
    image = Image.fromarray(tile_raster_images(
        X=pca_recon,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save("img/recon_pca.png")

    def scatter_with_legned(X, Y):
        color_map = ["b.", "g.", "r.", "c.", "m.", "y.", "c+", "r+", "b+", "g+"]
        for num in range(10):
            indx = (Y==num)
            x_this_num = X[indx, :]
            plt.plot(x_this_num[:, 0], x_this_num[:,1], color_map[num], label=("%s"%num))
        plt.legend(fontsize="x-small", 
            numpoints=1, loc=2,    # 3是左上角的意思
            bbox_to_anchor=(0, 0), # legend的loc位置在图的左下方 
            ncol=10, # 10列
            columnspacing=1, borderaxespad=0.) # borderaxespad指定legend和轴之间的距离

    plt.subplot(211)
    # plt.scatter(pca_final[:, 0], pca_final[:, 1], c=Y, cmap=pylab.cm.hsv)
    scatter_with_legned(pca_final, Y)
    plt.title("RandomizedPCA")

    plt.subplot(212)
    # plt.scatter(encode_out[:, 0], encode_out[:, 1], c=Y, cmap=pylab.cm.hsv)
    scatter_with_legned(encode_out, Y)
    plt.title("Neural Auto encoder")
    plt.show()

def stack_encoder_demo(dataset, pretrain_model = None):
    tr = dataset[0][0]
    inpt = T.matrix("input")
    # 在3层stack encoder情况下，pre-train基本没有影响。
    # 唯一的影响就是开始的训练误差确实小了很多，但是随着训练的继续
    # 依然可以达到一个比较好的误差范围
    s = timeit.default_timer()
    stk_encoder = StackEncoder(inpt, 
                    784,  [1000, 500, 250, 2], layers_corruption = [0.3, 0.3, 0.1, 0.1])
                    # layers_corruption = [0.2, 0.3, 0.3, 0.4])
    # if pretrain_model is None:
    #     print("begin pre-train")
    #     stk_encoder.pre_train(tr, 
    #                     mini_batch = 100, 
    #                     epoch = 10, 
    #                     learn_rate = [0.1, 0.1, 0.02, 0.01])
    # else:
    #     print("load pretrain paramters")
    #     with open(pretrain_model, 'rb') as f:
    #         stk_encoder = pickle.load(f)

    print("begin fine-tune train")    
    tr_err, va_err = stk_encoder.train(dataset, 
            mini_batch = 20, 
            epoch = 100,
            learn_rate = 0.001,
            momentum_ratio = 0.8)
    e = timeit.default_timer()
    print("eclapse time: %fs" % (e - s))
    plt.subplot(211)
    plt.plot(tr_err, "r.-")
    plt.title("train cross-entropy error")
    plt.subplot(212)
    plt.plot(tr_err, "b.-")
    plt.title("valid cross-entropy error")
    plt.savefig()
    plt.show()

if __name__ == "__main__":
    print("seed =", seed)
    dataset = load_data("mnist.pkl.gz")
    tr = dataset[0][0]
    va = dataset[1]
    if len(sys.argv) < 2:
        encoder_demo(tr)
    else:
        if sys.argv[1] == "en":
            encoder_demo(tr)
        elif sys.argv[1] == "cmp":
            if len(sys.argv) < 3:
                print("请指定模型路径！", file=sys.stderr)
                os.abort() 
            compare_model(va, sys.argv[2])
        elif sys.argv[1] == "stk":
            if len(sys.argv) > 2:
                stack_encoder_demo(dataset, sys.argv[2])
            else:
                stack_encoder_demo(dataset)
        else:
            print("参数错误")