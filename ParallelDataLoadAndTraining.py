filenames=['/data/imagenet/ilsvrc12/matdata/val-0.mat',
 '/data/imagenet/ilsvrc12/matdata/val-1.mat',
 '/data/imagenet/ilsvrc12/matdata/val-2.mat',
 '/data/imagenet/ilsvrc12/matdata/val-3.mat',
 '/data/imagenet/ilsvrc12/matdata/val-4.mat']
import gc
def release(data):
    del data
    gc.collect()
import threading
from math import *
import scipy.io as sp
import copy
preprocess_input=bias_subtract_preprocess_input
class readImageNetBatch(threading.Thread):
    #patch_num: read the data sets number 
    #capacity : setting the queue capacity, which load mat data from disk,and 
    def __init__(self,filenames,shuffle=True,patch_num=1,capacity=3,prepare_func=None,datagen=None,batchsize=256):
        threading.Thread.__init__(self)
#         self.train_df=train_df.sample(frac=1).reset_index(drop=True) 
        self.filenames=filenames
        self.index=np.arange(0,len(self.filenames))
        self.shuffle=shuffle
        self.prepare_func=prepare_func
        self.datagen=datagen
        self.batchsize=batchsize
        self.generator=None
        self.patch_num=patch_num
        self.capacity=capacity
        self.queue=[]
        self.lock = threading.Lock()
        if self.shuffle:
            np.random.shuffle(self.index)
        self.batch_id=0
        self.finished=False
        self.x=[]
        self.y=[]
        self.sample_num=0
        self.debug=False
#         self.walkup= threading.Event()
        self.kill=False#杀死自身进程的
    def run(self):
        while(not self.kill):
            if self.batch_id>=len(self.filenames)-1:
                self.batch_id=0
            else:
                self.batch_id+=1
#             print(self.batch_id,len(self.filenames))
            filename=self.filenames[self.index[self.batch_id]]
#             print(down,up,self.train_df[down:up].head(),self.train_df[down:up].tail())
            if self.debug:
                print ("\rstart load Image batch",end="")
            self.finished=False
            s=time.time()
    #         print_time(self.name, self.counter, 5)
            for i in range(self.patch_num):
                file=sp.loadmat(filename)
                if i==0:
                    self.x=file["x"]
                    self.y=file["y"].reshape(-1)
                else:
                    self.x=np.concatenate((self.x,file["x"]))
                    self.y=np.concatenate((self.y,file["y"].reshape(-1)))
            self.sample_num=len(self.y)
            if self.debug:
                print(self.x.shape,self.y.shape)
            if self.shuffle:
                if self.debug:
                    print ("\r shuffle data",end="")
                    sm=time.time()
                samples=np.arange(0,len(self.x))
                np.random.shuffle(samples)
                if self.debug:
                    print(samples.shape)
                self.x=self.x[samples]
                self.y=self.y[samples]
                if self.debug:
                    print ("\r end shuffle data time %.2fs"%(time.time()-sm),end="")
            #对x,y预处理
            self.y=to_categorical(self.y,1000)
            if self.debug:
                print(self.y.shape)
            if self.datagen is not None:
                tmp=copy.deepcopy(self.datagen)
                if self.debug:
                    print ("\rcreate data generator",end="")
                    sm=time.time()
                self.generator=tmp.flow(self.x,self.y,self.batchsize)
                if self.debug:
                    print ("\r end data generator %.2fs"%(time.time()-sm),end="")
                self.lock.acquire()#独写
                self.queue.append(self.generator)
                self.lock.release()
            else:
                if self.prepare_func is not None:
                    if self.debug:
                        print ("\r process data prepare_func",end="")
                        sm=time.time()
                    self.x=self.prepare_func(self.x)
                    if self.debug:
                        print ("\r end data prepare_func  %.2fs"%(time.time()-sm),end="")
                self.lock.acquire()#独写
                self.queue.append([self.x,self.y])
                self.lock.release()
            if self.debug:
                print ("\n\r load Image batch %d success, time cost %.2fs"%(self.batch_id,time.time()-s),end="")
            print(" load time %.2f"%(time.time()-s),end="")
            while len(self.queue)>=self.capacity:
            #只能通过event唤醒了
#             while(self.finished):
                time.sleep(0.5)
#                 print("self.finished=",self.finished)
            if self.debug:
                print("walk up!")
        print("Stop thread!")
    def get_data(self):
        if len(self.queue)>0:
            self.lock.acquire()#独写
            x,y=self.queue.pop(0)
            self.lock.release()#独写
            #唤醒线程
#             self.finished=False
            return x,y
#             release(self.x)
#             release(self.y)
        else:
            time.sleep(0.5)
            return self.get_data()
    def get_generator(self):
        if len(self.queue)>0:
            self.lock.acquire()#独写
            gen=self.queue.pop(0)
            self.lock.release()#独写
            #唤醒线程
#             self.finished=False
            return gen
        else:
            time.sleep(0.5)
            return self.get_generator()
    def kill_Thread(self):
        self.kill=True
def randon_flip(image):
    r=np.random.rand(1)
#     print(r)
    if r>0.5:
        image=cv2.flip(image, 1)
#     print(image.shape)
    return image
def random_crop_image(image):
    sizes=(256,256)
    image=cv2.resize(image,sizes)
    src_h,src_w=image.shape[:2]
#     print(image.shape)
    [t_h,t_w]=list(crop_size)
#     means=np.array([[[123,117,104]]])
    h = np.random.randint(src_h-t_h)
    w = np.random.randint(src_w-t_h)
    image_crop = image[h:h+t_h,w:w+t_w,:]
    return preprocess_input(image_crop)
def prepare_image(image):
    for i in range(len(image)):
        image[i]=random_crop_image(randon_flip(image[i]))
#     return image
def thread_prapare(image):
    p = Pool()
    max_threads=1000
    _step=floor(len(image)/max_threads)
    print(image[0].reshape(-1)[:10])
    results=[]
    for i in range(max_threads):
        up=_step*(i+1)
        down=up-_step
        results.append(p.apply_async(prepare_image,args=(image[down:up],)))
    for res in results:
        res.get()
        p.close()
        p.join()
    print(image[0].reshape(-1)[:10])
    return image
class PrepareImage (threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.image = image
    def run(self):
#         print("start prepare image")
        prepare_image(self.image)
import _thread
def train_pro(image):
    p=[]
    max_threads=100
    _step=floor(len(image)/max_threads)
    print(image[0].reshape(-1)[:10])
    for i in range(max_threads):
        up=_step*(i+1)
        down=up-_step
        p.append(PrepareImage(image[down:up]))
        p[-1].start()
#         image[i]=prepare_image(image[i])
    for i in range(max_threads):
        p[i].join()
#         image[i]=p[i].image
    print(image[0].reshape(-1)[:10])
    return image
def gc_thread_batch_generator(filenames,batch_size,train_datagen=None,prepare_func=None):
    #datagen:ImageDataGenerator object for different operator of train and test
#   datagen=ImageDataGenerator()
    thread_obj=readImageNetBatch(filenames,shuffle=False,patch_num=1,
                                 datagen=train_datagen,prepare_func=prepare_func,batchsize=train_batch_size)
    thread_obj.start()
    debug=False
    while True:
        if train_datagen is None:
            x_train,y_train=thread_obj.get_data()
        else:
            gen=thread_obj.get_generator()
        lens=thread_obj.sample_num
        if debug:
            print("have got data!")
        step_train=ceil(lens*1.0/batch_size)
#         print("step_train computing!")
        if train_datagen is None:
            for i in range(step_train):
    #             print("loop!")
                up=batch_size*(i+1)
                if up>len(x_train):
                    up=len(x_train)
                down=up-batch_size
                yield x_train[down:up],y_train[down:up]
        else:
            #将x1,y1分段送个程序运行 
            #需要提前终止，不然这货不会退出循环
            iters=0
            for x,y in gen:
                yield x,y
                iters+=1
                if iters>=step_train:
                    break#跳出当前循环，重新从大循环里获取数据
# for x,y in gc_thread_batch_generator(filenames,256):
#     print(x.shape)
train_datagen = ImageDataGenerator(horizontal_flip=True,preprocessing_function=random_crop_image)
# train_datagen=None
train_gen=gc_thread_batch_generator(filenames,128,train_datagen,prepare_func=None)
#*************************************AlexNet
import create_model
import keras
from keras.optimizers import SGD
from gc_quantize_methods import VOE
from keras.utils import multi_gpu_model
sess=tf.InteractiveSession()
K.set_session(sess)
qmw=VOE(data_max_bits=2,discard_scaler=False,average_motion=0,reback_shift=True)
# qmw=None
qma=None
qmg=None
alexnet=create_model.AlexNet(using_bn=True,weights_decay=0.0001, 
                             kernel_quantization=qmw,bias_quantization=qmw,activity_quantization=None)
from gc_learning_schedules import learnrateSchedules,multiStep
parallel_model = multi_gpu_model(alexnet, gpus=4)
parallel_model.compile(loss="categorical_crossentropy",
                 optimizer=SGD(lr=0.1),#,gradient_quantization=qmg),
                 metrics=["acc"])
histroy=parallel_model.fit_generator(train_datagen.flow(x,y,batchsize),
# histroy=parallel_model.fit_generator(train_gen,
#                                validation_data=(gc_val_generator()),
                                 epochs=70,
                                 steps_per_epoch=train_num/batchsize,
#                                validation_steps=val_num/val_batch_size,
                                 verbose=1,
                   callbacks=[learnrateSchedules(multiStep,extend_params=[[60,65],0.1])])
alexnet.save("Alex-net-VOE-w2.h5")
