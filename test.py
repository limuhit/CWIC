import caffe
import numpy as np
import cv2
import os
import math
def get_flist(name):
    f=open(name)
    flist=[pt[:-1] for pt in f.readlines()]
    f.close()
    return flist
def test(net,flist,save_dir,save=False):
    rt=0
    ls=0
    mse=0
    psnr=0
    shown=False
    iters=len(flist)
    res=[]
    for pt in flist:
        img=cv2.imread(pt)
        th,tw=img.shape[:2]
        h=th//8*8
        w=tw//8*8
        if h>1024:h=1024
        if w>1024:w=1024
        if h<th or w<tw:
            bh = 0#(th-h)/2
            bw = 0#(tw-w)/2
            img=img[bh:bh+h,bw:bw+w]
        img=img.transpose(2,0,1).astype(np.float32)
        net.blobs['data'].reshape(1,3,img.shape[1],img.shape[2])
        net.blobs['data'].data[...]=img
        net.reshape()
        net.forward()
        rt+=(net.params['imp'][0].data[0]+0)
        imp = np.sum(net.blobs['imp'].data[0],axis=0)/32*255
        gdata = net.blobs['gdata_scale'].data[0]
        gdata[gdata<0]=0
        gdata[gdata>255]=255
        if shown:
            cv2.imshow('imp',imp.astype(np.uint8))
            cv2.imshow('gdata',gdata.transpose(1,2,0).astype(np.uint8))
            cv2.waitKey(0)
        if save:
            name=pt.split('/')[-1]
            cv2.imwrite('%s/%s_imp.png'%(save_dir,name.split('.')[0]),imp.astype(np.uint8))
            cv2.imwrite('%s/%s'%(save_dir,name),img.transpose(1,2,0).astype(np.uint8))
            cv2.imwrite('%s/g_%s'%(save_dir,name),gdata.transpose(1,2,0).astype(np.uint8))
        print pt,net.params['imp'][0].data[0],  net.blobs['loss2'].data
        ls+=(net.blobs['mloss'].data+0)
        mse+=(net.blobs['loss'].data+0)
        psnr+=(net.blobs['loss2'].data+0)
        name=pt.split('/')[-1]
        res.append([name,net.blobs['loss2'].data+0,net.blobs['mloss'].data+0])
    iters=len(flist)
    print rt/iters,ls/iters,mse/iters,psnr/iters
    return res
def test_all(idx):
    global gpu_id
    global version
    global img_names
    global data_set
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    model = './%s/%d.caffemodel'%(version,idx)
    net = caffe.Net('./our_deploy.prototxt',model,caffe.TEST)
    flist=get_flist(img_names)
    save_dir='./results/%s'%data_set
    if not os.path.exists(save_dir):os.mkdir(save_dir)
    res=test(net,flist,save_dir,True)
    return res
def test_ratio(net,coder,imp,flist,save=False):
    rt_list=[]
    shown=False
    iters=len(flist)
    res=[]
    for pt in flist:
        img=cv2.imread(pt)
        th,tw=img.shape[:2]
        h=th//8*8
        w=tw//8*8
        if h>1024:h=1024
        if w>1024:w=1024
        if h<th or w<tw:
            bh = 0#(th-h)/2
            bw = 0#(tw-w)/2
            img=img[bh:bh+h,bw:bw+w]
        img=img.transpose(2,0,1).astype(np.float32)
        net.blobs['data'].reshape(1,3,img.shape[1],img.shape[2])
        net.blobs['data'].data[0]=img
        net.reshape()
        net.forward()
        code_data=net.blobs['tdata_imp'].data[0].astype(np.float32)
        imp_data = np.sum(net.blobs['imp'].data[0],axis=0).reshape(1,code_data.shape[1],code_data.shape[2])/2
        coder.blobs['data'].reshape(1,32,code_data.shape[1],code_data.shape[2])
        coder.blobs['data'].data[0]=code_data
        imp.blobs['data'].reshape(1,1,code_data.shape[1],code_data.shape[2])
        imp.blobs['data'].data[0]=imp_data.astype(np.float32)
        coder.forward()
        imp.forward()
        ptr = np.average(net.blobs['imp'].data[0])
        rt = (coder.blobs['loss'].data+0)/2/0.693*ptr+ (imp.blobs['loss'].data+0)/64/0.693
        rt_list.append(rt+0)
        name=pt.split('\\')[-1]
        res.append([name,rt+0])
        print pt,rt,coder.blobs['loss'].data,imp.blobs['loss'].data
    iters=len(flist)
    print np.average(rt_list)
    return res
def test_all_ratio(idx):
    global gpu_id
    global version
    global img_names
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    model = './%s/%d.caffemodel'%(version,idx)
    coder_model = './%s/code_%d.caffemodel'%(version,idx)
    imp_model = './%s/imp_%d.caffemodel'%(version,idx)
    net = caffe.Net('./our_extractor_deploy.prototxt',model,caffe.TEST)
    coder = caffe.Net('./lossless_deploy.prototxt',coder_model,caffe.TEST)
    imp = caffe.Net('./imp_deploy.prototxt',imp_model,caffe.TEST)
    f=open(img_names)
    flist=[pt[:-1] for pt in f.readlines()]
    res=test_ratio(net,coder,imp,flist,True)
    return res
if __name__ == '__main__':
    process=True
    #process=False
    version='v1'
    gpu_id = 1
    idx = 1
    kodak = True
    kodak = False
    if kodak: 
        img_names = './kodak.txt'
        data_set = 'kodak'
    else: 
        img_names = './tecnick.txt'
        data_set = 'tecnick'
    if process:
        res1=test_all(idx)
        res2=test_all_ratio(idx)
        f=open('./%s/%d_%s.txt'%(version,idx,data_set),'w')
        res=''
        for pa,pb in zip(res1,res2):
            res+='%s,%.3f,%.2f,%.3f\n'%(pb[0],pb[1],pa[1],pa[2])
        f.write(res)
        f.close()
    else:
        f=open('./%s/%d_%s.txt'%(version,idx,data_set),'r')
        get_data=lambda pt:  [float(pt[1]),float(pt[2]),float(pt[3])]
        res=[ get_data(pt[:-1].split(',')) for pt in f.readlines()]
        print np.average(res,axis=0)
    
    