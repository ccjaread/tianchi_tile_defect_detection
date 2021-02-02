import cv2 
import numpy as np
import tqdm
import os
import glob

files=glob.glob('.tcdata/tile_round1_train_20201231/train_imgs/*.jpg')

import json
with open('.tcdata/tile_round1_train_20201231/train_annos.json', 'r') as fp:
     results=json.load(fp)
        
        
def get_sliced_yolo_box(dict_data):# for labeled train
    #parameters
    pic_save_path='.tcdata/train_imgs_sliced_all_320_val8/images/'
    anno_save_path='.tcdata/train_imgs_sliced_all_320_val8/labels/'
    
    if not os.path.exists(pic_save_path): 
        os.makedirs(pic_save_path)
        
    if not os.path.exists(anno_save_path): 
        os.makedirs(anno_save_path)
    
    target_size=260 #100,180,260
#     expand_ratio=320/target_size #2.24
    if dict_data['category'] in [3,4,5]:
        wh_resize_ratio=2
    else:
        wh_resize_ratio=1.05
    
    im = cv2.imread('.tcdata/tile_round1_train_20201231/train_imgs/'+dict_data['name'])
    fname=dict_data['name'].split('.j')[0]

    #hight_y(hy),width_x(wx)
    hy,wx,_=im.shape
    gap_y=hy%target_size//2
    gap_x=wx%target_size//2

    tag_xy=dict_data['bbox']

    new_x0=((round(tag_xy[0])-gap_x)//target_size)*target_size+gap_x
    new_y0=((round(tag_xy[1])-gap_y)//target_size)*target_size+gap_y

    #new img size
#     w=round(target_size*expand_ratio)
#     h=round(target_size*expand_ratio)
    w,h=[320,320]
    
    #new boundary
    new_x1=new_x0+w
    new_y1=new_y0+h


    out_x=(np.mean([tag_xy[0],min(tag_xy[2],new_x1)])-new_x0)/w
    out_y=(np.mean([tag_xy[1],min(tag_xy[3],new_y1)])-new_y0)/h
    out_w=(min(tag_xy[2],new_x1)-tag_xy[0])/w
    out_w=min(1,out_w*wh_resize_ratio)
    out_h=(min(tag_xy[3],new_y1)-tag_xy[1])/h
    out_h=min(1,out_h*wh_resize_ratio)
    
    out_string="{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(dict_data['category'],out_x,out_y,out_w,out_h)
    
    
    new_img=im[new_y0:new_y1,new_x0:new_x1,:]
    
#     if hy>=new_y1 and wx>=new_x1:
#         new_img=im[new_y0:new_y1,new_x0:new_x1,:]
#     else:
#         new_img=im[new_y0:hy,new_x0:wx,:]
#         new_img=np.pad(new_img, ((0,new_y1-hy),(0,new_x1-wx),(0,0)), 'constant', constant_values=0)

    pic_fname=fname+'_'+str(wx)+'_'+str(hy)+'_'+str(new_x0)+'_'+str(new_y0)+'.jpg'
    if os.path.exists(pic_fname):
        pass
    else:
        cv2.imwrite(pic_save_path+pic_fname, new_img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

    anno_fname=fname+'_'+str(wx)+'_'+str(hy)+'_'+str(new_x0)+'_'+str(new_y0)+'.txt'
    with open(anno_save_path+anno_fname,'a+') as f:
        f.write(out_string)

        
        
def get_sliced_yolo_box_nogap(dict_data):# for 1280 test
    #parameters
    pic_save_path='.tcdata/train_imgs_sliced_all_1280_val/images/'
    anno_save_path='.tcdata/train_imgs_sliced_all_1280_val/labels/'
    
    if not os.path.exists(pic_save_path): 
        os.makedirs(pic_save_path)
        
    if not os.path.exists(anno_save_path): 
        os.makedirs(anno_save_path)
    
    target_size=1280 #100,180,260
#     expand_ratio=320/target_size #2.24
    if dict_data['category'] in [3,4,5]:
        wh_resize_ratio=1.2
    else:
        wh_resize_ratio=1.05
    
    im = cv2.imread('.tcdata/tile_round1_train_20201231/train_imgs/'+dict_data['name'])
    fname=dict_data['name'].split('.j')[0]

    #hight_y(hy),width_x(wx)
    hy,wx,_=im.shape

    tag_xy=dict_data['bbox']

    new_x0=(round(tag_xy[0])//target_size)*target_size
    new_y0=(round(tag_xy[1])//target_size)*target_size

    #new img size
#     w=round(target_size*expand_ratio)
#     h=round(target_size*expand_ratio)
    w,h=[1280,1280]
    
    #new boundary
    new_x1=new_x0+w
    new_y1=new_y0+h


    out_x=(np.mean([tag_xy[0],min(tag_xy[2],new_x1)])-new_x0)/w
    out_y=(np.mean([tag_xy[1],min(tag_xy[3],new_y1)])-new_y0)/h
    out_w=(min(tag_xy[2],new_x1)-tag_xy[0])/w
    out_w=min(1,out_w*wh_resize_ratio)
    out_h=(min(tag_xy[3],new_y1)-tag_xy[1])/h
    out_h=min(1,out_h*wh_resize_ratio)
    
    out_string="{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(dict_data['category'],out_x,out_y,out_w,out_h)
    
    
    new_img=im[new_y0:new_y1,new_x0:new_x1,:]
    if hy>=new_y1 and wx>=new_x1:
        new_img=im[new_y0:new_y1,new_x0:new_x1,:]
    elif hy>=new_y1 and wx<new_x1:
        new_img=im[new_y0:new_y1,new_x0:wx,:]
        new_img=np.pad(new_img, ((0,0),(0,new_x1-wx),(0,0)), 'constant', constant_values=0)
    elif hy<new_y1 and wx>=new_x1:
        new_img=im[new_y0:hy,new_x0:new_x1,:]
        new_img=np.pad(new_img, ((0,new_y1-hy),(0,0),(0,0)), 'constant', constant_values=0)
    else:
        new_img=im[new_y0:hy,new_x0:wx,:]
        new_img=np.pad(new_img, ((0,new_y1-hy),(0,new_x1-wx),(0,0)), 'constant', constant_values=0)
#     if hy>=new_y1 and wx>=new_x1:
#         new_img=im[new_y0:new_y1,new_x0:new_x1,:]
#     else:
#         new_img=im[new_y0:hy,new_x0:wx,:]
#         new_img=np.pad(new_img, ((0,new_y1-hy),(0,new_x1-wx),(0,0)), 'constant', constant_values=0)

    pic_fname=fname+'_'+str(wx)+'_'+str(hy)+'_'+str(new_x0)+'_'+str(new_y0)+'.jpg'
    if os.path.exists(pic_fname):
        pass
    else:
        cv2.imwrite(pic_save_path+pic_fname, new_img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

    anno_fname=fname+'_'+str(wx)+'_'+str(hy)+'_'+str(new_x0)+'_'+str(new_y0)+'.txt'
    with open(anno_save_path+anno_fname,'a+') as f:
        f.write(out_string)
#### try multiprocessing
import multiprocessing as mp

if __name__=='__main__':
    print('start')
    pool = mp.Pool()
    # 按3进程运行一方面加数，一方面减少python内存泄漏的发生
    # 这边的目录在实际运行时需要修改
    print(len(results))
    pool.map(get_sliced_yolo_box_nogap,results)
    pool.close()
    pool.join()
    print('all done')