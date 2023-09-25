# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/05/18 19:08:19
@Author  :   houqi 
@Version :   1.0
@Contact :   houqi@100tal.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import os

def get_all_images(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(('.jpg','.png','.jpeg')):
                filelist.append(os.path.join(root, name))
    print('There are %d images' % (len(filelist)))
    return filelist

def get_acc(txt_file, thr_min=0 ,thr_max=1000, interval=1):

    lines = open(txt_file,'r').readlines()

    for thr in range(thr_min, thr_max, interval):
        false_num = 0
        for line in lines:
            line = line.strip('\n')
            sims= line.split(' ')[1].split(',')

            for sim in sims:
                if sim == 'None':
                    sim = 0.0
                elif sim != '0.0':
                    sim = sim[1:-1]
                if float(sim)>= thr / 1000.0:
                    false_num +=1
                    # print('false_num:{} line:{}'.format(false_num,line))
                    break

        # print(false_num,len(lines))
        acc = 1-(false_num)/float(len(lines))

        if thr%1==0:
            print ("~~~~~~~~~~  thr:{0:.3f} ~~~~~~~~~~~~~ ".format(thr/1000.0))
            print('acc: {} '.format(acc))
                

            

if __name__ == "__main__":
    txt_file = '/home/work/tf_political_det_v1/src/test.txt'
    # txt_file = '/workspace/houqi/face_project/political_det/results/attack_txt/result.txt'
    get_acc(txt_file,thr_min=400)





