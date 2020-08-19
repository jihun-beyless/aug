#import sys
#import os
import cv2
import numpy as np
from aug_DB import aug_db
from augment import augment

import time


def aug_start(device_id, grid, grid_id, object_category, background_id, iteration, batch_num):
    """
    합성과정 전체 돌아가는 메인 함수

    args: 
        device_id (int): 촬영된 기기 id
        grid (tuple) : 가로 세로 그리드 비율로 튜플값 (w)(h) 
        grid_id (int): 그리드 id
        object_category (list) : 물품의 category 값 ex) [12, 34, 23]
        background_id (int) : 배경 이미지에 해당되는 id값
        iteration (int) : 여기서 iteration은 촬영 횟수를 말하고, 물품 전부 동일하면 int 값만 받아도 무방
                                    아니면 위의 object_category처럼 list로 받음
        batch_num (tuple): 이미지 생성할 갯수로 3가지 합성 방법에 따라 합성 갯수를 정해서 받음 ex) (4000, 3000, 3000)

    return:
        bool : True or False
    """

    if str(type(iteration))=="<class 'int'>" :
        iteration_list = [iteration for a in range(len(object_category))]
    else:
        iteration_list = iteration
        
    # DB접속
    #db = DB.DB(ip='192.168.10.69', port=3306, user='root', password='return123', db_name='test')
    DB_data = aug_db.aug_db(ip='192.168.10.69', port=3306, user='root', password='return123', db_name='test', preprocessing_flag=True)


    # 먼저 DB에서 file을 읽어오기
    print('read DB data for augmentation')
    DB_mask, background, flag = DB_data.get_DB_data(object_category, grid, grid_id, iteration_list, background_id)
    #DB_mask = get_data(object_category, grid, grid_id, iteration_list, image_size)
    if not flag:
        print('DB에서 합성에 필요한 데이터 읽기 실패')
        return False

    #그외 합성 조건
    result_data = []
    #save_count = 1
    aug_count = 1
    cut_value = 1000
    img_path_list = []
    batch_method_list = [1 for i in range(batch_num[0])]
    batch_method_list.extend([2 for i in range(batch_num[1])])
    batch_method_list.extend([3 for i in range(batch_num[2])])
    #cv2.imshow('bg',background)
    #cv2.waitKey(0)
    
    print('start augmentation')
    for batch_method in batch_method_list:
        if len(result_data)==cut_value:
            print('save dataset')
            aug_save_flag = DB_data.set_aug_result(result_data, grid, grid_id, device_id, img_path_list)
            if not aug_save_flag:
                print('합성이미지 {}~{}번까지 데이터 저장 실패'.format(aug_count-cut_value, aug_count))
                return False
            result_data = []
            img_path_list = []
            #save_count+=1
        #실제 이미지 합성이 이루어 지는 함수
        img_path, result = augment.aug_process(grid, object_category, batch_method, background, DB_mask, iteration_list, aug_count)
        print('{}번 이미지 합성'.format(aug_count))
        result_data.append(result)
        img_path_list.append(img_path)
        aug_count +=1
    print('save dataset')
    aug_save_flag = DB_data.set_aug_result(result_data, grid, grid_id, device_id, img_path_list)
    if not aug_save_flag:
        print('합성이미지 {}~{}번까지 데이터 저장 실패'.format(aug_count-cut_value, aug_count-1))
        return False

    print('finish all augmentations')
    return True


if __name__ == "__main__":
    # 이건 tool에서 입력으로 받아와야 하는 변수들
    # 20001, 2, 3, 1, [1, 2], 3, 29

    device_id = 20001

    grid = (7, 6)

    grid_id = 3

    obj_cate = [1, 2, 3]

    bg_id = 463

    iteration = 3
    batch_num = (300, 300, 400)
    aug_start(device_id = device_id, 
            grid = grid, 
            grid_id = grid_id, 
            object_category = obj_cate, 
            background_id = bg_id, 
            iteration = iteration,
            batch_num = batch_num)