import os
import sys
import cv2
import numpy as np

def img_loader(img_dir):
    if isinstance(img_dir, str):
        with open(img_dir, 'rb') as file:
            img = file.read()
    return img


def load_aug_images(images_path):
    images = []
    for path in images_path:
        img = cv2.imread(path)
        images.append(img)
    return images

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print("Failed to create director!!"+directory)


def arrange_masks(DB_masks, grid, obj_iter):
    """
    mask 정보를 합성코드내에서 사용하기 쉽도록 재배열

    Args:
        DB_masks (tuple) : mask 데이터들
        grid (tuple): grid 가로,세로 정보 
        obj_iter (list): 특정 category id에 해당되는 iteration 정보

    return:
        list : 다차원배열로 정리 
    """
    #grid : 10
    #category :15, 16, 17
    #np_masks = np.array(masks)
    masks_list = list([[[None for iter in range(obj_iter)] for row in range(grid[1])] for col in range(grid[0])])
    for x in range(1,grid[0]+1):
         for y in range(1,grid[1]+1):
             for iter_num in range(1,obj_iter+1):
                mask_value = [list(obj_mask[3:6]) for obj_mask in DB_masks if (obj_mask[0:3]==(x,y,iter_num))]
                sort_mask = sorted(mask_value, key=lambda mask_value: mask_value[0])
                masks_list[x-1][y-1][iter_num-1] = [obj_mask[1:3] for obj_mask in sort_mask]
    return masks_list


def update_value(src, th, rect, max_value):
    src16 = src.astype(np.int16)
    mean_value = np.sum(src16[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])])/(rect[2]*rect[3])
    diff_value = th-mean_value
    add_img = src16+diff_value
    update_img1 = np.where(add_img<0, 0 , add_img)
    update_img2 = np.where(update_img1>max_value, max_value , update_img1)
    return update_img2.astype(np.uint8)


def edit_img_value(img, bright_param):
    """
    white blance 맞추기 위한 함수

    Args:
        img (numpy): image 
        bright_param : 특정 위치값 및 BGR 값 등으로 white blance 맞출때 사용됨

    return:
        numpy : 밝기 조절된 image 
    """
    # bright_param : [bright_flag, mode_flag, flag1, flag2, flag3, th1, th2, th3, rect x, rect y, rect w, rect h] 
    ch_flag = bright_param[2:5]
    th_param = bright_param[5:8]
    rect = bright_param[8:12]
    if bright_param[1]==1:
        src_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        max_param = [180, 255, 255]
    elif bright_param[1]==2:
        src_img = img
        max_param = [255, 255, 255]

    img_split = cv2.split(src_img)
    
    result=[]
    for img_ch, flag_value, th, max_value in zip(img_split, ch_flag, th_param, max_param):
        if flag_value==1:
            re = update_value(img_ch, th, rect, max_value)
            result.append(re)
        else:
            result.append(img_ch)

    re_img = cv2.merge(result)
    if bright_param[1]==1:
        re_img = cv2.cvtColor(re_img, cv2.COLOR_HSV2BGR)
    
    return re_img

def tmp_image_save(DB_imgs, obj_id, g, g_id, obj_iter, pre_flag):    
    """
    DB에서 받은 이미지를 tmp에 저장
    입력받은 category_id별로 이미지를 생성

    Args:
        DB_imgs (tuple): DB에서 가져온 image들
        obj_id (int): DB에서 가져온 image의 obj_id
        g (tuple): grid 가로,세로 정보 
        g_id (int) : grid_id정보
        obj_iter (list): 특정 category id에 해당되는 iteration 정보
        pre_flag (bool): white blance 적용 여부 확인

    return:
        bool : True or False
    """
    print('임시로 tmp에 이미지 저장')
    
    # white balance 맞추는 용도로 쓰는 값, 나중에 카메라 바뀌면 삭제
    bright_param = (pre_flag, 1, 1, 1, 1, 78, 36, 113, 1140, 440, 100, 200)
    
    #폴더 생성
    folder_name = str('/tmp/augment_DB/{}/').format(obj_id)
    createFolder(folder_name)
    file_info = [[x,y,iter_num, str('{}x{}_{}.png').format(x, y, iter_num), False] for x in range(1,g[0]+1) for y in range(1,g[1]+1) for iter_num in range(1,obj_iter+1)]
    for f in file_info:
        for img in DB_imgs:
            if img[0:3]==tuple(f[0:3]):
                img_bytes = img[3]
                img_np = np.frombuffer(img_bytes, dtype = np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if bright_param[0]==1:
                    img = edit_img_value(img, bright_param)
                
                #img = img_np.reshape((1080,1920,3))
                #cv2.imshow('DB_images', img)
                #cv2.waitKey(0)
                file_path = folder_name+f[3]
                cv2.imwrite(file_path, img)
                f[4] = True

    #마지막으로 안 읽힌 데이터가 있는지 확인
    for f in file_info:
        if not f[4]:
            print('tmp/augment_DB폴더에 obj_id가 {}인 물체의 {}이미지가 저장이 되지 않았음'.format(obj_id,f[3]))
            print('DB에서 읽어오는 부분 또는 이미지 저장하는 부분의 코드 확인 필요')
            return False
    return True


