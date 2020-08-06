import sys
import os
import augment
import json
import cv2
import numpy as np
#from DCD_DB_API import*

sys.path.insert(0,'./DCD_DB_API-master/') 
from db_api import DB

import time
#from db_api import DB

def img_loader(img_dir):
    if isinstance(img_dir, str):
        with open(img_dir, 'rb') as file:
            img = file.read()
    return img


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print("Failed to create director!!"+directory)

def update_value(src, th, rect, max_value):
    src16 = src.astype(np.int16)
    mean_value = np.sum(src16[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])])/(rect[2]*rect[3])
    diff_value = th-mean_value
    add_img = src16+diff_value
    update_img1 = np.where(add_img<0, 0 , add_img)
    update_img2 = np.where(update_img1>max_value, max_value , update_img1)
    return update_img2.astype(np.uint8)


def edit_img_value(img, bright_param):
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

def tmp_image_save(DB_imgs, obj_id, g, g_id, obj_iter, bright_param):    
    '''
    DB에서 받은 이미지를 tmp에 저장
    입력받은 category_id별로 이미지를 생성하며, 
    '''
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

def arrange_masks(DB_masks, grid, obj_iter):
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

def get_DB_data(db, obj_category, grid, grid_id, iteration, background_id, bright_param):
    
    get_flag = True
    createFolder('/tmp/augment_DB')

    # DB접속
    #db = DB.DB('192.168.10.69', 3306, 'root', 'return123', 'test')
    
    #배경읽기
    print('DB에서 배경 이미지 읽어오기')
    bg_bytes  = db.get_table(background_id, 'Image')
    if bg_bytes==False:
        print('DB에서 image id가 {}인 배경 이미지를 읽어오는데 에러'.format(background_id))
        get_flag = False
        return None, None, False
    elif bg_bytes==None:
        print('DB에서 image id가 {}인 배경 이미지가없음 '.format(background_id))
        get_flag = False
        return None, None, False
    bg_np = np.frombuffer(bg_bytes[2], dtype = np.uint8)
    bg = cv2.imdecode(bg_np, cv2.IMREAD_COLOR)

    masks = []
    for cate_id, obj_iter in zip(obj_category, iteration):
        print('DB에서 catgegory id가 {}인 데이터 읽어오기'.format(cate_id))
        DB_images = db.get_aug_img(str(grid_id), str(cate_id), "-1")
        if DB_images==False:
            print('DB에서 category id가 {}인 물품이미지들을 읽어오는데 에러'.format(cate_id))
            get_flag = False
            break
        elif DB_images==None:
            print('DB에 obj_id가 {}인 물품의 이미지가 없음'.format(cate_id))
            get_flag = False
            break
        print('임시로 tmp에 이미지 저장')
        img_save_flag = tmp_image_save(DB_images, cate_id, grid, grid_id, obj_iter, bright_param)
        if not img_save_flag:   
            get_flag = False
            break
        print('mask 데이터 읽어오기')
        DB_masks = db.get_aug_mask(str(grid_id), str(cate_id), "-1")
        if DB_masks==False:
            print('DB에서 category id가 {}인 물품의 mask를 읽어오는데 에러'.format(cate_id))
            get_flag = False
            break
        elif DB_masks==None:
            print('DB에 obj_id가 {}인 물품의 mask가 없음'.format(cate_id))
            get_flag = False
            break
        masks_list = arrange_masks(DB_masks, grid,  obj_iter)
        masks.append(masks_list)

    return masks, bg, get_flag

def load_aug_images(images_path):
    images = []
    for path in images_path:
        img = cv2.imread(path)
        images.append(img)
    return images

def set_aug_result(db, aug_segs, grid, grid_id, device_id, images_path):
    # try:
    #     f = open("aug_num.txt", 'r')
    #     obj_aug_start_num = int(f.readline())
    #     f.close()
    # except:
    #     obj_aug_start_num = 1
    
    
    obj_data_list = []
    bbox_data_list = []
    bbox_info = []
    
    #aug_images = load_aug_images(images_path)

    loc_id_tuple = db.get_aug_loc_id(grid_id)
    loc_id_table = list([[None for row in range(grid[1])] for col in range(grid[0])])

    for loc in loc_id_tuple:
        loc_id_table[loc[0]-1][loc[1]-1] = loc[2] 

    image_data_list = [[str(device_id), img_loader(img_p), '3', '1'] for img_p in images_path]


    print('합성된 이미지를 DB에 저장')
    start_img_id = db.get_last_id('Image')+1
    result_flag = db.set_bulk_img(datas=image_data_list)
    if not result_flag:
        print('합성된 이미지파일 DB에 저장 실패')
        return False
    end_img_id = db.get_last_id('Image')+1
    #end_time = time.time()
    #print('total_time: ', end_time - start_time)

    aug_num = db.get_obj_max_aug()+1
    for img_segs, img_id  in zip(aug_segs, range(start_img_id, end_img_id)):
        #우선 obj 정보부터 저장
        for seg in img_segs:
            loc_id = loc_id_table[seg['x']][seg['y']]
            cate_id = seg['category_id']
            iter_num = seg['iteration']
            obj_data_list.append((str(loc_id), str(cate_id), str(img_id), str(iter_num), str(-1), str(aug_num)))
            bbox_info.append(seg['bbox'])
            aug_num +=1

    #print(obj_data_list)
    
    #증가된 obj_id만큼 txt에서도 값 증가
    # f = open("aug_num.txt", 'w')
    # f.write(str(aug_num))
    # f.close()

    print('합성된 Object정보를 DB에 저장')
    start_obj_id = db.get_last_id('Object')+1

    result_flag = db.set_bulk_obj(datas=obj_data_list)
    if not result_flag:
        print('합성된 이미지에서 Object 정보를 DB에 저장 실패')
        print('./error_obj.txt파일 참조')
        f = open("error_obj.txt", 'w')
        f.write(str(obj_data_list))
        f.close()
        return False
    end_obj_id = db.get_last_id('Object')+1 
    print('시작 obj_id : {}'.format(start_obj_id))
    print('마지막 obj_id : {}'.format(end_obj_id))
    print('obj_id 차이 : {}'.format(end_obj_id-start_obj_id))
    print('DB에 저장하는 obj 데이터 갯수 : {}'.format(len(obj_data_list)))


    for bbox, obj_id in zip(bbox_info, range(start_obj_id, end_obj_id)):
        bbox_data_list.append((str(obj_id), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])))

    #print(bbox_data_list)
    print('합성된 Object의 bbox정보를 DB에 저장')
    print('DB에 저장하는 bbox 데이터 갯수 : {}'.format(len(bbox_data_list)))
    result_flag = db.set_bulk_bbox(datas=bbox_data_list)
    if not result_flag:
        print('합성된 이미지에서 bbox정보를 DB에 저장 실패')
        print('./error_obj.txt 및 ./error_bbox.txt 파일 참조')
        f = open("error_obj.txt", 'w')
        f.write(str(obj_data_list))
        f.close()
        f = open("error_bbox.txt",'w')
        f.write(str(bbox_data_list))
        f.close()
        return False

    return True

def aug_main(device_id, grid, grid_id, object_category, background_id, iteration, batch_num, bright_param):
    """
    합성과정 전체 돌아가는 메인 함수
    입력값은 7개로 gRPC를 통해서 json으로 전달 받도록 짜려고 예상
    args : 
        device_id (int): 촬영될 기기 id (
        grid (tuple) : 가로 세로 그리드 비율로 튜플로 반환 (w)(h) 
        grid_id (int): 그리드 id
        object_category (list or tulple) : 물품의 category 값 ex) [12, 34, 23]
        background (bytes) : 배경 이미지, 바이트로 받음
        iteration (int) or (list or tuple): 여기서 iteration은 촬영 횟수를 말하고, 물품 전부 동일하면 int 값만 받아도 무방
                                    아니면 위의 object_category처럼 list로 받음
        batch_num (list or tuple): 이미지 생성할 갯수로 3가지 합성 방법에 따라 합성 갯수를 정해서 받음 ex) [4000, 3000, 3000]
    """

    if str(type(iteration))=="<class 'int'>" :
        iteration_list = [iteration for a in range(len(object_category))]
    else:
        iteration_list = iteration
        
    # DB접속
    db = DB.DB(ip='192.168.10.69', port=3306, user='root', password='return123', db_name='test')

    #db.db_to_json('./json/aug.json','./json/img')
    #db.db_to_json_type('./json','./json/img',3)
    # 먼저 DB에서 file을 읽어오기
    print('read DB data for augmentation')
    DB_mask, background, flag = get_DB_data(db, object_category, grid, grid_id, iteration_list, background_id, bright_param)
    #DB_mask = get_data(object_category, grid, grid_id, iteration_list, image_size)
    if not flag:
        print('DB에서 합성에 필요한 데이터 읽기 실패')
        return False

    #조건에 따라서 몇장 만들지 
    result_data = []
    #save_count = 1
    aug_count = 1
    cut_value = 3
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
            aug_save_flag = set_aug_result(db, result_data, grid, grid_id, device_id, img_path_list)
            #if not aug_save_flag:
            #    print('합성이미지 {}~{}번까지 데이터 저장 실패'.format(aug_count-cut_value, aug_count))
            #    return False
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
    aug_save_flag = set_aug_result(db, result_data, grid, grid_id, device_id, img_path_list)
    #if not aug_save_flag:
    #    print('합성이미지 {}~{}번까지 데이터 저장 실패'.format(aug_count-cut_value, aug_count-1))
    #    return False

    print('finish all augmentations')
    return True


if __name__ == "__main__":
    # 이건 tool에서 입력으로 받아와야 하는 변수들
    # 20001, 2, 3, 1, [1, 2], 3, 29
    device_id = 20001
    grid = (6,5)
    grid_id = 2
    object_category=[1, 2, 4, 5, 6]
    background_id = 1
    iteration = 3
    batch_num = [2000, 3000, 3000]
    # bright_param : [bright_flag, mode_flag, flag1, flag2, flag3, th1, th2, th3, rect x, rect y, rect w, rect h] 
    bright_param = [1, 1, 1, 1, 1, 78, 36, 113, 1140, 440, 100, 200]
    aug_main(device_id, grid, grid_id, object_category, background_id, iteration, batch_num, bright_param)