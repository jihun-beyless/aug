import sys
#import os
import cv2
import numpy as np
from functools import wraps

sys.path.insert(0,'./DCD_DB_API/') 
from db_api import DB
from . import aug_db_util

#decorator
def check_read_DB(func):
    @wraps(func)
    def read_check(*args, **kwargs):
        print(func.__name__+' start')
        check_flag = True
        result = func(*args, **kwargs)
        if result==False:
            print(func.__name__+' fail case')
            check_flag=False
        elif result==None:
            print(func.__name__+' None case')
            check_flag=False
        return result, check_flag
    return read_check

#decorator
def check_save_DB(func):
    @wraps(func)
    def save_check(*args, **kwargs):
        error_type = ['image', 'object', 'bbox']
        print(func.__name__+' start')
        flag = func(*args, **kwargs)
        if flag==False:
            for d,e_t in zip(args[2],error_type):
                print('합성된 이미지에서 정보를 DB에 저장 실패')
                file_name = 'error_{}.txt'.format(e_t)
                print(file_name+'파일 참조')
                f = open(file_name, 'w')
                f.write(str(d))
                f.close()
        else:
            print('aug data saved')
        return flag
    return save_check


@check_read_DB
def read_background_image(db, background_id):
    """
    DB에서 background image 읽어옴

    Args:
        db (DB): 접속된 db
        background_id (int): 

    Return:
        Byte: image bytes
    """
    print('background image id : '+str(background_id))
    bg_img  = db.get_table(background_id, 'Image')
    return bg_img

@check_read_DB
def read_image(db, grid_id, cate_id):
    """
    DB에서 grid_id와 cate_id가 동일한 물품의 이미지 들을 읽어옴

    Args:
        db (DB): 접속된 db
        background_id (int): 

    Return:
        tuple ()(): ((loc_x, loc_y, iteration, (byte)img), (...))
    """
    print('category_id : '+str(cate_id))
    DB_images = db.get_aug_img(str(grid_id), str(cate_id), "-1")
    return DB_images

@check_read_DB
def read_mask(db, grid_id, cate_id):
    """
    DB에서 grid_id와 cate_id가 동일한 물품의 마스크정보들을 읽어옴

    Args:
        db (DB): 접속된 db
        background_id (int): 

    Return:
        tuple ((loc_x, loc_y, iteration, mask_id, mask_x, mask_y), (...))
    """
    print('category_id : '+str(cate_id))
    DB_masks = db.get_aug_mask(str(grid_id), str(cate_id), "-1")
    return DB_masks

@check_save_DB
def save_aug_image(db, data):
    """
    합성된 이미지 데이터 저장

    Args:
        db (DB): 접속된 db
        data (list): 현재 데이터들, 0번 이미지, 1번 obj, 2번 bbox

    Return:
        bool: True or False
    """
    return db.set_bulk_img(datas=data[0])

@check_save_DB
def save_aug_object(db, data):
    """
    합성된 이미지 데이터 저장

    Args:
        db (DB): 접속된 db
        data (list): 현재 데이터들, 0번 이미지, 1번 obj, 2번 bbox

    Return:
        bool: True or False
    """
    return db.set_bulk_obj(datas=data[1])

@check_save_DB
def save_aug_bbox(db, data):
    """
    합성된 이미지 데이터 저장

    Args:
        db (DB): 접속된 db
        data (list): 현재 데이터들, 0번 이미지, 1번 obj, 2번 bbox

    Return:
        bool: True or False
    """
    return db.set_bulk_bbox(datas=data[2])


class aug_db:
    """
    합성시 필요한 데이터를 읽고, 합성된 데이터를 저장하는 클래스
    """

    def __init__(self, ip, port, user, password, db_name, preprocessing_flag):
        """
        DB접속에 필요한 정보로 DB에 접속 및 preproceissing의 여부를 확인함

        Args:
            ip (str): MySQL 서버에 로그인하기위한 ip 주소
            port (int): 포트 포워딩을 위한 포트
            user (str): MySQL 서버에 로그인을 위한 아이디
            password (str): MySQL 서버에 로그인을 위한 비밀번호
            db_name (str): 데이터베이스 네임
            preprocessing_flag (bool): 전처리를 진행을 확인하는 flag, True시 white balance를 맞춤
        """
    
        self.db = DB.DB(ip, port, user, password, db_name)
        self.p_flag = preprocessing_flag


    def get_DB_data(self, obj_category, grid, grid_id, iteration, background_id):
        """
        합성시 필요한 데이터를 읽는 함수
        
        Args:
            obj_category (tuple): object category id가 나열된 tuple
            grid (tuple) : 가로 세로 그리드 비율로 튜플값 (w)(h) 
            grid_id (int): 그리드 id
            iteration (list): 각 물품별 실제 배치 가능한 영역
            background_id (int) : 배경 이미지에 해당되는 id값

        return:
            masks (list) : DB에서 읽어온 mask정보 6차원 배열 [category 순서][position x][position y][iteration][[x1,y1], [x2, y2], ... ]
            bg (numpy) : background image
            read_flag (bool): DB에서 데이터 읽는게 제대로 됐는지 확인 True or False
        """
        
        read_flag = True
        aug_db_util.createFolder('/tmp/augment_DB')

        #self.db.db_to_json_type('./json/data.json','./json/img/',3)
        #read background
        bg, bg_flag =read_background_image(self.db, background_id)
        if bg_flag==False:
            return None, None, False
        bg_np = np.frombuffer(bg[2], dtype = np.uint8)
        bg = cv2.imdecode(bg_np, cv2.IMREAD_COLOR)

        masks = []
        for cate_id, obj_iter in zip(obj_category, iteration):
            
            # read images
            DB_images, db_img_flag = read_image(self.db, grid_id, cate_id)
            if db_img_flag==False:
                read_flag = False
                break
            
            # image saving in tmp folder 
            img_save_flag = aug_db_util.tmp_image_save(DB_images, cate_id, grid, grid_id, obj_iter, self.p_flag)
            if not img_save_flag:   
                read_flag = False
                break
            
            #read masks
            DB_masks, db_mask_flag= read_mask(self.db, grid_id, cate_id)
            if db_mask_flag==False:
                read_flag = False
                break

            masks_list = aug_db_util.arrange_masks(DB_masks, grid,  obj_iter)
            masks.append(masks_list)

        return masks, bg, read_flag

    def set_aug_result(self, aug_segs, grid, grid_id, device_id, images_path):
        """
        합성된 데이터를 저장하는 함수
        
        Args:
            aug_segs (list): 이미지별 합성정보가 list로 저장되어 있음, [image num]['area', 'bbox', 'mask', 'category_id', 'iteration', 'x', 'y']
            grid (tuple): 가로 세로 그리드 비율로 튜플값 (w)(h) 
            grid_id (int): 그리드 id
            device_id (int): 촬영된 기기 id
            images_path (list): 현재 합성된 이미지가 저장된 파일경로들을 list로 저장 [image num][str]
        return:
            bool : True or False
        """


        save_datas = []
        bbox_info = []
        obj_data_list = []
        bbox_data_list = []
        flag = True
        #aug_images = load_aug_images(images_path)

        #저장될 형식을 맞추기 위해서 loc_id 를 위치별로 따로 뽑아냄
        loc_id_tuple = self.db.get_aug_loc_id(grid_id)
        loc_id_table = list([[None for row in range(grid[1])] for col in range(grid[0])])

        for loc in loc_id_tuple:
            loc_id_table[loc[0]-1][loc[1]-1] = loc[2] 

        #image data정리
        image_data_list = [[str(device_id), aug_db_util.img_loader(img_p), '3', '1'] for img_p in images_path]
        save_datas.append(image_data_list)

        #image 저장
        result_flag = save_aug_image(self.db, save_datas)
        flag = flag & result_flag
        end_img_id = self.db.get_last_id('Image')+1

        # obj data 정리
        aug_num = self.db.get_obj_max_aug()+1
        for img_segs, img_id  in zip(aug_segs, range(end_img_id - len(aug_segs), end_img_id)):
            #우선 obj 정보부터 저장
            for seg in img_segs:
                loc_id = loc_id_table[seg['x']][seg['y']]
                cate_id = seg['category_id']
                iter_num = seg['iteration']
                obj_data_list.append((str(loc_id), str(cate_id), str(img_id), str(iter_num), str(-1), str(aug_num)))
                bbox_info.append(seg['bbox'])
                aug_num +=1
        save_datas.append(obj_data_list)

        # obj data 저장
        result_flag = save_aug_object(self.db, save_datas)
        flag = flag & result_flag
        end_obj_id = self.db.get_last_id('Object')+1 

        # bbox data 정리
        for bbox, obj_id in zip(bbox_info, range(end_obj_id-len(obj_data_list), end_obj_id)):
            bbox_data_list.append((str(obj_id), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])))
        save_datas.append(bbox_data_list)

        # bbox data 저장
        result_flag = save_aug_bbox(self.db, save_datas)
        flag = flag & result_flag


        return flag

