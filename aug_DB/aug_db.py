import sys
#import os
import cv2
import numpy as np
from functools import wraps

sys.path.insert(0,'./aug_DB/DCD_DB_API/') 
from db_api import DB
from . import aug_db_util

#decorator
def check_read_DB(func):
        @wraps(func)
        def check(self, *args, **kwargs):
            print(func.__name__+' start')
            result, check_flag = func(*args, **kwargs)
            if result==False:
                print(func.__name__+' fail case')
                chekc_flag=False
            elif result==None:
                print(func.__name__+' None case')
                check_flag=False
            return result, check_flag


@check_read_DB
def read_background_image(db, background_id):
    bg_img = db.get_table(background_id, "image")
    return bg_img, True

@check_read_DB
def read_aug_image(db, grid_id, cate_id):
    DB_images = db.get_aug_img(str(grid_id), str(cate_id), "-1")
    return DB_images, True

@check_read_DB
def read_aug_mask(db, grid_id, cate_id):
    DB_masks = db.get_aug_mask(str(grid_id), str(cate_id), "-1")
    return DB_masks, True


class aug_db:
    

    def __init__(self, ip, port, user, password, db_name, preprocessing_flag):
    
        self.db = DB.DB(ip, port, user, password, db_name)
        self.p_flag = preprocessing_flag


    def get_DB_data(self, obj_category, grid, grid_id, iteration, background_id):
        
        get_flag = True
        aug_db_util.createFolder('/tmp/augment_DB')

        # DB접속
        
        #배경읽기
        # print('DB에서 배경 이미지 읽어오기')
        # bg_bytes  = self.db.get_table(background_id, 'Image')
        # if bg_bytes==False:
        #     print('DB에서 image id가 {}인 배경 이미지를 읽어오는데 에러'.format(background_id))
        #     get_flag = False
        #     return None, None, False
        # elif bg_bytes==None:
        #     print('DB에서 image id가 {}인 배경 이미지가없음 '.format(background_id))
        #     get_flag = False
        #     return None, None, False
        bg, bg_flag =read_background_image(self.db, background_id)
        if bg_flag==False:
            return None, None, False
        bg_np = np.frombuffer(bg[2], dtype = np.uint8)
        bg = cv2.imdecode(bg_np, cv2.IMREAD_COLOR)

        masks = []
        for cate_id, obj_iter in zip(obj_category, iteration):
            #print('DB에서 catgegory id가 {}인 데이터 읽어오기'.format(cate_id))
            #DB_images = db.get_aug_img(str(grid_id), str(cate_id), "-1")
            DB_images, db_img_flag = read_aug_image(self.db, grid_id, cate_id)
            if db_img_flag==False:
                break
            # if DB_images==False:
            #     print('DB에서 category id가 {}인 물품이미지들을 읽어오는데 에러'.format(cate_id))
            #     get_flag = False
            #     break
            # elif DB_images==None:
            #     print('DB에 obj_id가 {}인 물품의 이미지가 없음'.format(cate_id))
            #     get_flag = False
            #     break
            print('임시로 tmp에 이미지 저장')
            img_save_flag = aug_db_util.tmp_image_save(DB_images, cate_id, grid, grid_id, obj_iter, bright_param. self.p_flag)
            if not img_save_flag:   
                get_flag = False
                break
            #print('mask 데이터 읽어오기')
            
            DB_masks, db_mask_flag= read_aug_mask(self.db, grid_id, cate_id)
            if db_mask_flag==False:
                break
            # if DB_masks==False:
            #     print('DB에서 category id가 {}인 물품의 mask를 읽어오는데 에러'.format(cate_id))
            #     get_flag = False
            #     break
            # elif DB_masks==None:
            #     print('DB에 obj_id가 {}인 물품의 mask가 없음'.format(cate_id))
            #     get_flag = False
            #     break
            masks_list = aug_db_util.arrange_masks(DB_masks, grid,  obj_iter)
            masks.append(masks_list)

        return masks, bg, get_flag

    def set_aug_result(self, aug_segs, grid, grid_id, device_id, images_path):
        
        obj_data_list = []
        bbox_data_list = []
        bbox_info = []
        
        #aug_images = load_aug_images(images_path)

        loc_id_tuple = self.db.get_aug_loc_id(grid_id)
        loc_id_table = list([[None for row in range(grid[1])] for col in range(grid[0])])

        for loc in loc_id_tuple:
            loc_id_table[loc[0]-1][loc[1]-1] = loc[2] 

        image_data_list = [[str(device_id), aug_db_util.img_loader(img_p), '3', '1'] for img_p in images_path]


        print('합성된 이미지를 DB에 저장')
        start_img_id = self.db.get_last_id('Image')+1
        result_flag = self.db.set_bulk_img(datas=image_data_list)
        if not result_flag:
            print('합성된 이미지파일 DB에 저장 실패')
            return False
        end_img_id = self.db.get_last_id('Image')+1
        #end_time = time.time()
        #print('total_time: ', end_time - start_time)

        aug_num = self.db.get_obj_max_aug()+1
        for img_segs, img_id  in zip(aug_segs, range(start_img_id, end_img_id)):
            #우선 obj 정보부터 저장
            for seg in img_segs:
                loc_id = loc_id_table[seg['x']][seg['y']]
                cate_id = seg['category_id']
                iter_num = seg['iteration']
                obj_data_list.append((str(loc_id), str(cate_id), str(img_id), str(iter_num), str(-1), str(aug_num)))
                bbox_info.append(seg['bbox'])
                aug_num +=1

        print('합성된 Object정보를 DB에 저장')
        start_obj_id = self.db.get_last_id('Object')+1

        result_flag = self.db.set_bulk_obj(datas=obj_data_list)
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
        result_flag = self.db.set_bulk_bbox(datas=bbox_data_list)
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

