import os
import sys
import numpy as np
import cv2
import random
import math
from . import aug_cal

class augment:
    """
    물품 합성용 클래스
    """
    def __init__(self, grid, object_category, batch_method, background_image, mask, iteration_list, shadow_flag=1, center=None, category_grid=None):
        """
        물품 합성에 필요한 정보를 저장하는 부분

        Args:
            grid (tuple): 가로 세로 그리드 비율로 튜플값 (w)(h) 
            object_category [list]: 물품의 category 값 ex) [12, 34, 23]
            batch_method (int): 배치방식, 3가지로 나뉘며 1,2,3 으로 구분
            backgroun_image (numpy): 배경 이미지
            mask (list): 마스크 정보, 6차원 배열
            iteration_list (list): iteration 정보가 저장된 배열
            shadow_flag (bool): 그림자 적용 여부
            center (tuple): 내부 판에서 정중앙 위치 따로 입력시 사용(이미지 중앙과 판에서의 중앙이 다를때 사용됨)
            category_grid(list): 물품이 특정 포지션 내에서만 들어가서 따로 구분이 필요할때 사용, 현재 사용안함
        """
        # 그리드 정보로 x,y 값, tuple
        self.grid = grid

        # 현재 사용할 category 정보, list, tuple 둘 중 뭐든 상관없을듯
        #ex)[3 , 7, 4, 5, 13....]
        self.object_category = object_category

        #카테고리 종류 최대치
        self.category_num = len(object_category)

        # 입력받는 mask 정보
        self.mask_data = mask

        # iteration 정보도 입력, 각 물품별로 최대 몇장씩 찍었는지 
        self.iteration = iteration_list
        
        # category에 맞는 grid 맵 정보로 3d list
        #[물품종류][가로][세로]
        if category_grid==None:
            self.category_grid = list([[[1 for row in range(grid[1])] for col in range(grid[0])]for cate in range(len(object_category))])
        else:
            self.category_grid = category_grid

        # 실제 매대의 중심좌표(calibration안할경우 따로 입력)
        if center==None:
            self.center = (int(background_image.shape[1]/2), int(background_image.shape[0]/2))
        else:
            self.center = center

        # batch_method는 1,2,3  3가지
        # 1. 열별 배치, 그리고 같은 열은 단일 물품
        # 2. 열별로 배치, 대신 물품 종류는 랜덤
        # 3. 랜덤 배치
        self.batch_method = batch_method

        # 배경 이미지
        self.ori_background_image = background_image

        # 그림자 옵션 추가 여부
        self.shadow_flag = shadow_flag
        # threshold 기준
        # param1,2 로 나뉘어 지고 물품의 가로길이를 기준으로 세로로 얼마만큼 잘라낼지 판단
        self.threshold_param1=(1.0, 0.3)
        self.threshold_param2=(0.7, 0.2)

        #물품이 배치될때 전체에서 얼마만큼 비율로 배치될지 정하는 파라미터
        self.object_dense = 0.5

        # rand option같은 경우 0과 1이 차이가 있는데
        #0은 배치할때 확률이 dense가 0.3이면 무조건 30%는 배치가 되어야함. 즉 49칸이면 15칸만 딱 물품이 배치
        #반대로 1은 각각의 위치별로 물품이 존재 할 확률이 30%, 실제 배치되는 갯수는 이미지마다 다르며, Normal 분포를 가짐
        self.rand_option = 1

        # array_method 1이면 음료수 물품과 같이 중앙이 가장 크고 가로 갈수록 가려지는 형태
        # array_method 2이면 트레이 설치된 경우로 뒷쪽 물품이 점점 가려지는 형태, 다만 현재 사용불가
        self.array_method = 1

        # 이미지 가로,세로길이를 background 이미지크기에서 받아오도록 설정함
        self.img_w = background_image.shape[1]
        self.img_h = background_image.shape[0]
        
        # 타원관련 파라미터
        # 물품의 가로, 세로 길이를 기반으로 위에 shadow_value를 포함해서 실제 타원의 크기에 영향을 줌 
        self.ellipse_param = (0.4, 0.5)
        
        # 그림자를 타원형태로 단순하게 만드는데 사용하는 방식이 픽셀값이 1차이인 타원을 수십개를 이미지에 붙여서 만드는 형태로 사용함
        # 여기서 사용되는 타원갯수와 가장 진한 타원의 픽셀값이 shadow_value가 됨
        self.shadow_value = 30
        
        # 물품끼리 서로 겹치게 될 경우 삭제하며, 원래 물품의 mask 영역 크기와 나중에 가려져서 남은 영역크기 비율로 제거
        # 0.06일 경우 가려진게 94% 이상 가려지면 검출 불가능이라고 판단해서 지움
        self.delete_ratio_th = 0.06

        # 물품끼리 겹치는 경우 bbox를 재 계산할때 필요한 파라미터
        # 가리는쪽 가중치를 -, 가려지는 쪽을 +로 해서 bbox 내부의 영역합이 최대인 값으로 계산
        # 가리는쪽 가중치를 -3~-4정도로 설정하면 무난할듯
        self.around_object_weight = -3

        # 마지막에 bbox 다시 보정용으로 재계산할때 필요
        # 실제 물품의 사이즈 줄여가면서 적합한 bbox를 다시 계산하는데 얼만큼 줄일지 판단하는 값
        self.re_cal_search_region = 0.7
        
    def compose_batch(self):
        """
        그리드 위에서 물품을 어떻게 배치할지 설정하는 부분
        배치방식에 맞춰서 최종적으로 물품이 배치될 그리드 정보를 list 형태로 받아옴

        배치방식 
        1. 줄 단위로 배치, 같은 줄은 같은 물품
        2. 줄 단위로 배치, 같은 줄이라도 물품은 서로 다르게
        3. 전체적으로 완전히 랜덤
        
        """
        batch_map = []
        if self.batch_method<3:
            if self.rand_option: 
                #option 1일때는 normal distribution으로 
                map_possible = aug_cal.make_gridmap(self.object_dense, self.grid[0])
            else: 
                #optino 0일때로 전체 리스트에서 원하는 %만
                p = int(self.grid[0]*self.object_dense+0.5)
                map_possible = [0 for i in range(self.grid[0]-p)]+[1 for i in range(p)]
                random.shuffle(map_possible)
            
            # 둘다 방식은 조금 다르지만 열 단위로 봤을때 1은 들어가는 경우, 0은 들어가지 않는 경우
            if self.batch_method==1:
                #여기는 같은 열은 같은 물체로만 배치
                for col in range(self.grid[0]):
                    if map_possible[col]==0: 
                        batch_map.append([0 for i in range(self.grid[1])])
                        continue
                    col_poss=[]
                    for obj_cate in range(self.category_num):
                        map_sum=sum(self.category_grid[obj_cate][col])
                        if map_sum==self.grid[1]: col_poss.append(1)
                        else: col_poss.append(0)
                    poss_cate = [self.object_category[i] for i in range(len(col_poss)) if col_poss[i]]
                    select_cate = random.choice(poss_cate)
                    batch_map.append([select_cate for i in range(self.grid[1])])
            else:
                #같은 열이라도 다른 물체 배치
                batch_map = [[0 for row in range(self.grid[1])] for col in range(self.grid[0])]
                for col in range(self.grid[0]):
                    if map_possible[col]==0: 
                        continue
                    for row in range(self.grid[1]):
                        #행렬 각 위치에서 확률을 계산
                        #col_poss=[]
                        col_poss = [self.category_grid[i][col][row] for i in range(self.category_num)]
                        poss_cate = [self.object_category[i] for i in range(len(col_poss)) if col_poss[i]]
                        select_cate = random.choice(poss_cate)
                        batch_map[col][row]=select_cate
           
        else:
            ##방식3 2차원 map으로 만듬
            if self.rand_option: 
                 #map_possible = [[round(random.random()-(0.5-self.object_dense)) for row in range(self.grid[1])]for col in range(self.grid[0])]
                map_possible = aug_cal.make_gridmap(self.object_dense, self.grid)
            else:
                p = int(self.grid[0]*self.grid[1]*self.object_dense+0.5)
                map_possible_line = [0 for i in range(self.grid[0]*self.grid[1]-p)]+[1 for i in range(p)]
                print(sum(map_possible_line))
                random.shuffle(map_possible_line)
                print(map_possible_line)
                map_possible = []
                for col in range(self.grid[0]):
                    map_possible.append(map_possible_line[(col*self.grid[1]):((col+1)*self.grid[1])])
            # 이제 물체 배치
            batch_map = [[0 for row in range(self.grid[1])] for col in range(self.grid[0])]
            for col in range(self.grid[0]):
                for row in range(self.grid[1]):
                    if map_possible[col][row]==0: 
                        continue
                    #행렬 각 위치에서 확률을 계산
                    col_poss = [self.category_grid[i][col][row] for i in range(self.category_num)]
                    poss_cate = [self.object_category[i] for i in range(len(col_poss)) if col_poss[i]]
                    select_cate = random.choice(poss_cate)
                    batch_map[col][row]=select_cate
        #print("물품 배치 완료")

        self.batch_map = batch_map
        #print(batch_map)
      
    def load_DB(self, batch_map=None):
        """
        딱 필요한 데이터만 로드하는 함수로 self.image_data에 저장

        각각의 파라미터는 다음과 같음
        ['mask_value'] = mask map만들때 사용되며 랜덤한 값이 각각 할당 됨, 
        category 와 다른 값을 사용하는 이유는 같은 물품이라도 별도로 구별하기 위해서 따로 랜덤한 값을 넣음
        ['pos_x'], ['pos_y'] : 각 물품별 포지션 위치
        ['category'] : 말그대로 카테고리 넘버, 
        ['bbox'] : bbox 정보, x, y , w ,h 순서로 list로 저장
        ['mask'] : 마스크 정보 - [[x1, y1], [x2, y2], [x3, y3], ... , [xn, yn]] 식으로 구성
        ['area'] : 마스크 내부의 영역크기
        ['image'] : 물체가 그 그리드에 배치가 된 이미지, 나중에 합성시 사용
        """
        
        if batch_map!=None:
            self.batch_map = batch_map
        

        #self.batch_map[3]=[1,1,1,1,1,1]
        print(self.batch_map)
        image_data = []
        # 여기서 물품 합성시 중앙에서 가장 먼 순서대로 배치 할 수 있도록 조정
        batch = aug_cal.array_DB_batch(self.grid, self.batch_map, self.array_method)
        self.batch_num = len(batch)
        #나중에 실제 합성시 따로 필요한 정보
        mask_value = list(range(1, self.batch_num*(255//self.batch_num)+1,(255//self.batch_num)))
        random.shuffle(mask_value)
        
        for b, v in zip(batch, mask_value):
            # batch에서 카테고리 정보 가져옴
            cate_id = self.batch_map[b[0]][b[1]]
            
            #그리드 정보는 재배열된 batch정보에서 바로 얻어내서 저장
            data_info = {'pos_x':b[0],'pos_y':b[1]}
            # mask_value는 그냥 랜덤한 값이므로 바로 같이 저장
            data_info['mask_value'] = v
            # 카테고리 정보 저장
            data_info['category'] = cate_id
            
            cate_index =self.object_category.index(cate_id)

            iter_value = random.randrange(self.iteration[cate_index])
            data_info['iteration'] = iter_value

            mask = self.mask_data[cate_index][b[0]][b[1]][iter_value]
            data_info['mask'] = mask
            mask_np = np.array(mask)
            area = cv2.contourArea(mask_np)
            data_info['area'] = area
            #print(area)
            
            #이미지를 opencv로 읽어오기
            image_name = str('{}x{}_{}.jpg').format(b[0]+1, b[1]+1, iter_value+1)
            image_path = '/tmp/augment_DB/'+str(cate_id)+'/'+image_name
            img = cv2.imread(image_path)
            #이미지 저장
            data_info['image'] = img
            
            #각각의 딕셔너리 파일을 list로 쌓음
            image_data.append(data_info)
        
        # 최종적으로 저장된 파일을 self로 저장
        self.image_data = image_data

    def make_background(self):
        """
        background 이미지를 전처리하거나 그림자를 붙이는 용도
        """
        if self.shadow_flag==0:
            #print("배경에 그림자 적용 제외")
            self.background_image = self.ori_background_image
            #return self.ori_background
        else :
            # 그림자를 단순하게 적용하려면 물품의 크기에 맞는 타원형태로 적용
            # 그래서 물품의 영역을 보여주는 타원을 구하고, 그 타원에 맞춰서 그림자를 배경에 집어 넣는 것으로 구분
            shadow_background_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            for img_info in self.image_data:
                shadow = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
                #물품이 위에서 촬영된 걸 보면 원근감으로 인해 구석에 있는건 회전된 것 처럼 보임
                #즉 물품이 회전되었다고 가정하고 그에 맞는 타원을 계산

                mask_np = np.array(img_info['mask'])
                
                obj_center = aug_cal.cal_obj_center(mask_np)
                
                #각도 계산
                angle = -math.atan2((obj_center[0]-self.center[0]), (obj_center[1]-self.center[1]))
                # 기존의 회전된 물체를 수직형태로 회전변환을 진행하기 위해서 각도 정보를 가진 메트릭스를 따로 정의
                rotate_m = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
                mask_np_diff = mask_np - np.array(obj_center)

                # 각도 메트릭스와 현재 mask 점들을 메트릭스 곱으로 한번에 연산을 통해 회전변환 계산
                rotate_mask_np = np.dot(mask_np_diff,rotate_m)
                rotate_mask_np = rotate_mask_np.astype(np.int16)+np.array(obj_center)
                
                # 물체에 맞는 타원의 가로, 세로 계산
                length_w = np.max(rotate_mask_np, axis=0)[0]-np.min(rotate_mask_np, axis=0)[0]
                length_h = np.max(rotate_mask_np, axis=0)[1]-np.min(rotate_mask_np, axis=0)[1]            
                
                #물품의 타원
                #크기가 다른 shadow_value개의 타원을을 순차적으로 겹쳐서 부드럽게 그림자를 만듬
                for j in range(self.shadow_value):
                    w_d = (self.shadow_value-j+1)*0.2/(self.shadow_value+1)
                    cv2.ellipse(shadow, tuple(obj_center), (int(length_w * self.ellipse_param[0] + length_h * w_d), int(length_h * self.ellipse_param[1])), \
                                (angle * 180 / math.pi), 0, 360,(j, j, j), -1)
                shadow_background_img = shadow_background_img + shadow 
                
                
            #배경에 각각의 타원 정보가 입력되었으니 이걸 최종적으로 기존 background랑 합침
            bg_sum = self.ori_background_image.astype(np.int16) - shadow_background_img.astype(np.int16)
            bg = np.where(bg_sum < 0, 0, bg_sum)
            bg = bg.astype(np.uint8)
            
            self.background_image = bg


    def make_maskmap(self):
        """
        이미지를 붙이기 전, 실제 물품영역만 붙여진 맵을 따로 만듬
        """
        #물품영역을 검정 배경에 붙이기 때문에 검정 배경이미지를 먼저 만듬
        maskmap = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        for img_info in self.image_data:
            mask_np = np.array(img_info['mask'])
            value = img_info['mask_value']
            # opencv의 contour그리는 함수를 통해 검정 배경에 붙임
            cv2.drawContours(maskmap, [mask_np], -1,(value,value,value), -1)
            
        # maskmap 저장
        self.maskmap = maskmap
        return maskmap
    
    def augment_image(self):
        """
        실제 물품을 합성해서 이미지에 붙이는 작업
        연산량을 줄이기 위해 min, max 정보를 mask에서 뽑아낸다음
        그 영역에서만 따로 연산으로 붙임
        """
        #입력으로 받은 배경이미지에다가 붙이기 위해서 배경을 가져옴
        aug_img = self.background_image
        for img_info in self.image_data:
            # mask 점의 x,y 최대 최소를 구함
            mask_np = np.array(img_info['mask'])
            x_max = np.max(mask_np, axis=0)[0]
            x_min = np.min(mask_np, axis=0)[0]
            y_max = np.max(mask_np, axis=0)[1]
            y_min = np.min(mask_np, axis=0)[1]

            
            # 물품 영역부분을 잘라내서 붙이는 부분
            obj_s = img_info['image'][y_min:y_max,x_min:x_max]
            obj_maskmap = self.maskmap[y_min:y_max,x_min:x_max]
            aug_obj = aug_img[y_min:y_max,x_min:x_max]
            aug_img[y_min:y_max,x_min:x_max] = np.where(obj_maskmap==img_info['mask_value'], obj_s, aug_obj)
        
        
        self.aug_img = aug_img
    

    def post_processing(self):
        """
        후처리단으로 이미지의 노이즈나, 밝기 등의 요소를 조절
        """
        pass
    
    def re_segmentation(self):
        """
        앞서 구한 segment map을 기반으로 다시 계산한 segmentation을 출력으로 보냄
        """
        cal_seg = []
        aug_seg_img = self.aug_img.copy()
        #aug_seg_img = self.aug_img.copy()
        deleted_info = []
        
        for img_info in self.image_data:
            #print(img_info['mask_value'])
            deleted_map= np.where(self.maskmap == img_info['mask_value'], 255, 0)
            mask_np = np.array(img_info['mask'])
            img_center = (self.center[0], self.center[1])
            threshold = [self.threshold_param1, self.threshold_param2]
            
            
            if(img_info['pos_x']==3) & (img_info['pos_y']==5) & (img_info['category']==1):
                check=True 

            obj_map = deleted_map.astype(np.uint8)
            
            # mask 계산
            obj_cal_mask, area = aug_cal.cal_mask(obj_map,img_info['area'], self.delete_ratio_th)
            if obj_cal_mask[0][0]==-1:
                self.batch_map[img_info['pos_x']][img_info['pos_y']]=0
                deleted_info.append(img_info)
                continue
                
            #print('물체의 위치정보: ({},{})'.format(img_info['pos_x'], img_info['pos_y']))

            # bbox계산
            obj_cal_bbox = aug_cal.cal_bbox(obj_map, obj_cal_mask, img_center, threshold)
            
            cal_seg.append({'mask': obj_cal_mask, 'bbox' : obj_cal_bbox})

        #없어진 물품 리스트에서 지우기
        for del_info in deleted_info:
            self.image_data.remove(del_info)
        
        #bbox 다시 계산
        re_seg = aug_cal.revise_bbox(cal_seg, self.batch_map, self.grid, self.image_data, self.around_object_weight, self.re_cal_search_region)
        self.re_segmentation = re_seg
        #for seg2 in re_seg:
        #    cv2.rectangle(aug_seg_img, tuple(seg2['bbox']), (0, 255, 255), 1)
        #cv2.imshow('aug_img',aug_seg_img)
        #cv2.waitKey(0)
    
    def arrange_aug_data(self, aug_count):
        """
        필요한 정보만 정리

        Args:
            aug_count : 현재 합성이 몇번째인지 확인하는 용도

        Return:
            str : 저장된 이미지 path정보
            dictionary : 합성 결과 데이터
        """
        img_save_folder = '/tmp/augment_DB/aug_img'
        try:
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)
        except OSError as e:
            print("Failed to create director!!"+img_save_folder)

        img_save_path = img_save_folder+'/'+'%06d.jpg'%(aug_count)
        cv2.imwrite(img_save_path, self.aug_img)

        aug_DB = self.re_segmentation

        #이부분은 혹시 bbox까지 포함된 결과 이미지를 저장하기 위한 용도
        img_save_folder2 = '/tmp/augment_DB/aug_result'
        img_save_path2 = img_save_folder2+'/'+'%06d.jpg'%(aug_count)
        re_img = self.aug_img.copy()
        for seg in self.re_segmentation:
            cv2.rectangle(re_img, tuple(seg['bbox']),(0, 255, 255), 1)
        cv2.imwrite(img_save_path2, re_img)
        
        return img_save_path, aug_DB
        

def aug_process(grid, object_category, batch_method, background, DB_masks,iteration_list, aug_count):
    """
    실제 합성이 이루어지는 함수로 아래와 같이 구성

    1. 물품 배치할 배치맵 생성
    2. 필요한 데이터만 로드
    3. background image 생성(그림자 추가 등)
    4. 이미지 합성
    5. bbox 및 mask 재 계산

    args: 
        grid (tuple): 가로 세로 그리드 비율로 튜플값 (w)(h) 
        object_category (list): 물품의 category 값 ex) [12, 34, 23]
        batch_method (int): 배치방식, 3가지로 나뉘며 1,2,3 으로 구분
        background (numpy): 배경 이미지
        DB_masks (list): DB에서 읽어온 mask정보 6차원 배열 [category 순서][position x][position y][iteration][[x1,y1], [x2, y2], ... ]
        iteration_list (list): iteration 정보가 저장된 배열
        aug_count (int): 현재 합성되는 이미지가 몇번째 이미지인지 확인용도

    Return:
        str : 저장된 이미지 path정보
        dictionary : 합성 결과 데이터, ['mask'], ['bbox'], ['category_id'], ['area'], ['x'], ['y'], ['iteration']
    """

    #print("합성할 물품 배치를 계산하는 부분 시작")
    aug1 = augment(grid, object_category, batch_method, background, DB_masks, iteration_list)
    aug1.compose_batch() 
    #aug1.load_DB_API()
    #print("데이터 읽기 시작")
    #batch = [[2, 1, 3, 2, 1, 3], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [3, 2, 1, 3, 2, 2], [1, 2, 3, 2, 3, 2], [0, 0, 0, 0, 0, 0], [1, 2, 2, 3, 2, 1]]
    #aug1.load_DB(batch_map=batch)
    
    aug1.load_DB()
    #print("데이터 읽기 완료")
    aug1.make_background()
    #print('배경에 그림자 적용 완료')
    aug1.make_maskmap()
    aug1.augment_image()
    #print('이미지 합성 완료')
    #print("mask 및 bbox 다시 계산")
    aug1.re_segmentation()
    img_path, result = aug1.arrange_aug_data(aug_count)

    #print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    return img_path, result
