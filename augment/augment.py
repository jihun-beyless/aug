import os
import sys
import numpy as np
import cv2
import random
import math
from . import aug_cal

class augment:
    '''
    물품 합성용 클래스
    '''
    def __init__(self, grid, object_category, batch_method, background_image, mask, iteration_list, shadow_flag=1, center=None, category_grid=None):
        '''
        입력 :
        그리드 정보, 배치할 물품 정보, 물품별 배치가능 범위, 배치 방식, 백그라운드 이미지, 그림자 여부, 중심값
        (annotation tool에서 받아옴)
        그외 세부 조건은 따로 config 파일에서 받아옴
        '''
        # 그리드 정보로 x,y 값, tuple
        self.grid = grid
        # 현재 사용할 category 정보, list, tuple 둘 중 뭐든 상관없을듯
        #ex)[3 , 7, 4, 5, 13....]
        self.object_category = object_category
        #카테고리 종류 최대치
        self.category_num = len(object_category)
        # mask 정보가 입력 받는 형태로 바뀜
        self.mask_data = mask
        # iteration 정보도 입력, 각 물품별로 최대 몇장씩 찍었는지 확인 용도
        self.iteration = iteration_list
        
        # category에 맞는 grid 맵 정보로 3d list
        #[물품종류][가로][세로]
        if category_grid==None:
            self.category_grid = list([[[1 for row in range(grid[1])] for col in range(grid[0])]for cate in range(len(object_category))])
        else:
            self.category_grid = category_grid
        # batch_method는 1,2,3  3가지
        # 1. 열별 배치, 그리고 같은 열은 단일 물품
        # 2. 열별로 배치, 대신 물품 종류는 랜덤
        # 3. 랜덤 배치
        # 중심점 위치 입력, tuple
        #이미지의 중심이 아닌 실제 매대의 중심좌표를 입력해야함
        if center==None:
            self.center = (int(background_image.shape[1]/2), int(background_image.shape[0]/2))
        else:
            self.center = center
        self.batch_method = batch_method
        # 배경 이미지
        # opencv에서 이미지 읽어올때 쓰는 형태면 상관없긴 한데 그게 아니면 아래와같이  수정이 필요할지도
        # np(가로,세로,3, dtype = uint8)
        self.ori_background_image = background_image
        # 그림자 옵션 추가 여부
        self.shadow_flag = shadow_flag
        # threshold 기준
        # param1,2 로 나뉘어 지고 물품의 가로길이를 기준으로 세로로 얼마만큼 잘라낼지 판단
        # 1은 무조건 잘라내는 기준이고, 수직보다는 대각선을 더 많이 잘라낸다고 보면됨
        # 그래서 대각선은 th1기준으로 1.0, 수직은 1.3
        # th2에서는 대각선 0.7, 수직은 0.9
        self.threshold_param1=(1.0, 0.3)
        self.threshold_param2=(0.7, 0.2)
        #물품이 배치될때 전체에서 얼마만큼 비율로 배치될지 정하는 파라미터
        self.object_dense = 0.5
        # rand option같은 경우 0과 1이 차이가 있는데
        #0은 배치할때 확률이 dense가 0.3이면 무조건 30%는 배치가 되어야함. 즉 49칸이면 15칸만 딱 물품이 배치
        #반대로 1은 각각의 위치별로 물품이 존재 할 확률이 30%, 실제 배치되는 갯수는 이미지마다 다르며, Normal 분포를 가짐
        self.rand_option = 1
        # array_method 1이면 음료수 물품과 같이 중앙이 가장 크고 가로 갈수록 가려지는 형태
        # array_method 2이면 트레이 설치된 경우로 뒷쪽 물품이 점점 가려지는 형태
        # 다만 array_method 2는 코드를 짜다가 말아서, 현재는 안된다고 봐야함
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
        # 물품끼리 서로 겹치게 될 경우 물품이 거의 안보이게 되는데 그러면 삭제가 필요함
        # 삭제하는 기준을 원래 물품의 mask 영역 크기와 나중에 가려져서 거의다 가려질 경우 남은 영역크기 비율로 제거하려고 함
        # 즉 여기서 가려진게 94% 이상 가려지면 검출 불가능이라고 판단해서 지우는 형태로 구현
        self.delete_ratio_th = 0.06
        # 물품끼리 겹치는 경우 bbox를 재 계산할때 필요한 파라미터로 가려지는 쪽보다 가리는 쪽에 가중치를 더 많이 둬야
        # 위쪽의 뚜껑부분이 덜 문제가 발생됨
        # 계산시 영역내부의 픽셀합으로 계산하는데 가리는쪽이 -값이여야 됨
        # -3~-4정도로 설정하면 무난할듯
        self.around_object_weight = -3
        # 마지막에 bbox 다시 보정용으로 재계산할때 필요
        # 실제 물품의 사이즈 줄여가면서 적합한 bbox를 다시 계산하는데 얼만큼 줄일지 판단하는 값
        self.re_cal_search_region = 0.5
        
    def compose_batch(self):
        '''
        그리드 위에서 물품을 어떻게 배치할지 설정하는 부분
        배치방식에 맞춰서 최종적으로 물품이 배치될 그리드 정보를 list 형태로 받아옴
        필요한 정보(self) : 그리드 정보(가로,세로), 배치할 물품 정보(list로 category_id), 물품별 배치가능 범위(list), category_id : 배치가능한 영역이 보이는 그리드맵), 배치 방식(1~3번까지 방식), 배치 분포율(float, ex)0.4)
        출력: 그리드별 배치정보를 가진 list 맵 

        배치방식 
        1. 줄 단위로 배치, 같은 줄은 같은 물품
        2. 줄 단위로 배치, 같은 줄이라도 물품은 서로 다르게
        3. 전체적으로 완전히 랜덤
        
        그리고 구성은 2가지로 배치할 공간을 설정하는 부분 -> batch_map으로 분리
        그 공간에 어떤 물품을 넣을지 물품을 넣는 부분으로 나뉨
        '''
        #batch_map = [[0 for row in range(self.grid[1])] for col in range(self.grid[0])]
        batch_map = []
        if self.batch_method<3:
            if self.rand_option: 
                #option 1일때로 말그대로 각 위치별로 확률이 별도로 계산 
                #저기서 round를 하면 기본적으로 50%가 되므로 0.5-dense를 빼면 원하는 확률로 설정
                #map_possible = [round(random.random()-(0.5-self.object_dense)) for col in range(self.grid[0])]
                map_possible = aug_cal.make_gridmap(self.object_dense, self.grid[0])
            else: 
                #optino 0일때로 전체 리스트에서 원하는 %만
                p = int(self.grid[0]*self.object_dense+0.5)
                map_possible = [0 for i in range(self.grid[0]-p)]+[1 for i in range(p)]
                random.shuffle(map_possible)
            # 둘다 방식은 조금 다르지만 열 단위로 봤을때 1은 들어가는 경우, 0은 들어가지 않는 경우
            #print("물품이 들어갈 위치만 결정")
            #print(map_possible)
            if self.batch_method==1:
                #여기는 같은 열은 같은 물체로만 배치
                for col in range(self.grid[0]):
                    if map_possible[col]==0: 
                        batch_map.append([0 for i in range(self.grid[1])])
                        continue
                    #우선 현재 열에서 각 물체들의 배치 가능 여부를 알아야함
                    col_poss=[]
                    for obj_cate in range(self.category_num):
                        map_sum=sum(self.category_grid[obj_cate][col])
                        if map_sum==self.grid[1]: col_poss.append(1)
                        else: col_poss.append(0)
                    # 즉 col_poss에서는 각 열별로 가능여부가 나타나기 때문에 저기서 1만 해당되는 것들을 따로 뽑아서 
                    # 뽑아내면 됨, 여기서는 중복도 상관없으니 random.choice 사용
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
            ##방식3
            # 3에서는 물품 배치 가능도 2차원 map으로 만듬
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
            #print("물품이 들어갈 위치만 결정")
            #print(map_possible)
            batch_map = [[0 for row in range(self.grid[1])] for col in range(self.grid[0])]
            for col in range(self.grid[0]):
                for row in range(self.grid[1]):
                    if map_possible[col][row]==0: 
                        continue
                    #행렬 각 위치에서 확률을 계산
                    #col_poss=[]
                    col_poss = [self.category_grid[i][col][row] for i in range(self.category_num)]
                    poss_cate = [self.object_category[i] for i in range(len(col_poss)) if col_poss[i]]
                    select_cate = random.choice(poss_cate)
                    batch_map[col][row]=select_cate
        #print("물품 배치 완료")

        self.batch_map = batch_map
        print(batch_map)
        #return batch_map
      
    def load_DB(self, batch_map=None):
        '''
        위의 load_DB를 임시로 대신하는 코드, 실제 데이터를 읽어서 테스트 하기 위해서 사용
        image_data 라는 리스트에 전부 데이터를 저장하며
        각 위치별로 이미지, bbox, mask 정보는 각각 딕셔너리 형태로 저장
        즉 딕셔너리의 데이터를 list로 만들어 놓은걸 self.image_data로 저장
        데이터의 구성 : 
        위치별로 물품이 하나씩 배치가 때문에 그 각각의 위치별 물품 정보를  
        따로 저장해서 나중에 사용함
        배치될 물품 하나당 딕셔너리의 데이터 하나씩 만들어지며 
        각각의 파라미터는 다음과 같음
        ['mask_value'] = mask map만들때 사용되며 랜덤한 값이 각각 할당 됨, 
        category 와 다른 값을 사용하는 이유는 같은 물품이라도 별도로 구별하기 위해서 따로 랜덤한 값을 넣음
        ['grid_x'], ['grid_y'] : 각 물품별 그리드 위치
        ['category'] : 말그대로 카테고리 넘버, 
        ['bbox'] : bbox 정보, x, y , w ,h 순서로 list로 저장
        ['mask'] : 마스크 정보 - [[x1, y1], [x2, y2], [x3, y3], ... , [xn, yn]] 식으로 구성
        ['area'] : 마스크 내부의 영역크기
        ['image'] : 물체가 그 그리드에 배치가 된 이미지, 나중에 합성시 사용
        '''
        
        if batch_map!=None:
            self.batch_map = batch_map
        image_data = []
        # 여기서 물품 합성시 중앙에서 가장 먼 순서대로 배치 할 수 있도록 조정
        batch = array_DB_batch(self.grid, self.batch_map, self.array_method)
        self.batch_num = len(batch)
        #나중에 실제 합성시 따로 필요한 정보
        mask_value = list(range(1, self.batch_num*(255//self.batch_num)+1,(255//self.batch_num)))
        random.shuffle(mask_value)
        
        #for col in range(self.grid[0]):
        #   for row in range(self.grid[1]):
        for b, v in zip(batch, mask_value):
            # batch에서 카테고리 정보 가져옴
            cate_id = self.batch_map[b[0]][b[1]]
            
            #그리드 정보는 재배열된 batch정보에서 바로 얻어내서 저장
            data_info = {'grid_x':b[0],'grid_y':b[1]}
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
            image_name = str('{}x{}_{}.png').format(b[0]+1, b[1]+1, iter_value+1)
            image_path = '/tmp/augment_DB/'+str(cate_id)+'/'+image_name
            img = cv2.imread(image_path)
            #이미지 저장
            data_info['image'] = img
            
            #각각의 딕셔너리 파일을 list로 쌓음
            image_data.append(data_info)
        
        # 최종적으로 저장된 파일을 self로 저장
        self.image_data = image_data

    def make_background(self):
        '''
        background 이미지를 전처리하거나 그림자를 붙이는 용도
        그림자 설정 여부에 따라서 연산이 확달라지는데, 그림자가 필요하면 물품의 정보와, 위치정보 전부 필요로 함
        입력 : 그리드별 배치정보, DB정보, 그림자 여부(전부 self에 들어가있음)
        출력: background 이미지
        '''
        if self.shadow_flag==0:
            #print("배경에 그림자 적용 제외")
            self.background_image = self.ori_background_image
            #return self.ori_background
        else :
            # 그림자를 단순하게 적용하려면 물품의 크기에 맞는 원형 형태로 적용하는게 무난
            # 이걸 하려면 물품을 둘러싸는 타원을 구하고 그에 맞춰서 그림자를 넣어버리면 됨
            # 그래서 물품의 영역을 보여주는 타원을 구하고, 그 타원에 맞춰서 그림자를 배경에 집어 넣는 것으로 구분
            #print('배경에 그림자 적용 시작')
            shadow_background_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            for img_info in self.image_data:
                shadow = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
                #물품이 위에서 촬영된 걸 보면 원근감으로 인해 구석에 있는건 회전된 것 처럼 보임
                #즉 물품이 회전되었다고 가정하고 그에 맞는 타원을 구해야함
                #필요한 값 : 회전각도, 회전되었다고 가정했을때 타원의 가로, 세로 길이
                #계산 방법 : 물품의 중심위치(opencv moments 이용)-> 기울기 계산-> 물품의 각 점을 똑바로 세운다고 가정하고 
                #반대방향으로 역으로 회전했을때 x, y 최대, 최소를 구함-> 그에 맞춰서 타원을 계산->
                #그림자를 좀 더 자연스럽게 하기 위해서 타원 여러개를 겹치게 하되
                #타원크기에 맞춰서 픽셀값을 순차적으로 줄임
                
                mask_np = np.array(img_info['mask'])
                #print('원본마스크 점 : {0}'.format(mask_np))
                
                obj_center = aug_cal.cal_obj_center(mask_np)
                #print('물체 중심 : {0}'.format(obj_center))
                
                #각도 계산
                angle = -math.atan2((obj_center[0]-self.center[0]), (obj_center[1]-self.center[1]))
                #print('각도 : {0}'.format(angle))
                
                # 기존의 회전된 물체를 수직형태로 회전변환을 진행하기 위해서 각도 정보를 가진 메트릭스를 따로 정의
                rotate_m = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
                #print('회전 메트릭스 : {0}'.format(rotate_m))
                
                #물체 중심을 기반으로 회전해야하므로 
                mask_np_diff = mask_np - np.array(obj_center)
                #print('마스크와 물체 중심과의 차이값 : {0}'.format(mask_np_diff))
                
                # 각도 메트릭스와 현재 mask 점들을 메트릭스 곱으로 한번에 연산을 통해 회전변환 계산
                rotate_mask_np = np.dot(mask_np_diff,rotate_m)
                rotate_mask_np = rotate_mask_np.astype(np.int16)+np.array(obj_center)
                #print('회전 결과 점위치 : {0}'.format(rotate_mask_np))
                
                # 물체에 맞는 타원의 가로, 세로 계산
                length_w = np.max(rotate_mask_np, axis=0)[0]-np.min(rotate_mask_np, axis=0)[0]
                length_h = np.max(rotate_mask_np, axis=0)[1]-np.min(rotate_mask_np, axis=0)[1]
                #print('타원 가로 : {0}, 타원 세로: {1}'.format(length_w, length_h))                
                
                #물품의 타원
                #shadow_value 라는게 그림자 최대 픽셀값
                #크기가 다른 shadow_value개의 타원을을 순차적으로 겹쳐서 부드럽게 그림자를 만듬
                for j in range(self.shadow_value):
                    w_d = (self.shadow_value-j+1)*0.2/(self.shadow_value+1)
                    cv2.ellipse(shadow, tuple(obj_center), (int(length_w * self.ellipse_param[0] + length_h * w_d), int(length_h * self.ellipse_param[1])), \
                                (angle * 180 / math.pi), 0, 360,(j, j, j), -1)
                
                shadow_background_img = shadow_background_img + shadow 
                
                
            #배경에 각각의 타원 정보가 입력되었으니 이걸 최종적으로 기존 background랑 합침
            bg_sum = self.ori_background_image.astype(np.int16) - shadow_background_img.astype(np.int16)
            #uint8은 값의 범위가 0~255로 음수값을 제외하기 위해서 코드 몇줄 추가됨
            bg = np.where(bg_sum < 0, 0, bg_sum)
            bg = bg.astype(np.uint8)
            
            #cv2.imshow('bg',bg)
            #cv2.waitKey(0)
            self.background_image = bg
            #print('배경에 그림자 적용 완료')
            #return bg

    def make_maskmap(self):
        '''
        이미지를 붙이기 전, 실제 물품영역만 붙여진 맵을 따로 만듬
        예를 들어 음료수 물품이 사이다면 사이다가 붙여질 영역을 따로 만든 맵 
        대신 물품별로 구별 되어야 할 필요가 있기 때문에 앞서서 랜덤하게 부여한 mask_value를 그 영역에 값으로 집어넣음
        물론 가려지는 부분은 고려해서 가장자리 물품은 앞의 물품의 여부에 따라 가려지고, 중앙 물품은 가려지지 않는 형태로 구성
        입력 : 그리드별 배치정보, DB정보(전부 self에 들어가있음)
        출력: 물품이 배치될 maskmap
        ''' 
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
        '''
        실제 물품을 합성해서 이미지에 붙이는 작업
        그림자의 경우 나중에 그림자 개선이 필요하면, 여기서 추가 작업이 이루어 질 수 있을수 있으나 현재 아직은 고려 X
        연산량을 줄이기 위해 min, max 정보를 mask에서 뽑아낸다음
        그 영역에서만 따로 연산으로 붙임
        붙일때 음료수만 붙여야 하므로 그 조건을 위에서 구한 mask map을 이용함
        입력 : 그리드별 배치정보, DB정보, 그림자 여부(나중에 고려), segment map
        출력 : 합성된 이미지
        '''
        #입력으로 받은 배경이미지에다가 붙이기 위해서 배경을 가져옴
        aug_img = self.background_image
        for img_info in self.image_data:
            # 연산을 효율적으로 이용하기 위해 mask 점의 x,y 최대 최소를 구함
            mask_np = np.array(img_info['mask'])
            x_max = np.max(mask_np, axis=0)[0]
            x_min = np.min(mask_np, axis=0)[0]
            y_max = np.max(mask_np, axis=0)[1]
            y_min = np.min(mask_np, axis=0)[1]
            #print("최소최대: {0},{1}, {2},{3}".format(x_max, x_min, y_max, y_min))
            
            # 실제 이미지에서 물품 영역부분을 잘라냄
            obj_s = img_info['image'][y_min:y_max,x_min:x_max]
            # mask map에서 물품 영역부분을 잘라냄
            obj_maskmap = self.maskmap[y_min:y_max,x_min:x_max]
            # 배경도 마찬가지로 잘라옴
            aug_obj = aug_img[y_min:y_max,x_min:x_max]
            # 붙이는 작업
            aug_img[y_min:y_max,x_min:x_max] = np.where(obj_maskmap==img_info['mask_value'], obj_s, aug_obj)
        
        # 이미지 합성 종료
        #cv2.imshow('segmap',aug_img)
        #cv2.waitKey(0)
        
        self.aug_img = aug_img
        #return aug_img
    

    def post_processing(self):
        '''
        후처리단으로 이미지의 노이즈나, 밝기 등의 요소를 조절
        아마 기존의 물품 정보는 불필요할 가능성이 높음
        입력 : 합성된 이미지
        출력 : 후처리된 합성된 이미지
        '''
        pass
    
    def re_segmentation(self):
        '''
        앞서 구한 segment map을 기반으로 다시 계산한 segmentation을 출력으로 보냄
        입력 : segment map, 기존 segmentation 정보, 관련 config 파라미터
        출력 : 다시 계산된 segmentation
        bbox와 mask를 다시 계산하는 함수를 따로 불러서
        각각 계산을 따로 진행
        '''
        cal_seg = []
        aug_seg_img = self.aug_img.copy()
        aug_seg_img2 = self.aug_img.copy()
        deleted_info = []
        
        for img_info in self.image_data:
            #print(img_info['mask_value'])
            deleted_map= np.where(self.maskmap == img_info['mask_value'], 255, 0)
            mask_np = np.array(img_info['mask'])
            img_center = (self.center[0], self.center[1])
            threshold = [self.threshold_param1, self.threshold_param2]
            
            obj_map = deleted_map.astype(np.uint8)
            #print(img_info['area'])
            # mask 다시 계산
            obj_cal_mask, area = aug_cal.cal_mask(obj_map,img_info['area'], self.delete_ratio_th)
            if obj_cal_mask[0][0]==-1:
                #print('삭제됨')
                #print('삭제된 위치:{}, {}'.format(img_info['grid_x'],img_info['grid_y']))
                self.batch_map[img_info['grid_x']][img_info['grid_y']]=0
                deleted_info.append(img_info)
                continue
            
            # bbox계산
            obj_cal_bbox = aug_cal.cal_bbox(obj_map, obj_cal_mask, img_center, threshold)
            
            cal_seg.append({'mask': obj_cal_mask, 'bbox' : obj_cal_bbox})
            #cv2.drawContours(aug_seg_img, re_mask,-1, (255, 255, 255), 1)
            #for p in range(re_mask.shape[0]-1):
            #    cv2.line(aug_seg_img,tuple(re_mask[p]),tuple(re_mask[p+1]),(0,255,255),1)
            #cv2.rectangle(aug_seg_img, tuple(obj_cal_bbox), (255, 255, 0), 1)
            #cv2.putText(aug_seg_img, str(a_r), (re_bbox[0],re_bbox[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (100,100,255), 1)

        #없어진 물품 리스트에서 지우기
        for del_info in deleted_info:
            self.image_data.remove(del_info)
        
        #bbox 다시 조절하는 함수 추가
        re_seg = aug_cal.revise_bbox(cal_seg, self.batch_map, self.grid, self.image_data, self.around_object_weight, self.re_cal_search_region)
        self.re_segmentation = re_seg
        for seg2 in re_seg:
            cv2.rectangle(aug_seg_img2, tuple(seg2['bbox']), (0, 255, 255), 1)
        #cv2.imshow('aug_img',aug_seg_img)
        #cv2.imshow('aug_img2',aug_seg_img2)
        #cv2.waitKey(0)
        #return re_seg
    
    def save_DB(self, aug_count):
        '''
        이미지 및 재 계산된 mask와 bbox 정보를 DB에 저장
        입력 : 후처리된 합성된 이미지, 다시 계산된 segmentation 정보
        '''
        img_save_folder = '/tmp/augment_DB/aug_img'
        try:
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)
        except OSError as e:
            print("Failed to create director!!"+img_save_folder)

        img_save_path = img_save_folder+'/'+'%06d.png'%(aug_count)
        cv2.imwrite(img_save_path, self.aug_img)
        #img_bytes = img.tobytes()
        #aug_DB = {'bbox':bbox, 'image':img_bytes}
        #aug_DB = {'seg':self.re_segmentation, 'image':self.aug_img}
        aug_DB = self.re_segmentation
        
        return img_save_path, aug_DB
        
        
    # def augment_main(self):
    #     '''
    #     순서:
    #     -> 그리드 각각 위치에 물품 배치구성
    #     -> DB에서 필요한 이미지 및 정보 찾아오기(background, 실제 물품 촬영 이미지, bbox, mask 정보)
    #     -> background 이미지 만들기 (전처리+그림자)
    #     -> 물품을 background에 붙이기
    #     -> 이미지 후처리 (노이즈, 밝기 조절)
    #     -> bbox와 mask 다시 계산
    #     -> DB에 저장
    #     '''
    #     pass


#print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

def aug_process(grid, object_category, batch_method, background, DB_masks,iteration_list, aug_count):

    #print("합성할 물품 배치를 계산하는 부분 시작")
    aug1 = augment(grid, object_category, batch_method, background, DB_masks, iteration_list)
    aug1.compose_batch() 
    #aug1.load_DB_API()
    #print("데이터 읽기 시작")
    aug1.load_DB()
    #print("데이터 읽기 완료")
    aug1.make_background()
    #print('배경에 그림자 적용 완료')
    aug1.make_maskmap()
    aug1.augment_image()
    #print('이미지 합성 완료')
    #print("mask 및 bbox 다시 계산")
    aug1.re_segmentation()
    img_path, result = aug1.save_DB(aug_count)

    #print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    return img_path, result
