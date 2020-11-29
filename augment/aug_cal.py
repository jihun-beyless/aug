import os
import sys
import numpy as np
import cv2
import random
import math
import copy

def make_gridmap(dense, *args):
    """
    물품 랜덤하게 선택하는부분
    물품 자체가 안잡히는 0이 되는 맵이 존재할 수 있으니, 0이 되는걸 없앨려고 따로 재귀함수 형태로 구현함

    Args: 
        dense (float): 물품 배치 밀도,y 값
        *args (args): 배치 방식을 여러가지 고려하기 위해서 사용, (grid x, )또는 (grid x, grid y, )값이 들어옴

    return :
        list : grid 맵, 배치 방식에 따라 1차원 2차원 따로 return
    """
    if str(type(args[0]))=="<class 'int'>" :
        grid = [round(random.random()-(0.5-dense)) for col in range(args[0])]
        sum_grid = sum(grid)
        if sum_grid==0:
            return make_gridmap(dense, args[0])
        else :
             return grid
    else :
        grid = [[round(random.random()-(0.5-dense)) for row in range(args[0][1])]for col in range(args[0][0])]
        sum_grid = sum([sum(g) for g in grid])
        if sum_grid==0:
            return make_gridmap(dense, args[0])
        else :
            return grid

def array_DB_batch(pos, batch_map, array_method):
    """
    물품의 합성 순서를 결정하는 부분, 맨뒤나 맨 가장자리부터 물품이미지를 붙여야 문제가 안생김
    중앙을 기준으로 position 상에서 얼만큼 떨어져 있는지를 계산해서 재 정렬
    
    Args:
        pos (tuple):  pos정보 (x, y)
        batch_map (list): 실제 물품이 배치될 2차원 map
        array_method (int): 물품이 원통형인지, tray가 필요한 물품인지에 따라서 나눔 

    return:
        list : [[x1, y1, distance1], [x2, y2, distance2], ...] 
    """
    # 합성 방법으로 가장 단순한 방법은 특정 위치와의 실제 거리를 계산하여 가장 먼 곳부터 합성을 진행
    if array_method==1: c_point = [(pos[0]-1)/2,(pos[1]-1)/2]
    elif array_method==2: c_point = [(pos[0]-1)/2,0] 
    #print(c_point)

    dis_info_list=[]
    for col in range(pos[0]):
        for row in range(pos[1]):
            if batch_map[col][row]==0:
                continue
            dx = col-c_point[0]
            dy = row-c_point[1]
            distance = dx*dx+dy*dy
            dis_info_list.append([col, row, distance])
    
    #정렬
    dis_info_tuple = tuple(dis_info_list)
    array_dis_info = sorted(dis_info_tuple, key=lambda x: -x[2])
    return array_dis_info

def cal_obj_center(mask):
    """
    단순히 mask 정보를 기반으로 물품의 중심좌표를 구하는 함수(opencv의 moments 이용)

    Args:
        mask (list): mask 정보, [[x1, y1], [x2, y2], ...]

    return:
        list : [x,y] 좌표
    """
    M = cv2.moments(mask)
                
    #물체의 중심 좌표
    obj_cx = int(M['m10'] / M['m00'])
    obj_cy = int(M['m01'] / M['m00'])
    return [obj_cx, obj_cy]

def cal_mask(obj_map, ori_area, delete_flag = True ,area_ratio_th = 0.06):
    """
    mask를 가려진부분 제외하고 실제로 진짜 보이는 부분으로 다시 계산

    Args:
        obj_map (numpy): 실제 그 물체만 255이고, 다른 부분은 전부 0으로 된 이진화된 이미지
        ori_area (int): 원래 물품의 mask 영역크기값
        area_ratio_th (float): 만약 물품이 거의 다 가려질 경우 물품 삭제를 할 기준 커트라인
    
    return: 
        mask (list): mask 정보, [[x1, y1], [x2, y2], ...]
        re_area (float): mask 내부의 영역 넓이
    """
    binary_map = cv2.cvtColor(obj_map, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS )
    
    if len(contours)==0:
        #print('싹다가려짐')
        return [[-1,-1]], 0
    elif len(contours)>1:
        # 물품이 서로 가려지는것 때문에 영역이 2개이상으로 잡히는 경우가 발생함
        # size 측정해서 가장 큰값만 남기는게 맞음
        #print('애매하게 가려져서 조각남')
        area = [cv2.contourArea(a) for a in contours]
        m = contours[area.index(max(area))]
    else:
        m = contours[0]
        
    #물품이 가려진 비율 계산
    re_area = cv2.contourArea(m)
    #print('크기 비교:{},{}'.format(area, re_area))
    a_ratio = re_area/ori_area
    if delete_flag:
        if a_ratio<area_ratio_th:
            #print('mask 크기가 너무 작음')
            return [[-1,-1]], 0

    mask = np.reshape(m,(m.shape[0],2))
    return mask, re_area

def cal_bbox(obj_map, ori_mask, mask, center, threshold, cap_option):
    '''
    bbox를 다시 계산하는 함수 자세한 과정은 아래에

    과정 
    1. 우선 물품의 중심좌표를 계산
    2. 물품과 중심점의 기울기(각도)를 계산
    3. 물품의 영역을 따옴
    (다만 따올때 회전을 다음에 진행할때 이미지 영역을 벗어나면 안되기 때문에 적당히 크기를 조절해서 가져와야함)
    4. 물품의 영역을 -각도로 회전(이러면 맨 아래가 물품의 윗부분으로 회전됨)
    5. 실제 bbox만 남기기 위해서 잘라내기
    물품이 수직방향이기 때문에 threshold 기준을 y축을 기준으로 잡아서 삭제
    (threshold는 총 2개, 2개 4개를 쓰는데, 기준1에서 2개,기준2에서 2개, 그리고 각도별로 threshold가 다르게 설정됨)
    6. 잘라낸 영역을 기준으로 다시 원래 상태로 재 회전
    7. 다시 회전된 영역에서 박스정보를 계산(contour 계산 + contour 내에서 외접하는 박스 계산)


    Args :
        obj_map (list): 실제 그 물체만 255이고, 다른 부분은 전부 0으로 된 이진화된 이미지
        mask (list): 마스크 정보
        center (tuple): 중심좌표
        threshold (tuple): 총 4개의 기준 존재

    return 
        list : 다시 계산된 bbox 정보[[x1,y1,w1,h1], [x2, y2, w2, h2], ...]

    '''
    
    # 1. 물품 중심좌표
    obj_center = cal_obj_center(mask)
    
    # 2. 물품과 중심점의 기울기(각도)를 계산
    if obj_center[1]==center[1]:
        obj_center[1]=obj_center[1]+1
    angle = -math.atan2((obj_center[0]-center[0]), (obj_center[1]-center[1]))
    #print(obj_center)
    #print(angle)

    # 3. 물품의 영역을 가져오기 
    # 즉 mask상에서 점들을 회전을 시켜서 그 mask 점들중 최대값을 계산
    
    # mask점들을 회전변환을 진행하기 위해서 각도 정보를 가진 메트릭스를 따로 정의
    ro_m1 = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
                
    #물체 중심을 기반으로 회전해야하므로 
    mask_diff = mask - np.array(obj_center)
                
    # 각도 메트릭스와 현재 mask 점들을 메트릭스 곱으로 한번에 연산을 통해 회전변환 계산
    rotate_mask = np.dot(mask_diff,ro_m1)
    rotate_mask = rotate_mask.astype(np.int16)+np.array(obj_center)
    
    #변환했을시 영역의 size 측정
    #현재 이미지 기준이라, 회전이 된경우 이미지 크기를 벗어난 경우도 존재함(-값으로도 나올 수 있음)
    # 순서는 x_min, y_min, x_max, y_max, width, height 
    rotate_size = [np.min(rotate_mask, axis=0)[0], np.min(rotate_mask, axis=0)[1], np.max(rotate_mask, axis=0)[0],
                  np.max(rotate_mask, axis=0)[1], np.max(rotate_mask, axis=0)[0]-np.min(rotate_mask, axis=0)[0],
                  np.max(rotate_mask, axis=0)[1]-np.min(rotate_mask, axis=0)[1]]
    
    # 회전 전에 영역의 size 측정
    #이것도 현재 이미지 기준
    mask_size = [np.min(mask, axis=0)[0], np.min(mask, axis=0)[1], np.max(mask, axis=0)[0],
                np.max(mask, axis=0)[1], np.max(mask, axis=0)[0]-np.min(mask, axis=0)[0],
                np.max(mask, axis=0)[1]-np.min(mask, axis=0)[1]]
    
    # 실제 계산이 이미지 전체에서 이루어지면 연산량이 확 늘기 때문에 연산량을 줄이기 위해서
    # 물품 크기의 딱 2배가 되는 임의의 작은 영역을 따로 만듬
    
    # 작은 영역 이미지의 w, h
    ro_img_w = max(rotate_size[4], mask_size[4])*2
    ro_img_h = max(rotate_size[5], mask_size[5])*2
    
    # 작은 영역크기에 맞는 
    obj_region = np.zeros((ro_img_h, ro_img_w, 3), dtype=np.uint8)
    
    #작은 영역 이미지에서 물체의 크기를 다시 따로 계산
    #x_min, y_min, x_max, y_max(w,h는 동일하기 때문에 제외)
    #이건 그냥 연산을 좀 더 편하게 하기 위한 용도
    region_size= [int(ro_img_w/2+mask_size[0]-obj_center[0]), int(ro_img_h/2+mask_size[1]-obj_center[1]), int(ro_img_w/2+mask_size[2]-obj_center[0]), int(ro_img_h/2+mask_size[3]-obj_center[1])]
    
    # 영역 붙이기
    obj_region[region_size[1]:region_size[3], region_size[0]:region_size[2]] = obj_map[mask_size[1]:mask_size[3], mask_size[0]:mask_size[2]]
    
    # 4.물품의 영역을 -각도로 회전(각도가 -값으로 이미 주어져 있음)
    degree = (angle * 180 / math.pi)
    ro_m2 = cv2.getRotationMatrix2D((ro_img_w/2,ro_img_h/2), degree, 1)
    rotate_region = cv2.warpAffine(obj_region, ro_m2,(ro_img_w, ro_img_h))
    
    
    # 5. 실제 bbox만 남기기 위해서 잘라내기
    #물품이 수직방향이기 때문에 threshold 기준을 y축을 기준으로 잡으면 됨
    #cut_late = threshold[0][0] + abs(abs(abs(degree)-90) - 45) / 45 * threshold[0][1]

    if cap_option:
        #cap_region = np.zeros((rotated_size[5],rotated_size[4],3),dtype=np.uint8)
        #세로축 길이를 일단 10% 잡고 잘라내돼, 이미 충분히 가려진 물품은 그보다 더 짧을 수 있기 때문에 세로축 길이 계산이 필요함
        cap_height = min(int(rotate_size[5]*0.2), rotate_size[5])
        cap_region = rotate_region[int(ro_img_h/2+rotate_size[3]-obj_center[1])-cap_height:int(ro_img_h/2+rotate_size[3]-obj_center[1]), \
            int(ro_img_w/2+rotate_size[0]-obj_center[0]):int(ro_img_w/2+rotate_size[2]-obj_center[0])]
        binary_cap_map = cv2.cvtColor(cap_region, cv2.COLOR_BGR2GRAY)
        cap_contours, caps_hierarchy = cv2.findContours(binary_cap_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS )
        if len(cap_contours)>1:
            cap_area = [cv2.contourArea(a) for a in cap_contours]
            #max_cap_area = max(cap_area)
            cap_m = cap_contours[cap_area.index(max(cap_area))]
        else:
            #max_cap_area = cv2.contourArea(cap_contours[0])
            cap_m = cap_contours[0]
        
        cap_region_bbox = cv2.boundingRect(cap_m)
        cap_bbox = [cap_region_bbox[0]+int(ro_img_w/2+rotate_size[0]-obj_center[0]), int(ro_img_h/2+rotate_size[3]-obj_center[1])-cap_region_bbox[3], cap_region_bbox[2], cap_region_bbox[3]]
        cap_copy = rotate_region.copy()
        #cap_copy2 = rotate_region.copy()
        cv2.rectangle(cap_copy, cap_bbox, (200,0,200))
        rotate_region[:,0:cap_bbox[0]] = np.zeros((ro_img_h, cap_bbox[0], 3), dtype=np.uint8)
        rotate_region[:,(cap_bbox[0]+cap_bbox[2]):ro_img_w] = np.zeros((ro_img_h, ro_img_w-(cap_bbox[0]+cap_bbox[2]), 3), dtype=np.uint8)
        #cap_copy2[:,0:cap_bbox[0]] = np.zeros((ro_img_h, cap_bbox[0], 3), dtype=np.uint8)
        #cap_copy2[:,(cap_bbox[0]+cap_bbox[2]):ro_img_w] = np.zeros((ro_img_h, ro_img_w-(cap_bbox[0]+cap_bbox[2]), 3), dtype=np.uint8)
        #cv2.imshow('cap1', cap_copy)
        #cv2.imshow('cap2', cap_copy2)
        #cv2.waitKey(0)


    
    #(1) 일단 1차 기준의 경우 1차 기준 밑으로는 다 잘라냄(예외없이 싹뚝) 
    cut_length1 = int((rotate_size[4] * (threshold[0][0] + abs(abs(abs(degree)-90) - 45) / 45 * threshold[0][1])))
    mask_cut1 = int(ro_img_h/2+rotate_size[3]-obj_center[1] - cut_length1)
    if cut_length1<rotate_size[5] :
        min_y = int(ro_img_h/2+rotate_size[1]-obj_center[1])-5
        if min_y<0 : min_y=0
        rotate_region[min_y:mask_cut1] = np.zeros((mask_cut1-min_y, ro_img_w, 3), dtype=np.uint8)
    
    cut_length2 = int(rotate_size[4] * (threshold[1][0] + abs(abs(abs(degree)-90)- 45) / 45 * threshold[1][1]))
    mask_cut2 = int(ro_img_h/2+rotate_size[3]-obj_center[1] - cut_length2)
        
    #(2) 이제 한줄씩 읽어가면서 물품의 영역이 얼만큼 차지하는지 비율에 따라서 잘라낼지 아닐지를 결정
    if cut_length2<rotate_size[5] :
        for y in range(mask_cut1-1, mask_cut2, 1):
            aver_value = np.sum(obj_region[y])/rotate_size[4]
            if aver_value >128:
                if (y-mask_cut1+1)<0:
                    print("합성쪽 에러")
                    print('mask_cu1:{0}, mask_cut2:{1}, y:{2}'.format(mask_cut1, mask_cut2, y))
                rotate_region[mask_cut1-1 : y ] = np.zeros((y-mask_cut1+1, ro_img_w, 3), dtype=np.uint8)
                break
            
    # 6. 이제 원래로 다시 역 변환 

    ro_m3 = cv2.getRotationMatrix2D((ro_img_w/2,ro_img_h/2), -degree, 1)
    re_obj_region = cv2.warpAffine(rotate_region, ro_m3,(ro_img_w, ro_img_h))
    
    # 7. 실제 최종 bbox 계산
    binary_map = cv2.cvtColor(re_obj_region, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS )
    
    fit_bbox = cv2.boundingRect(contours[0])
    
    re_bbox = [int(fit_bbox[0]-ro_img_w/2+obj_center[0]), int(fit_bbox[1]-ro_img_h/2+obj_center[1]),fit_bbox[2], fit_bbox[3]]
    
    #cv2.imshow("cap_region", cap_copy)
    #cv2.imshow("rotated region", rotate_region)
    #cv2.rectangle(re_obj_region,fit_bbox, (0,200,200))
    #cv2.imshow("final bbox region", re_obj_region)
    #cv2.waitKey(0)

    return re_bbox


def related_pos(p_x, p_y, g_x, g_y, batch_map):
    """
    그리드 위치, batch_map을 보고 현재 물품의 위치를 1~9까지 구역으로 나눔
    각 구역 번호로 가려질때 어느 위치만 참조할지 결정함

    Args:
        p_x (int): 현재 물체의 position x좌표
        p_y (int): 현재 물체의 position y좌표
        g_x (int): grid x
        g_y (int): grid y
        batch_map (list): 2차원 물품 배치된 map [g_x][g_y]
    
    return
        int : 구역넘버 (1~9까지 중 하나의 값)
        list : 현재 위치에서 상대적인 위치값을 list로 반환 [[x1, y1], [x2, y2],..] ex: [[-1, 0], [0, 1]]

    """
    #region_table = [[1,4,7],[2,5,8],[3,6,9]]
    region_table = [[3,6,9],[2,5,8],[1,4,7]]
    
    re_pos = []
    half_x = (g_x-1)/2
    half_y = (g_y-1)/2
    
    # 각 방향별로 조건달아서 선택
    # 그리고 실제 batch_map에서 그 위치에 물체가 있는지를 확인
    # 왼쪽위
    if (p_x>=half_x) and (p_y>=half_y):
        if batch_map[p_x-1][p_y-1]!=0:
            re_pos.append([-1,-1])
    # 위
    if p_y>=half_y:
        if batch_map[p_x][p_y-1]!=0:
            re_pos.append([0,-1])
    # 오른쪽위
    if (p_x<=half_x) and (p_y>=half_y):
        if batch_map[p_x+1][p_y-1]!=0:
            re_pos.append([1,-1])
    # 왼쪽
    if p_x>=half_x:
        if batch_map[p_x-1][p_y]!=0:
            re_pos.append([-1,0])
    # 오른쪽
    if p_x<=half_x:
        if batch_map[p_x+1][p_y]!=0:
            re_pos.append([1,0])
    # 왼쪽아래
    if (p_x>=half_x) and (p_y<=half_y):
        if batch_map[p_x-1][p_y+1]!=0:
            re_pos.append([-1,1])
    # 위
    if p_y<=half_y:
        if batch_map[p_x][p_y+1]!=0:
            re_pos.append([0,1])
    # 오른쪽아래
    if (p_x<=half_x) and (p_y<=half_y):
        if batch_map[p_x+1][p_y+1]!=0:
            re_pos.append([1,1])
    
    #여기는 현재 물체가 어느 구역인지 확인(1~9구역)
    r_x = p_x-half_x
    r_y = p_y-half_y
    if r_x!=0: r_x = r_x//abs(r_x)+1
    else : r_x=1
    if r_y!=0: r_y = r_y//abs(r_y)+1
    else : r_y=1
    region_num = region_table[int(r_x)][int(r_y)]
    # re_pos에서 필요한 것만 확인한거니 그걸 반환
    return region_num,re_pos


def check_overlap(bbox1, bbox2):
    """
    2개의 박스를 비교해서 겹치는지 여부를 확인  
    
    Args:
        bbox1 (list): 첫번째 bbox 정보
        bbox2 (list): 두번째 bbox 정보
    
    return:
        bool : True or False
    """ 
    flag = True
    

    #박스 2개가 겹침을 확인하는방법은  4개의 모서리 크기를 비교
    if bbox1[0]>(bbox2[0]+bbox2[2]-5):
        flag = False
        #print('조건1')
    if bbox2[0]>(bbox1[0]+bbox1[2]-5): 
        flag = False
        #print('조건2')
    if bbox1[1]>(bbox2[1]+bbox2[3]-5): 
        flag = False
        #print('조건3')
    if bbox2[1]>(bbox1[1]+bbox1[3]-5): 
        flag = False
        #print('조건4')
        
    return flag
    

def related_mask_map(map_size, tar_mask, a_mask, other_mask_value=-3):
    """
    현재 bbox를 조정하려고 하는 물품과 그 주변에 bbox가 겹치는 물품을 포함해서 mask맵 생성하는부분
    배경은 0, 현재 bbox를 조정하려고 하는 물품은 1 주변에 bbox가 겹치는 물품은 -3 ~ -4이고,  디폴트는 -3으로 설정

    Args:
        map_size (tuple): 두점의 위치, [x1, y1, x2, y2]
        tar_mask (numpy): 현재 타켓물체의 mask 정보가 있는 map
        a_mask (numpy): 주변 가리는 물체의 mask 정보가 있는 map

    return:
        numpy : 위에 설명과 같은 형태의 mask 맵 [x][y]
    """
    mask_map = np.ones((map_size[3]-map_size[1], map_size[2]-map_size[0],3), dtype=np.int16)*other_mask_value[2]
    #check_map = np.zeros((map_size[3]-map_size[1], map_size[2]-map_size[0],3), dtype=np.uint8)
    t_m = tar_mask-[map_size[0],map_size[1]]
    cv2.drawContours(mask_map, [t_m], -1,(other_mask_value[1],other_mask_value[1],other_mask_value[1]), -1)
    #cv2.drawContours(check_map, [t_m], -1,(100,100,100), -1)
    for a in a_mask:
        a_m = a-[map_size[0],map_size[1]]
        cv2.drawContours(mask_map, [a_m], -1,(other_mask_value[0],other_mask_value[0],other_mask_value[0]), -1)
        #cv2.drawContours(check_map, [a_m], -1,(200,200,200), -1)
    
    #mask_map.astype(np.int32)
    #mask_map2 = np.where(mask_map==3, other_mask_value, mask_map)
    
    #return mask_map,check_map
    return mask_map

def re_cal_bbox(target, around_grid, around_box, region_num, around_object_value, re_cal_search_region=0.5):
    """
    실제 bbox를 다시 계산하는 부분 
    
    찾는 방식은 일단 
    1. 주변물체의 마스크의 영역 (ex: -3) 현재 타겟물체의 마스크 영역 1, 그외 영역 0으로 설정된 mask 맵 생성
    2. 실제로 물체의 감소할 좌표를 앞서 계산한 물체의 영역정보를 기반으로 설정
    대각(1,3,7,9 영역)이랑 4방향(위,아래,좌우, 2,4,6,8영역)
    3. x, y 축의 값을 줄이거나 늘여가면서 앞서 만든 mask 맵의 bbox 영역안의 값의 총합을 계산
    4. 반복하면서 값의 총합이 최대치인 값을 기준으로 bbox를 다시 설정

    Args: 
        target (dictionary): 현재 bbox를 수정할 물품 정보(bbox, mask 등의 정보 포함)
        around_box (list): 주변에 현재 물품을 가리는 물품의 bbox 정보
        region_num (int): 현재 target 물품이 해당되는 영역
        around_object_value (int): 주변 물품에 해당되는 mask영역 가중치 (-3이 디폴트)
        re_cal_search_region (float): bbox 수정시 얼만큼 for 구문으로 search 할지 기준

    return:
        list : 재 계산된 bbox [x, y, w, h]
    """
    
    #우선 마스크 맵을 먼저 만들어야함
    #맵 만들기전 영역크기부터설정 
    map_size = copy.deepcopy(target['bbox'])
    map_size[2] = map_size[2]+map_size[0]
    map_size[3] = map_size[3]+map_size[1]
    a_mask=[]
    for a in around_box:
        a_b = a['bbox']
        if map_size[0]>a_b[0]: map_size[0]=a_b[0]
        if map_size[1]>a_b[1]: map_size[1]=a_b[1]
        if map_size[2]<(a_b[0]+a_b[2]): map_size[2]=a_b[0]+a_b[2]
        if map_size[3]<(a_b[1]+a_b[3]): map_size[3]=a_b[1]+a_b[3]
        a_mask.append(a['mask'])
    
    #mask_map, check_map = related_mask_map(map_size, target['mask'], a_mask)
    mask_map = related_mask_map(map_size, target['mask'], a_mask, around_object_value)

    
    #이제 여기서 감소시켜가면서 결과비교 
    #감소할 값 설정은 앞서구했던 region_num으로 구함
    #대각(1,3,7,9 영역)이랑 4방향(위,아래,좌우, 2,4,6,8영역을 분리해서 계싼)
    #정중앙(5영역)은 계산X 어차피 불필요
    initial_value= np.sum(mask_map)
    re_bbox = target['bbox'].copy()
    if (region_num==1) or (region_num==3) or (region_num==7) or (region_num==9):
        #대각인경우
        if region_num==1:
            x_value=-1
            y_value=-1
        elif region_num==3:
            x_value=1
            y_value=-1
        elif region_num==7:
            x_value=-1
            y_value=1
        elif region_num==9:
            x_value=1
            y_value=1
        if x_value==-1:
            x_start = target['bbox'][2]+target['bbox'][0]-map_size[0]
            x_end = x_start-int(target['bbox'][2]*re_cal_search_region)
        else :
            x_start = target['bbox'][0]-map_size[0]
            x_end = x_start+int(target['bbox'][2]*re_cal_search_region)
        if y_value==-1:
            y_start = target['bbox'][3]+target['bbox'][1]-map_size[1]
            y_end = y_start-int(target['bbox'][3]*re_cal_search_region)
        else :
            y_start = target['bbox'][1]-map_size[1]
            y_end = y_start+int(target['bbox'][3]*re_cal_search_region)
        
        optimal_value = initial_value
        optimal_pos=[x_start,y_start]
        #이제 계산시작
        for y in range(y_start, y_end, y_value):
            for x in range(x_start, x_end, x_value):
                if x_value==-1: 
                    x1 = target['bbox'][0]-map_size[0]
                    x2 = x
                else :
                    x1 = x
                    x2 = target['bbox'][0]+target['bbox'][2]-map_size[0]
                if y_value==-1: 
                    y1 = target['bbox'][1]-map_size[1]
                    y2 = y
                else :
                    y1 = y
                    y2 = target['bbox'][1]+target['bbox'][3]-map_size[1]
                
                
                value = np.sum(mask_map[y1:y2,x1:x2,1])
                #print(x1,x2,y1,y2, value)
                #print('x시작:{}, x끝:{}, y시작:{}, y끝:{}, 합:{}'.format(x1,x2, y1, y2,value))
                if value>optimal_value:
                    optimal_value = value
                    optimal_pos[0] = x
                    optimal_pos[1] = y
        #결과 업데이트
        if x_value==-1: re_bbox[2]=optimal_pos[0]-re_bbox[0]+map_size[0]
        else : 
            re_bbox[2] = re_bbox[2]-(optimal_pos[0]+map_size[0]-re_bbox[0])
            re_bbox[0] = optimal_pos[0]+map_size[0]
        if y_value==-1: re_bbox[3]=optimal_pos[1]-re_bbox[1]+map_size[1]
        else : 
            re_bbox[3] = re_bbox[3]-(optimal_pos[1]+map_size[1]-re_bbox[1])
            re_bbox[1] = optimal_pos[1]+map_size[1]
        
    else:
        #여기서부터는  4방향
        #여기는 좌우, 위아래 2개로 쪼개서 계산
        #3방향을 고려하기위해서  물품을 2개의 사각형으로 나뉘어서 설정 
        #위아래
        #optimal_pos=[0,0,0]
        if (region_num==2) or (region_num==8):
            #위 아래는 x축으로 양방향 y축으로 한 방향  고려
            if region_num==2:
                y_value = -1
                y_start =  target['bbox'][3]+target['bbox'][1]-map_size[1]
                y_end =  y_start-int(target['bbox'][3]*re_cal_search_region)
            else : 
                y_value=1
                y_start = target['bbox'][1]-map_size[1]
                y_end = y_start+int(target['bbox'][3]*re_cal_search_region)
            #x축은 상자 2개로 설정
            #상자설정
            x1_start = target['bbox'][0]-map_size[0]
            x1_end = x1_start+int(target['bbox'][2]*re_cal_search_region*0.5)
            x2_start = target['bbox'][2]+target['bbox'][0]-map_size[0]
            x2_end = x2_start-int(target['bbox'][2]*re_cal_search_region*0.5)
            x_c = target['bbox'][0]+int(target['bbox'][2]*0.5)-map_size[0]

            optimal_pos=[x1_start, x2_start, y_start]
            optimal_x_pos=[x1_start, x2_start]
            optimal_value = initial_value*2
            #이제 계산시작
            for y in range(y_start, y_end, y_value):
                optimal_x = [initial_value,initial_value]
                if y_value==-1: 
                    y1 = target['bbox'][1]-map_size[1]
                    y2 = y
                else :
                    y1 = y
                    y2 = target['bbox'][1]+target['bbox'][3]-map_size[1]
                for x1 in range(x1_start, x1_end, 1):
                    x1_value = np.sum(mask_map[y1:y2,x1:x_c,1])
                    if x1_value>optimal_x[0]:
                        optimal_x[0] = x1_value
                        optimal_x_pos[0] = x1
                for x2 in range(x2_start, x2_end, -1):
                    x2_value = np.sum(mask_map[y1:y2,x_c:x2,1])
                    if x2_value>optimal_x[1]:
                        optimal_x[1] = x2_value
                        optimal_x_pos[1] = x2
                #print(optimal_x_pos)
                value = sum(optimal_x)
                #print(value)
                if value>optimal_value:
                    optimal_value = value
                    optimal_pos[0] = optimal_x_pos[0]
                    optimal_pos[1] = optimal_x_pos[1]
                    optimal_pos[2] = y
            #결과 업데이트

            re_bbox[0] = optimal_pos[0]+map_size[0]
            re_bbox[2] = optimal_pos[1]-optimal_pos[0]
            if y_value==-1: re_bbox[3]=optimal_pos[2]-re_bbox[1]+map_size[1]
            else : 
                re_bbox[3] = re_bbox[3]-(optimal_pos[2]+map_size[1]-re_bbox[1])
                re_bbox[1] = optimal_pos[2]+map_size[1]
        
        else:
            #여기는 좌우영역
            #위 아래랑 반대로 y축으로 양방향 x축으로 한 방향  고려
            if region_num==4:
                x_value = -1
                x_start =  target['bbox'][2]+target['bbox'][0]-map_size[0]
                x_end =  x_start-int(target['bbox'][2]*re_cal_search_region)
            else : 
                x_value= 1
                x_start = target['bbox'][0]-map_size[0]
                x_end = x_start+int(target['bbox'][2]*re_cal_search_region)
            #y축은 상자 2개로 설정
            #상자설정
            y1_start = target['bbox'][1]-map_size[1]
            y1_end = y1_start+int(target['bbox'][3]*re_cal_search_region*0.5)
            y2_start = target['bbox'][3]+target['bbox'][1]-map_size[1]
            y2_end = y2_start-int(target['bbox'][3]*re_cal_search_region*0.5)
            y_c = target['bbox'][1]+int(target['bbox'][3]*0.5)-map_size[1]
            
            optimal_pos=[x_start, y1_start, y2_start]
            optimal_y_pos=[y1_start, y2_start]
            optimal_value = initial_value*2
            #이제 계산시작
            for x in range(x_start, x_end, x_value):
                optimal_y = [initial_value,initial_value]
                if x_value==-1: 
                    x1 = target['bbox'][0]-map_size[0]
                    x2 = x
                else :
                    x1 = x
                    x2 = target['bbox'][0]+target['bbox'][2]-map_size[0]
                for y1 in range(y1_start, y1_end, 1):
                    y1_value = np.sum(mask_map[y1:y_c,x1:x2,1])
                    if y1_value>optimal_y[0]:
                        optimal_y[0] = y1_value
                        optimal_y_pos[0] = y1
                for y2 in range(y2_start, y2_end, -1):
                    y2_value = np.sum(mask_map[y_c:y2,x1:x2,1])
                    if y2_value>optimal_y[1]:
                        optimal_y[1] = y2_value
                        optimal_y_pos[1] =y2
                value = sum(optimal_y)
                if value>optimal_value:
                    optimal_value = value
                    optimal_pos[0] = x
                    optimal_pos[1] = optimal_y_pos[0]
                    optimal_pos[2] = optimal_y_pos[1]
            #결과 업데이트
            re_bbox[1] = optimal_pos[1]+map_size[1]
            re_bbox[3] = optimal_pos[2]-optimal_pos[1]
            if x_value==-1: re_bbox[2]=optimal_pos[0]-re_bbox[0]+map_size[0]
            else : 
                re_bbox[2] = re_bbox[2]-(optimal_pos[0]+map_size[0]-re_bbox[0])
                re_bbox[0] = optimal_pos[0]+map_size[0]

    #결과확인 
    #check_bbox = [re_bbox[0]-map_size[0],re_bbox[1]-map_size[1], re_bbox[2], re_bbox[3]]
    #cv2.rectangle(check_map, check_bbox, (255,0,255),1)
    #cv2.imshow('mask_result',check_map)
    #cv2.waitKey(0)
    return re_bbox
    
def revise_bbox(segment, batch_map, grid, image_data, around_object_value, re_cal_search_region):
    """
    bbox 자체는 이미 한번 계산이 되었으나, 겹쳐서 재수정이 필요한 부분을 수정하는 함수

    구성
    1. 필요한 정보를 다시 batch map 형식으로 재배치
    2. 각 물품 별로 현재 영역 계산-> 주변에 겹치는지 확인-> bbox 재 계산 순서로 진행

    Args:
        segment (dictionary): 앞서 한번 계산된 segmentation(bbox, mask)정보 
        batch_map (list): 물품이 배치된 map정보 [x][y]
        grid (tuple): 그리드 가로, 세로 정보 (x, y)
        image_data (numpy): 합성된 이미지 데이터
        around_object_value (int): 주변 물품에 해당되는 mask영역 가중치 (-3이 디폴트)
        re_cal_search_region (float): bbox 수정시 얼만큼 for 구문으로 search 할지 기준

    return:
        dictionary : 재계산된 합성 결과물

    """
    # 우선 re_seg를 일단 batch map 형식으로 재배치
    seg_batch_map = copy.deepcopy(batch_map)
    #print(batch_map)
    
    for seg, img_info in zip(segment, image_data):
        seg_batch_map[img_info['pos_x']][img_info['pos_y']] = seg
        seg_batch_map[img_info['pos_x']][img_info['pos_y']]['category_id'] = img_info['category']
        seg_batch_map[img_info['pos_x']][img_info['pos_y']]['area'] = img_info['area']
        seg_batch_map[img_info['pos_x']][img_info['pos_y']]['x'] = img_info['pos_x']
        seg_batch_map[img_info['pos_x']][img_info['pos_y']]['y'] = img_info['pos_y']
        seg_batch_map[img_info['pos_x']][img_info['pos_y']]['iteration'] = img_info['iteration']

    # 이제 여기서 부터 실제 겹치는 부분을 정리
    for seg, img_info in zip(segment, image_data):
        # 우선 실제로 주변에 물품끼리 붙어있는 영역이 있는지 부터 확인 
        # 방향중 현재 어느 방향을 보는게 맞는지 부터 확인
        g_x = img_info['pos_x']
        g_y = img_info['pos_y']
        region_num, search_pos = related_pos(g_x,g_y, grid[0],grid[1], batch_map)

        #정중앙인경우이고, 보통 무시해도 상관없긴함
        if region_num==5:
            break
        if len(search_pos)==0:
            continue
        re_bbox= seg_batch_map[g_x][g_y]['bbox']
        around_bbox=[]
        around_grid=[]
        for search in search_pos:
            #우선 bbox2개가 서로 겹치는지 부터 확인

            overlap_flag = check_overlap(seg_batch_map[g_x][g_y]['bbox'],seg_batch_map[g_x+search[0]][g_y+search[1]]['bbox'])
            if overlap_flag: 
                around_grid.append(search)
                around_bbox.append(seg_batch_map[g_x+search[0]][g_y+search[1]])
        if len(around_grid)==0:
            continue
        else:
            bbox_update=re_cal_bbox(seg_batch_map[g_x][g_y], around_grid, around_bbox, region_num, around_object_value, re_cal_search_region)
            seg_batch_map[g_x][g_y]['bbox'] = bbox_update
    
    # 이제 수정한 batch_map 기준으로 segmentation 정보 재정렬
    re_seg=[]
    for y in range(grid[1]):
        for x in range(grid[0]):
            if batch_map[x][y]!=0:
                seg1d=seg_batch_map[x][y]['mask'].ravel()
                seg_batch_map[x][y]['mask'] = seg1d.tolist()
                re_seg.append(seg_batch_map[x][y])
    
    return re_seg
