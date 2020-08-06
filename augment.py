import os
import sys
import numpy as np
import cv2
import random
import math
import copy
import json


def make_gridmap(dense, *args):
    '''
    물품 랜덤하게 선택하는부분
    다만 위치별로 랜덤하게 잡으면 물품 자체가 안잡히는 0이 되는 맵이 존재할 수 있으니
    0이 되는걸 없앨려고 따로 재귀함수 형태로 구현함
    입력 : 물품 배치 밀도
    출력 : 그리드맵
    '''
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

def array_DB_batch(grid, batch_map, array_method):
    '''
    클래스 밖의 함수로 단순히 물품의 합성 순서를 결정하는 부분 
    물품배치를 map 형태로 만들었지만 실제 물품 합성시 합성 순서는 물품의 순차적인 순서가 아님
    예를들면 일반 음료수나 세우는 물품은 중앙에서 가장 먼 위치부터 합성이 들어가야 함
    그리고 트레이가 필요한 눕혀진 물품은 맨 뒤부터 합성이 시작되어야 함(특정한 방향)
    따라서 이게 맞게 합성 순서를 다시 계산하는 함수를 추가해 놓음
    입력: grid정보, batch_map, 합성방식
    출력: 합성할 순서에 맞게 x,y  grid를 재 배열한 tuple
    '''
    # 합성 방법으로 가장 단순한 방법은 특정 위치와의 실제 거리를 계산하여 가장 먼 곳부터 합성을 진행
    if array_method==1: c_point = [(grid[0]-1)/2,(grid[1]-1)/2]
    elif array_method==2: c_point = [(grid[0]-1)/2,0] 
    #print(c_point)

    dis_info_list=[]
    for col in range(grid[0]):
        for row in range(grid[1]):
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
    '''
    단순히 mask 정보를 기반으로 물품의 중심좌표를 구하는 함수
    opencv로 간단하게 구현(opencv의 moments 함수)
    입력: mask 정보
    출력 : 물품 중심좌표(tuple)
    '''
    M = cv2.moments(mask)
                
    #물체의 중심 좌표
    obj_cx = int(M['m10'] / M['m00'])
    obj_cy = int(M['m01'] / M['m00'])
    return [obj_cx, obj_cy]

def cal_mask(obj_map, ori_area, area_ratio_th = 0.06):
    '''
    mask를 가려진부분 제외하고 실제로 진짜 보이는 부분으로 다시 얻어냄
    mask_map을 사전에 만들어 두었고 거기서 현재 물품을 제외한 다른 물품은 전부 0이므로 
    단순히 opencv의 findconours 함수로 간단하게 구현이 가능
    입력: obj_map(실제 그 물체만 255이고, 다른 부분은 전부 0으로 된 이진화된 이미지 )
    출력 : 다시 계산된 mask 정보(array 형식)
    '''
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
    if a_ratio<area_ratio_th:
        print('mask 크기가 너무 작음')
        return [[-1,-1]], 0

    mask = np.reshape(m,(m.shape[0],2))
    return mask, re_area

def cal_bbox(obj_map, mask, center, threshold):
    '''
    bbox를 다시 계산하는 함수
    bbox의 경우 실제 물품전체가 아닌 윗부분만 잘라서 검출한다고 가정하기 때문에 
    그 윗부분을 계산하는게 연산량이 상당히 차지함
    입력 : obj_map(위에 마스크에 쓰인것과 동일), mask(기존 마스크 정보),
    center(밑의 배치하는 판 기준으로 중심점), threshold(총 4개의 기준 존재)
    출력 : 다시 계산된 bbox 정보(x,y,w,h 순서)
    계산과정이 좀 복잡하고 길기 때문에 아래에 순차적으로 정리함
    과정 
    1. 우선 물품의 중심좌표를 계산
    2. 물품과 중심점의 기울기(각도)를 계산
    3. 물품의 영역을 따옴
    (다만 따올때 회전을 다음에 진행할때 이미지 영역을 벗어나면 안되기 때문에 적당히 크기를 조절해서 가져와야함)
    4. 물품의 영역을 -각도로 회전(이러면 맨 아래가 물품의 윗부분으로 회전됨)
    5. 실제 bbox만 남기기 위해서 잘라내기
    물품이 수직방향이기 때문에 threshold 기준을 y축을 기준으로 잡으면 됨
    영역을 잘라낼 때 밑부분부터 물체가 영역이 실제 얼마나 겹치는지 계산하면서 고려
    (threshold는 총 2개, 2개 4개를 쓰는데, 기준1에서 2개,기준2에서 2개, 그리고 각도별로 threshold가 다르게 설정됨)
    6. 잘라낸 영역을 기준으로 다시 원래 상태로 재 회전
    7. 다시 회전된 영역에서 박스정보를 계산(contour 계산 + contour 내에서 외접하는 박스 계산)
    '''
    
    # 1. 물품 중심좌표
    obj_center = cal_obj_center(mask)
    
    # 2. 물품과 중심점의 기울기(각도)를 계산
    if obj_center[1]==center[1]:
        obj_center[1]=obj_center[1]+1
    angle = -math.atan2((obj_center[0]-center[0]), (obj_center[1]-center[1]))
    
    # 3. 물품의 영역을 가져오기 
    # 그 전에 변환을 했을때 물품의 영역이 얼만큼 차지하는지를 우선적으로 알아야함
    # 즉 mask상에서 점들을 회전을 시켜서 그 mask 점들중 최대값을 계산
    # 밑에 타원의 크기 계산하는 방식과 동일함
    
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
    
    #(1) 일단 1차 기준의 경우 1차 기준 밑으로는 다 잘라냄(예외없이 싹뚝) 
    cut_length1 = int((rotate_size[4] * (threshold[0][0] + abs(abs(abs(degree)-90) - 45) / 45 * threshold[0][1])))
    mask_cut1 = int(ro_img_h/2+rotate_size[3]-obj_center[1] - cut_length1)
    if cut_length1<rotate_size[5] :
        min_y = int(ro_img_h/2+rotate_size[1]-obj_center[1])-5
        if min_y<0 : min_y=0
        rotate_region[min_y:mask_cut1] = np.zeros((mask_cut1-min_y, ro_img_w, 3), dtype=np.uint8)
    #else : 
    #    print('1차기준 안에 물품이 다 포함됨')
    #    mask_cut1 = int(ro_img_h/2+region_size[1]-obj_center[1])-5
    #    if mask_cut1<0 : mask_cut1=0
    
    cut_length2 = int(rotate_size[4] * (threshold[1][0] + abs(abs(abs(degree)-90)- 45) / 45 * threshold[1][1]))
    mask_cut2 = int(ro_img_h/2+rotate_size[3]-obj_center[1] - cut_length2)
        
    #(2) 이제 한줄씩 읽어가면서 물품의 영역이 얼만큼 차지하는지 비율에 따라서 잘라낼지 아닐지를 결정
    for y in range(mask_cut1-1, mask_cut2, 1):
        aver_value = np.sum(obj_region[y])/rotate_size[4]
        if aver_value >128:
            if (y-mask_cut1+1)<0:
                print("합성쪽 에러")
                print('mask_cu1:{0}, mask_cut2:{1}, y:{2}'.format(mask_cut1, mask_cut2, y))
            rotate_region[mask_cut1-1 : y ] = np.zeros((y-mask_cut1+1, ro_img_w, 3), dtype=np.uint8)
            break
         
    # 8. 이제 원래로 다시 역 변환 

    ro_m3 = cv2.getRotationMatrix2D((ro_img_w/2,ro_img_h/2), -degree, 1)
    re_obj_region = cv2.warpAffine(rotate_region, ro_m3,(ro_img_w, ro_img_h))
    
    # 9. 실제 최종 bbox 계산
    binary_map = cv2.cvtColor(re_obj_region, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS )
    
    fit_bbox = cv2.boundingRect(contours[0])
    
    re_bbox = [int(fit_bbox[0]-ro_img_w/2+obj_center[0]), int(fit_bbox[1]-ro_img_h/2+obj_center[1]),fit_bbox[2], fit_bbox[3]]
    
    return re_bbox


def related_pos(g_x, g_y, x_num, y_num, batch_map):
    '''
    그리드 위치, batch_map을 보고 실제 어떤 방향만 고려해도 되는지 확인하는부분
    그리고 현재 물품이 대략적으로 어느 위치에 있는지도 확인하기위한 용도 
    1~9까지 구역이 있다고 가정하고 실제 구역에 따라서 연산이 달라지기 때문에 그거 확인이 필요함
    입력: 현재 물체의 grid x,y좌표, grid 전체 x, y축 크기, batch_map
    출력: 고려 가능한 물체 위치 리스트 
    '''
    region_table = [[1,4,7],[2,5,8],[3,6,9]]
    region_table = [[3,6,9],[2,5,8],[1,4,7]]
    
    re_pos = []
    half_x = (x_num-1)/2
    half_y = (y_num-1)/2
    
    # 각 방향별로 조건달아서 선택
    # 그리고 실제 batch_map에서 그 위치에 물체가 있는지를 확인
    # 왼쪽위
    if (g_x>=half_x) and (g_y>=half_y):
        if batch_map[g_x-1][g_y-1]!=0:
            re_pos.append([-1,-1])
    # 위
    if g_y>=half_y:
        if batch_map[g_x][g_y-1]!=0:
            re_pos.append([0,-1])
    # 오른쪽위
    if (g_x<=half_x) and (g_y>=half_y):
        if batch_map[g_x+1][g_y-1]!=0:
            re_pos.append([1,-1])
    # 왼쪽
    if g_x>=half_x:
        if batch_map[g_x-1][g_y]!=0:
            re_pos.append([-1,0])
    # 오른쪽
    if g_x<=half_x:
        if batch_map[g_x+1][g_y]!=0:
            re_pos.append([1,0])
    # 왼쪽아래
    if (g_x>=half_x) and (g_y<=half_y):
        if batch_map[g_x-1][g_y+1]!=0:
            re_pos.append([-1,1])
    # 위
    if g_y<=half_y:
        if batch_map[g_x][g_y+1]!=0:
            re_pos.append([0,1])
    # 오른쪽아래
    if (g_x<=half_x) and (g_y<=half_y):
        if batch_map[g_x+1][g_y+1]!=0:
            re_pos.append([1,1])
    
    #여기는 현재 물체가 어느 구역인지 확인(1~9구역)
    r_x = g_x-half_x
    r_y = g_y-half_y
    if r_x!=0: r_x = r_x//abs(r_x)+1
    else : r_x=1
    if r_y!=0: r_y = r_y//abs(r_y)+1
    else : r_y=1
    region_num = region_table[int(r_x)][int(r_y)]
    # re_pos에서 필요한 것만 확인한거니 그걸 반환
    return region_num,re_pos

def check_overlap(bbox1, bbox2):
    #2개의 박스를 비교해서 겹치는지 여부를 확인   
    flag = True
    
    #print('박스2개 겹침비교')
    #print(bbox1)
    #print(bbox2)
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
    
def related_mask_map(map_size,tar_mask, a_mask, other_mask_value=-3):
    '''
    현재 bbox를 조정하려고 하는 물품과 그 주변에 bbox가 겹치는 물품을 포함해서 mask맵 생성하는부분
    배경은 0, 현재 bbox를 조정하려고 하는 물품은 1
    주변에 bbox가 겹치는 물품은 -3 ~ -4 정도가 좋으며 디폴트는 -3으로 설정
    '''
    mask_map = np.zeros((map_size[3]-map_size[1], map_size[2]-map_size[0],3), dtype=np.int16)
    #check_map = np.zeros((map_size[3]-map_size[1], map_size[2]-map_size[0],3), dtype=np.uint8)
    t_m = tar_mask-[map_size[0],map_size[1]]
    cv2.drawContours(mask_map, [t_m], -1,(1,1,1), -1)
    #cv2.drawContours(check_map, [t_m], -1,(100,100,100), -1)
    for a in a_mask:
        a_m = a-[map_size[0],map_size[1]]
        cv2.drawContours(mask_map, [a_m], -1,(other_mask_value,other_mask_value,other_mask_value), -1)
        #cv2.drawContours(check_map, [a_m], -1,(200,200,200), -1)
    
    #mask_map.astype(np.int32)
    #mask_map2 = np.where(mask_map==3, other_mask_value, mask_map)
    
    #return mask_map,check_map
    return mask_map

def re_cal_bbox(target, around_grid, around_box, region_num, around_object_value, re_cal_search_region=0.5):
    '''
    접근 방식을 바꿈
    아예 주변 물체의 마스크를 덮어씌운 중간 이상 크기의 맵을 따로 만든다음 
    타겟 물체와, 주변 물체만 있는 map을 가지고
    마스크 맵에서 target의 크기를 줄일 수 있는 방향으로(좌우로 하나 위아래로 하나)
    순차적으로 줄여가면서(한 절반정도) 적절한 크기를 찾음
    찾는 방식은 일단 1. 주변물체의 마스크의 영역이 가하는폭/현재 타겟물체의 마스크 영역이 감소하는 폭
    '''
    
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
    #결과확인용
    #cv2.imshow('mask_map',check_map)
    #cv2.waitKey(1)
    
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
            x_c = target['bbox'][0]+int(target['bbox'][2]*re_cal_search_region)-map_size[0]

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
            y_c = target['bbox'][1]+int(target['bbox'][3]*re_cal_search_region)-map_size[1]
            
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
    check_bbox = [re_bbox[0]-map_size[0],re_bbox[1]-map_size[1], re_bbox[2], re_bbox[3]]
    #cv2.rectangle(check_map, check_bbox, (255,0,255),1)
    #cv2.imshow('mask_result',check_map)
    #cv2.waitKey(0)
    return re_bbox
    
def revise_bbox(segment, batch_map, grid, image_data, around_object_value, re_cal_search_region):
    '''
    bbox 자체는 이미 한번 계산이 되었으나, 문제가 발생될 것 같아서 재수정이 필요한 부분을 따로 추가함
    즉 bbox가 이미 다시 계산이 되었는데도 서로 겹치는 경우를 여기에 작성
    입력: 계산된 segmentation(bbox, mask), batch map, grid정보
    출력: re_seg에서 bbox만 다시 수정된 리스트 형식
    우선 물품의 배치형태가 정리된 batch map을 기반으로 딕셔너리 형식이 각 grid 위치별로 들어간 형태로 구성된 list 파일이 필요함
    그리고 batch map을 보고 주변 방향에서 겹치는게 있는지를 확인해서 정리
    주변 방향은 8방향 전부를 확인하는건 비효율적(그리고 중복 검출 할수도 있어서 방향제한을 하는게 효율이 나음)
    즉 물품 위치에 따라서 방향을 어디를 얼만큼 볼지 결정함
    그리고 실제 조정은 인접한 물품을 확인하고 난 다음 2개를 비교하는 형태로 진행
    그 후 find_overlapped_box로 검출영역 확인하고
    실제 수정은 revise_bbox에서 수정된 bbox 2개를 얻어내는 형태로 진행
    '''
    # 우선 re_seg를 일단 batch map 형식으로 재배치
    seg_batch_map = copy.deepcopy(batch_map)
    #print(batch_map)
    
    for seg, img_info in zip(segment, image_data):
        seg_batch_map[img_info['grid_x']][img_info['grid_y']] = seg
        seg_batch_map[img_info['grid_x']][img_info['grid_y']]['category_id'] = img_info['category']
        seg_batch_map[img_info['grid_x']][img_info['grid_y']]['area'] = img_info['area']
        seg_batch_map[img_info['grid_x']][img_info['grid_y']]['x'] = img_info['grid_x']
        seg_batch_map[img_info['grid_x']][img_info['grid_y']]['y'] = img_info['grid_y']
        seg_batch_map[img_info['grid_x']][img_info['grid_y']]['iteration'] = img_info['iteration']

    # 이제 여기서 부터 실제 겹치는 부분을 정리
    for seg, img_info in zip(segment, image_data):
        # 우선 실제로 주변에 물품끼리 붙어있는 영역이 있는지 부터 확인 
        # 방향중 현재 어느 방향을 보는게 맞는지 부터 확인
        g_x = img_info['grid_x']
        g_y = img_info['grid_y']
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
                map_possible = make_gridmap(self.object_dense, self.grid[0])
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
                map_possible = make_gridmap(self.object_dense, self.grid)
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
                
                obj_center = cal_obj_center(mask_np)
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
            obj_cal_mask, area = cal_mask(obj_map,img_info['area'], self.delete_ratio_th)
            if obj_cal_mask[0][0]==-1:
                #print('삭제됨')
                #print('삭제된 위치:{}, {}'.format(img_info['grid_x'],img_info['grid_y']))
                self.batch_map[img_info['grid_x']][img_info['grid_y']]=0
                deleted_info.append(img_info)
                continue
            
            # bbox계산
            obj_cal_bbox = cal_bbox(obj_map, obj_cal_mask, img_center, threshold)
            
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
        re_seg = revise_bbox(cal_seg, self.batch_map, self.grid, self.image_data, self.around_object_weight, self.re_cal_search_region)
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
