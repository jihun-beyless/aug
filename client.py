from __future__ import print_function
import logging

import grpc

import aug_protobuf_pb2
import aug_protobuf_pb2_grpc

import cv2 

def run(de_id, g, g_id, obj_cate, bg_id, iter_num, b_num, b_param):
    #object_category의 경우 list 파일이라 str로 변경후 넘김
    str_cate = ' '.join(map(str, obj_cate))
    bright = ' '.join(map(str,b_param))
    
    with grpc.insecure_channel('192.168.10.52:50051') as channel:
        stub = aug_protobuf_pb2_grpc.TransactionAugmentStub(channel)
        
        result_response = stub.SendAugmentData(aug_protobuf_pb2.AugType(device_id = de_id, grid_x = g[0], grid_y = g[1], grid_id = g_id, \
            object_category = str_cate, background_id = bg_id, iteration = iter_num, batch_num1 = b_num[0], batch_num2= b_num[1], batch_num3 = b_num[2], bright_param = bright))

        
    #print('------------------gRPC & protobuf3 test------------------')
    print('gRPC결과 bool타입값 : {}'.format(result_response.result))
    #print('------------------end of test------------------')
        
        

if __name__ == '__main__':
    device_id = 20001
    grid = (6,5)
    grid_id = 3
    object_category=[1, 3]
    background_id = 348
    iteration = 3
    batch_num = [2, 2, 2]
    # bright_param : [bright_flag, mode_flag, flag1, flag2, flag3, th1, th2, th3, rect x, rect y, rect w, rect h] 
    bright_param = [1, 1, 1, 1, 1, 78, 36, 113, 1140, 440, 100, 200]
    run(device_id, grid, grid_id, object_category, background_id, iteration, batch_num, bright_param)
