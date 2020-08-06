from concurrent import futures
import logging

import grpc

import aug_protobuf_pb2
import aug_protobuf_pb2_grpc

import augmain


class TransactionAugment(aug_protobuf_pb2_grpc.TransactionAugmentServicer):
    def __init__(self):
        self.call_counter = 1

    def SendAugmentData(self, AugType, context):
        test_str = 'device_id :{}, grid: ({},{}), grid_id: {}, obj_category: {}, background_id: {}, iteration: {}, batch_num : ({},{},{}), brigth_parameter : {}'.format(AugType.device_id, \
            AugType.grid_x, AugType.grid_y, AugType.grid_id, AugType.object_category, AugType.background_id, AugType.iteration, \
                AugType.batch_num1, AugType.batch_num2, AugType.batch_num3, AugType.bright_param)
        file = open('aug_data_info.txt', 'w')
        file.write(test_str)
        file.close()
        obj_cate = list(map(int,AugType.object_category.split(' ')))
        b_param = list(map(int, AugType.bright_param.split(' ')))
        batch = [AugType.batch_num1, AugType.batch_num2, AugType.batch_num3]
        g = (AugType.grid_x, AugType.grid_y)
        aug_flag = augmain.aug_main(device_id = AugType.device_id, grid = g, grid_id = AugType.grid_id, object_category = obj_cate, background_id = AugType.background_id, iteration= AugType.iteration, batch_num = batch, bright_param = b_param)
        return aug_protobuf_pb2.BoolType(result = aug_flag)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    aug_protobuf_pb2_grpc.add_TransactionAugmentServicer_to_server(TransactionAugment(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
