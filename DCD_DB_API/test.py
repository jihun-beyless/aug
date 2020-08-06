# -*- coding: utf-8 -*-
from db_api.DB import DB
from db_api.DB import *
from os import listdir
from os.path import join
from db_api.utils.util import cpu_mem_check
from db_api.utils.img_util import img_loader
from functools import wraps

import time
import os


# decorator
def print_basic(func):
    @wraps(func)
    def show(self, *method_args, **method_kwargs):
        func(self)
        print(func.__name__[1].upper() + '{} table: '.format(func.__name__[2:]), self.table)
        print(func.__name__[1].upper() + '{} table last id: '.format(func.__name__[2:]), self.last_id)

    return show


# decorator
def print_check(func):
    @wraps(func)
    def show(self, *method_args, **method_kwargs):
        self._reset_table()
        self.ans = None
        func(self)
        print("{} 결과:".format(func.__name__[1:]), self.ans)

    return show


# decorator
def mem_check(func):
    @wraps(func)
    def mem(self, *method_args, **method_kwargs):
        func(self)
        cpu_mem_check()

    return mem


# decorator
def time_check(func):
    @wraps(func)
    def tc(self, *method_args, **method_kwargs):
        start_time = time.time()
        func(self)
        cpu_mem_check()
        end_time = time.time()
        print('total_time: ', end_time - start_time)

    return tc


def reset_table(db):
    db.drop_table(table='Bbox')
    db.drop_table(table='Mask')
    db.drop_table(table='Object')
    db.drop_table(table='Image')
    db.drop_table(table='Location')
    db.drop_table(table='Category')
    db.drop_table(table='Environment')
    db.drop_table(table='Grid')
    db.drop_table(table='SuperCategory')


class ResetTable:
    def __init__(self, db):
        self.db = db

    def _reset_table(self):
        reset_table(self.db)
        self.db.init_table_no_print()
        self.db.set_environment(env_id='20001', broker_ip="111.111.111.111",
                                floor="1", width="1", height="1", depth="1")
        self.db.set_supercategory(super_name='1')
        self.db.set_grid(width='1', height='1')
        self.db.set_image(env_id='20001', img='1', type='0', check_num='1')
        self.db.set_location(grid_id='1', x='1', y='1')
        self.db.set_category(super_id='1', cat_name='1', width='1', height='1', depth='1', iteration='1', thumbnail='1')
        self.db.set_object(loc_id='1', cat_id='1', img_id='1', iteration='1', mix_num='-1')
        self.db.set_bbox(obj_id='1', x='1', y='1', width='1', height='1')
        self.db.set_mask(obj_id='1', x='1', y='1')


class CheckBasic:
    def __init__(self, db):
        self.db = db
        self.db.init_table()
        self.table = None
        self.last_id = None

    def check_all(self):
        self._environment()
        self._supercategory()
        self._grid()
        self._image()
        self._location()
        self._category()
        self._object()
        self._bbox()
        self._mask()

    @print_basic
    def _environment(self):
        # check environment fucntions
        self.db.set_environment(env_id="20001", broker_ip="111.111.111.111",
                                floor="1", width="1", height="1", depth="1")
        self.db.get_table(id='1', table='Environment')
        # self.db.delete_table(id='20001', table='Environment')
        self.db.update_environment(env_id='20001', broker_ip='112.112.112.122')
        self.table = self.db.list_table(table='Environment')
        self.last_id = self.db.get_last_id(table="Environment")

    @print_basic
    def _supercategory(self):
        self.db.set_supercategory(super_name='1')
        self.db.get_table(id='1', table='SuperCategory')
        # self.db.delete_table(id='1', table='SuperCategory')
        self.db.update_supercategory(super_id='1', super_name='1')
        self.table = self.db.list_table(table='SuperCategory')
        self. last_id = self.db.get_last_id(table='SuperCategory')

    @print_basic
    def _grid(self):
        self.db.set_grid(width='1', height='1')
        self.db.get_table(id='1', table='Grid')
        # self.db.delete_table(id='1', table='Grid')
        self.db.update_grid(grid_id='1', width='1')
        self.table = self.db.list_table(table='Grid')
        self.last_id = self.db.get_last_id(table='Grid')

    @print_basic
    def _image(self):
        self.db.set_image(env_id='20001', img='1', type='0', check_num='1')
        self.db.get_table(id='1', table='Image')
        # self.db.delete_table(id='1', table='Image')
        self.db.update_image(img_id='1', env_id='20001')
        self.table = self.db.list_table(table='Image')
        self.last_id = self.db.get_last_id(table='Image')

    @print_basic
    def _location(self):
        self.db.set_location(grid_id='1', x='1', y='1')
        self.db.get_table(id='1', table='Location')
        # self.db.delete_table(id='1', table='Location')
        self.db.update_location(loc_id='1', x='1')
        self.table = self.db.list_table(table='Location')
        self.last_id = self.db.get_last_id(table='Location')

    @print_basic
    def _category(self):
        self.db.set_category(super_id='1', cat_name='1', width='1', height='1', depth='1', iteration='1', thumbnail='1')
        self.db.get_table(id='1', table='Category')
        # self.db.delete_table(id='1', table='Category')
        self.db.update_category(cat_id='1', cat_name='1')
        self.table = self.db.list_table(table='Category')
        self.last_id = self.db.get_last_id(table='Category')

    @print_basic
    def _object(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='1', mix_num='-1')
        self.db.get_table(id='1', table='Object')
        # self.db.delete_table(id='1', table='Object')
        self.db.update_object(obj_id='1', loc_id='1')
        self.table = self.db.list_table(table='Object')
        self.last_id = self.db.get_last_id(table='Object')

    @print_basic
    def _bbox(self):
        self.db.set_bbox(obj_id='1', x='1', y='1', width='1', height='1')
        self.db.get_table(id='1', table='Bbox')
        # self.db.delete_table(id='1', table='Bbox')
        self.db.update_bbox(bbox_id='1', x='1')
        self.table = self.db.list_table(table='Bbox')
        self.last_id = self.db.get_last_id(table='Bbox')

    @print_basic
    def _mask(self):
        self.db.set_mask(obj_id='1', x='1', y='1')
        self.db.get_table(id='1', table='Mask')
        # self.db.delete_table(id='1', table='Mask')
        self.db.update_mask(mask_id='1', x='1')
        self.table = self.db.list_table(table='Mask')
        self.last_id = self.db.get_last_id(table='Mask')


class CheckGet(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------get 함수 결과--------------------------------")
        self._get_grid_id()
        self._get_super_id_SN()
        self._get_super_id_CI()
        self._get_super_name()
        self._get_loc_id()
        self._get_loc_id_GL()
        self._get_loc()
        self._get_cat_id_SI()
        self._get_cat_id_SN()
        self._get_img_id()
        self._get_cat_id_obj()
        self._get_obj_id_img()
        self._get_obj_id()
        self._get_bbox_id()
        self._get_bbox()
        self._get_bbox_img()
        self._get_mask_id()
        self._get_obj_max_iter()
        self._get_bbox_img_id()
        self._get_obj_max_aug()
        self._get_obj_id_args()
        self._get_mask_obj_id()
        self._get_max_mix_num()
        self._get_max_G_size()
        self._get_obj_id_cat_id_NULL()
        self._get_img_id_type()

    @print_check
    def _get_grid_id(self):
        self.ans = self.db.get_grid_id(grid_w_h='1x1')

    @print_check
    def _get_super_id_SN(self):
        self.ans = self.db.get_super_id_SN(super_name='1')

    @print_check
    def _get_super_id_CI(self):
        self.ans = self.db.get_super_id_CI(cat_id='1')

    @print_check
    def _get_super_name(self):
        self.ans = self.db.get_super_name(super_id='1')

    @print_check
    def _get_loc_id(self):
        self.ans = self.db.get_loc_id(grid_id='1', loc_x_y='1x1')

    @print_check
    def _get_loc_id_GL(self):
        self.db.set_grid(width='1', height='2')
        self.db.set_grid(width='1', height='3')
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_location(grid_id='1', x='1', y='3')
        self.db.set_location(grid_id='1', x='1', y='4')
        self.db.set_location(grid_id='1', x='1', y='5')
        self.ans = self.db.get_loc_id_GL(grid_w_h='1x1', loc_x_y='1x1')

    @print_check
    def _get_loc(self):
        self.db.set_location(grid_id='1', x='1', y='2')
        self.ans = self.db.get_loc(grid_id='1')

    @print_check
    def _get_cat_id_SI(self):
        self.ans = self.db.get_cat_id_SI(super_id='1', cat_name='1')

    @print_check
    def _get_cat_id_SN(self):
        self.ans = self.db.get_cat_id_SN(super_name='1', cat_name='1')

    @print_check
    def _get_img_id(self):
        self.ans = self.db.get_img_id(obj_id='1')

    @print_check
    def _get_cat_id_obj(self):
        self.ans = self.db.get_cat_id_obj(obj_id='1')

    @print_check
    def _get_obj_id_img(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='3', mix_num='-1')
        self.ans = self.db.get_obj_id_img(img_id='1')

    @print_check
    def _get_obj_id(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='3', mix_num='-1')
        self.ans = self.db.get_obj_id(cat_id='1')

    @print_check
    def _get_bbox_id(self):
        self.db.set_bbox(obj_id='1', x='1', y='2', width='1', height='1')
        self.ans = self.db.get_bbox_id(obj_id='1')

    @print_check
    def _get_bbox(self):
        self.db.set_bbox(obj_id='1', x='1', y='2', width='1', height='1')
        self.ans = self.db.get_bbox(obj_id='1')

    @print_check
    def _get_bbox_img(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='1', mix_num='1')
        self.db.set_bbox(obj_id='1', x='1', y='2', width='1', height='1')
        self.db.set_bbox(obj_id='1', x='1', y='3', width='1', height='1')
        self.ans = self.db.get_bbox_img(img_id='1')

    @print_check
    def _get_mask_id(self):
        self.db.set_mask(obj_id='1', x='1', y='2')
        self.ans = self.db.get_mask_id(obj_id='1')

    @print_check
    def _get_obj_max_iter(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='1', mix_num='1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='3', mix_num='1')
        self.ans = self.db.get_obj_max_iter()

    @print_check
    def _get_bbox_img_id(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='1')
        self.db.set_bbox(obj_id='1', x='1', y='2', width='1', height='1')
        self.db.set_bbox(obj_id='2', x='1', y='3', width='1', height='1')
        self.ans = self.db.get_bbox_img_id(img_id='1')

    @print_check
    def _get_obj_max_aug(self):
        self.ans = self.db.get_obj_max_aug()

    @print_check
    def _get_obj_id_args(self):
        self.ans = self.db.get_obj_id_args(loc_id='1', cat_id='1', iteration='1', mix_num='-1', aug_num='-1')

    @print_check
    def _get_mask_obj_id(self):
        self.db.set_mask(obj_id='1', x='1', y='2')
        self.db.set_mask(obj_id='1', x='1', y='3')
        self.db.set_mask(obj_id='1', x='1', y='4')
        self.ans = self.db.get_mask_obj_id(obj_id='1')

    @print_check
    def _get_max_mix_num(self):
        self.db.set_location(grid_id='1', x='0', y='0')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='0')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='1')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='2')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='3')
        self.ans = self.db.get_max_mix_num()

    @print_check
    def _get_max_G_size(self):
        self.db.set_grid(width='5', height='6')
        self.db.set_grid(width='6', height='6')
        self.db.set_grid(width='7', height='7')
        self.db.delete_table(id='1', table='Grid')
        self.ans = self.db.get_max_G_size()

    @print_check
    def _get_obj_id_cat_id_NULL(self):
        self.db.set_mix_obj_list(max_w='2', max_h='2', loc_id='1', iteration='2', mix_num='-1', aug_num='-1')
        self.ans = self.db.get_obj_id_cat_id_NULL(loc_id='1', iteration='1', mix_num='-1', aug_num='-1')

    @print_check
    def _get_img_id_type(self):
        self.db.set_image(env_id='20001', img='1', type='1', check_num='1')
        self.db.set_image(env_id='20001', img='1', type='1', check_num='1')
        self.ans = self.db.get_img_id_type(type='1')


class CheckList(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------list 함수 결과--------------------------------")
        self._list_bbox()
        self._list_obj()
        self._list_img_id_TC()
        self._list_obj_CN()
        self._list_obj_CN_NULL()

    @print_check
    def _list_bbox(self):
        self.db.set_bbox(obj_id='1', x='1', y='2', width='1', height='1')
        self.ans = self.db.list_bbox(obj_id='1')

    @print_check
    def _list_obj(self):
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_location(grid_id='1', x='1', y='3')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='0')
        self.db.set_object(img_id='1', loc_id='3', cat_id='1', iteration='1', mix_num='1')
        self.ans = self.db.list_obj(cat_id='1', loc_ids=('1', '2', '3'))

    @print_check
    def _list_img_id_TC(self):
        self.db.set_image(env_id='20001', img='1', type='1', check_num='1')
        self.db.set_image(env_id='20001', img='1', type='1', check_num='1')
        self.ans = mydb.list_img_id_TC(type='1', check_num='1')

    @print_check
    def _list_obj_CN(self):
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_image(env_id='20001', img='22', type='1', check_num='1')
        self.db.set_image(env_id='20001', img='22', type='1', check_num='1')
        self.db.set_object(img_id='2', loc_id='2', cat_id='1', iteration='1', mix_num='-1')
        self.db.set_object(img_id='3', loc_id='2', cat_id='1', iteration='2', mix_num='-1')
        self.ans = self.db.list_obj_CN(grid_id='1', cat_id='1', check_num='1', type='1')

    @print_check
    def _list_obj_CN_NULL(self):
        self.db.set_image(env_id='20001', img='1', type='1', check_num='1')
        self.db.set_object(img_id='1', loc_id='1', cat_id=None, iteration='1', mix_num='-1', aug_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id=None, iteration='1', mix_num='-1', aug_num='-1')
        self.db.set_object(img_id='2', loc_id='1', cat_id=None, iteration='1', mix_num='-1', aug_num='-1')
        self.ans = self.db.list_obj_CN_NULL(grid_id='1', check_num='1')


class CheckCheck(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------check 함수 결과--------------------------------")
        self._check_obj_id()
        self._check_nomix_OBM()
        self._check_cat_id()

    @print_check
    def _check_obj_id(self):
        self.ans = self.db.check_obj_id(loc_id='1', cat_id='1', iteration='1', mix_num='-1')

    @print_check
    def _check_nomix_OBM(self):
        self.ans = self.db.check_nomix_OBM(cat_id='1')

    @print_check
    def _check_cat_id(self):
        self.ans = self.db.check_cat_id(super_name='1', cat_name='1')


class CheckUpdate(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------update 함수 결과--------------------------------")
        self._update_img_CN_OI()
        self._update_img_CN_II()
        self._update_img_img_OI()
        self._update_img_img_II()
        self._update_mix_obj_img()
        self._update_mix_img_obj()

    @print_check
    def _update_img_CN_OI(self):
        self.ans = self.db.update_img_CN_OI(obj_id='1', check_num='1')

    @print_check
    def _update_img_CN_II(self):
        self.ans = self.db.update_img_CN_II(img_id='1', check_num='1')

    @print_check
    def _update_img_img_OI(self):
        self.ans = self.db.update_img_img_OI(obj_id='1', img='1')

    @print_check
    def _update_img_img_II(self):
        self.ans = self.db.update_img_img_II(img_id='1', img='1')

    @print_check
    def _update_mix_obj_img(self):
        self.db.set_image(env_id='20001', img='100', type='1', check_num='2')
        self.db.set_object(img_id=None, loc_id='1', cat_id=None, iteration='1', mix_num='-1')
        self.ans = self.db.update_mix_obj_img(iteration='1', img_id='2')

    @print_check
    def _update_mix_img_obj(self):
        self.db.set_image(env_id='20001', img='100', type='1', check_num='2')
        self.db.set_object(img_id='1', loc_id='1', cat_id=None, iteration='1', mix_num='-1')
        self.db.set_object(img_id='2', loc_id='1', cat_id=None, iteration='1', mix_num='-1')
        self.ans = self.db.update_mix_img_obj(iteration='1', img='99999999999999')


class CheckDelete(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------delete 함수 결과--------------------------------")
        self._delete_object()
        self._delete_bbox()
        self._delete_mask()
        self._delete_bbox_img()
        self._delete_nomix_img()
        self._delete_empty_obj()
        self._delete_mix_obj()

    @print_check
    def _delete_object(self):
        self.ans = self.db.delete_object(img_id='1')

    @print_check
    def _delete_bbox(self):
        self.ans = self.db.delete_bbox(obj_id='1')

    @print_check
    def _delete_mask(self):
        self.ans = self.db.delete_mask(obj_id='1')

    @print_check
    def _delete_bbox_img(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='3', mix_num='-1')
        self.db.set_bbox(obj_id='2', x='1', y='3', width='1', height='1')
        self.db.set_bbox(obj_id='3', x='1', y='3', width='1', height='1')
        self.ans = self.db.delete_bbox_img(img_id='1')

    @print_check
    def _delete_nomix_img(self):
        self.db.set_supercategory(super_name='mix')
        self.db.set_category(super_id='2', cat_name='1', width='1', height='1', depth='1', iteration='1', thumbnail='1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='3', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='4', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='2', iteration='5', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='2', iteration='6', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='2', iteration='7', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='2', iteration='8', mix_num='-1')
        # img_id=1이면 mix, img_id=2이면 mix 아님
        self.ans = self.db.delete_nomix_img(img_id='1')

    @print_check
    def _delete_empty_obj(self):
        self.db.set_object(img_id=None, loc_id='1', cat_id=None, iteration='1', mix_num='-1')
        self.ans = self.db.delete_empty_obj()

    @print_check
    def _delete_mix_obj(self):
        self.db.set_grid(width='0', height='0')
        self.db.set_location(grid_id='2', x='0', y='0')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='1', aug_num='1')
        self.ans = self.db.delete_mix_obj(iteration='1')


class CheckAug(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------aug 함수 결과--------------------------------")
        self._get_aug_mask()
        self._get_aug_img()
        self._get_aug_loc_id()

    @print_check
    def _get_aug_mask(self):
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='100', mix_num='-1', aug_num='1')
        self.db.set_mask(obj_id='1', x='1', y='2')
        self.db.set_mask(obj_id='2', x='1', y='1')
        self.db.set_mask(obj_id='2', x='1', y='2')
        self.ans = self.db.get_aug_mask(grid_id='1', cat_id='1', aug_num='-1')

    @print_check
    def _get_aug_img(self):
        self.db.set_environment(env_id="20002", broker_ip="111.111.111.111",
                                floor="1", width="1", height="1", depth="1")
        self.db.set_image(env_id='20002', img='1ddd', type='2', check_num='3')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='100', mix_num='-1', aug_num='1')
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_object(img_id='1', loc_id='2', cat_id='1', iteration='1', mix_num='-1')
        self.db.set_object(img_id='2', loc_id='2', cat_id='1', iteration='3', mix_num='-1')
        self.ans = self.db.get_aug_img(grid_id='1', cat_id='1', aug_num='-1')

    @print_check
    def _get_aug_loc_id(self):
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_location(grid_id='1', x='2', y='1')
        self.db.set_location(grid_id='1', x='2', y='2')
        self.ans = self.db.get_aug_loc_id(grid_id='1')


class CheckSet(ResetTable):
    def __init__(self, db):
        super().__init__(db)
        self.db = db
        self.ans = None

    def check_all(self):
        print("--------------------------------set 함수 결과--------------------------------")
        self._set_obj_list()
        self._set_bulk_obj()
        self._set_bulk_bbox()
        self._set_bulk_img()
        self._set_mix_obj_list()

    @print_check
    def _set_obj_list(self):
        self.db.set_location(grid_id='1', x='1', y='2')
        self.db.set_location(grid_id='1', x='1', y='3')
        self.db.set_location(grid_id='1', x='1', y='4')
        self.ans = self.db.set_obj_list(grid_id='1', cat_id='1', iteration='3', mix_num='1')

    @time_check
    @mem_check
    @print_check
    def _set_bulk_obj(self):
        # (loc_id, cat_id, img_id, iteration, mix_num, aug_num)
        ex_table = ([('1', '1', "1", "{}".format(i), '1', '-1') for i in range(4, 65533)])
        self.ans = self.db.set_bulk_obj(datas=ex_table)

    @time_check
    @mem_check
    @print_check
    def _set_bulk_bbox(self):
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='2', mix_num='-1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='1', iteration='3', mix_num='-1')
        ex_table = ([("{}".format(i), '1', "1", '1', '1') for i in range(1, 3)])
        self.ans = self.db.set_bulk_bbox(datas=ex_table)

    @time_check
    @mem_check
    @print_check
    def _set_bulk_img(self):
        img_path = 'img'
        ex_table = (['20001', img_loader(join(img_path, img_p)), '1', '1'] for img_p in sorted(listdir(img_path)))
        self.ans = self.db.set_bulk_img(datas=ex_table)

    @print_check
    def _set_mix_obj_list(self):
        self.ans = self.db.set_mix_obj_list(max_w='2', max_h='2', loc_id='1', iteration='3', mix_num='1', aug_num='1')


class CheckDbJson(ResetTable):
    def __init__(self, db, json_path, img_path):
        super().__init__(db)
        self.db = db
        self.ans = None
        self.json_path = json_path
        self.img_path = img_path
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

    def check_all(self):
        print("--------------------------------db_to_json 함수 결과--------------------------------")
        self._db_to_json()
        self._db_to_json_type()

    @print_check
    def _db_to_json(self):
        self.db.set_supercategory(super_name='생수')
        self.db.set_category(super_id='2', cat_name='삼다수', width='1', height='1', depth='1', iteration='1', thumbnail='1')
        self.db.set_object(img_id='1', loc_id='1', cat_id='2', iteration='2', mix_num='-1')
        self.db.set_mask(obj_id='2', x='1', y='2')
        self.db.set_mask(obj_id='2', x='1', y='3')
        self.db.set_mask(obj_id='2', x='1', y='4')
        self.db.set_bbox(obj_id='2', x='1', y='1', width='1', height='1')

        self.db.update_image(img_id='1', img=img, env_id='20001')
        self.db.set_image(env_id='20001', img=img, type='0', check_num='1')
        self.db.set_image(env_id='20001', img=img, type='0', check_num='1')
        self.db.set_mask(obj_id='1', x='1', y='2')
        self.db.set_mask(obj_id='1', x='1', y='3')
        self.db.set_mask(obj_id='1', x='1', y='4')
        self.ans = self.db.db_to_json(json_path=self.json_path, img_path=self.img_path)

    @print_check
    def _db_to_json_type(self):
        self.db.set_supercategory(super_name='생수')
        self.db.set_category(super_id='2', cat_name='삼다수', width='1', height='1', depth='1', iteration='1', thumbnail='1')
        self.db.set_image(env_id='20001', img=img, type='3', check_num='99')
        self.db.set_image(env_id='20001', img=img, type='3', check_num='2')
        self.db.set_image(env_id='20001', img=img, type='3', check_num='3')
        self.db.set_object(img_id='1', loc_id='1', cat_id='2', iteration='2', mix_num='-1')
        self.db.set_object(img_id='2', loc_id='1', cat_id='2', iteration='3', mix_num='-1')
        self.db.set_mask(obj_id='2', x='1', y='2')
        self.db.set_mask(obj_id='2', x='1', y='3')
        self.db.set_mask(obj_id='2', x='1', y='4')
        self.db.set_bbox(obj_id='2', x='1', y='1', width='1', height='1')

        self.db.update_image(img_id='1', img=img, env_id='20001')
        self.db.set_mask(obj_id='1', x='1', y='2')
        self.db.set_mask(obj_id='1', x='1', y='3')
        self.db.set_mask(obj_id='1', x='1', y='4')
        self.ans = self.db.db_to_json_type(json_path=self.json_path, img_path=self.img_path, type='3')


if __name__ == "__main__":

    img_dir = 'img/test.png'
    img = img_loader(img_dir)

    # cunnect to MYSQL Server
    mydb = DB(ip='192.168.10.69',
              port=20000,
              user='root',
              password='return123',
              db_name='test')

    # reset tables
    reset_table(mydb)

    # basic 코드 test
    cb = CheckBasic(mydb)
    cb.check_all()

    # # get 함수 test
    # cg = CheckGet(mydb)
    # cg.check_all()
    #
    # # list 함수 test
    # cl = CheckList(mydb)
    # cl.check_all()
    #
    # # check 함수 test
    # ct = CheckCheck(mydb)
    # ct.check_all()
    #
    # # update 함수 test
    # cu = CheckUpdate(mydb)
    # cu.check_all()
    #
    # # delete 함수 test
    # cd = CheckDelete(mydb)
    # cd.check_all()
    #
    # # aug 함수 test
    # ca = CheckAug(mydb)
    # ca.check_all()
    #
    # # set 함수 test
    # cs = CheckSet(mydb)
    # cs.check_all()

    json_path = "./coco_info.json"
    img_path = "./img"
    cdj = CheckDbJson(db=mydb, json_path=json_path, img_path=img_path)
    cdj.check_all()