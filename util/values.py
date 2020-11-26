# -*- coding: utf-8 -*-
# @Time: 2020/11/26 11:07
# @Author: Rollbear
# @Filename: values.py


class DataProperty:
    feature_name = [
        "account_length",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "total_day_minutes",
        "total_day_calls",
        "total_day_charge",
        "total_eve_minutes",
        "total_eve_calls",
        "total_eve_charge",
        "total_night_minutes",
        "total_night_calls",
        "total_night_charge",
        "total_intl_minutes",
        "total_intl_calls",
        "total_intl_charge",
        "number_customer_service_calls"
    ]
    target_name = ["0", "1"]


class ResourceRoot:
    graphviz_home = 'D:/Graphviz/bin'
    exp_resource_root = "../Exp/resource/"

