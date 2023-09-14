import time

def get_time_str():
    return "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])