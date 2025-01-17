import platform
import os

# model and dataset save dir
# save_path = "D:/data_2"
save_path = "/data_2"

# minio 配置
service = "minio:9090"
access_key = "minioadmin"
secret_key = "minioadmin"

# 评测结果所在目录
RESULT_DIR = "result"


# 获取当前操作系统 平台
def get_platform():
    return platform.system().lower()


# 获取结果存放目录
def get_result_file_path(configFilePath):
    global configFilePath_splits
    global result_dir
    cur_platform = get_platform()
    if cur_platform == 'windows':
        configFilePath_splits = configFilePath.split('\\')  # 分隔符
        configFilePath_splits[-2] = RESULT_DIR  # 更换目录
        result_dir = "\\".join(configFilePath_splits[:-1])
    elif cur_platform == 'linux':
        configFilePath_splits = configFilePath.split(r'/')  # 分隔符
        configFilePath_splits[-2] = RESULT_DIR  # 更换目录
        result_dir = r"/".join(configFilePath_splits[:-1])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    global resultFilePath
    # 结果界面
    if cur_platform == 'windows':
        resultFilePath = "\\".join(configFilePath_splits)
    elif cur_platform == 'linux':
        resultFilePath = '/'.join(configFilePath_splits)

    return resultFilePath
