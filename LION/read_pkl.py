import pickle

# 读取文件
with open('/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_data_space/Special_Dataset_v1/nuscenes_infos_10sweeps_train.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看类型和长度
print(f"数据类型: {type(data)}")
print(f"列表长度: {len(data)}")

# 查看第一个元素的内容
if len(data) > 0:
    print(f"第一个元素类型: {type(data[0])}")
    print(f"第一个元素内容: {data[0]}")