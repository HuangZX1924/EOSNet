import pickle

# 从文件加载变量
with open('YOLOC.pkl', 'rb') as file:
    px, py, ap, f1, p, r = pickle.load(file)
    print(px, py, ap, f1, p, r)