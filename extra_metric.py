import torch
from thop import profile
from ultralytics import YOLO
import numpy as np

device = torch.device("cpu")
#input_shape of model,batch_size=1
net = YOLO("runs/detect/train3/weights/best.pt").model ##定义好的网络模型

input = torch.randn(1, 3, 640, 640)
flops, params = profile(net, inputs=(input, ))
flops = 2 * flops
print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
print("Params=", str(params/1e6)+'{}'.format("M"))

total = sum([param.nelement() for param in net.parameters()])
print("parameter:%fM" % (total/1e6))

model = YOLO("runs/detect/train3/weights/best.pt").model
device = torch.device('cuda')
model.to(device)
model.eval()
dummy_input = torch.randn(1, 3, 640, 640,dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print(mean_syn)
