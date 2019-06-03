import torch
import gc
from functools import reduce

def profile():
    total_memory = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                memory = obj.element_size() * obj.nelement()
                print(type(obj), memory)
                total_memory = total_memory + memory
        except:
            pass
    print("End of Profile: " + str(total_memory))

def profile_gb():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if len(obj.size()) > 0:
                    if obj.type() == 'torch.cuda.FloatTensor':
                        total += reduce(lambda x, y: x*y, obj.size()) * 32
                    elif obj.type() == 'torch.cuda.LongTensor':
                        total += reduce(lambda x, y: x*y, obj.size()) * 64
                    elif obj.type() == 'torch.cuda.IntTensor':
                        total += reduce(lambda x, y: x*y, obj.size()) * 32
                    #else:
                    # Few non-cuda tensors in my case from dataloader
        except Exception as e:
            pass
    print("{} GB".format(total/((1024**3) * 8)))