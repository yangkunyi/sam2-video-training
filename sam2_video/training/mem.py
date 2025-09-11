# mem_snap.py
import torch
import traceback
from contextlib import contextmanager

@contextmanager
def snap_when_peak(delta_gb=2.0, top_k=3):
    """
    上下文管理器：
    进入时记录当前显存峰值，
    退出时如果峰值上涨超过 delta_gb，就打印上涨栈 & TOP-K 大 Tensor
    """
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        new_peak = torch.cuda.max_memory_allocated()
        if (new_peak - peak) / 1024**3 > delta_gb:
            print(f"\n>>> 峰值上涨 {(new_peak-peak)/1024**3:.2f} GB")
            print(">>> 分配栈（最近 6 帧）")
            traceback.print_stack(limit=30)
            print(">>> TOP-K 大 Tensor")
            for i, (obj, sz) in enumerate(_top_tensors(top_k), 1):
                print(f"  {i}  {obj}  {sz/1024**3:.3f} GB")

def _top_tensors(k=3):
    import gc
    gc.collect()
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((tuple(obj.shape), obj.numel() * obj.element_size()))
        except Exception:
            continue
    tensors.sort(key=lambda x: x[1], reverse=True)
    return tensors[:k]