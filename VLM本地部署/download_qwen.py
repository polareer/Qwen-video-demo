from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")
print("模型下载到：", model_dir)