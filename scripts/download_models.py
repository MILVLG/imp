import os
# os.environ["https_proxy"] = "http://xxx.xxx.xxx.xxx:xx"  # in case you need proxy to access Huggingface Hub
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/phi-2", 
    revision="d3186761bf5c4409f7679359284066c25ab668ee",
    local_dir='/home/ouyangxc/ckpts/base/phi-2',
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="google/siglip-so400m-patch14-384", 
    revision="7067f6db2baa594bab7c6d965fe488c7ac62f1c8",
    local_dir='/home/ouyangxc/ckpts/base/siglip-so400m-patch14-384',
    local_dir_use_symlinks=False
)