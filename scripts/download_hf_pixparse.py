from huggingface_hub import snapshot_download

local_dir = "downloads/conceptual12m/cc12m_pixparse_wds"

snapshot_download(
    repo_id="pixparse/cc12m-wds",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # keep real files in that folder
)
print("Finished snapshot_download to", local_dir)