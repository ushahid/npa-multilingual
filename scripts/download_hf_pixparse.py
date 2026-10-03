"""Download the pixparse CC12M WebDataset snapshot from Hugging Face."""

import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dir", default="downloads/conceptual12m/cc12m_pixparse_wds")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="pixparse/cc12m-wds",
        repo_type="dataset",
        local_dir=args.local_dir,
    )
    print("Finished snapshot_download to", args.local_dir)


if __name__ == "__main__":
    main()
