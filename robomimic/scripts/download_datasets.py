"""
Script to download datasets packaged with the repository. By default, all
datasets will be stored at robomimic/datasets, unless the @download_dir
argument is supplied. We recommend using the default, as most examples that
use these datasets assume that they can be found there.

The @tasks, @dataset_types, and @hdf5_types arguments can all be supplied
to choose which datasets to download.

Args:
    download_dir (str): Base download directory. Created if it doesn't exist.
        Defaults to datasets folder in repository - only pass in if you would
        like to override the location.

    tasks (list): Tasks to download datasets for. Defaults to lift task. Pass 'all' to
        download all tasks (sim + real) 'sim' to download all sim tasks, 'real' to
        download all real tasks, or directly specify the list of tasks.

    dataset_types (list): Dataset types to download datasets for (e.g. ph, mh, mg).
        Defaults to ph. Pass 'all' to download datasets for all available dataset
        types per task, or directly specify the list of dataset types.

    hdf5_types (list): hdf5 types to download datasets for (e.g. raw, low_dim, image).
        Defaults to low_dim. Pass 'all' to download datasets for all available hdf5
        types per task and dataset, or directly specify the list of hdf5 types.

    NEW:
      --num_workers (int): Number of parallel download workers (default: 8).
      --force: If set, re-download and overwrite existing files.

Example usage:

    # default behavior - just download lift proficient-human low-dim dataset
    python download_datasets.py

    # download low-dim proficient-human datasets for all simulation tasks
    # (do a dry run first to see which datasets would be downloaded)
    python download_datasets.py --tasks sim --dataset_types ph --hdf5_types low_dim --dry_run
    python download_datasets.py --tasks sim --dataset_types ph --hdf5_types low_dim

    # download all low-dim and image multi-human datasets for the can and square tasks
    python download_datasets.py --tasks can square --dataset_types mh --hdf5_types low_dim image

    # download the sparse reward machine-generated low-dim datasets
    python download_datasets.py --tasks all --dataset_types mg --hdf5_types low_dim_sparse

    # download all real robot datasets
    python download_datasets.py --tasks real

    # parallelism and overwriting
    python download_datasets.py --tasks all --dataset_types all --hdf5_types all --num_workers 16 --force
"""
import os
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

import robomimic
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY, HF_REPO_ID

ALL_TASKS = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]
ALL_DATASET_TYPES = ["ph", "mh", "mg", "paired"]
ALL_HDF5_TYPES = ["raw", "low_dim", "image", "low_dim_sparse", "low_dim_dense", "image_sparse", "image_dense"]


@dataclass(frozen=True)
class DownloadJob:
    task: str
    dataset_type: str
    hdf5_type: str
    url: str
    download_dir: str  # directory to place file(s)
    target_path: str   # expected primary target file path (download_dir / basename(url))
    is_real: bool      # whether "real" in task (Stanford host) vs sim (HF)


def _normalize_args_tasks(tasks: List[str]) -> List[str]:
    if "all" in tasks:
        assert len(tasks) == 1, f"all should be only tasks argument but got: {tasks}"
        return ALL_TASKS
    if "sim" in tasks:
        assert len(tasks) == 1, f"sim should be only tasks argument but got: {tasks}"
        return [t for t in ALL_TASKS if "real" not in t]
    if "real" in tasks:
        assert len(tasks) == 1, f"real should be only tasks argument but got: {tasks}"
        return [t for t in ALL_TASKS if "real" in t]
    return tasks


def _normalize_args_list(arg_list: List[str], all_values: List[str], arg_name: str) -> List[str]:
    if "all" in arg_list:
        assert len(arg_list) == 1, f"all should be only {arg_name} argument but got: {arg_list}"
        return all_values
    return arg_list


def _expected_target_path(download_dir: str, url: str) -> str:
    """
    We assume the main downloaded artifact is named by the basename of the URL / HF repo path.
    """
    base = os.path.basename(url.rstrip("/"))
    return os.path.join(download_dir, base)


def _build_jobs(default_base_dir: str,
                tasks: List[str],
                dataset_types: List[str],
                hdf5_types: List[str]) -> List[DownloadJob]:
    jobs: List[DownloadJob] = []
    for task in DATASET_REGISTRY:
        if task not in tasks:
            continue
        for dset_type in DATASET_REGISTRY[task]:
            if dset_type not in dataset_types:
                continue
            for h5_type in DATASET_REGISTRY[task][dset_type]:
                if h5_type not in hdf5_types:
                    continue
                url = DATASET_REGISTRY[task][dset_type][h5_type]["url"]
                download_dir = os.path.abspath(os.path.join(default_base_dir, task, dset_type))
                target_path = _expected_target_path(download_dir, url) if url else ""
                jobs.append(
                    DownloadJob(
                        task=task,
                        dataset_type=dset_type,
                        hdf5_type=h5_type,
                        url=url,
                        download_dir=download_dir,
                        target_path=target_path,
                        is_real=("real" in task),
                    )
                )
    return jobs


def _describe_job(job: DownloadJob) -> str:
    return (
        f"\nDownloading dataset:\n"
        f"    task: {job.task}\n"
        f"    dataset type: {job.dataset_type}\n"
        f"    hdf5 type: {job.hdf5_type}\n"
        f"    download path: {job.download_dir}\n"
        f"    target file: {job.target_path if job.target_path else '<none>'}"
    )


def _download_one(job: DownloadJob, force: bool) -> Tuple[DownloadJob, str, bool]:
    """
    Returns (job, message, success_flag).
    On skip, success_flag=True since it's an intended outcome.
    """
    # No URL means "generate locally" – skip with a message.
    if job.url is None:
        msg = (
            f"Skipping {job.task}-{job.dataset_type}-{job.hdf5_type}, no url for dataset exists. "
            f"Create this dataset locally by running the appropriate command from "
            f"robomimic/scripts/extract_obs_from_raw_datasets.sh."
        )
        return job, msg, True

    # Ensure directory exists
    os.makedirs(job.download_dir, exist_ok=True)

    # Skip if exists and not forcing
    if (not force) and os.path.isfile(job.target_path):
        return job, f"Already exists, skipping (use --force to overwrite): {job.target_path}", True

    try:
        if job.is_real:
            # Real-world datasets (Stanford-hosted)
            # FileUtils.download_url has signature accepting url + download_dir.
            # We handle overwrite ourselves by deleting existing file if necessary.
            if force and os.path.isfile(job.target_path):
                try:
                    os.remove(job.target_path)
                except OSError:
                    pass
            FileUtils.download_url(
                url=job.url,
                download_dir=job.download_dir,
            )
        else:
            # Simulation datasets (Hugging Face)
            # Respect force by passing check_overwrite=force if supported,
            # but we also pre-delete for safety/consistency.
            if force and os.path.isfile(job.target_path):
                try:
                    os.remove(job.target_path)
                except OSError:
                    pass
            FileUtils.download_file_from_hf(
                repo_id=HF_REPO_ID,
                filename=job.url,
                download_dir=job.download_dir,
                check_overwrite=force,
            )
        return job, f"Downloaded to {job.download_dir}", True
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return job, f"ERROR: {e}\n{tb}", False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Base download directory. Created if it doesn't exist. Defaults to datasets folder in repository.",
    )

    # tasks to download datasets for
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=["lift"],
        help="Tasks to download datasets for. Defaults to lift task. Pass 'all' to download all tasks (sim + real) \
            'sim' to download all sim tasks, 'real' to download all real tasks, or directly specify the list of \
            tasks.",
    )

    # dataset types to download datasets for
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs='+',
        default=["ph"],
        help="Dataset types to download datasets for (e.g. ph, mh, mg). Defaults to ph. Pass 'all' to download \
            datasets for all available dataset types per task, or directly specify the list of dataset types.",
    )

    # hdf5 types to download datasets for
    parser.add_argument(
        "--hdf5_types",
        type=str,
        nargs='+',
        default=["low_dim"],
        help="hdf5 types to download datasets for (e.g. raw, low_dim, image). Defaults to raw. Pass 'all' \
            to download datasets for all available hdf5 types per task and dataset, or directly specify the list \
            of hdf5 types.",
    )

    # dry run - don't actually download datasets, but print which datasets would be downloaded
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="set this flag to do a dry run to only print which datasets would be downloaded",
    )

    # NEW: number of parallel workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )

    # NEW: force overwrite existing files
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, re-download and overwrite existing files",
    )

    args = parser.parse_args()

    # set default base directory for downloads
    default_base_dir = args.download_dir
    if default_base_dir is None:
        default_base_dir = os.path.join(robomimic.__path__[0], "../datasets")

    # load args
    download_tasks = _normalize_args_tasks(args.tasks)
    download_dataset_types = _normalize_args_list(args.dataset_types, ALL_DATASET_TYPES, "dataset_types")
    download_hdf5_types = _normalize_args_list(args.hdf5_types, ALL_HDF5_TYPES, "hdf5_types")

    # build job list
    jobs = _build_jobs(
        default_base_dir=default_base_dir,
        tasks=download_tasks,
        dataset_types=download_dataset_types,
        hdf5_types=download_hdf5_types,
    )

    if not jobs:
        print("No matching datasets found for the provided arguments.")
        exit(0)

    # Print a summary, respecting dry_run
    for job in jobs:
        print(_describe_job(job))
        if job.url is None:
            print("    note: no url for this dataset; generate locally via scripts mentioned above.")
        elif args.dry_run:
            # Show whether we'd skip due to existing file
            exists = os.path.isfile(job.target_path)
            status = "would re-download (force)" if (exists and args.force) else ("would skip (exists)" if exists else "would download")
            print(f"    dry run: {status}")

    if args.dry_run:
        print("\nDry run complete. No downloads performed.")
        exit(0)

    # Execute in parallel
    print(f"\nStarting downloads with {args.num_workers} worker(s)...\n")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        future_map = {ex.submit(_download_one, job, args.force): job for job in jobs if job.url is not None}
        # also account for URL-less jobs as successful skips
        for job in jobs:
            if job.url is None:
                print(f"[SKIP] {job.task}/{job.dataset_type}/{job.hdf5_type}: no url (generate locally).")
                success_count += 1

        for fut in as_completed(future_map):
            job = future_map[fut]
            try:
                _job, msg, ok = fut.result()
                tag = "OK" if ok else "FAIL"
                print(f"[{tag}] {job.task}/{job.dataset_type}/{job.hdf5_type}: {msg}")
                success_count += 1 if ok else 0
                fail_count += 0 if ok else 1
            except Exception as e:
                fail_count += 1
                print(f"[FAIL] {job.task}/{job.dataset_type}/{job.hdf5_type}: Unexpected error: {e}")

    total = len(jobs)
    print(f"\nDone. {success_count}/{total} completed (including intentional skips); {fail_count} failed.")
