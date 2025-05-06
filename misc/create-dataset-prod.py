# import shutil
# !pip install -U git+https://github.com/huggingface/transformers.git
# !pip install -U git+https://github.com/huggingface/accelerate.git
# !pip install -U git+https://github.com/huggingface/datasets.git
# !pip install -U git+https://github.com/huggingface/evaluate.git
# !mkdir -p cache

from tqdm import tqdm
import tempfile
import boto3
from datasets import Dataset
import json
import os, shutil
import concurrent.futures

session = boto3.Session(profile_name='personal')

def download_and_parse_s3_file(bucket_name, file_key):
    s3 = session.resource('s3')
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    s3.Bucket(bucket_name).download_file(file_key, temp_file.name)

    with open(temp_file.name, 'r') as f:
        data = json.load(f)

    return data


def download_files_with_keys(bucket_name, keys, download_dir, cache_dir):
    s3 = session.resource('s3')

    for file_key in keys:
        local_file_path = os.path.join(download_dir, os.path.basename(file_key))
        cache_file_path = os.path.join(cache_dir, os.path.basename(file_key))

        if os.path.exists(local_file_path):
            # if not os.path.exists(cache_file_path):
            #     shutil.copy(local_file_path, cache_file_path)
            continue

        if os.path.exists(cache_file_path):
            shutil.copy(cache_file_path, local_file_path)
            os.remove(cache_file_path)
            continue

        s3.Bucket(bucket_name).download_file(file_key, local_file_path)
        # shutil.copy(local_file_path, cache_file_path)


def filter_entires(data, first_n):
    filtered = []
    for entry in data:
        if entry['similarities']['MOBILE'] < 0.99:
            continue
        if entry['areThereBlankScreenshots']:
            continue
        # if entry['isAugmented']:
        #     continue
        filtered.append(entry)

        if len(filtered) >= first_n:
            break

    return filtered

def download_and_save_files_for_entry(bucket_name, run_id, entry):
    with (tempfile.TemporaryDirectory() as temp_dir):
        try:
            download_files_with_keys(bucket_name, [
                'crawls/' + run_id + '/' + entry['prefix'] + '-MOBILE-svg-clean.svg',
                'crawls/' + run_id + '/' + entry['prefix'] + '-TABLET-svg-clean.svg',
                'crawls/' + run_id + '/' + entry['prefix'] + '-DESKTOP-svg-clean.svg',
                'crawls/' + run_id + '/' + entry['prefix'] + '-page-composite.html',
            ], temp_dir, "cache")
        except Exception as e:
            print(f"Failed to download files for entry {entry['prefix']}: {e}")
            return None

        with open(os.path.join(temp_dir, entry['prefix'] + '-MOBILE-svg-clean.svg'), 'r') as file:
            svg_mobile = file.read()

        with open(os.path.join(temp_dir, entry['prefix'] + '-TABLET-svg-clean.svg'), 'r') as file:
            svg_tablet = file.read()

        with open(os.path.join(temp_dir, entry['prefix'] + '-DESKTOP-svg-clean.svg'), 'r') as file:
            svg_desktop = file.read()

        with open(os.path.join(temp_dir, entry['prefix'] + '-page-composite.html'), 'r') as file:
            html = file.read()

    return {
        "svg_mobile": svg_mobile,
        "svg_tablet": svg_tablet,
        "svg_desktop": svg_desktop,

        "html": html,

        "url": entry['url'],

        "msps_mobile": entry['similarities']['MOBILE'],
        "msps_tablet": entry['similarities']['TABLET'],
        "msps_desktop": entry['similarities']['DESKTOP'],

        "is_augmented": entry['isAugmented'],
        "has_media_queries": entry['hasMediaQueries'],

        "page_sizes_desktop_width": entry["pageSizes"]["DESKTOP"]["width"],
        "page_sizes_desktop_height": entry["pageSizes"]["DESKTOP"]["height"],

        "page_sizes_tablet_width": entry["pageSizes"]["TABLET"]["width"],
        "page_sizes_tablet_height": entry["pageSizes"]["TABLET"]["height"],

        "page_sizes_mobile_width": entry["pageSizes"]["MOBILE"]["width"],
        "page_sizes_mobile_height": entry["pageSizes"]["MOBILE"]["height"],

        "svg_lengths_desktop": entry["svgLengths"]["DESKTOP"],
        "svg_lengths_tablet": entry["svgLengths"]["TABLET"],
        "svg_lengths_mobile": entry["svgLengths"]["MOBILE"],
        "page_data_size": entry["pageDataSize"]
    }


def s3_dataset_generator(run_ids, first_n):
    bucket_name = 'reverse-browser'
    run_id_counter = 1

    for run_id in run_ids:
        if first_n <= 0:
            break

        file_key = 'crawls/' + run_id + '/dataset.json'
        data = download_and_parse_s3_file(bucket_name, file_key)

        filtered_data = filter_entires(data, first_n)
        first_n = first_n - len(filtered_data)

        chunk_size = 2500
        for i in range(0, len(filtered_data), chunk_size):
            filtered_data_chunk =  filtered_data[i:i + chunk_size]

            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                futures = set()
                for entry in filtered_data_chunk:
                    futures.add(executor.submit(download_and_save_files_for_entry, bucket_name, run_id, entry))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            yield result
                    except Exception as e:
                        print('%r generated an exception: %s' % (e))

        run_id_counter += 1


dataset = Dataset.from_generator(s3_dataset_generator, gen_kwargs={"run_ids":
    [
        "20241023-1",
        "20241025-1",
        "20241025-2",
        "20241025-3",
        "20241025-4",
        "20241025-5",
        "20241025-6",
        "20241025-7",
        "20241025-8",
        "20241025-9",
        "20241026-1",
        "20241101-1",
        "20241101-2",
        "20241101-3",
        "20241101-4",
        "20241101-5",
        "20241101-6",
        "20241101-7",
        "20241101-8",
        "20241101-9",
        "20241101-10"
    ], "first_n": 10000000})

print(dataset)

save_path = 'data-rb-large-prod'
dataset.save_to_disk(save_path)

