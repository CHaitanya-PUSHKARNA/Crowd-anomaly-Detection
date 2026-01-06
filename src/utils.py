import os
import random
import shutil

VIDEO_EXTS = (".mp4", ".avi", ".mov")

def create_mini_dataset(
    raw_root="data/raw/Violence Fight Detection dataset",
    mini_root="data/mini",
    fraction=0.1,
    seed=42
):
    random.seed(seed)

    fight_dst = os.path.join(mini_root, "fight")
    nonfight_dst = os.path.join(mini_root, "nonfight")

    os.makedirs(fight_dst, exist_ok=True)
    os.makedirs(nonfight_dst, exist_ok=True)

    datasets = ["RLVS", "RWF-2000"]
    splits = ["train", "val"]
    class_map = {
        "Fight": fight_dst,
        "NonFight": nonfight_dst
    }

    for dataset in datasets:
        for split in splits:
            for cls_name, dst_dir in class_map.items():
                src_dir = os.path.join(
                    raw_root,
                    dataset,
                    split,
                    cls_name
                )

                if not os.path.exists(src_dir):
                    continue

                videos = [
                    v for v in os.listdir(src_dir)
                    if v.lower().endswith(VIDEO_EXTS)
                ]

                if len(videos) == 0:
                    continue

                k = max(1, int(len(videos) * fraction))
                sampled = random.sample(videos, k)

                for video in sampled:
                    src_path = os.path.join(src_dir, video)

                    # prevent filename collision
                    new_name = f"{dataset}_{split}_{cls_name}_{video}"
                    dst_path = os.path.join(dst_dir, new_name)

                    shutil.copy2(src_path, dst_path)

                print(
                    f"{dataset}/{split}/{cls_name}: "
                    f"copied {k}/{len(videos)}"
                )



# import os
# import random
# import shutil

# def create_mini_dataset(src_dir, dst_dir, fraction=0.1):
#     """
#     Creates a smaller dataset for fast testing.
#     """
#     os.makedirs(dst_dir, exist_ok=True)

#     for cls in ["fight", "nonfight"]:
#         src_cls = os.path.join(src_dir, cls)
#         dst_cls = os.path.join(dst_dir, cls)

#         os.makedirs(dst_cls, exist_ok=True)

#         videos = os.listdir(src_cls)
#         k = max(1, int(len(videos) * fraction))
#         sampled = random.sample(videos, k)

#         for video in sampled:
#             shutil.copy(
#                 os.path.join(src_cls, video),
#                 os.path.join(dst_cls, video)
#             )
