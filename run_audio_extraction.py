"""
Full audio-only re-extraction for all 4097 AVE videos.
Overwrites data/features/audio/*.pt directly (existing zeros have no value).
Does NOT touch data/features/video/.
Writes audio_extraction_failures.txt on completion.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))

import torch
import config, utils
from feature_extractor import build_vggish, get_device, extract_audio_features

FAILURES_FILE = os.path.join(os.path.dirname(__file__), "audio_extraction_failures.txt")
LOG_INTERVAL  = 200

def main():
    device = get_device()
    print(f"Device: {device}", flush=True)

    print("Loading VGGish from local cache …", flush=True)
    vggish = build_vggish(device)
    print("VGGish ready.\n", flush=True)

    # Collect all unique video IDs across all splits
    all_samples = []
    for sf in [config.TRAIN_SET_FILE, config.VAL_SET_FILE, config.TEST_SET_FILE]:
        all_samples.extend(utils.load_split_file(sf))

    seen, unique = set(), []
    for s in all_samples:
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            unique.append(s)

    total     = len(unique)
    audio_dir = os.path.join(config.FEATURES_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"Starting audio re-extraction for {total} videos …", flush=True)
    print(f"Output: {audio_dir}", flush=True)
    print("-" * 60, flush=True)

    succeeded, failed = 0, []
    t_global = time.time()

    for i, sample in enumerate(unique, 1):
        vid      = sample["video_id"]
        mp4_path = utils.get_video_path(vid)
        out_path = os.path.join(audio_dir, f"{vid}.pt")

        if not os.path.exists(mp4_path):
            failed.append((vid, "mp4 not found"))
            if i % LOG_INTERVAL == 0 or i == total:
                elapsed = time.time() - t_global
                print(f"  [{i:4d}/{total}] elapsed {elapsed:.0f}s  "
                      f"ok={succeeded}  fail={len(failed)}", flush=True)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feat = extract_audio_features(mp4_path, vggish, device)

            # Validate before saving
            if feat.shape != (config.NUM_SEGMENTS, config.AUDIO_EMBED_DIM):
                raise ValueError(f"unexpected shape {tuple(feat.shape)}")
            if feat.abs().mean().item() < 1e-9:
                raise ValueError("all-zero output — ffmpeg decode likely failed silently")
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                raise ValueError("NaN/Inf in output")

            torch.save(feat, out_path)
            succeeded += 1

        except Exception as e:
            failed.append((vid, str(e)))

        if i % LOG_INTERVAL == 0 or i == total:
            elapsed = time.time() - t_global
            rate    = i / elapsed if elapsed > 0 else 0
            eta_s   = (total - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{total}]  elapsed {elapsed:5.0f}s  "
                  f"ok={succeeded:4d}  fail={len(failed):3d}  "
                  f"rate={rate:.2f}/s  ETA {eta_s/60:.1f}min", flush=True)

    elapsed_total = time.time() - t_global

    # Write failures file
    with open(FAILURES_FILE, "w", encoding="utf-8") as f:
        if failed:
            f.write(f"# Audio extraction failures — {len(failed)} of {total} videos\n")
            for vid, reason in failed:
                f.write(f"{vid}\t{reason}\n")
        else:
            f.write("no failures\n")

    print("\n" + "=" * 60, flush=True)
    print(f"DONE", flush=True)
    print(f"  Total videos    : {total}", flush=True)
    print(f"  Succeeded       : {succeeded}", flush=True)
    print(f"  Failed          : {len(failed)}", flush=True)
    print(f"  Total time      : {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)", flush=True)
    print(f"  Failures file   : {FAILURES_FILE}", flush=True)
    if failed:
        print(f"\nFailed video_ids:", flush=True)
        for vid, reason in failed:
            print(f"  {vid}: {reason}", flush=True)

if __name__ == "__main__":
    main()
