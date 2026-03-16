# raw_dataloader.py
# Loads and filters FMA dataset for rock genres

from datasets import load_dataset
import yaml
from pathlib import Path

def load_genre_config(config_path="configs/genres.yaml"):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['rock_genres'], config.get('genre_categories', {})

def filter_rock_songs(dataset, rock_genre_ids):
    def is_rock(example):
        genres = example['genres']
        if isinstance(genres, list):
            return bool(set(rock_genre_ids) & set(genres))
        return genres in rock_genre_ids
    
    return dataset.filter(is_rock)

def get_genre_distribution(metadata, rock_genres):
    genre_counts = {gid: 0 for gid in rock_genres.keys()}
    
    for i in range(len(metadata)):
        genres = metadata[i]['genres']
        genre_list = genres if isinstance(genres, list) else [genres]
        for gid in genre_list:
            if gid in genre_counts:
                genre_counts[gid] += 1
    
    return dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))

def main():
    rock_genres, categories = load_genre_config()
    rock_genre_ids = set(rock_genres.keys())
    print(f"✓ {len(rock_genres)} rock genres loaded\n")
    
    dataset = load_dataset("benjamin-paine/free-music-archive-small", split="train")
    print(f"✓ {len(dataset):,} total samples\n")
    
    dataset_meta = dataset.remove_columns(['audio'])
    rock_meta = filter_rock_songs(dataset_meta, rock_genre_ids)
    
    rock_count = len(rock_meta)
    
    if rock_count > 0:
        # Genre distribution
        genre_dist = get_genre_distribution(rock_meta, rock_genres)
        for gid, count in list(genre_dist.items())[:10]:
            if count > 0:
                print(f"  {rock_genres[gid]:15}: {count:4}")
        
        # Get full dataset with audio
        print("\nLoading audio data...")
        rock_indices = [
            i for i in range(len(dataset))
            if bool(set(rock_genre_ids) & set(
                dataset[i]['genres'] if isinstance(dataset[i]['genres'], list) 
                else [dataset[i]['genres']]
            ))
        ]
        
        rock_dataset = dataset.select(rock_indices)
        
        # Save filtered dataset
        output_path = Path("data/rock_dataset")
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving to {output_path}...")
        rock_dataset.save_to_disk(str(output_path))
        rock_meta.to_csv(output_path / "metadata.csv")
        return rock_dataset
    else:
        
        return None

if __name__ == "__main__":
    rock_dataset = main()
    print("\n✓ Complete!")