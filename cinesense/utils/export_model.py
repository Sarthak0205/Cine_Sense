import os
import hashlib
from pathlib import Path
from cinesense.recommenders.two_stage import CineSenseTwoStage
from cinesense.utils.model_storage import save_model
from evaluation.datasets import (
    ITEM_ID_COL,
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)


def compute_file_sha256(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    print("Loading datasets...", flush=True)
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog[ITEM_ID_COL].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)

    # We fit on all eligible eval users in the dataset split
    eval_users = build_eval_users(split, use_validation=True)
    eval_user_ids = [user.user_id for user in eval_users]

    print("Fitting champion CineSenseTwoStage model...", flush=True)
    model = CineSenseTwoStage()
    model.fit(catalog, split.train, user_ids=eval_user_ids)
    print("Model fitted successfully.", flush=True)

    # Compute catalog file SHA-256
    catalog_path = Path("archive-2/animes.csv")
    catalog_sha = compute_file_sha256(catalog_path)
    catalog_version = f"sha256_{catalog_sha}"

    # Compute embedding array SHA-256
    if model.catalog_embeddings is None:
        raise RuntimeError("Embeddings were not precomputed during fit.")

    embedding_hash = hashlib.sha256(model.catalog_embeddings.tobytes()).hexdigest()
    embedding_version = f"{model.embedding_model_name}_sha256_{embedding_hash}"

    # Save the model
    output_dir = "cinesense/models/twostage_v1"
    print(f"Exporting serialized model assets to {output_dir}...", flush=True)
    save_model(
        model=model,
        catalog_df=model.catalog,
        dir_path=output_dir,
        model_version="twostage_v1",
        catalog_version=catalog_version,
        embedding_version=embedding_version,
    )
    print("Model export completed successfully.", flush=True)


if __name__ == "__main__":
    main()
