import pandas as pd
import torch
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
import unicodedata
import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def main():
    # ----------------------
    # Start Dask cluster
    # ----------------------
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(client)

    # ----------------------
    # Excel input/output
    # ----------------------
    input_file = "input_10.xlsx"
    output_file = "output_nemo.xlsx"

    df = pd.read_excel(input_file)
    df["original_text"] = df["original_text"].apply(normalize_text)

    all_masked_texts = []

    # ----------------------
    # Iterate over rows (no chunking)
    # ----------------------
    for idx, text in enumerate(df["original_text"]):
        print(f"Processing row {idx+1}...")
        df_chunk = pd.DataFrame({'text': [text]})
        ddf_chunk = dd.from_pandas(df_chunk, npartitions=1)
        dataset = DocumentDataset(ddf_chunk)

        # Create modifier to only mask PII
        modifier = PiiModifier(
            language="en",
            supported_entities=["PERSON", "ADDRESS", "DATE_TIME"],
            anonymize_action="replace",  # mask with <ENTITY> tags
            return_decision=False,       # we don't need detected entities
            batch_size=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        modify_pipeline = Modify(modifier)

        masked_dataset = modify_pipeline(dataset)
        masked_df = masked_dataset.df.compute()

        # Only save masked text
        masked_text = masked_df['text'].iloc[0]
        all_masked_texts.append(masked_text)

    # ----------------------
    # Save results
    # ----------------------
    df["Masked_Text"] = all_masked_texts
    df.to_excel(output_file, index=False)
    print(f"Masking finished. Results saved to {output_file}")

if __name__ == "__main__":
    main()
