import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import json2dict


def main(args):
    print(f"Loading Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()

    query_map = json2dict(f"data/{args.data_type}/query_map.json")

    df = pd.read_csv(f"data/{args.data_type}/{args.split}.csv")
    query_ids = df["query_id"].unique()

    results = []
    print(
        f"Extracting confidence for {len(query_ids)} queries in {args.split} split..."
    )

    for q_id in tqdm(query_ids):
        query_text = query_map.get(str(q_id), "")
        if not query_text:
            continue

        prompt = f"Question: {query_text}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            top_prob, _ = torch.max(probs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            results.append(
                {
                    "query_id": q_id,
                    "top_logit_prob": top_prob.item(),
                    "logit_entropy": entropy.item(),
                }
            )

    output_path = f"data/{args.data_type}/{args.split}_14b_conf.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Successfully saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to 14B model"
    )
    parser.add_argument("--data_type", type=str, default="triviaqa_full")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    args = parser.parse_args()
    main(args)
