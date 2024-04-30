import subprocess

print("Making Baseline JSONS")
subprocess.run(["python", "download_elastic_for_baseline.py"])
print("Making Selective JSONS")
subprocess.run(["python", "download_elastic_for_selective_baseline.py"])

print("Making Baseline txt")
subprocess.run(["python", "readJSON.py", "baseline.txt", "Top-2000_InitialQuery"])
print("Making Selective txt")
subprocess.run(["python", "readJSON.py", "selective.txt", "Selective_JSONS"])

#Add in fine tuning script
print("Fine tuning cross encoder")
subprocess.run(["python", "finetune_crossencoder.py"])

print("Running cross encoder")
subprocess.run(["python", "cross_encoder.py", "baseline.txt", "rr_baseline.txt"]) # rr = reranked

print("Combining scores of reranked baseline and selective")
subprocess.run(["python", "combine_scores.py", "rr_baseline.txt", "selective.txt", "final_results.txt"])

print("Running evaluation")
subprocess.run(["python", "evaluation.py", "final_results.txt"])

print("Readability scores")
subprocess.run(["python", "get_readability_scores_with_json.py", "final_results.txt"])
