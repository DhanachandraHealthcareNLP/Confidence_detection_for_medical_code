import csv
import json
import os.path
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm


def read_label_data(csv_file):
    # Nested defaultdict structure
    label_info_dict = defaultdict(lambda: defaultdict(str))

    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading label csv file!"):
            # print(row)
            doc_name = str(row["doc_name"]).strip()
            dt = datetime.strptime(str(row["DOS"]).strip(), "%Y-%m-%d")
            # Reformat
            dos = dt.strftime("%m/%d/%Y")
            code = str(row["code"]).strip()
            code_status = str(row["CODE_STATUS"]).strip()

            # Build nested structure
            label_info_dict[doc_name][(dos, code)] = code_status
    print(f"Label for {len(label_info_dict)} of documents extracted!")
    return label_info_dict


def read_json_file(file_path: str, label_info_dict: defaultdict):
    datapoint = []
    with open(file_path) as f:
        data = json.load(f)
    dos_list = data["result"]["dosList"]
    for dos_data in dos_list:
        dos = dos_data["dos"]
        for icd10cmcode_data in dos_data["icd10cmcodes"]:
            code = icd10cmcode_data["code"]
            justification_list = []
            monitoring_evidence_list = []
            evaluation_evidence_list = []
            assessment_evidence_list = []
            treatment_evidence_list = []
            for code_detials in icd10cmcode_data["code_combination_details"]:
                justification = code_detials["justification"]
                meat_evidence_data = code_detials["meatEvidence"]
                internal_code_monitoring_text = [monit["text"].strip() for monit in meat_evidence_data["Monitoring"]]
                internal_code_eval_text = [eval["text"].strip() for eval in meat_evidence_data["Evaluation"]]
                internal_code_acmnt_text = [acmnt["text"].strip() for acmnt in meat_evidence_data["Assessment"]]
                internal_code_trtmnt_text = [trtmnt["text"].strip() for trtmnt in meat_evidence_data["Treatment"]]
                justification_list.append(justification)
                monitoring_evidence_list.append(internal_code_monitoring_text)
                evaluation_evidence_list.append(internal_code_eval_text)
                assessment_evidence_list.append(internal_code_acmnt_text)
                treatment_evidence_list.append(internal_code_trtmnt_text)

            label = label_info_dict[(dos, code)]
            if not label:
                print(f"Label not found for DOS: {dos}, Code: {code} and File path: {file_path}")
                continue

            datapoint.append((code, justification_list, monitoring_evidence_list, evaluation_evidence_list,
                              assessment_evidence_list, treatment_evidence_list, label))

    return datapoint


# Example usage
if __name__ == "__main__":
    csv_path = "../data/Sentara Model Training .csv"
    json_path = "../data/Sentara_UI_JSON-20250915T101150Z-1-001/Sentara_UI_JSON/Batch_1/SHP_MA_MRR_2024DOS_900032653_01_1316997356_06112025/final_merged_result.json"
    label_info_dict = read_label_data(csv_path)
    datapoint = read_json_file(json_path, label_info_dict["SHP_MA_MRR_2024DOS_900032653_01_1316997356_06112025"])
    print(len(datapoint))
    print(datapoint)

    # Pretty-print a sample
    # for doc, dos_dict in label_info_dict.items():
    #     print(f"Document: {doc}")
    #     for (dos, code), code_status in label_info_dict.items():
    #         print(f"  DOS: {dos}")
    #         print(f"{code} -> {code_status}")
